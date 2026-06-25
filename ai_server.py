from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "trained_model" / "samvaad_isl_model.keras"
LABELS_PATH = PROJECT_ROOT / "trained_model" / "label_encoder.json"
TEMPERATURE_PATH = PROJECT_ROOT / "trained_model" / "temperature.json"

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
SEQUENCE_SHAPE = (30, 126)

# Minimum calibrated confidence to return a real label.
# Below this threshold the response uses label="uncertain" to prevent the
# frontend from accumulating high-confidence wrong predictions.
CONFIDENCE_THRESHOLD = 0.40


# ---------------------------------------------------------
# Landmark normalisation  (MUST be identical to train_isl_model.py)
#
# Raw MediaPipe x,y,z coords are frame-relative.  Without normalisation the
# model memorises hand position in frame rather than hand shape, causing it
# to collapse onto a single class when hand position shifts.
#
# For each hand per frame:
#   all-zero block → no hand detected → keep zeros (preserves LSTM masking).
#   Otherwise: translate to wrist origin, scale by wrist→MCP-9 distance.
# ---------------------------------------------------------

def normalize_landmarks_sequence(X: np.ndarray) -> np.ndarray:
    """
    Normalise a batch of landmark sequences for position / scale invariance.

    X : (N, T, 126) — left_63 ‖ right_63, each hand = 21 lm × 3 coords.
    Returns the same shape with each detected hand centred at its wrist and
    scaled by the wrist-to-MCP9 distance.
    """
    out = X.copy().astype(np.float32)
    N, T, _ = out.shape

    for hand_offset in (0, 63):
        block = out[:, :, hand_offset : hand_offset + 63]          # (N, T, 63)
        is_zero = np.all(block == 0.0, axis=-1, keepdims=True)     # (N, T, 1)

        lm = block.reshape(N, T, 21, 3)
        wrist = lm[:, :, 0:1, :]                                    # (N, T, 1, 3)
        lm_c = lm - wrist                                           # (N, T, 21, 3)

        mcp9 = lm_c[:, :, 9, :]                                    # (N, T, 3)
        scale = np.linalg.norm(mcp9, axis=-1, keepdims=True)       # (N, T, 1)
        scale = np.where(scale < 1e-6, 1.0, scale)

        lm_s = lm_c / scale[:, :, np.newaxis, :]                   # (N, T, 21, 3)
        flat = lm_s.reshape(N, T, 63)

        zero_exp = np.broadcast_to(is_zero, (N, T, 63))
        out[:, :, hand_offset : hand_offset + 63] = np.where(zero_exp, 0.0, flat)

    return out


# ---------------------------------------------------------
# Temperature scaling
#
# T was fitted on the validation set after training.
# T > 1 softens over-confident softmax distributions.
# Applied as:  probs_scaled[i] ∝ probs[i]^(1/T)
# ---------------------------------------------------------

def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Apply temperature scaling to a 1-D probability array."""
    if abs(temperature - 1.0) < 1e-6:
        return probs
    scaled = np.clip(probs, 1e-10, 1.0) ** (1.0 / temperature)
    scaled /= scaled.sum()
    return scaled.astype(np.float32)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def load_label_map(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        items = sorted(payload.items(), key=lambda kv: int(kv[0]))
        return [str(v) for _, v in items]

    if isinstance(payload, list):
        return [str(x) for x in payload]

    raise ValueError(f"Unsupported label format in {path}")


def load_temperature(path: Path) -> float:
    """Load calibration temperature.  Returns 1.0 (no scaling) if missing."""
    if not path.exists():
        return 1.0
    try:
        return float(json.loads(path.read_text(encoding="utf-8"))["temperature"])
    except Exception:
        return 1.0


def load_required_model(path: Path) -> tf.keras.Model:
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return tf.keras.models.load_model(str(path))


# ---------------------------------------------------------
# Core inference
# ---------------------------------------------------------

def predict_sequence(
    model: tf.keras.Model,
    labels: list[str],
    temperature: float,
    sequence: list[list[float]],
) -> dict[str, Any]:
    arr = np.asarray(sequence, dtype=np.float32)

    if arr.shape != SEQUENCE_SHAPE:
        return {
            "success": False,
            "error": f"Expected sequence shape {SEQUENCE_SHAPE}, got {arr.shape}",
        }

    # Normalise (same transform as applied during training)
    x = normalize_landmarks_sequence(np.expand_dims(arr, axis=0))  # (1, 30, 126)

    raw_probs = model.predict(x, verbose=0)[0]                      # (num_classes,)
    probs = apply_temperature(raw_probs, temperature)

    best_index = int(np.argmax(probs))
    confidence = float(probs[best_index])
    low_confidence = confidence < CONFIDENCE_THRESHOLD

    # When confidence is too low we return "uncertain" so the frontend
    # does not accumulate a wrong high-confidence label in its history.
    label = (
        "uncertain"
        if low_confidence
        else (
            labels[best_index]
            if best_index < len(labels)
            else f"class_{best_index}"
        )
    )

    top_indices = np.argsort(probs)[::-1][:5]
    top_k = [
        {
            "label": labels[idx] if idx < len(labels) else f"class_{idx}",
            "confidence": float(probs[idx]),
            "index": int(idx),
        }
        for idx in top_indices
    ]

    return {
        "success": True,
        "label": label,
        "confidence": confidence,
        "low_confidence": low_confidence,
        "class_index": best_index,
        "top_k": top_k,
    }


# ---------------------------------------------------------
# Load model + artefacts at startup
# ---------------------------------------------------------
print("Loading continuous ISL model...")
continuous_model = load_required_model(MODEL_PATH)
continuous_labels = load_label_map(LABELS_PATH)
continuous_temperature = load_temperature(TEMPERATURE_PATH)

print(f"Loaded labels      : {len(continuous_labels)}")
print(f"Temperature        : {continuous_temperature:.4f}")

# ---------------------------------------------------------
# FastAPI
# ---------------------------------------------------------
app = FastAPI(title="SAMVAAD AI Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    sequence: list[list[float]]


@app.get("/")
def root():
    return {
        "status": "online",
        "model_loaded": True,
        "labels": len(continuous_labels),
        "temperature": continuous_temperature,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        result = predict_sequence(
            continuous_model,
            continuous_labels,
            continuous_temperature,
            request.sequence,
        )
        return result
    except Exception as exc:
        return {
            "success": False,
            "error": str(exc),
        }
