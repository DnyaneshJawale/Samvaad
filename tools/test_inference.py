"""Quick end-to-end test of the trained model + ai_server normalisation pipeline."""
from __future__ import annotations

import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, r"C:\Backup\Samvaad\samvaad")

import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from ai_server import (
    normalize_landmarks_sequence,
    apply_temperature,
    load_temperature,
    load_label_map,
    CONFIDENCE_THRESHOLD,
)

DATASET_DIR = Path(r"C:\Users\dvjaw\Downloads\processed_isl")
MODEL_PATH  = Path(r"C:\Backup\Samvaad\samvaad\trained_model\samvaad_isl_model.keras")
TEMP_PATH   = Path(r"C:\Backup\Samvaad\samvaad\trained_model\temperature.json")
LABELS_PATH = Path(r"C:\Backup\Samvaad\samvaad\trained_model\label_encoder.json")


def main() -> None:
    print("Loading model...")
    model  = tf.keras.models.load_model(str(MODEL_PATH))
    labels = load_label_map(LABELS_PATH)
    temp   = load_temperature(TEMP_PATH)
    print(f"Labels: {len(labels)}  Temperature: {temp}  Threshold: {CONFIDENCE_THRESHOLD}")

    manifest = pd.read_csv(DATASET_DIR / "manifest.csv")
    manifest = manifest[manifest["status"] == "ok"].copy()

    # Sample 3 sequences per class from the held-out val-style set
    test_classes = sorted(manifest["label"].unique())
    correct = 0
    uncertain = 0
    total = 0
    wrong_cases = []

    for cls in test_classes:
        rows = manifest[manifest["label"] == cls].head(3)
        for _, row in rows.iterrows():
            path = DATASET_DIR / row["sequence_path"]
            arr = np.load(path)  # (30, 126) raw

            x = normalize_landmarks_sequence(np.expand_dims(arr, axis=0))
            raw_probs = model.predict(x, verbose=0)[0]
            probs = apply_temperature(raw_probs, temp)

            best_idx  = int(np.argmax(probs))
            best_conf = float(probs[best_idx])
            predicted = labels[best_idx] if best_idx < len(labels) else f"class_{best_idx}"
            low_conf  = best_conf < CONFIDENCE_THRESHOLD

            if low_conf:
                uncertain += 1
                predicted = "uncertain"

            is_correct = predicted == cls
            correct += int(is_correct)
            total += 1

            if not is_correct:
                wrong_cases.append((cls, predicted, best_conf))

    print(f"\n=== Results ===")
    print(f"Total samples   : {total}")
    print(f"Correct         : {correct}  ({correct/total:.1%})")
    print(f"Uncertain (<{CONFIDENCE_THRESHOLD:.2f}): {uncertain}")
    print(f"Wrong           : {len(wrong_cases)}")

    if wrong_cases:
        print("\nWrong predictions:")
        for true, pred, conf in wrong_cases:
            print(f"  true={true:<22} pred={pred:<22} conf={conf:.3f}")

    # Spot-check: feed a zero sequence (no hand) and confirm uncertain
    zero_seq = np.zeros((1, 30, 126), dtype=np.float32)
    zero_norm = normalize_landmarks_sequence(zero_seq)
    zero_probs = apply_temperature(model.predict(zero_norm, verbose=0)[0], temp)
    zero_conf = float(np.max(zero_probs))
    zero_pred = labels[int(np.argmax(zero_probs))]
    print(f"\nZero-sequence (no hand) -> pred={zero_pred}  conf={zero_conf:.3f}  low_conf={zero_conf < CONFIDENCE_THRESHOLD}")

    print("\nDone.")


if __name__ == "__main__":
    main()
