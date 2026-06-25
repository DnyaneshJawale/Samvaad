from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

try:
    from scipy.optimize import minimize_scalar  # type: ignore

    _SCIPY = True
except ImportError:
    _SCIPY = False
    print("scipy not found -- temperature calibration will be skipped (pip install scipy).")


SEQUENCE_LENGTH = 30
FEATURE_DIM = 126
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seeds(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Landmark normalisation
#
# Raw MediaPipe coordinates are frame-relative.  A hand at the top-left of
# the frame produces completely different (x, y) values from the same hand
# at the centre.  A model trained on those raw coords learns WHERE hands are
# positioned, not what SHAPE they form.  After normalisation the features are
# position- and scale-invariant so the model generalises across cameras.
#
# For each hand block (63 floats = 21 landmarks x 3):
#   - If all-zero: no hand detected -> keep zeros (preserves Masking behaviour).
#   - Translate so wrist (landmark 0) is the origin.
#   - Scale by the wrist->MCP-9 distance (middle-finger MCP, a stable palm ref).
# ---------------------------------------------------------------------------

def normalize_landmarks_sequence(X: np.ndarray) -> np.ndarray:
    """
    Normalise a batch of landmark sequences.

    X : (N, T, 126) -- left_63 || right_63, each hand = 21 lm x 3 coords.
    Returns the same shape, normalised per-hand per-frame.
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


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Apply randomised augmentations to a single (T, 126) normalised sequence.

    Augmentations:
      1. Gaussian noise on detected frames (sigma ≈ 0.025 in normalised space)
      2. Temporal warp  +/-25 % speed change
      3. Left/right mirror swap (50 % chance)  -- helps left-handed signers
      4. Random frame dropout (30 % chance)
    """
    T, _ = seq.shape
    seq = seq.copy().astype(np.float32)

    non_zero = ~np.all(seq == 0.0, axis=-1, keepdims=True)   # (T, 1)

    # 1. Gaussian noise
    seq += non_zero * rng.normal(0.0, 0.025, seq.shape).astype(np.float32)

    # 2. Temporal warp
    speed = float(rng.uniform(0.75, 1.33))
    new_T = max(10, min(int(T * speed), T * 2))
    src_idx = np.linspace(0, T - 1, new_T).round().astype(int)
    stretched = seq[src_idx]
    dst_idx = np.linspace(0, new_T - 1, T).round().astype(int)
    seq = stretched[dst_idx].astype(np.float32)

    # 3. Mirror: swap left <-> right, negate x coords
    if rng.random() < 0.5:
        left = seq[:, :63].copy()
        right = seq[:, 63:].copy()
        for hand in (left, right):
            h = hand.reshape(T, 21, 3)
            h[:, :, 0] = -h[:, :, 0]
        seq = np.concatenate([right, left], axis=-1)

    # 4. Frame dropout
    if rng.random() < 0.3:
        n_drop = int(rng.integers(1, max(2, T // 5)))
        drop_idx = rng.choice(T, size=n_drop, replace=False)
        seq[drop_idx] = 0.0

    return seq.astype(np.float32)


def build_augmented_train_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_augments: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Create n_augments additional copies of the training set and shuffle."""
    rng = np.random.default_rng(seed)
    all_X = [X_train]
    all_y = [y_train]

    for i in range(n_augments):
        aug = np.stack(
            [augment_sequence(X_train[j], rng) for j in range(len(X_train))]
        )
        all_X.append(aug)
        all_y.append(y_train)
        print(f"  Augment pass {i + 1}/{n_augments} done  ({len(aug)} samples)")

    X_out = np.concatenate(all_X, axis=0)
    y_out = np.concatenate(all_y, axis=0)
    idx = rng.permutation(len(X_out))
    return X_out[idx], y_out[idx]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sequences(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[Path]]:
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    if "status" in manifest.columns:
        manifest = manifest[manifest["status"] == "ok"].copy()
    if manifest.empty:
        raise RuntimeError("No valid samples found in manifest.csv")

    sequences: List[np.ndarray] = []
    labels: List[str] = []
    source_paths: List[Path] = []

    for row in manifest.itertuples(index=False):
        seq_rel = getattr(row, "sequence_path", "")
        label = getattr(row, "label", None)

        if not isinstance(seq_rel, str) or not seq_rel.strip():
            continue
        if not isinstance(label, str) or not label.strip():
            continue

        seq_path = dataset_dir / seq_rel
        if not seq_path.exists():
            continue

        try:
            arr = np.load(seq_path, allow_pickle=False)
        except Exception as exc:
            print(f"Skipping unreadable: {seq_path.name} ({exc})")
            continue

        if arr.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            print(f"Skipping bad shape: {seq_path.name} -> {arr.shape}")
            continue

        sequences.append(arr.astype(np.float32, copy=False))
        labels.append(label.strip())
        source_paths.append(seq_path)

    if not sequences:
        raise RuntimeError("No valid .npy sequences could be loaded.")

    X = np.stack(sequences, axis=0)
    y = np.asarray(labels, dtype=str)
    return X, y, source_paths


# ---------------------------------------------------------------------------
# Train / val split  (stratified, keeps every sample)
# ---------------------------------------------------------------------------

def split_all_samples_per_class(
    labels: np.ndarray,
    val_fraction: float,
    seed: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    by_class: Dict[str, List[int]] = defaultdict(list)

    for idx, label in enumerate(labels.tolist()):
        by_class[str(label)].append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []

    for label in sorted(by_class.keys()):
        idxs = by_class[label]
        rng.shuffle(idxs)
        if len(idxs) == 1:
            train_idx.extend(idxs)
            continue
        n_val = max(1, int(round(len(idxs) * val_fraction)))
        n_val = min(n_val, len(idxs) - 1)
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return np.asarray(train_idx, dtype=np.int32), np.asarray(val_idx, dtype=np.int32)


# ---------------------------------------------------------------------------
# Model
#
# Same BiLSTM structure as before but:
#   - LayerNormalization instead of BatchNormalization
#     (more stable with variable-length masked sequences)
#   - L2 regularization on all kernel weights
#   - LSTM internal dropout (recurrent_dropout) to reduce co-adaptation
#   - Dense head with L2
#
# The primary generalisation improvement comes from normalised inputs + augmentation,
# not from the architecture change, but these additions reduce residual memorisation.
# ---------------------------------------------------------------------------

def build_model(num_classes: int) -> tf.keras.Model:
    reg = tf.keras.regularizers.l2(5e-5)

    inputs = tf.keras.Input(
        shape=(SEQUENCE_LENGTH, FEATURE_DIM), name="landmark_sequence"
    )
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            128,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.1,
            kernel_regularizer=reg,
        )
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(
            64,
            return_sequences=False,
            dropout=0.2,
            recurrent_dropout=0.1,
            kernel_regularizer=reg,
        )
    )(x)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=reg)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="samvaad_isl_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------------------------
# Temperature calibration
#
# After training, we find the scalar T that minimises NLL on the validation set.
# T > 1  ->  softer (less over-confident) distribution.
# T < 1  ->  sharper (more confident) distribution.
# A model that memorises training data is typically over-confident -> T > 1.
# ---------------------------------------------------------------------------

def calibrate_temperature(
    model: tf.keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    if not _SCIPY:
        print("scipy unavailable -- using T = 1.0 (no calibration)")
        return 1.0

    y_prob = model.predict(X_val, verbose=0)
    val_acc = float(np.mean(np.argmax(y_prob, axis=1) == y_val))

    # When validation accuracy is saturated (100 %) the calibration objective
    # is already at its minimum at T = 1.0 and the bounded search would drift
    # to T < 1, which SHARPENS distributions and makes out-of-distribution
    # inputs appear MORE confident -- the opposite of what we want for live
    # inference.  Skip calibration in this case.
    if val_acc >= 0.999:
        print(
            f"Val accuracy is {val_acc:.4f} (saturated) -- "
            "calibration cannot be performed reliably; using T = 1.0."
        )
        return 1.0

    def nll(T: float) -> float:
        T = max(T, 1e-3)
        scaled = np.clip(y_prob, 1e-10, 1.0) ** (1.0 / T)
        scaled /= scaled.sum(axis=1, keepdims=True)
        correct = scaled[np.arange(len(y_val)), y_val]
        return -float(np.mean(np.log(np.clip(correct, 1e-10, 1.0))))

    # Only allow T >= 1.0 so we can only soften predictions, never sharpen.
    result = minimize_scalar(nll, bounds=(1.0, 10.0), method="bounded")
    T = float(result.x)
    print(f"Calibrated temperature T = {T:.4f}")
    return T


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Works for both (N, C) batches and (C,) single predictions."""
    if abs(temperature - 1.0) < 1e-6:
        return probs
    scaled = np.clip(probs, 1e-10, 1.0) ** (1.0 / temperature)
    if scaled.ndim == 1:
        scaled /= scaled.sum()
    else:
        scaled /= scaled.sum(axis=1, keepdims=True)
    return scaled


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_label_map(output_dir: Path, label_encoder: LabelEncoder) -> Path:
    label_map = {str(i): label for i, label in enumerate(label_encoder.classes_)}
    path = output_dir / "label_encoder.json"
    path.write_text(json.dumps(label_map, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_temperature(output_dir: Path, temperature: float) -> Path:
    path = output_dir / "temperature.json"
    path.write_text(
        json.dumps({"temperature": round(temperature, 6)}, indent=2), encoding="utf-8"
    )
    return path


def save_dataset_summary(
    output_dir: Path,
    labels: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
) -> Path:
    summary = {
        "total_samples": int(len(labels)),
        "total_classes": int(len(np.unique(labels))),
        "class_counts": {str(k): int(v) for k, v in Counter(labels.tolist()).items()},
        "train_samples": int(len(train_labels)),
        "val_samples": int(len(val_labels)),
        "train_class_counts": {
            str(k): int(v) for k, v in Counter(train_labels.tolist()).items()
        },
        "val_class_counts": {
            str(k): int(v) for k, v in Counter(val_labels.tolist()).items()
        },
    }
    path = output_dir / "dataset_summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_history_csv(output_dir: Path, history: tf.keras.callbacks.History) -> Path:
    path = output_dir / "training_history.csv"
    pd.DataFrame(history.history).to_csv(path, index=False)
    return path


def plot_history(output_dir: Path, history: tf.keras.callbacks.History) -> None:
    hist = history.history

    if "accuracy" in hist:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hist["accuracy"], label="Train Accuracy")
        if "val_accuracy" in hist:
            ax.plot(hist["val_accuracy"], label="Validation Accuracy")
        ax.set_title("Training Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "training_accuracy.png", dpi=160)
        plt.close(fig)

    if "loss" in hist:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(hist["loss"], label="Train Loss")
        if "val_loss" in hist:
            ax.plot(hist["val_loss"], label="Validation Loss")
        ax.set_title("Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "training_loss.png", dpi=160)
        plt.close(fig)


def plot_confusion_matrix(
    output_dir: Path,
    cm: np.ndarray,
    class_names: List[str],
) -> None:
    num_classes = len(class_names)

    if num_classes > 25:
        pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
            output_dir / "confusion_matrix.csv"
        )
        return

    fig, ax = plt.subplots(
        figsize=(max(10, num_classes * 0.6), max(8, num_classes * 0.6))
    )
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im, ax=ax)
    tick_marks = np.arange(num_classes)
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train SAMVAAD ISL model with landmark normalisation and augmentation."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=r"C:\Users\dvjaw\Downloads\processed_isl",
        help="Path to processed_isl folder (contains manifest.csv and sequences/).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"C:\Backup\Samvaad\samvaad\trained_model",
        help="Folder to save trained model and artefacts.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--n-augments",
        type=int,
        default=7,
        help="Number of augmented copies of training set (total = original + n_augments copies).",
    )
    args = parser.parse_args()

    set_seeds()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    model_path = output_dir / "samvaad_isl_model.keras"
    labels_path = output_dir / "label_encoder.json"
    temp_path = output_dir / "temperature.json"
    report_path = output_dir / "classification_report.txt"

    print(f"Dataset dir  : {dataset_dir}")
    print(f"Output dir   : {output_dir}")

    # ------------------------------------------------------------------
    # Load & normalise
    # ------------------------------------------------------------------
    print("\nLoading sequences...")
    X_raw, y_str, _ = load_sequences(dataset_dir)
    print(f"Loaded samples  : {len(X_raw)}")
    print(f"Loaded classes  : {len(np.unique(y_str))}")
    print(f"Sequence shape  : {X_raw.shape[1:]}")

    print("\nNormalising landmarks (wrist-centred, MCP9-scale)...")
    X_norm = normalize_landmarks_sequence(X_raw)

    # Quick sanity: normalised values should be near zero mean, ~unit scale
    lm_vals = X_norm[X_norm != 0]
    print(f"Post-norm non-zero stats -> mean: {lm_vals.mean():.4f}  std: {lm_vals.std():.4f}")

    # ------------------------------------------------------------------
    # Encode labels
    # ------------------------------------------------------------------
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_str)
    num_classes = len(label_encoder.classes_)

    print("\nClass distribution:")
    counts = Counter(y_str.tolist())
    for label in sorted(counts.keys()):
        print(f"  {label}: {counts[label]}")

    # ------------------------------------------------------------------
    # Split  (stratified, on original data before augmentation)
    # ------------------------------------------------------------------
    train_idx, val_idx = split_all_samples_per_class(
        labels=y_str,
        val_fraction=args.val_fraction,
        seed=RANDOM_STATE,
    )

    X_train_base = X_norm[train_idx]
    y_train_base = y_encoded[train_idx]

    X_val = (
        X_norm[val_idx]
        if len(val_idx) > 0
        else np.empty((0, SEQUENCE_LENGTH, FEATURE_DIM), dtype=np.float32)
    )
    y_val = (
        y_encoded[val_idx]
        if len(val_idx) > 0
        else np.empty((0,), dtype=np.int32)
    )

    print(f"\nBase train  : {len(X_train_base)}")
    print(f"Validation  : {len(X_val)}")

    if len(np.unique(y_train_base)) < 2:
        raise RuntimeError("Training split contains fewer than 2 classes.")

    save_label_map(output_dir, label_encoder)
    save_dataset_summary(
        output_dir=output_dir,
        labels=y_str,
        train_labels=label_encoder.inverse_transform(y_train_base),
        val_labels=(
            label_encoder.inverse_transform(y_val)
            if len(y_val) > 0
            else np.array([], dtype=str)
        ),
    )

    # ------------------------------------------------------------------
    # Augment training set only
    # ------------------------------------------------------------------
    print(f"\nBuilding augmented training set ({args.n_augments}x)...")
    X_train, y_train = build_augmented_train_set(
        X_train_base, y_train_base,
        n_augments=args.n_augments,
        seed=RANDOM_STATE,
    )
    print(f"Augmented train samples : {len(X_train)}")

    # ------------------------------------------------------------------
    # Class weights (computed on original, un-augmented distribution)
    # ------------------------------------------------------------------
    class_weights_raw = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train_base),
        y=y_train_base,
    )
    class_weight = {
        int(cls): float(min(w, 20.0))
        for cls, w in zip(np.unique(y_train_base), class_weights_raw)
    }

    print("\nClass weights:")
    for idx, w in class_weight.items():
        print(f"  {label_encoder.classes_[idx]}: {w:.4f}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    tf.keras.backend.clear_session()
    model = build_model(num_classes)
    print("\nModel summary:")
    model.summary()

    use_validation = len(X_val) > 0
    monitor_metric = "val_accuracy" if use_validation else "loss"
    monitor_mode = "max" if use_validation else "min"
    reduce_lr_metric = "val_loss" if use_validation else "loss"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor=monitor_metric,
            mode=monitor_mode,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor_metric,
            mode=monitor_mode,
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=reduce_lr_metric,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("\nStarting training...\n")

    fit_kwargs: dict = dict(
        x=X_train,
        y=y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )
    if use_validation:
        fit_kwargs["validation_data"] = (X_val, y_val)

    history = model.fit(**fit_kwargs)

    print("\nTraining finished.")

    # ------------------------------------------------------------------
    # Temperature calibration + evaluation
    # ------------------------------------------------------------------
    if use_validation:
        print("\nCalibrating temperature on validation set...")
        temperature = calibrate_temperature(model, X_val, y_val)
        print(f"Optimal temperature T = {temperature:.4f}")
        save_temperature(output_dir, temperature)

        # Raw accuracy (no temperature)
        loss_raw, acc_raw = model.evaluate(X_val, y_val, verbose=0)
        print(f"\nVal Loss (raw)         : {loss_raw:.4f}")
        print(f"Val Accuracy (raw)     : {acc_raw:.4f}")

        # Calibrated predictions
        y_prob_raw = model.predict(X_val, verbose=0)
        y_prob_cal = apply_temperature(y_prob_raw, temperature)
        y_pred = np.argmax(y_prob_cal, axis=1)

        cal_acc = float(np.mean(y_pred == y_val))
        print(f"Val Accuracy (cal T={temperature:.2f}): {cal_acc:.4f}")

        # Per-class confidence stats to spot remaining collapse
        print("\nPer-class max confidence (calibrated) -- mean over val samples:")
        for cls_idx in range(num_classes):
            mask = y_val == cls_idx
            if mask.sum() == 0:
                continue
            mc = float(y_prob_cal[mask, cls_idx].mean())
            print(f"  {label_encoder.classes_[cls_idx]}: {mc:.3f}")

        report = classification_report(
            y_val,
            y_pred,
            labels=list(range(num_classes)),
            target_names=label_encoder.classes_,
            zero_division=0,
        )
        print("\nClassification Report (temperature-calibrated):\n")
        print(report)
        report_path.write_text(report, encoding="utf-8")

        cm = confusion_matrix(y_val, y_pred, labels=list(range(num_classes)))
        plot_confusion_matrix(output_dir, cm, label_encoder.classes_.tolist())
    else:
        print("No validation split -- skipping evaluation.")
        save_temperature(output_dir, 1.0)

    # ------------------------------------------------------------------
    # Save artefacts
    # ------------------------------------------------------------------
    model.save(model_path)
    print(f"\nSaved model        -> {model_path}")

    save_history_csv(output_dir, history)
    plot_history(output_dir, history)

    print(f"Saved labels       -> {labels_path}")
    print(f"Saved temperature  -> {temp_path}")
    print(f"Saved history      -> {output_dir / 'training_history.csv'}")

    for name in (
        "training_accuracy.png",
        "training_loss.png",
        "confusion_matrix.png",
        "confusion_matrix.csv",
        "classification_report.txt",
    ):
        if (output_dir / name).exists():
            print(f"Saved {name:30s} -> {output_dir / name}")

    print("\nTraining complete.")
    print(f"Total samples  : {len(X_raw)}")
    print(f"Total classes  : {num_classes}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

