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
from sklearn.utils.class_weight import compute_class_weight

SEQUENCE_LENGTH = 30
FEATURE_DIM = 126
RANDOM_STATE = 42

TARGET_CLASSES = ["bear", "break", "brinjal", "budget"]


def set_seeds(seed: int = RANDOM_STATE) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_sequences(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    manifest_path = dataset_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    if "status" in df.columns:
        df = df[df["status"] == "ok"].copy()

    if "label" not in df.columns or "sequence_path" not in df.columns:
        raise RuntimeError("manifest.csv must contain 'label' and 'sequence_path' columns.")

    df = df[df["label"].isin(TARGET_CLASSES)].copy()

    if df.empty:
        raise RuntimeError(
            f"No samples found for target classes: {TARGET_CLASSES}. "
            "Check your processed dataset."
        )

    sequences: List[np.ndarray] = []
    labels: List[str] = []

    for row in df.itertuples(index=False):
        seq_rel = getattr(row, "sequence_path", "")
        label = getattr(row, "label", "")

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
            print(f"Skipping unreadable file: {seq_path.name} ({exc})")
            continue

        if arr.shape != (SEQUENCE_LENGTH, FEATURE_DIM):
            print(f"Skipping bad shape: {seq_path.name} -> {arr.shape}")
            continue

        sequences.append(arr.astype(np.float32, copy=False))
        labels.append(label.strip())

    if not sequences:
        raise RuntimeError("No valid .npy sequences could be loaded.")

    X = np.stack(sequences, axis=0)
    y = np.asarray(labels, dtype=str)

    return X, y


def split_indices_per_class(
    labels: np.ndarray,
    val_fraction: float = 0.2,
    seed: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Safe per-class split:
    - classes with one sample stay in train
    - each class with >=2 samples gets at least 1 validation sample
    """
    rng = np.random.default_rng(seed)
    by_class: Dict[str, List[int]] = defaultdict(list)

    for idx, label in enumerate(labels.tolist()):
        by_class[str(label)].append(idx)

    train_idx: List[int] = []
    val_idx: List[int] = []

    for label in TARGET_CLASSES:
        idxs = by_class.get(label, [])
        if not idxs:
            continue

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


def augment_sequence(seq: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Gentle augmentation for temporal landmark sequences.
    Keeps the overall sign structure while adding small variation.
    """
    aug = seq.astype(np.float32, copy=True)

    # Small temporal shift
    shift = int(rng.integers(-3, 4))
    if shift != 0:
        aug = np.roll(aug, shift=shift, axis=0)

    # Slight scale jitter
    scale = float(rng.uniform(0.96, 1.04))
    aug *= scale

    # Small Gaussian noise
    noise = rng.normal(0.0, 0.003, size=aug.shape).astype(np.float32)
    aug += noise

    # Rare feature dropout
    if rng.random() < 0.25:
        drop_mask = rng.random(size=aug.shape) < 0.01
        aug[drop_mask] = 0.0

    return np.clip(aug, -2.0, 2.0)


def expand_training_set(
    X_train: np.ndarray,
    y_train: np.ndarray,
    copies_per_sample: int = 6,
    seed: int = RANDOM_STATE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create an expanded training set with the original sample + augmented copies.
    This is the "train hard" part.
    """
    rng = np.random.default_rng(seed)

    X_parts: List[np.ndarray] = [X_train]
    y_parts: List[np.ndarray] = [y_train]

    for _ in range(copies_per_sample):
        aug_samples = [augment_sequence(seq, rng) for seq in X_train]
        X_parts.append(np.stack(aug_samples, axis=0))
        y_parts.append(y_train.copy())

    X_expanded = np.concatenate(X_parts, axis=0)
    y_expanded = np.concatenate(y_parts, axis=0)

    shuffle_idx = rng.permutation(len(X_expanded))
    return X_expanded[shuffle_idx], y_expanded[shuffle_idx]


def build_model(num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(SEQUENCE_LENGTH, FEATURE_DIM), name="landmark_sequence")
    x = tf.keras.layers.Masking(mask_value=0.0)(inputs)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=False)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.35)(x)

    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.20)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="samvaad_isl_4class_model")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_label_map(output_dir: Path) -> Path:
    label_map = {str(i): label for i, label in enumerate(TARGET_CLASSES)}
    path = output_dir / "label_encoder.json"
    path.write_text(json.dumps(label_map, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_dataset_summary(
    output_dir: Path,
    labels: np.ndarray,
    train_labels: np.ndarray,
    val_labels: np.ndarray,
) -> Path:
    summary = {
        "target_classes": TARGET_CLASSES,
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


def plot_history(output_dir: Path, history: tf.keras.callbacks.History) -> None:
    hist = history.history

    plt.figure(figsize=(10, 5))
    plt.plot(hist.get("accuracy", []), label="Train Accuracy")
    if "val_accuracy" in hist:
        plt.plot(hist["val_accuracy"], label="Validation Accuracy")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_accuracy.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(hist.get("loss", []), label="Train Loss")
    if "val_loss" in hist:
        plt.plot(hist["val_loss"], label="Validation Loss")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=160)
    plt.close()

    pd.DataFrame(hist).to_csv(output_dir / "training_history.csv", index=False)


def plot_confusion_matrix(output_dir: Path, cm: np.ndarray, class_names: List[str]) -> None:
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=160)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a hard 4-class ISL model.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=r"C:\Users\dvjaw\Downloads\processed_isl",
        help="Folder containing manifest.csv and sequences/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=r"C:\Backup\Samvaad\samvaad\trained_model_4class",
        help="Folder where the new 4-class model will be saved.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--augment-copies", type=int, default=6)
    args = parser.parse_args()

    set_seeds()

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_dir(output_dir)

    model_path = output_dir / "samvaad_isl_4class_model.keras"
    report_path = output_dir / "classification_report.txt"

    print(f"Dataset dir : {dataset_dir}")
    print(f"Output dir  : {output_dir}")
    print(f"Target classes: {TARGET_CLASSES}")

    X, y = load_sequences(dataset_dir)
    print(f"Loaded samples : {len(X)}")
    print(f"Loaded classes : {len(np.unique(y))}")
    print(f"Sequence shape : {X.shape[1:]}")

    print("\nClass distribution in filtered dataset:")
    counts = Counter(y.tolist())
    for label in TARGET_CLASSES:
        print(f"  {label}: {counts.get(label, 0)}")

    label_to_idx = {label: idx for idx, label in enumerate(TARGET_CLASSES)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    y_idx = np.asarray([label_to_idx[label] for label in y.tolist()], dtype=np.int32)

    train_idx, val_idx = split_indices_per_class(
        labels=y,
        val_fraction=args.val_fraction,
        seed=RANDOM_STATE,
    )

    X_train = X[train_idx]
    y_train = y_idx[train_idx]
    X_val = X[val_idx]
    y_val = y_idx[val_idx]

    print(f"\nTrain samples : {len(X_train)}")
    print(f"Val samples   : {len(X_val)}")

    if len(np.unique(y_train)) < 2:
        raise RuntimeError("Training split has fewer than 2 classes. Add more samples and retry.")

    # Save metadata before training
    save_label_map(output_dir)
    save_dataset_summary(
        output_dir=output_dir,
        labels=y,
        train_labels=np.asarray([idx_to_label[i] for i in y_train], dtype=str),
        val_labels=np.asarray([idx_to_label[i] for i in y_val], dtype=str),
    )

    # Expand training set with augmentation
    X_train_aug, y_train_aug = expand_training_set(
        X_train=X_train,
        y_train=y_train,
        copies_per_sample=args.augment_copies,
        seed=RANDOM_STATE,
    )

    print(f"Augmented train samples: {len(X_train_aug)}")

    # Class weights from the original training split
    class_weight_values = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weight = {
        int(cls): float(min(weight, 20.0))
        for cls, weight in zip(np.unique(y_train), class_weight_values)
    }

    print("\nClass weights:")
    for idx, weight in class_weight.items():
        print(f"  {TARGET_CLASSES[idx]}: {weight:.4f}")

    tf.keras.backend.clear_session()
    model = build_model(num_classes=len(TARGET_CLASSES))

    print("\nModel summary:")
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=12,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    print("\nStarting training...\n")

    history = model.fit(
        X_train_aug,
        y_train_aug,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=1,
    )

    print("\nTraining complete.")

    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation Loss     : {val_loss:.4f}")
    print(f"Validation Accuracy : {val_accuracy:.4f}")

    y_prob = model.predict(X_val, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    report = classification_report(
        y_val,
        y_pred,
        labels=list(range(len(TARGET_CLASSES))),
        target_names=TARGET_CLASSES,
        zero_division=0,
    )
    print("\nClassification Report:\n")
    print(report)
    report_path.write_text(report, encoding="utf-8")

    cm = confusion_matrix(y_val, y_pred, labels=list(range(len(TARGET_CLASSES))))
    plot_confusion_matrix(output_dir, cm, TARGET_CLASSES)

    model.save(model_path)
    print(f"\nSaved model   -> {model_path}")
    print(f"Saved labels  -> {output_dir / 'label_encoder.json'}")
    print(f"Saved summary -> {output_dir / 'dataset_summary.json'}")
    print(f"Saved report  -> {report_path}")
    print(f"Saved history -> {output_dir / 'training_history.csv'}")
    print(f"Saved plots   -> {output_dir / 'training_accuracy.png'}")
    print(f"Saved plots   -> {output_dir / 'training_loss.png'}")
    print(f"Saved plots   -> {output_dir / 'confusion_matrix.png'}")

    print("\n4-class model is ready.")
    print("Next step: point ai_server.py to this new model folder for live testing.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())