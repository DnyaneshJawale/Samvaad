from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import tensorflow as tf

try:
    import tensorflowjs as tfjs
except ImportError:
    print(
        "Missing package: tensorflowjs\n"
        "Install it with:\n"
        "  pip install tensorflowjs",
        file=sys.stderr,
    )
    raise SystemExit(1)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MODEL = PROJECT_ROOT / "trained_model" / "samvaad_isl_model.keras"
SOURCE_LABELS = PROJECT_ROOT / "trained_model" / "label_encoder.json"
OUTPUT_DIR = PROJECT_ROOT / "public" / "models" / "isl"


def read_label_list(label_map_path: Path) -> list[str]:
    data = json.loads(label_map_path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [str(item) for item in data]

    if isinstance(data, dict):
        items = sorted(data.items(), key=lambda kv: int(kv[0]))
        return [str(label) for _, label in items]

    raise ValueError("label_encoder.json must be either a list or a mapping.")


def clean_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> int:
    if not SOURCE_MODEL.exists():
        print(f"Model not found: {SOURCE_MODEL}", file=sys.stderr)
        return 1

    if not SOURCE_LABELS.exists():
        print(f"Label map not found: {SOURCE_LABELS}", file=sys.stderr)
        return 1

    labels = read_label_list(SOURCE_LABELS)

    clean_output_dir(OUTPUT_DIR)

    print(f"Loading Keras model: {SOURCE_MODEL}")
    model = tf.keras.models.load_model(SOURCE_MODEL)

    print(f"Exporting TensorFlow.js model to: {OUTPUT_DIR}")
    tfjs.converters.save_keras_model(model, str(OUTPUT_DIR))

    (OUTPUT_DIR / "labels.json").write_text(
        json.dumps(labels, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    shutil.copy2(SOURCE_LABELS, OUTPUT_DIR / "label_encoder.json")

    print("\nExport completed.")
    print(f"Model folder : {OUTPUT_DIR}")
    print(f"Model file   : {OUTPUT_DIR / 'model.json'}")
    print(f"Labels file  : {OUTPUT_DIR / 'labels.json'}")
    print(f"Label map    : {OUTPUT_DIR / 'label_encoder.json'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())