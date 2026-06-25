#prepare_isl_dataset_v2.py
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp


# ---------------------------------------------------------------------
# Default paths for your machine
# ---------------------------------------------------------------------
DEFAULT_DATASET_ROOT = Path.home() / "Downloads" / "archive"
DEFAULT_OUTPUT_DIR = Path.home() / "Downloads" / "processed_isl"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "models"
DEFAULT_MODEL_PATH = DEFAULT_MODEL_DIR / "hand_landmarker.task"

# Official MediaPipe Hand Landmarker model bundle URL
# If you already have the model locally, the script will use it.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
CONTAINER_NAMES = {"archive", "sample videos", "video_dataset"}

# MediaPipe Tasks API (matches your installed build)
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def normalize_label(text: str) -> str:
    text = text.replace("_", " ").replace("-", " ").strip()
    text = " ".join(text.split())
    return text.lower()


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def safe_filename(text: str) -> str:
    text = text.strip().lower().replace(" ", "_")
    text = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)
    text = "_".join(part for part in text.split("_") if part)
    return text or "sample"


def ensure_model_file(model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists() and model_path.stat().st_size > 0:
        return model_path

    print(f"Model not found at: {model_path}")
    print("Downloading official Hand Landmarker model...")

    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as exc:
        raise RuntimeError(
            "Could not download the MediaPipe Hand Landmarker model. "
            f"Place the file manually at: {model_path}"
        ) from exc

    if not model_path.exists() or model_path.stat().st_size == 0:
        raise RuntimeError(
            f"Model download finished but file is invalid: {model_path}"
        )

    return model_path


def collect_videos(dataset_root: Path) -> List[Path]:
    roots: List[Path] = []

    sample_root = dataset_root / "Sample Videos"
    video_dataset_root = dataset_root / "Video_Dataset"

    if sample_root.exists():
        roots.append(sample_root)

    if video_dataset_root.exists():
        roots.append(video_dataset_root)

    found: List[Path] = []
    for root in roots:
        for item in root.rglob("*"):
            if is_video_file(item):
                found.append(item.resolve())

    unique = sorted({str(p): p for p in found}.values(), key=lambda p: str(p).lower())
    return unique


def derive_label(video_path: Path) -> str:
    """
    Rules:
    - Sample Videos/<file>.mp4  -> label = file stem
    - Video_Dataset/<label>/<file>.mp4 -> label = folder name
    - Video_Dataset/Video_Dataset/<label>/<file>.mp4 -> label = folder name
    """
    parts_lower = [part.lower() for part in video_path.parts]

    if "sample videos" in parts_lower:
        return normalize_label(video_path.stem)

    for parent in video_path.parents:
        name = parent.name.strip()
        if not name:
            continue

        lowered = name.lower()
        if lowered in CONTAINER_NAMES:
            continue

        return normalize_label(name)

    return normalize_label(video_path.stem)


def category_name_of(category_obj) -> str:
    """
    MediaPipe category object field names can vary slightly by version.
    This helper is defensive.
    """
    for attr in ("category_name", "label", "display_name", "name"):
        value = getattr(category_obj, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "Hand"


def landmarks_to_vector(hand_landmarks) -> np.ndarray:
    """
    Converts 21 hand landmarks into a flat vector of 63 floats:
    [x1, y1, z1, x2, y2, z2, ...]
    """
    points = getattr(hand_landmarks, "landmark", None)

    if points is None:
        # fallback: try treating it as a plain iterable of points
        points = hand_landmarks

    vec: List[float] = []
    for lm in points:
        vec.extend([float(lm.x), float(lm.y), float(lm.z)])

    return np.asarray(vec, dtype=np.float32)


def empty_hand_vector() -> np.ndarray:
    return np.zeros((63,), dtype=np.float32)


def result_to_frame_feature(result) -> np.ndarray:
    """
    Returns one frame feature vector:
    [left_hand_63 | right_hand_63] => 126 values
    """
    left_vec = empty_hand_vector()
    right_vec = empty_hand_vector()

    landmarks_list = (
        getattr(result, "hand_landmarks", None)
        or getattr(result, "landmarks", None)
        or []
    )

    handedness_list = (
        getattr(result, "handedness", None)
        or getattr(result, "handednesses", None)
        or []
    )

    if not landmarks_list:
        return np.concatenate([left_vec, right_vec], axis=0)

    for idx, hand_landmarks in enumerate(landmarks_list):
        side = "unknown"

        if idx < len(handedness_list):
            try:
                # handedness is usually List[List[Category]]
                first_group = handedness_list[idx]
                first_cat = first_group[0] if first_group else None
                if first_cat is not None:
                    side = category_name_of(first_cat).lower()
            except Exception:
                side = "unknown"

        vec = landmarks_to_vector(hand_landmarks)

        if side == "left":
            left_vec = vec
        elif side == "right":
            right_vec = vec
        else:
            # fallback assignment when handedness is not available
            if np.allclose(left_vec, 0.0):
                left_vec = vec
            elif np.allclose(right_vec, 0.0):
                right_vec = vec

    return np.concatenate([left_vec, right_vec], axis=0)


def fixed_length_sequence(frames: List[np.ndarray], target_frames: int) -> Optional[np.ndarray]:
    if not frames:
        return None

    if len(frames) == target_frames:
        return np.stack(frames, axis=0).astype(np.float32)

    if len(frames) < target_frames:
        pad_frame = frames[-1]
        padded = frames + [pad_frame] * (target_frames - len(frames))
        return np.stack(padded, axis=0).astype(np.float32)

    indices = np.linspace(0, len(frames) - 1, target_frames).round().astype(int)
    sampled = [frames[i] for i in indices]
    return np.stack(sampled, axis=0).astype(np.float32)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_label_map(labels: List[str]) -> Dict[str, int]:
    unique = sorted({normalize_label(label) for label in labels})
    return {label: idx for idx, label in enumerate(unique)}


def create_landmarker(model_path: Path):
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def process_video(
    video_path: Path,
    landmarker,
    target_frames: int,
    frame_stride: int,
    global_timestamp_offset_ms: int,
) -> Tuple[Optional[np.ndarray], int, int, int, int]:
    """
    Returns:
        sequence, total_read_frames, used_frames, valid_frames, duration_ms
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0, 0, 0, 0

    frames: List[np.ndarray] = []
    total_read = 0
    used_frames = 0
    valid_frames = 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            total_read += 1

            if frame_stride > 1 and ((total_read - 1) % frame_stride != 0):
                frame_index += 1
                continue

            used_frames += 1

            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            except Exception:
                frame_index += 1
                continue

            timestamp_ms = global_timestamp_offset_ms + int((frame_index / fps) * 1000)

            try:
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
            except Exception:
                frame_index += 1
                continue

            feature = result_to_frame_feature(result)

            if not np.allclose(feature, 0.0):
                valid_frames += 1

            frames.append(feature)
            frame_index += 1

    finally:
        cap.release()

    sequence = fixed_length_sequence(frames, target_frames)
    duration_ms = int((total_read / fps) * 1000) if total_read > 0 else 0

    return sequence, total_read, used_frames, valid_frames, duration_ms


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare ISL landmark sequences from video dataset using MediaPipe Tasks."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help="Path to the archive folder that contains Sample Videos and Video_Dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Folder where processed sequences and manifests will be saved.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to hand_landmarker.task. If missing, the script will try to download it.",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=30,
        help="Fixed number of frames per sample sequence.",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help="Process every Nth frame to reduce compute.",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Limit processing to this many videos for testing. 0 means all videos.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing sequence files if they already exist.",
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()

    if not dataset_root.exists():
        print(f"Dataset root not found: {dataset_root}", file=sys.stderr)
        return 1

    videos = collect_videos(dataset_root)
    if not videos:
        print(
            "No video files found under Sample Videos or Video_Dataset.",
            file=sys.stderr,
        )
        return 1

    if args.max_videos and args.max_videos > 0:
        videos = videos[: args.max_videos]

    labels = [derive_label(video) for video in videos]
    label_map = build_label_map(labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    sequences_dir = output_dir / "sequences"
    sequences_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    label_map_path = output_dir / "label_map.json"
    summary_path = output_dir / "dataset_summary.json"

    try:
        model_path = ensure_model_file(model_path)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Dataset root : {dataset_root}")
    print(f"Output dir   : {output_dir}")
    print(f"Model path   : {model_path}")
    print(f"Videos found : {len(videos)}")
    print(f"Target frames: {args.target_frames}")
    print(f"Frame stride : {args.frame_stride}")
    print("Starting preprocessing...\n")

    manifest_rows: List[Dict[str, str]] = []
    stats = {
        "total_videos_found": len(videos),
        "processed": 0,
        "skipped": 0,
        "labels": {},
    }

    global_timestamp_offset_ms = 0

    try:
        with create_landmarker(model_path) as landmarker:
            for index, video_path in enumerate(videos, start=1):
                label = derive_label(video_path)
                label_id = label_map[label]

                short_hash = hashlib.sha1(str(video_path).encode("utf-8")).hexdigest()[:8]
                file_stem = safe_filename(video_path.stem)
                seq_name = f"{label_id:03d}_{file_stem}_{short_hash}.npy"
                seq_path = sequences_dir / seq_name

                if seq_path.exists() and not args.overwrite:
                    manifest_rows.append(
                        {
                            "sample_id": seq_name[:-4],
                            "label": label,
                            "label_id": str(label_id),
                            "source_path": str(video_path),
                            "sequence_path": str(seq_path.relative_to(output_dir)),
                            "total_read_frames": "0",
                            "used_frames": "0",
                            "valid_frames": "0",
                            "status": "skipped_existing",
                        }
                    )
                    stats["skipped"] += 1
                    print(f"[{index}/{len(videos)}] SKIP  {label} -> existing")
                    continue

                (
                    sequence,
                    total_read,
                    used_frames,
                    valid_frames,
                    duration_ms,
                ) = process_video(
                    video_path=video_path,
                    landmarker=landmarker,
                    target_frames=args.target_frames,
                    frame_stride=args.frame_stride,
                    global_timestamp_offset_ms=global_timestamp_offset_ms,
                )

                global_timestamp_offset_ms += duration_ms + 1000

                if sequence is None or valid_frames == 0:
                    manifest_rows.append(
                        {
                            "sample_id": seq_name[:-4],
                            "label": label,
                            "label_id": str(label_id),
                            "source_path": str(video_path),
                            "sequence_path": "",
                            "total_read_frames": str(total_read),
                            "used_frames": str(used_frames),
                            "valid_frames": str(valid_frames),
                            "status": "skipped_no_landmarks",
                        }
                    )
                    stats["skipped"] += 1
                    print(f"[{index}/{len(videos)}] SKIP  {label} -> no landmarks")
                    continue

                np.save(seq_path, sequence.astype(np.float32))

                manifest_rows.append(
                    {
                        "sample_id": seq_name[:-4],
                        "label": label,
                        "label_id": str(label_id),
                        "source_path": str(video_path),
                        "sequence_path": str(seq_path.relative_to(output_dir)),
                        "total_read_frames": str(total_read),
                        "used_frames": str(used_frames),
                        "valid_frames": str(valid_frames),
                        "status": "ok",
                    }
                )

                stats["processed"] += 1
                stats["labels"][label] = stats["labels"].get(label, 0) + 1

                print(
                    f"[{index}/{len(videos)}] OK    {label} "
                    f"(read={total_read}, used={used_frames}, valid={valid_frames})"
                )

    except Exception as exc:
        print(f"\nPreprocessing failed: {exc}", file=sys.stderr)
        return 1

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id",
                "label",
                "label_id",
                "source_path",
                "sequence_path",
                "total_read_frames",
                "used_frames",
                "valid_frames",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    write_json(label_map_path, label_map)
    write_json(summary_path, stats)

    print("\nDone.")
    print(f"Processed : {stats['processed']}")
    print(f"Skipped   : {stats['skipped']}")
    print(f"Manifest  : {manifest_path}")
    print(f"Labels    : {label_map_path}")
    print(f"Summary   : {summary_path}")
    print(f"Sequences : {sequences_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())