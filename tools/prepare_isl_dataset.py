#prepare_isl_dataset.py
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp  # type: ignore
except ImportError as exc:
    print(
        "MediaPipe is not installed. Run: python -m pip install mediapipe",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
CONTAINER_NAMES = {"archive", "sample videos", "video_dataset"}


def normalize_label(text: str) -> str:
    text = text.replace("_", " ").replace("-", " ").strip()
    text = " ".join(text.split())
    return text.lower()


def is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_EXTS


def get_hands_module():
    """
    MediaPipe changed import layouts across versions.
    This helper tries multiple safe fallbacks so the script works across versions.
    """
    # Most common newer/older public API
    try:
        return mp.solutions.hands  # type: ignore[attr-defined]
    except AttributeError:
        pass

    # Older/alternate internal paths
    try:
        from mediapipe.python.solutions import hands as mp_hands  # type: ignore

        return mp_hands
    except Exception:
        pass

    try:
        from mediapipe.framework.formats import landmark_pb2  # noqa: F401
        from mediapipe.python.solutions import hands as mp_hands  # type: ignore

        return mp_hands
    except Exception as exc:
        raise RuntimeError(
            "Could not locate MediaPipe Hands module. "
            "Please reinstall mediapipe with a compatible version."
        ) from exc


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
    parts_lower = [part.lower() for part in video_path.parts]

    # Sample Videos/<sign>.mp4
    if "sample videos" in parts_lower:
        return normalize_label(video_path.stem)

    # Video_Dataset/<sign>/<video>.mp4
    # If nested Video_Dataset/Video_Dataset/<sign>/<video>.mp4, still works because
    # we walk upward and take the first meaningful directory after skipping container names.
    for parent in video_path.parents:
        name = parent.name.strip()
        if not name:
            continue

        lowered = name.lower()
        if lowered in CONTAINER_NAMES:
            continue

        return normalize_label(name)

    # fallback
    return normalize_label(video_path.stem)


def landmarks_to_vector(hand_landmarks) -> np.ndarray:
    vec: List[float] = []
    for lm in hand_landmarks.landmark:
        vec.extend([float(lm.x), float(lm.y), float(lm.z)])
    return np.array(vec, dtype=np.float32)  # 63 values


def empty_hand_vector() -> np.ndarray:
    return np.zeros((63,), dtype=np.float32)


def frame_to_feature(result) -> np.ndarray:
    """
    Output per frame:
    [left_hand_63 | right_hand_63] => 126 values
    """
    left_vec = empty_hand_vector()
    right_vec = empty_hand_vector()

    multi_hand_landmarks = getattr(result, "multi_hand_landmarks", None)
    multi_handedness = getattr(result, "multi_handedness", None) or []

    if not multi_hand_landmarks:
        return np.concatenate([left_vec, right_vec], axis=0)

    for idx, hand_landmarks in enumerate(multi_hand_landmarks):
        side = "unknown"

        if idx < len(multi_handedness):
            try:
                side = multi_handedness[idx].classification[0].label.lower()
            except Exception:
                side = "unknown"

        vec = landmarks_to_vector(hand_landmarks)

        if side == "left":
            left_vec = vec
        elif side == "right":
            right_vec = vec
        else:
            # fallback assignment when handedness is not available or unclear
            if np.allclose(left_vec, 0.0):
                left_vec = vec
            elif np.allclose(right_vec, 0.0):
                right_vec = vec

    return np.concatenate([left_vec, right_vec], axis=0)


def fixed_length_sequence(
    frames: List[np.ndarray],
    target_frames: int,
) -> Optional[np.ndarray]:
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


def safe_filename(text: str) -> str:
    text = text.strip().lower()
    text = text.replace(" ", "_")
    text = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)
    text = "_".join(part for part in text.split("_") if part)
    return text or "sample"


def process_video(
    video_path: Path,
    target_frames: int,
    frame_stride: int,
    detection_confidence: float,
    tracking_confidence: float,
) -> Tuple[Optional[np.ndarray], int, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0, 0, 0

    frames: List[np.ndarray] = []
    total_read = 0
    used_frames = 0
    valid_frames = 0

    mp_hands = get_hands_module()

    try:
        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        ) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                total_read += 1

                if frame_stride > 1 and ((total_read - 1) % frame_stride != 0):
                    continue

                used_frames += 1

                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception:
                    continue

                result = hands.process(rgb)
                feature = frame_to_feature(result)

                if not np.allclose(feature, 0.0):
                    valid_frames += 1

                frames.append(feature)

    finally:
        cap.release()

    sequence = fixed_length_sequence(frames, target_frames)
    return sequence, total_read, used_frames, valid_frames


def build_label_map(labels: List[str]) -> Dict[str, int]:
    unique = sorted({normalize_label(label) for label in labels})
    return {label: idx for idx, label in enumerate(unique)}


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare ISL landmark sequences from video dataset."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the archive folder that contains Sample Videos and Video_Dataset.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Folder where processed sequences and manifests will be saved.",
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
        "--detection-confidence",
        type=float,
        default=0.5,
        help="MediaPipe hand detection confidence.",
    )
    parser.add_argument(
        "--tracking-confidence",
        type=float,
        default=0.5,
        help="MediaPipe hand tracking confidence.",
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
    sequences_dir = output_dir / "sequences"
    manifest_path = output_dir / "manifest.csv"
    label_map_path = output_dir / "label_map.json"
    summary_path = output_dir / "dataset_summary.json"

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
    sequences_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: List[Dict[str, str]] = []
    stats = {
        "total_videos_found": len(videos),
        "processed": 0,
        "skipped": 0,
        "labels": {},
    }

    print(f"Dataset root : {dataset_root}")
    print(f"Output dir   : {output_dir}")
    print(f"Videos found : {len(videos)}")
    print(f"Target frames: {args.target_frames}")
    print(f"Frame stride : {args.frame_stride}")
    print("Starting preprocessing...\n")

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

        sequence, total_read, used_frames, valid_frames = process_video(
            video_path=video_path,
            target_frames=args.target_frames,
            frame_stride=args.frame_stride,
            detection_confidence=args.detection_confidence,
            tracking_confidence=args.tracking_confidence,
        )

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