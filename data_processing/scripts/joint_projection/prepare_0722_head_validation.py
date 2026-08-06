#!/usr/bin/env python3
"""Extract aligned module01 CAM_B/C source frames for the 0722 head validation."""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import cv2


DATASET = Path(r"C:\Users\hand\Desktop\Dataset\0722\record_9cam_0711_021044")
ALIGNED_CSV = DATASET / "aligned_data" / "aligned_30hz.csv"
TIMESTAMPS_CSV = DATASET / "timestamps.csv"
OUTPUT_ROOT = Path(__file__).resolve().parent / "validation_0722_head_ch3_08_random100_final"
SOURCE_ROOT = OUTPUT_ROOT / "source_frames" / "module01"
CAMERAS = {
    "CAM_B": DATASET / "module01_D45D2E00_CAM_B.mp4",
    "CAM_C": DATASET / "module01_D45D2E00_CAM_C.mp4",
}
# Cross-camera wrist-tag alignment shows that module01 CAM_B's first decoded
# MP4 frame corresponds to timestamps.csv row 31.  CAM_C is one-to-one.
DECODED_FRAME_INDEX_OFFSET = {"CAM_B": -31, "CAM_C": 0}
FRAME_COUNT = 100
CANDIDATE_COUNT = 120
RANDOM_SEED = 20260722


def timestamp_key(value: str | float) -> float:
    return round(float(value), 6)


def load_source_frame_indices() -> dict[str, dict[float, int]]:
    rows: dict[str, list[float]] = {camera: [] for camera in CAMERAS}
    with TIMESTAMPS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            camera = row["camera"]
            if row["module"] == "1" and camera in rows:
                rows[camera].append(timestamp_key(row["device_ts_ms"]))
    return {
        camera: {device_ts: index for index, device_ts in enumerate(timestamps)}
        for camera, timestamps in rows.items()
    }


def select_aligned_rows(
    source_indices: dict[str, dict[float, int]],
) -> list[dict[str, int | float]]:
    selected: list[dict[str, int | float]] = []
    required_rigid_fields = [
        f"mocap_CH3_08_Rigid_K_world_{suffix}"
        for suffix in ("x", "y", "z", "qw", "qx", "qy", "qz")
    ]
    with ALIGNED_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["mocap_valid"] not in {"1", "1.0", "True", "true"}:
                continue
            if not all(row.get(field) and math.isfinite(float(row[field])) for field in required_rigid_fields):
                continue
            timestamp_fields = [f"module01_{camera}_device_ts_ms" for camera in CAMERAS]
            if not all(row.get(field) for field in timestamp_fields):
                continue
            device_ts = {
                camera: timestamp_key(row[f"module01_{camera}_device_ts_ms"])
                for camera in CAMERAS
            }
            if not all(device_ts[camera] in source_indices[camera] for camera in CAMERAS):
                continue
            selected.append(
                {
                    "aligned_seq": int(row["seq"]),
                    "mocap_nearest_dt_ms": float(row["mocap_nearest_dt_ms"]),
                    **{
                        f"{camera}_source_frame_index": source_indices[camera][device_ts[camera]]
                        for camera in CAMERAS
                    },
                    **{f"{camera}_device_ts_ms": device_ts[camera] for camera in CAMERAS},
                }
            )
    if len(selected) < CANDIDATE_COUNT:
        raise RuntimeError(
            f"Only found {len(selected)} jointly aligned frames; expected at least {CANDIDATE_COUNT}"
        )
    return sorted(
        random.Random(RANDOM_SEED).sample(selected, CANDIDATE_COUNT),
        key=lambda row: row["aligned_seq"],
    )


def extract_camera(camera: str, video_path: Path, selected: list[dict[str, int | float]]) -> None:
    destination = SOURCE_ROOT / camera
    destination.mkdir(parents=True, exist_ok=True)
    target_by_index = {
        int(row[f"{camera}_source_frame_index"]) + DECODED_FRAME_INDEX_OFFSET[camera]: int(
            row["aligned_seq"]
        )
        for row in selected
        if int(row[f"{camera}_source_frame_index"]) + DECODED_FRAME_INDEX_OFFSET[camera] >= 0
    }
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    written = 0
    for source_index, aligned_seq in sorted(target_by_index.items()):
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, source_index):
            continue
        ok, frame = capture.read()
        if not ok:
            continue
        path = destination / f"seq_{aligned_seq:06d}.jpg"
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"Could not write {path}")
        written += 1
    capture.release()
    if written < FRAME_COUNT:
        raise RuntimeError(
            f"{camera}: only extracted {written}/{len(selected)} random candidates; need {FRAME_COUNT}"
        )
    print(f"{camera}: extracted {written}/{len(selected)} random candidates")


def main() -> int:
    source_indices = load_source_frame_indices()
    selected = select_aligned_rows(source_indices)
    for camera, video_path in CAMERAS.items():
        extract_camera(camera, video_path, selected)
    manifest = {
        "schema": "head_validation_source_frames.v1",
        "rigid_anchor": "CH3_08_Rigid_K",
        "aligned_csv": str(ALIGNED_CSV),
        "timestamps_csv": str(TIMESTAMPS_CSV),
        "candidate_count": len(selected),
        "output_sample_count_per_camera": FRAME_COUNT,
        "selection": "120 deterministic random candidates with valid mocap, finite CH3_08 pose, and exact CAM_B/C timestamp matches; projection selects 100 decodable images per camera",
        "random_seed": RANDOM_SEED,
        "decoded_frame_index_offset": DECODED_FRAME_INDEX_OFFSET,
        "frames": selected,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "source_frame_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Prepared random aligned candidates in {SOURCE_ROOT}")
    print(f"Aligned seq range: {selected[0]['aligned_seq']}..{selected[-1]['aligned_seq']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
