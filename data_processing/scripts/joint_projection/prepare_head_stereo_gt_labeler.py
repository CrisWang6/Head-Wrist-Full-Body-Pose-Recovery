#!/usr/bin/env python3
"""Select high-quality head stereo pairs and prepare metadata for manual 2-D GT."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

import project_0722_2_ch308_raw_bvh as baseline


HERE = Path(__file__).resolve().parent
RECORDING = Path(r"C:\Users\hand\Desktop\Dataset\0722_2\0711_035935")
ALIGNED = RECORDING / "aligned_data" / "aligned_30hz.csv"
POSE = RECORDING / "aligned_data" / "module01_cam_c_shoulder_elbow_2d.csv"
TIMESTAMPS = RECORDING / "timestamps.csv"
OUTPUT = HERE / "validation_0722_2_head_stereo_manual_gt"
SAMPLE_COUNT = 6
RANDOM_SEED = 20260805
MIN_SCORE = 0.80
WEB_SIZE = (960, 600)
SCORE_FIELDS = (
    "left_shoulder_score",
    "right_shoulder_score",
    "left_elbow_score",
    "right_elbow_score",
)


def load_by_seq(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {int(row["seq"]): row for row in csv.DictReader(handle)}


def load_timestamp_rows() -> dict[tuple[str, float], dict[str, str]]:
    result: dict[tuple[str, float], dict[str, str]] = {}
    with TIMESTAMPS.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["module"] == "1" and row["camera"] in {"CAM_B", "CAM_C"}:
                result[(row["camera"], baseline.key(row["device_ts_ms"]))] = row
    return result


def select_candidates(
    aligned: dict[int, dict[str, str]],
    pose: dict[int, dict[str, str]],
    ordinals: dict[str, dict[float, int]],
) -> list[int]:
    candidates: list[tuple[int, float]] = []
    for seq, pose_row in pose.items():
        aligned_row = aligned.get(seq)
        if aligned_row is None or pose_row.get("status") != "ok":
            continue
        scores = [float(pose_row[field]) for field in SCORE_FIELDS]
        if min(scores) < MIN_SCORE:
            continue
        if aligned_row.get("mocap_valid") not in {"1", "1.0", "True", "true"}:
            continue
        mapped = True
        for camera in ("CAM_B", "CAM_C"):
            field = f"module01_{camera}_device_ts_ms"
            if not aligned_row.get(field):
                mapped = False
                break
            if baseline.key(aligned_row[field]) not in ordinals[camera]:
                mapped = False
                break
        if mapped:
            candidates.append((seq, float(np.mean(scores))))
    if len(candidates) < SAMPLE_COUNT:
        raise RuntimeError(f"Only {len(candidates)} high-quality candidates")

    # Deterministic stratified random selection: random within each temporal bin,
    # while retaining motion coverage across the recording.
    rng = random.Random(RANDOM_SEED)
    candidates.sort()
    bins = np.array_split(np.arange(len(candidates)), SAMPLE_COUNT)
    selected: list[int] = []
    for indices in bins:
        pool = [candidates[int(index)] for index in indices]
        # Favor the upper half of confidence within each bin, then sample randomly.
        threshold = float(np.median([score for _, score in pool]))
        upper = [item for item in pool if item[1] >= threshold]
        selected.append(rng.choice(upper)[0])
    return sorted(selected)


def compact_timestamp(row: dict[str, str]) -> dict[str, object]:
    return {
        "module": int(row["module"]),
        "camera": row["camera"],
        "raw_camera_seq": int(row["seq"]),
        "device_ts_ms": float(row["device_ts_ms"]),
        "exposure_start_ts_ms": float(row["exposure_start_ts_ms"]),
        "exposure_middle_ts_ms": float(row["exposure_middle_ts_ms"]),
        "exposure_end_ts_ms": float(row["exposure_end_ts_ms"]),
        "host_ts_ms": float(row["host_ts_ms"]),
        "encoded_bytes": int(row["bytes"]),
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    aligned = load_by_seq(ALIGNED)
    pose = load_by_seq(POSE)
    ordinals = baseline.load_timestamp_ordinals()
    timestamp_rows = load_timestamp_rows()
    selected = select_candidates(aligned, pose, ordinals)
    selected_rows = [aligned[seq] for seq in selected]

    baseline.OUTPUT = OUTPUT
    images: dict[str, dict[int, Path]] = {}
    timeline_reports: dict[str, object] = {}
    for camera in ("CAM_B", "CAM_C"):
        images[camera], timeline_reports[camera] = baseline.extract_images(
            camera, selected_rows, ordinals
        )

    request_index = {
        (camera, int(item["seq"])): item
        for camera, report in timeline_reports.items()
        for item in report["requests"]
    }
    web_dir = OUTPUT / "web_images"
    web_dir.mkdir(parents=True, exist_ok=True)
    samples: list[dict[str, object]] = []
    for seq in selected:
        aligned_row = aligned[seq]
        pose_row = pose[seq]
        views: dict[str, object] = {}
        for camera in ("CAM_B", "CAM_C"):
            source = cv2.imread(str(images[camera][seq]), cv2.IMREAD_COLOR)
            if source is None:
                raise RuntimeError(f"Missing source image {camera} seq={seq}")
            resized = cv2.resize(source, WEB_SIZE, interpolation=cv2.INTER_AREA)
            destination = web_dir / f"seq_{seq:06d}_{camera}.jpg"
            cv2.imwrite(str(destination), resized, [cv2.IMWRITE_JPEG_QUALITY, 72])
            device_ts = baseline.key(aligned_row[f"module01_{camera}_device_ts_ms"])
            timing = compact_timestamp(timestamp_rows[(camera, device_ts)])
            request = request_index[(camera, seq)]
            views[camera] = {
                "image_path": str(destination),
                "original_size": [1920, 1200],
                "embedded_size": [WEB_SIZE[0], WEB_SIZE[1]],
                "decoded_frame_index": int(request["decoded_index"]),
                "capture_ordinal": int(request["capture_ordinal"]),
                **timing,
            }
        samples.append(
            {
                "aligned_seq": seq,
                "mocap_frame_index": int(aligned_row["mocap_frame_index"]),
                "mocap_time_sec": float(aligned_row["mocap_time_sec"]),
                "mocap_time_sec_target": float(aligned_row["mocap_time_sec_target"]),
                "mocap_nearest_dt_ms": float(aligned_row["mocap_nearest_dt_ms"]),
                "selection_quality": {
                    "source": "Sapiens2-0.4B CAM_C shoulder/elbow confidence",
                    "minimum_score": min(float(pose_row[field]) for field in SCORE_FIELDS),
                    "mean_score": float(np.mean([float(pose_row[field]) for field in SCORE_FIELDS])),
                    "scores": {field: float(pose_row[field]) for field in SCORE_FIELDS},
                },
                "views": views,
            }
        )

    manifest = {
        "schema": "head_stereo_manual_2d_gt.labeler_source.v1",
        "recording": str(RECORDING),
        "selection": {
            "sample_count": SAMPLE_COUNT,
            "random_seed": RANDOM_SEED,
            "minimum_joint_confidence": MIN_SCORE,
            "method": "stratified random sample from high-confidence external 2-D pose moments",
        },
        "joint_order": [
            "nose",
            "left_shoulder", "right_shoulder",
            "left_elbow", "right_elbow",
            "left_wrist", "right_wrist",
            "left_hip", "right_hip",
            "left_knee", "right_knee",
            "left_ankle", "right_ankle",
            "left_toe", "right_toe",
        ],
        "samples": samples,
        "timeline_reports": timeline_reports,
    }
    (OUTPUT / "labeler_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"selected": selected, "output": str(OUTPUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
