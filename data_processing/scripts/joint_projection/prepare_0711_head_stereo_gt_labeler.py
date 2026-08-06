#!/usr/bin/env python3
"""Prepare synchronized head stereo pairs plus external 2-D pose references."""

from __future__ import annotations

import csv
import json
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(r"C:\Users\hand\Desktop\0711_214559")
FULL = ROOT / "realigned_offset0_rawsource_full6207"
ALIGNED = FULL / "aligned_50hz.csv"
HEAD_TIMESTAMPS = ROOT / "timestamps.csv"
EXTERNAL_TIMESTAMPS = FULL / "external_timestamps.csv"
FIVEPOINT = FULL / "final_full_videos" / "global_fivepoint_ch07_full.csv"
HEAD_PROJECTED = FULL / "final_full_videos" / "head_stereo_projected_2d_full.csv"
EXTERNAL_REFERENCE = (
    FULL / "final_full_videos" / "external_stereo_2d_raw_vs_filtered_full.mp4"
)
FFMPEG = Path(
    r"C:\Users\hand\miniconda3\Lib\site-packages\imageio_ffmpeg"
    r"\binaries\ffmpeg-win-x86_64-v7.1.exe"
)
OUTPUT = ROOT / "head_stereo_manual_fullbody_gt_20260805"
SAMPLE_COUNT = 20
RANDOM_SEED = 20260805
HEAD_WEB_SIZE = (480, 300)
REFERENCE_WEB_SIZE = (480, 150)
JOINTS = ("nose", "left_shoulder", "right_shoulder", "left_hip", "right_hip")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def stable_external_candidates(
    aligned: list[dict[str, str]],
) -> tuple[list[int], dict[int, float]]:
    point_rows = load_csv(FIVEPOINT)
    points = np.full((len(aligned), len(JOINTS), 3), np.nan, dtype=np.float64)
    joint_index = {joint: index for index, joint in enumerate(JOINTS)}
    for row in point_rows:
        sequence = int(row["sequence"])
        if 0 <= sequence < len(points) and row["joint"] in joint_index:
            points[sequence, joint_index[row["joint"]]] = [
                float(row["x_m"]), float(row["y_m"]), float(row["z_m"])
            ]
    acceleration = np.full(len(points), np.nan)
    second = points[2:] - 2.0 * points[1:-1] + points[:-2]
    acceleration[1:-1] = np.nanmedian(np.linalg.norm(second, axis=2), axis=1) * 1000.0

    projected: dict[int, dict[str, list[float]]] = {}
    for row in load_csv(HEAD_PROJECTED):
        sequence = int(row["sequence"])
        projected.setdefault(sequence, {})[row["joint"]] = [
            float(row["head_A_u_px"]), float(row["head_A_v_px"]),
            float(row["head_D_u_px"]), float(row["head_D_v_px"]),
        ]

    finite_acc = acceleration[np.isfinite(acceleration)]
    threshold = float(np.percentile(finite_acc, 30.0))
    candidates: list[int] = []
    scores: dict[int, float] = {}
    for sequence, row in enumerate(aligned):
        if not np.isfinite(acceleration[sequence]) or acceleration[sequence] > threshold:
            continue
        required_frames = (
            "module01_CAM_A_source_frame", "module01_CAM_D_source_frame",
            "external_CAM_A_source_frame", "external_CAM_D_source_frame",
        )
        if not all(row.get(field, "").strip() for field in required_frames):
            continue
        # Ensure the five calibration anchors are geometrically in both head images.
        joint_pixels = projected.get(sequence, {})
        if not all(joint in joint_pixels for joint in JOINTS):
            continue
        values = np.asarray([joint_pixels[joint] for joint in JOINTS])
        if not (
            np.all((values[:, 0] >= 0) & (values[:, 0] < 1920))
            and np.all((values[:, 1] >= 0) & (values[:, 1] < 1200))
            and np.all((values[:, 2] >= 0) & (values[:, 2] < 1920))
            and np.all((values[:, 3] >= 0) & (values[:, 3] < 1200))
        ):
            continue
        candidates.append(sequence)
        scores[sequence] = float(acceleration[sequence])
    return candidates, scores


def select_sequences(candidates: list[int]) -> list[int]:
    rng = random.Random(RANDOM_SEED)
    candidates = sorted(candidates)
    bins = np.array_split(np.arange(len(candidates)), SAMPLE_COUNT)
    selected = [
        rng.choice([candidates[int(index)] for index in indices])
        for indices in bins if len(indices)
    ]
    return sorted(selected)


def extract_frames(
    source: Path,
    requests: list[tuple[int, int]],
    size: tuple[int, int],
) -> dict[int, np.ndarray]:
    ordered = sorted(requests, key=lambda item: item[1])
    expression = "+".join(f"eq(n\\,{frame})" for _, frame in ordered)
    command = [
        str(FFMPEG), "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-vf", f"select={expression}", "-vsync", "0", "-pix_fmt", "bgr24",
        "-f", "rawvideo", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    width, height = size
    frame_bytes = width * height * 3
    result: dict[int, np.ndarray] = {}
    for sequence, _ in ordered:
        buffer = process.stdout.read(frame_bytes)
        if len(buffer) != frame_bytes:
            break
        result[sequence] = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3).copy()
    process.stdout.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0 or len(result) != len(ordered):
        raise RuntimeError(
            f"Extraction failed for {source.name}: {len(result)}/{len(ordered)} "
            f"return={return_code} {stderr[-500:]}"
        )
    return result


def head_timestamp_index() -> dict[tuple[str, float], dict[str, str]]:
    index: dict[tuple[str, float], dict[str, str]] = {}
    for row in load_csv(HEAD_TIMESTAMPS):
        if row["module"] == "1" and row["camera"] in {"CAM_A", "CAM_D"}:
            index[(row["camera"], round(float(row["device_ts_ms"]), 6))] = row
    return index


def compact_head_timing(row: dict[str, str]) -> dict[str, object]:
    return {
        "module": int(row["module"]), "camera": row["camera"],
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
    web_dir = OUTPUT / "web_images"
    web_dir.mkdir(parents=True, exist_ok=True)
    aligned = load_csv(ALIGNED)
    candidates, quality = stable_external_candidates(aligned)
    selected = select_sequences(candidates)
    head_frames: dict[str, dict[int, np.ndarray]] = {}
    for camera in ("CAM_A", "CAM_D"):
        requests = [
            (sequence, int(aligned[sequence][f"module01_{camera}_source_frame"]))
            for sequence in selected
        ]
        head_frames[camera] = extract_frames(
            ROOT / f"module01_D45D2E00_{camera}.h265", requests, (1920, 1200)
        )
    references = extract_frames(
        EXTERNAL_REFERENCE, [(sequence, sequence) for sequence in selected], (1920, 600)
    )

    head_times = head_timestamp_index()
    external_times = {
        (row["camera"], int(row["frame_index"])): row
        for row in load_csv(EXTERNAL_TIMESTAMPS)
    }
    samples: list[dict[str, object]] = []
    for sequence in selected:
        aligned_row = aligned[sequence]
        views: dict[str, object] = {}
        for camera in ("CAM_A", "CAM_D"):
            web = cv2.resize(head_frames[camera][sequence], HEAD_WEB_SIZE, interpolation=cv2.INTER_AREA)
            path = web_dir / f"seq_{sequence:04d}_head_{camera}.jpg"
            cv2.imwrite(str(path), web, [cv2.IMWRITE_JPEG_QUALITY, 58])
            device_ts = round(float(aligned_row[f"module01_{camera}_device_ts_ms"]), 6)
            timing = compact_head_timing(head_times[(camera, device_ts)])
            views[camera] = {
                "image_path": str(path), "original_size": [1920, 1200],
                "embedded_size": list(HEAD_WEB_SIZE),
                "source_frame": int(aligned_row[f"module01_{camera}_source_frame"]),
                **timing,
            }
        reference_web = cv2.resize(references[sequence], REFERENCE_WEB_SIZE, interpolation=cv2.INTER_AREA)
        reference_path = web_dir / f"seq_{sequence:04d}_external_pose_reference.jpg"
        cv2.imwrite(str(reference_path), reference_web, [cv2.IMWRITE_JPEG_QUALITY, 60])
        ext = {}
        for camera in ("CAM_A", "CAM_D"):
            original_source_frame = int(aligned_row[f"external_{camera}_source_frame"])
            # external_timestamps.csv is already cropped/reindexed to the
            # 6207-frame aligned clip; aligned_50hz keeps the original source index.
            row = external_times[(camera, sequence)]
            ext[camera] = {
                "aligned_clip_frame": sequence,
                "original_source_frame": original_source_frame,
                "sequence": int(row["sequence"]),
                "exposure_end_device_timestamp_us": int(row["exposure_end_device_timestamp_us"]),
            }
        samples.append({
            "aligned_sequence": sequence,
            "mocap_frame_index": int(aligned_row["mocap_frame_index"]),
            "mocap_time_sec_target": float(aligned_row["mocap_time_sec_target"]),
            "mocap_nearest_time_sec": float(aligned_row["mocap_nearest_time_sec"]),
            "mocap_nearest_dt_ms": float(aligned_row["mocap_nearest_dt_ms"]),
            "external_pose_quality": {
                "metric": "median five-point second difference",
                "value_mm_per_frame2": quality[sequence],
                "candidate_percentile": 30,
            },
            "views": views,
            "external_reference": {
                "image_path": str(reference_path), "embedded_size": list(REFERENCE_WEB_SIZE),
                "layout": "left=external CAM_A raw/filtered pose; right=external CAM_D raw/filtered pose",
                "views": ext,
            },
        })

    manifest = {
        "schema": "0711.head_stereo_fullbody_manual_2d_gt.source.v1",
        "source_root": str(ROOT),
        "selection": {
            "sample_count": SAMPLE_COUNT, "random_seed": RANDOM_SEED,
            "policy": "stratified random among lowest 30% external five-point temporal residual, with all five anchors in both head views",
            "candidate_count": len(candidates),
        },
        "joint_order": [
            "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
            "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
            "right_knee", "left_ankle", "right_ankle", "left_toe", "right_toe",
        ],
        "samples": samples,
    }
    (OUTPUT / "labeler_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"selected": selected, "candidates": len(candidates), "output": str(OUTPUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
