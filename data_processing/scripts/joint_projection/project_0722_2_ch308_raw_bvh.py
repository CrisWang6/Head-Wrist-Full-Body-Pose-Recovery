#!/usr/bin/env python3
"""Baseline projection for 0722_2: raw BVH skeleton + CH3_08 head camera pose."""

from __future__ import annotations

import csv
import json
import random
import subprocess
from pathlib import Path

import cv2
import numpy as np

import inspect_h265_timeline as timeline_tools
import project_0722_abx2_subject_scaled as kin
import project_joints as base


HERE = Path(__file__).resolve().parent
DATASET = Path(r"C:\Users\hand\Desktop\Dataset\0722_2")
RECORDING = DATASET / "0711_035935"
ALIGNED = RECORDING / "aligned_data" / "aligned_30hz.csv"
TIMESTAMPS = RECORDING / "timestamps.csv"
CONFIG = HERE / "projection_config_0722_head_ch3_08.json"
CALIBRATION = HERE / "validation_0722_h265_fixed_time_calibration" / "calibration_fixed_time.json"
OUTPUT = HERE / "validation_0722_2_ch308_raw_bvh"
FFMPEG = Path(r"C:\Users\hand\miniconda3\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe")
CAMERAS = {"CAM_B": "module01_CAM_B", "CAM_C": "module01_CAM_C"}
CAMERA_FILE_PREFIX = "module01_D45D2E00"
SAMPLE_COUNT = 16
RANDOM_SEED = 20260722
WIDTH, HEIGHT = 1920, 1200


def key(value: str | float) -> float:
    return round(float(value), 6)


def load_timestamp_ordinals() -> dict[str, dict[float, int]]:
    rows: dict[str, list[float]] = {camera: [] for camera in CAMERAS}
    with TIMESTAMPS.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["module"] == "1" and row["camera"] in rows:
                rows[row["camera"]].append(key(row["device_ts_ms"]))
    return {
        camera: {timestamp: ordinal for ordinal, timestamp in enumerate(values)}
        for camera, values in rows.items()
    }


def select_rows(ordinals: dict[str, dict[float, int]]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    required = [
        *(f"mocap_{name}_world_{axis}" for name in kin.JOINT_NAMES for axis in "xyz"),
        *(f"mocap_CH3_08_Rigid_K_world_{suffix}" for suffix in ("x", "y", "z", "qw", "qx", "qy", "qz")),
    ]
    with ALIGNED.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("mocap_valid") not in {"1", "1.0", "True", "true"}:
                continue
            if not all(row.get(field) for field in required):
                continue
            if not all(row.get(f"module01_{camera}_device_ts_ms") for camera in CAMERAS):
                continue
            if not all(key(row[f"module01_{camera}_device_ts_ms"]) in ordinals[camera] for camera in CAMERAS):
                continue
            candidates.append(row)
    if len(candidates) < SAMPLE_COUNT:
        raise RuntimeError(f"Only {len(candidates)} valid aligned rows")
    # Stratified random sampling covers the whole motion while remaining reproducible.
    rng = random.Random(RANDOM_SEED)
    bins = np.array_split(np.arange(len(candidates)), SAMPLE_COUNT)
    selected = [candidates[rng.choice(bin_indices.tolist())] for bin_indices in bins if len(bin_indices)]
    # CAM_C around this exact sample is visibly damaged after a missing HEVC
    # reference picture; use a nearby clean motion sample for visual QA.
    selected = [
        min(candidates, key=lambda item: abs(int(item["seq"]) - 1800))
        if int(row["seq"]) == 1668 else row
        for row in selected
    ]
    return sorted(selected, key=lambda row: int(row["seq"]))


def decoded_capture_map(camera: str) -> tuple[np.ndarray, dict[str, object]]:
    timeline_tools.ROOT = RECORDING
    report = timeline_tools.inspect(camera)
    dropped_after = sorted(
        int(event["near_decoded_index"])
        for event in report["decoder_events"]
        if "Duplicate POC" in str(event["message"])
    )
    decoded = np.arange(int(report["decoded_frames"]), dtype=np.int64)
    capture = decoded.copy()
    for after in dropped_after:
        capture += decoded > after
    compact = {
        "timestamp_rows": None,
        "decoded_frames": int(report["decoded_frames"]),
        "duplicate_poc_drops": len(dropped_after),
        "drop_after_decoded_indices": dropped_after,
    }
    return capture, compact


def extract_images(camera: str, rows: list[dict[str, str]], ordinals: dict[str, dict[float, int]]) -> tuple[dict[int, Path], dict[str, object]]:
    capture_map, timeline_report = decoded_capture_map(camera)
    requests: list[dict[str, int]] = []
    for row in rows:
        capture_ordinal = ordinals[camera][key(row[f"module01_{camera}_device_ts_ms"])]
        decoded_index = int(np.argmin(np.abs(capture_map - capture_ordinal)))
        requests.append({"seq": int(row["seq"]), "capture_ordinal": capture_ordinal, "decoded_index": decoded_index})
    requests.sort(key=lambda item: item["decoded_index"])
    expression = "+".join(f"eq(n\\,{item['decoded_index']})" for item in requests)
    source = RECORDING / f"{CAMERA_FILE_PREFIX}_{camera}.h265"
    command = [
        str(FFMPEG), "-hide_banner", "-loglevel", "error", "-i", str(source),
        "-vf", f"select={expression}", "-vsync", "0", "-pix_fmt", "bgr24",
        "-f", "rawvideo", "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    destination = OUTPUT / "source_frames" / camera
    destination.mkdir(parents=True, exist_ok=True)
    result: dict[int, Path] = {}
    frame_bytes = WIDTH * HEIGHT * 3
    for item in requests:
        buffer = process.stdout.read(frame_bytes)
        if len(buffer) != frame_bytes:
            break
        image = np.frombuffer(buffer, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
        path = destination / f"seq_{item['seq']:06d}.jpg"
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 96])
        result[item["seq"]] = path
    process.stdout.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    return_code = process.wait()
    if return_code != 0 or len(result) != len(requests):
        raise RuntimeError(f"H265 extraction failed for {camera}: return={return_code}, images={len(result)}/{len(requests)}, stderr={stderr[-500:]}")
    timeline_report["timestamp_rows"] = len(ordinals[camera])
    timeline_report["requests"] = requests
    return result, timeline_report


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    ordinals = load_timestamp_ordinals()
    rows = select_rows(ordinals)
    source_images: dict[str, dict[int, Path]] = {}
    timeline_reports: dict[str, object] = {}
    for camera in CAMERAS:
        source_images[camera], timeline_reports[camera] = extract_images(camera, rows, ordinals)

    config = base.load_json(CONFIG)
    models = base.load_camera_models(config)
    calibration = base.load_json(CALIBRATION)
    head_axes = np.column_stack(([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
    head_to_rigid_mm = np.asarray([-2.0, 53.8, 135.5], dtype=np.float64)
    head_joint_in_rigid_mm = -head_axes.T @ head_to_rigid_mm
    joint_index = {name: i for i, name in enumerate(kin.JOINT_NAMES)}
    reports: dict[str, object] = {}

    for camera, camera_key in CAMERAS.items():
        model = models[camera_key]
        r_rigid_camera = np.asarray(calibration[f"R_rigid_cam_{camera[-1]}"], dtype=np.float64)
        p_rigid_camera = np.asarray(calibration[f"p_rigid_cam_{camera[-1]}_mm"], dtype=np.float64)
        head_to_camera_rigid = p_rigid_camera - head_joint_in_rigid_mm
        destination = OUTPUT / camera_key
        destination.mkdir(parents=True, exist_ok=True)
        camera_rows: list[dict[str, object]] = []
        for row in rows:
            seq = int(row["seq"])
            points_bvh = np.asarray([
                [float(row[f"mocap_{name}_world_{axis}"]) * 10.0 for axis in "xyz"]
                for name in kin.JOINT_NAMES
            ], dtype=np.float64)
            rigid_position = np.asarray([
                float(row[f"mocap_CH3_08_Rigid_K_world_{axis}"]) * 1000.0 for axis in "xyz"
            ], dtype=np.float64)
            rigid_q = np.asarray([
                float(row[f"mocap_CH3_08_Rigid_K_world_q{axis}"]) for axis in "wxyz"
            ], dtype=np.float64)
            rigid_q /= np.linalg.norm(rigid_q)
            r_world_head_rigid = kin.q_to_matrix(rigid_q)
            bvh_head = points_bvh[joint_index["Head"]]
            camera_position = bvh_head + r_world_head_rigid @ head_to_camera_rigid
            camera_rotation = r_world_head_rigid @ r_rigid_camera
            image = cv2.imread(str(source_images[camera][seq]), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read {source_images[camera][seq]}")
            kin.draw(image, points_bvh, camera_position, camera_rotation, model, (0, 255, 255), 4)
            cv2.rectangle(image, (0, 0), (image.shape[1], 62), (0, 0, 0), -1)
            cv2.putText(image, f"0722_2 raw BVH + CH3_08  {camera_key}  seq={seq:06d}", (24, 41), cv2.FONT_HERSHEY_SIMPLEX, .76, (0, 255, 255), 2, cv2.LINE_AA)
            output_path = destination / f"seq_{seq:06d}_projection.jpg"
            cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 94])
            camera_rows.append({
                "seq": seq,
                "mocap_time_sec": float(row["mocap_time_sec"]),
                "mocap_nearest_dt_ms": float(row["mocap_nearest_dt_ms"]),
                "image": str(output_path),
            })
        reports[camera_key] = {"sample_count": len(camera_rows), "frames": camera_rows}

    comparison_dir = OUTPUT / "CAM_B_vs_CAM_C"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        seq = int(row["seq"])
        images = [cv2.imread(str(OUTPUT / CAMERAS[camera] / f"seq_{seq:06d}_projection.jpg")) for camera in ("CAM_B", "CAM_C")]
        if all(image is not None for image in images):
            cv2.imwrite(str(comparison_dir / f"seq_{seq:06d}_comparison.jpg"), np.hstack(images), [cv2.IMWRITE_JPEG_QUALITY, 93])

    # Compact overview for quickly judging the complete random sample.
    for camera, camera_key in CAMERAS.items():
        thumbnails = []
        for row in rows:
            seq = int(row["seq"])
            image = cv2.imread(str(OUTPUT / camera_key / f"seq_{seq:06d}_projection.jpg"))
            if image is not None:
                thumbnails.append(cv2.resize(image, (480, 300), interpolation=cv2.INTER_AREA))
        if len(thumbnails) == SAMPLE_COUNT:
            overview = np.vstack([np.hstack(thumbnails[i:i + 4]) for i in range(0, SAMPLE_COUNT, 4)])
            cv2.imwrite(str(OUTPUT / f"overview_{camera_key}.jpg"), overview, [cv2.IMWRITE_JPEG_QUALITY, 92])

    summary = {
        "schema": "0722_2_ch308_raw_bvh_projection.v1",
        "dataset": str(DATASET),
        "method": "raw BVH joint world positions; no bone scaling; no wrist constraints; CH3_08 rigid camera pose",
        "sample_count_per_camera": len(rows),
        "selected_sequences": [int(row["seq"]) for row in rows],
        "camera_calibration": str(CALIBRATION),
        "camera_model": str(CONFIG),
        "timeline": timeline_reports,
        "reports": reports,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "selected_sequences": summary["selected_sequences"], "timeline": timeline_reports}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
