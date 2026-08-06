#!/usr/bin/env python3
"""Build stage-1 heatmap labels directly from timestamped 2D joint coordinates."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


CAMERAS = ("CAM_B", "CAM_C")
JOINTS = (
    "LeftFoot", "RightFoot",
    "LeftUpLeg", "RightUpLeg",
    "LeftArm", "RightArm",
    "Spine", "Spine2",
    "LeftForeArm", "RightForeArm",
    "LeftHand", "RightHand",
)
SKELETON = (
    (0, 2), (1, 3), (2, 3), (2, 6), (3, 6), (6, 7),
    (7, 4), (7, 5), (4, 5), (4, 8), (8, 10), (5, 9), (9, 11),
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data/labels/real_0722_01_head2cam_direct2d/heatmap_labels_114x64.npz",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=root / "logs/real_0722_01_head2cam_direct2d_label_previews",
    )
    parser.add_argument("--source-width", type=int, default=1920)
    parser.add_argument("--source-height", type=int, default=1200)
    parser.add_argument("--heatmap-width", type=int, default=114)
    parser.add_argument("--heatmap-height", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument(
        "--max-stereo-delta-ms",
        type=float,
        default=1.0,
        help="Reject B/C pairs whose device timestamps differ by more than this.",
    )
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def finite_float(value: str | None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def load_camera_records(
    csv_path: Path,
    images_root: Path,
    source_size: tuple[int, int],
) -> dict[str, list[dict[str, object]]]:
    records: dict[str, list[dict[str, object]]] = {camera: [] for camera in CAMERAS}
    width, height = source_size
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        required = {"seq"}
        for camera in CAMERAS:
            prefix = f"module01_{camera}"
            required.update(
                {
                    f"{prefix}_device_ts_ms",
                    f"{prefix}_decoded_frame_index",
                    f"{prefix}_status",
                }
            )
            for joint in JOINTS:
                required.add(f"{prefix}_{joint}_x_px")
                required.add(f"{prefix}_{joint}_y_px")
        missing = sorted(required - columns)
        if missing:
            raise KeyError(f"Direct-2D CSV is missing required columns: {missing}")

        for row in reader:
            aligned_seq = int(row["seq"])
            for camera in CAMERAS:
                prefix = f"module01_{camera}"
                if row[f"{prefix}_status"] != "ok":
                    continue
                timestamp = finite_float(row[f"{prefix}_device_ts_ms"])
                frame_text = row[f"{prefix}_decoded_frame_index"].strip()
                if not math.isfinite(timestamp) or not frame_text:
                    continue
                decoded_index = int(frame_text)
                points = np.full((len(JOINTS), 2), np.nan, dtype=np.float32)
                visible = np.zeros(len(JOINTS), dtype=bool)
                for joint_index, joint in enumerate(JOINTS):
                    x = finite_float(row[f"{prefix}_{joint}_x_px"])
                    y = finite_float(row[f"{prefix}_{joint}_y_px"])
                    points[joint_index] = (x, y)
                    visible[joint_index] = (
                        math.isfinite(x) and math.isfinite(y)
                        and 0.0 <= x < width and 0.0 <= y < height
                    )
                image_path = (
                    images_root / "module01" / camera / f"frame_{decoded_index:06d}.jpg"
                )
                records[camera].append(
                    {
                        "timestamp_ms": timestamp,
                        "decoded_index": decoded_index,
                        "aligned_seq": aligned_seq,
                        "points": points,
                        "visible": visible,
                        "image_path": image_path,
                    }
                )

    for camera in CAMERAS:
        records[camera].sort(key=lambda item: float(item["timestamp_ms"]))
        timestamps = np.asarray([item["timestamp_ms"] for item in records[camera]], dtype=np.float64)
        decoded = np.asarray([item["decoded_index"] for item in records[camera]], dtype=np.int64)
        if len(records[camera]) == 0:
            raise RuntimeError(f"No usable direct-2D rows found for {camera}")
        if np.any(np.diff(timestamps) <= 0):
            raise RuntimeError(f"{camera} device timestamps are not strictly increasing")
        if np.any(np.diff(decoded) <= 0):
            raise RuntimeError(f"{camera} decoded frame indices are not strictly increasing")
    return records


def pair_by_timestamp(
    records: dict[str, list[dict[str, object]]],
    max_delta_ms: float,
) -> tuple[list[tuple[dict[str, object], dict[str, object]]], dict[str, int]]:
    left = records["CAM_B"]
    right = records["CAM_C"]
    i = j = 0
    pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    skipped_b = skipped_c = 0
    while i < len(left) and j < len(right):
        delta = float(left[i]["timestamp_ms"]) - float(right[j]["timestamp_ms"])
        if abs(delta) <= max_delta_ms:
            pairs.append((left[i], right[j]))
            i += 1
            j += 1
        elif delta < 0.0:
            skipped_b += 1
            i += 1
        else:
            skipped_c += 1
            j += 1
    skipped_b += len(left) - i
    skipped_c += len(right) - j
    if not pairs:
        raise RuntimeError(
            f"No CAM_B/C frame pairs are within {max_delta_ms:.6f} ms"
        )
    return pairs, {"CAM_B": skipped_b, "CAM_C": skipped_c}


def write_previews(
    preview_dir: Path,
    image_paths: np.ndarray,
    points: np.ndarray,
    visible: np.ndarray,
    timestamps_ms: np.ndarray,
    source_size: tuple[int, int],
    count: int,
) -> None:
    if count <= 0:
        return
    preview_dir.mkdir(parents=True, exist_ok=True)
    selected = np.linspace(0, len(image_paths) - 1, min(count, len(image_paths)), dtype=int)
    source_width, source_height = source_size
    for sample_index in selected:
        for camera_index, camera in enumerate(CAMERAS):
            image = cv2.imread(str(image_paths[sample_index, camera_index]), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read preview image {image_paths[sample_index, camera_index]}")
            scaled = points[sample_index, camera_index].copy()
            scaled[:, 0] *= image.shape[1] / source_width
            scaled[:, 1] *= image.shape[0] / source_height
            mask = visible[sample_index, camera_index]
            for a, b in SKELETON:
                if mask[a] and mask[b]:
                    cv2.line(
                        image, tuple(np.round(scaled[a]).astype(int)),
                        tuple(np.round(scaled[b]).astype(int)), (0, 0, 255), 2, cv2.LINE_AA,
                    )
            for joint_index in np.flatnonzero(mask):
                cv2.circle(
                    image, tuple(np.round(scaled[joint_index]).astype(int)),
                    3, (0, 0, 255), -1, cv2.LINE_AA,
                )
            cv2.imwrite(
                str(preview_dir / f"sample_{sample_index:06d}_{camera}.jpg"),
                image,
                [cv2.IMWRITE_JPEG_QUALITY, 94],
            )


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    csv_path = (
        args.csv.expanduser().resolve()
        if args.csv is not None
        else dataset_root / "aligned_data" / "module01_cam_bc_hybrid_skeleton_2d.csv"
    )
    images_root = (
        args.images_root.expanduser().resolve()
        if args.images_root is not None
        else dataset_root / "images_direct_2d_456x256"
    )
    source_size = (args.source_width, args.source_height)
    records = load_camera_records(csv_path, images_root, source_size)
    pairs, skipped = pair_by_timestamp(records, args.max_stereo_delta_ms)
    if args.limit > 0:
        pairs = pairs[: args.limit]

    count = len(pairs)
    keypoints = np.full((count, 2, len(JOINTS), 2), np.nan, dtype=np.float32)
    visible = np.zeros((count, 2, len(JOINTS)), dtype=bool)
    device_ts_ms = np.empty((count, 2), dtype=np.float64)
    decoded_indices = np.empty((count, 2), dtype=np.int32)
    source_aligned_seq = np.empty((count, 2), dtype=np.int32)
    image_paths = np.empty((count, 2), dtype=object)
    missing_images = []
    for sample_index, pair in enumerate(pairs):
        for camera_index, item in enumerate(pair):
            keypoints[sample_index, camera_index] = item["points"]
            visible[sample_index, camera_index] = item["visible"]
            device_ts_ms[sample_index, camera_index] = item["timestamp_ms"]
            decoded_indices[sample_index, camera_index] = item["decoded_index"]
            source_aligned_seq[sample_index, camera_index] = item["aligned_seq"]
            image_paths[sample_index, camera_index] = str(item["image_path"])
            if not Path(str(item["image_path"])).is_file():
                missing_images.append(str(item["image_path"]))
    if missing_images:
        raise RuntimeError(
            f"{len(missing_images)} paired images are missing; first paths: {missing_images[:10]}"
        )

    stereo_delta_ms = np.abs(device_ts_ms[:, 0] - device_ts_ms[:, 1])
    if float(stereo_delta_ms.max()) > args.max_stereo_delta_ms + 1e-9:
        raise AssertionError("Internal timestamp-pairing tolerance violation")
    joint_mask = np.ones((2, len(JOINTS)), dtype=bool)
    visible &= joint_mask[None, :, :]

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    wrist_keypoints = np.full((count, 2, 7, 2), np.nan, dtype=np.float32)
    np.savez_compressed(
        output,
        schema_version=np.asarray(["egorear_real_head2cam_direct2d_head12_v1"]),
        keypoints=keypoints,
        visible=visible,
        joint_mask=joint_mask,
        head_keypoints=keypoints,
        head_visible=visible,
        head_joint_mask=joint_mask,
        wrist_keypoints=wrist_keypoints,
        wrist_visible=np.zeros((count, 2, 7), dtype=bool),
        wrist_joint_mask=np.zeros((2, 7), dtype=bool),
        camera_is_head=np.ones(2, dtype=bool),
        camera_is_wrist=np.zeros(2, dtype=bool),
        camera_names=np.asarray(["module01_CAM_B", "module01_CAM_C"]),
        joints=np.asarray(JOINTS),
        head_camera_joints=np.asarray(JOINTS),
        head_source_mocap_joints=np.asarray(JOINTS),
        wrist_camera_joints=np.asarray(
            ("L_Ankle", "R_Ankle", "L_Knee", "R_Knee", "L_Hip", "R_Hip", "Spine1")
        ),
        video_paths=np.asarray(["", ""]),
        image_paths=np.asarray(image_paths, dtype=str),
        frame_indices=np.arange(count, dtype=np.int64),
        decoded_frame_indices=decoded_indices,
        source_aligned_seq=source_aligned_seq,
        camera_device_ts_ms=device_ts_ms,
        stereo_delta_ms=stereo_delta_ms,
        video_size=np.asarray(source_size, dtype=np.int32),
        heatmap_size=np.asarray([args.heatmap_width, args.heatmap_height], dtype=np.int32),
        sigma=np.asarray([args.sigma], dtype=np.float32),
        projection_model=np.asarray(["none_direct_2d_ground_truth"]),
        fisheye_fov_deg=np.asarray([180.0], dtype=np.float32),
        source_csv=np.asarray([str(csv_path)]),
        source_render_dir=np.asarray([""]),
    )

    manifest = {
        "schema": "egorear.real_head2cam_direct2d_manifest.v1",
        "output": str(output),
        "source_csv": str(csv_path),
        "images_root": str(images_root),
        "frames": count,
        "camera_input_rows": {camera: len(records[camera]) for camera in CAMERAS},
        "unpaired_rows": skipped,
        "pairing": "ordered one-to-one nearest device timestamp",
        "max_allowed_stereo_delta_ms": args.max_stereo_delta_ms,
        "actual_stereo_delta_ms": {
            "min": float(stereo_delta_ms.min()),
            "mean": float(stereo_delta_ms.mean()),
            "p95": float(np.percentile(stereo_delta_ms, 95)),
            "p99": float(np.percentile(stereo_delta_ms, 99)),
            "max": float(stereo_delta_ms.max()),
        },
        "source_video_size": list(source_size),
        "heatmap_size": [args.heatmap_width, args.heatmap_height],
        "ground_truth": "direct per-camera 2D coordinates; no 3D human-pose projection",
        "joints": list(JOINTS),
        "masked_channels": [],
        "visible_points": {
            camera: int(visible[:, index].sum())
            for index, camera in enumerate(CAMERAS)
        },
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_previews(
        args.preview_dir.expanduser().resolve(),
        np.asarray(image_paths, dtype=str),
        keypoints,
        visible,
        device_ts_ms,
        source_size,
        args.preview_count,
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
