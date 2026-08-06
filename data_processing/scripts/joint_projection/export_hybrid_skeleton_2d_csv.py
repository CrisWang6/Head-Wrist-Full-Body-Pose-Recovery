#!/usr/bin/env python3
"""Export final CAM_B/C hybrid skeleton image coordinates by aligned seq."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import project_joints as base
from render_camc_hybrid_skeleton_video import (
    JOINT_NAMES,
    RTMPOSE_FIELDS as JOINT_FIELDS,
    camera_geometry,
    in_image,
    load_csv,
)


CAMERAS = ("B", "C")
REMOVED_JOINTS = {"LeftShoulder", "RightShoulder"}
OUTPUT_JOINTS = [name for name in JOINT_NAMES if name not in REMOVED_JOINTS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    return parser.parse_args()


def blank_joint_fields(row: dict[str, object], camera: str) -> None:
    prefix = f"module01_CAM_{camera}"
    for joint in OUTPUT_JOINTS:
        row[f"{prefix}_{joint}_x_px"] = ""
        row[f"{prefix}_{joint}_y_px"] = ""


def set_point(
    row: dict[str, object],
    camera: str,
    joint: str,
    point: np.ndarray,
) -> bool:
    if not in_image(point):
        return False
    prefix = f"module01_CAM_{camera}_{joint}"
    row[f"{prefix}_x_px"] = float(point[0])
    row[f"{prefix}_y_px"] = float(point[1])
    return True


def main() -> int:
    args = parse_args()
    aligned = load_csv(args.recording / "aligned_data" / "aligned_30hz.csv")
    poses = {
        camera: {
            int(row["seq"]): row
            for row in load_csv(
                args.recording
                / "aligned_data"
                / f"module01_cam_{camera.lower()}_shoulder_elbow_2d.csv"
            )
        }
        for camera in CAMERAS
    }

    config = base.load_json(
        args.project_dir / "projection_config_0722_head_ch3_08.json"
    )
    calibration = base.load_json(
        args.project_dir
        / "validation_0722_h265_fixed_time_calibration"
        / "calibration_fixed_time.json"
    )
    models = base.load_camera_models(config)
    joint_index = {name: index for index, name in enumerate(JOINT_NAMES)}

    output_rows: list[dict[str, object]] = []
    source_counts = {
        camera: {
            "rows_with_timestamp": 0,
            "timestamp_verified": 0,
            "sapiens_joint_points": 0,
            "rigid_wrist_points": 0,
            "bvh_joint_points": 0,
        }
        for camera in CAMERAS
    }
    timestamp_mismatches: list[dict[str, object]] = []

    for aligned_row in aligned:
        seq = int(aligned_row["seq"])
        output: dict[str, object] = {"seq": seq}
        for camera in CAMERAS:
            prefix = f"module01_CAM_{camera}"
            timestamp = aligned_row[f"{prefix}_device_ts_ms"].strip()
            pose = poses[camera].get(seq)
            output[f"{prefix}_device_ts_ms"] = timestamp
            output[f"{prefix}_decoded_frame_index"] = ""
            output[f"{prefix}_status"] = "missing_aligned_timestamp"
            blank_joint_fields(output, camera)

            if not timestamp:
                continue
            source_counts[camera]["rows_with_timestamp"] += 1
            if pose is None:
                output[f"{prefix}_status"] = "pose_row_missing"
                continue

            pose_timestamp = pose["device_ts_ms"].strip()
            if (
                not pose_timestamp
                or abs(float(timestamp) - float(pose_timestamp)) > 1e-6
            ):
                output[f"{prefix}_status"] = "timestamp_mismatch"
                timestamp_mismatches.append(
                    {
                        "seq": seq,
                        "camera": f"CAM_{camera}",
                        "aligned_device_ts_ms": timestamp,
                        "pose_device_ts_ms": pose_timestamp,
                    }
                )
                continue

            source_counts[camera]["timestamp_verified"] += 1
            output[f"{prefix}_decoded_frame_index"] = pose[
                "decoded_frame_index"
            ]
            output[f"{prefix}_status"] = pose["status"]
            if pose["status"] != "ok":
                continue

            geometry = camera_geometry(
                aligned_row,
                models[f"module01_CAM_{camera}"],
                calibration,
                camera,
            )
            uv = np.asarray(geometry["uv_raw"])
            valid = np.asarray(geometry["valid_raw"], dtype=bool)

            for joint in OUTPUT_JOINTS:
                if joint in JOINT_FIELDS or joint in {"LeftHand", "RightHand"}:
                    continue
                index = joint_index[joint]
                if valid[index] and set_point(output, camera, joint, uv[index]):
                    source_counts[camera]["bvh_joint_points"] += 1

            for joint, fields in JOINT_FIELDS.items():
                if pose[fields[0]].strip() and pose[fields[1]].strip():
                    point = np.asarray(
                        [float(pose[fields[0]]), float(pose[fields[1]])]
                    )
                    if set_point(output, camera, joint, point):
                        source_counts[camera]["sapiens_joint_points"] += 1

            for side, joint in (("Left", "LeftHand"), ("Right", "RightHand")):
                if set_point(
                    output, camera, joint, geometry["rigid_wrist_uv"][side]
                ):
                    source_counts[camera]["rigid_wrist_points"] += 1

        output_rows.append(output)

    columns = ["seq"]
    for camera in CAMERAS:
        prefix = f"module01_CAM_{camera}"
        columns.extend(
            [
                f"{prefix}_device_ts_ms",
                f"{prefix}_decoded_frame_index",
                f"{prefix}_status",
            ]
        )
        for joint in OUTPUT_JOINTS:
            columns.extend(
                [f"{prefix}_{joint}_x_px", f"{prefix}_{joint}_y_px"]
            )

    output_path = (
        args.recording
        / "aligned_data"
        / "module01_cam_bc_hybrid_skeleton_2d.csv"
    )
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output_rows)

    report = {
        "schema": "module01_cam_bc_hybrid_skeleton_2d.v1",
        "output": str(output_path),
        "rows": len(output_rows),
        "image_size": [1920, 1200],
        "joint_count_per_camera": len(OUTPUT_JOINTS),
        "joints": OUTPUT_JOINTS,
        "removed_joints": sorted(REMOVED_JOINTS),
        "sources": {
            "shoulders_elbows": "per-camera Sapiens2-0.4B CSV",
            "wrists": "CH3_01/CH3_07 rigid-derived",
            "camera_pose": "CH3_08 rigid",
            "other_joints": "original BVH",
        },
        "counts": source_counts,
        "timestamp_mismatch_count": len(timestamp_mismatches),
        "timestamp_mismatch_examples": timestamp_mismatches[:10],
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
