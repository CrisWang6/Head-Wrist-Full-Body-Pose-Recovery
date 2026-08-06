#!/usr/bin/env python3
"""Create head-coordinate 3D supervision from the stereo-lifted pose table."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


JOINTS = (
    "LeftFoot",
    "RightFoot",
    "LeftUpLeg",
    "RightUpLeg",
    "LeftArm",
    "RightArm",
    "Spine",
    "Spine2",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
)
STEREO_LIFTED_JOINTS = ("LeftArm", "RightArm", "LeftForeArm", "RightForeArm")


def quaternion_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError("Invalid head quaternion")
    w, x, y, z = np.asarray((qw, qx, qy, qz), dtype=np.float64) / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heatmap-labels", required=True)
    parser.add_argument("--stereo-pose-csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    heatmap_path = Path(args.heatmap_labels).expanduser().resolve()
    pose_csv = Path(args.stereo_pose_csv).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    labels = np.load(heatmap_path, allow_pickle=True)
    frame_indices = np.asarray(labels["frame_indices"], dtype=np.int64)

    required_sequences = set(int(value) for value in frame_indices)
    rows: dict[int, dict[str, str]] = {}
    with pose_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sequence = int(row["seq"])
            if sequence in required_sequences:
                rows[sequence] = row

    poses = np.full((len(frame_indices), len(JOINTS), 3), np.nan, dtype=np.float32)
    valid = np.zeros(len(frame_indices), dtype=bool)
    for index, sequence in enumerate(frame_indices):
        row = rows.get(int(sequence))
        if row is None or int(float(row.get("mocap_valid", "0") or 0)) != 1:
            continue
        if not all(int(float(row.get(f"stereo_{joint}_valid", "0") or 0)) == 1 for joint in STEREO_LIFTED_JOINTS):
            continue
        try:
            head = np.asarray(
                [float(row[f"mocap_Head_world_{axis}"]) for axis in "xyz"],
                dtype=np.float64,
            )
            rotation = quaternion_rotation(
                float(row["mocap_Head_world_qw"]),
                float(row["mocap_Head_world_qx"]),
                float(row["mocap_Head_world_qy"]),
                float(row["mocap_Head_world_qz"]),
            )
            world = np.asarray(
                [
                    [float(row[f"mocap_{joint}_world_{axis}"]) for axis in "xyz"]
                    for joint in JOINTS
                ],
                dtype=np.float64,
            )
            # CSV positions are centimeters. Row vectors multiply R_world_from_head.
            poses[index] = ((world - head) @ rotation / 100.0).astype(np.float32)
            valid[index] = bool(np.isfinite(poses[index]).all())
        except (KeyError, TypeError, ValueError):
            continue

    if not bool(valid.any()):
        raise RuntimeError("No valid stereo-lifted 3D supervision was produced")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray(["egorear_headbc_stage3_stereo_lifted_v1"]),
        frame_indices=frame_indices,
        joint_names=np.asarray(JOINTS),
        pose_head_m=poses,
        valid=valid,
        coordinate_convention=np.asarray(
            ["p_head = R_world_from_head^T @ (p_world - p_head_world)"]
        ),
        source_csv=np.asarray([str(pose_csv)]),
        stereo_lifted_joints=np.asarray(STEREO_LIFTED_JOINTS),
    )
    summary = {
        "output": str(output),
        "frames_total": len(frame_indices),
        "frames_valid_all_four_stereo_joints": int(valid.sum()),
        "joint_names": list(JOINTS),
        "stereo_lifted_joints": list(STEREO_LIFTED_JOINTS),
        "unit": "meter",
        "coordinate_frame": "mocap Head position and orientation",
    }
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
