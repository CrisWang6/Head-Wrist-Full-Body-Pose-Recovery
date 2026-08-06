#!/usr/bin/env python3
"""Prepare synchronized raw/fused 3D skeleton data for the 0722_2 comparison."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import project_0722_2_camc_hybrid_2d_upper_rigid_wrist as hybrid
import project_0722_abx2_subject_scaled as kin
import project_joints as base


JOINTS = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Neck",
    "Neck1",
    "Head",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
]

EDGES = [
    ("Hips", "Spine"),
    ("Spine", "Spine1"),
    ("Spine1", "Spine2"),
    ("Spine2", "Neck"),
    ("Neck", "Neck1"),
    ("Neck1", "Head"),
    ("Spine2", "LeftArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("Spine2", "RightArm"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg"),
    ("LeftUpLeg", "LeftLeg"),
    ("LeftLeg", "LeftFoot"),
    ("Hips", "RightUpLeg"),
    ("RightUpLeg", "RightLeg"),
    ("RightLeg", "RightFoot"),
]

PIXEL_COLUMNS = {
    "LeftArm": ("left_shoulder_x", "left_shoulder_y"),
    "RightArm": ("right_shoulder_x", "right_shoulder_y"),
    "LeftForeArm": ("left_elbow_x", "left_elbow_y"),
    "RightForeArm": ("right_elbow_x", "right_elbow_y"),
}


def read_rows(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return {int(row["seq"]): row for row in csv.DictReader(handle)}


def omni_unproject_unit(
    pixels: np.ndarray,
    model: dict[str, object],
) -> np.ndarray:
    """Invert the unified omni+radtan model and return camera-frame unit rays."""
    xi = float(model["xi"])
    fx, fy = float(model["fx"]), float(model["fy"])
    cx, cy = float(model["cx"]), float(model["cy"])
    coeffs = np.asarray(model["distortion"], dtype=float)
    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=float
    )
    normalized = cv2.undistortPoints(
        np.asarray(pixels, dtype=float).reshape(-1, 1, 2),
        camera_matrix,
        coeffs,
    ).reshape(-1, 2)
    x = normalized[:, 0]
    y = normalized[:, 1]
    r2 = x * x + y * y
    root = np.sqrt(np.maximum(1.0 + (1.0 - xi * xi) * r2, 1e-12))
    lam = (xi + root) / (1.0 + r2)
    rays = np.column_stack((lam * x, lam * y, lam - xi))
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return rays


def rounded_points(points: dict[str, np.ndarray], origin: np.ndarray) -> list[list[float]]:
    return [
        np.round(np.asarray(points[joint], dtype=float) - origin, 1).tolist()
        for joint in JOINTS
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            r"C:\Users\hand\Desktop\HearWristCam\test_code\joint_projection"
            r"\validation_0722_2_camc_hybrid_2d_upper_rigid_wrist\summary.json"
        ),
    )
    parser.add_argument(
        "--aligned-csv",
        type=Path,
        default=Path(
            r"C:\Users\hand\Desktop\Dataset\0722_2\0711_035935"
            r"\aligned_data\aligned_30hz.csv"
        ),
    )
    parser.add_argument(
        "--keypoints-csv",
        type=Path,
        default=Path(
            r"C:\Users\hand\Desktop\Dataset\0722_2\0711_035935\aligned_data"
            r"\module01_cam_c_shoulder_elbow_2d.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"C:\Users\hand\Desktop\HearWristCam\test_code\joint_projection"
            r"\validation_0722_2_camc_hybrid_3d_comparison\data.json"
        ),
    )
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    selected = [int(value) for value in summary["selected_sequences"]]
    aligned = read_rows(args.aligned_csv)
    keypoints = read_rows(args.keypoints_csv)

    config = base.load_json(hybrid.CONFIG)
    calibration = base.load_json(hybrid.CALIBRATION)
    model = base.load_camera_models(config)["module01_CAM_C"]
    joint_index = {name: index for index, name in enumerate(kin.JOINT_NAMES)}

    frames: list[dict[str, object]] = []
    for seq in selected:
        row = aligned[seq]
        kp = keypoints[seq]
        geometry = hybrid.camera_geometry(
            row=row,
            model=model,
            calibration=calibration,
        )
        raw = {
            joint: np.asarray(geometry["points"][joint_index[joint]], dtype=float)
            for joint in JOINTS
        }
        fused = {joint: point.copy() for joint, point in raw.items()}
        camera_position = np.asarray(geometry["camera_position"], dtype=float)
        camera_rotation = np.asarray(geometry["camera_rotation"], dtype=float)

        pixels = np.array(
            [
                [float(kp[PIXEL_COLUMNS[joint][0]]), float(kp[PIXEL_COLUMNS[joint][1]])]
                for joint in PIXEL_COLUMNS
            ],
            dtype=float,
        )
        rays = omni_unproject_unit(pixels, model)
        for index, joint in enumerate(PIXEL_COLUMNS):
            raw_camera = camera_rotation.T @ (raw[joint] - camera_position)
            radius = float(np.linalg.norm(raw_camera))
            fused[joint] = camera_position + camera_rotation @ (rays[index] * radius)

        fused["LeftHand"] = np.asarray(
            geometry["rigid_wrist_world"]["Left"], dtype=float
        )
        fused["RightHand"] = np.asarray(
            geometry["rigid_wrist_world"]["Right"], dtype=float
        )

        origin = raw["Spine2"].copy()
        deltas = {
            joint: round(float(np.linalg.norm(fused[joint] - raw[joint])), 1)
            for joint in (
                "LeftArm",
                "RightArm",
                "LeftForeArm",
                "RightForeArm",
                "LeftHand",
                "RightHand",
            )
        }
        frames.append(
            {
                "seq": seq,
                "raw": rounded_points(raw, origin),
                "fused": rounded_points(fused, origin),
                "delta_mm": deltas,
            }
        )

    payload = {
        "method": "2D omni ray + original BVH camera range; rigid-derived wrists",
        "units": "mm",
        "joints": JOINTS,
        "edges": [[JOINTS.index(a), JOINTS.index(b)] for a, b in EDGES],
        "modified_indices": [
            JOINTS.index(joint)
            for joint in (
                "LeftArm",
                "RightArm",
                "LeftForeArm",
                "RightForeArm",
                "LeftHand",
                "RightHand",
            )
        ],
        "frames": frames,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    print(f"Wrote {len(frames)} frames to {args.output}")


if __name__ == "__main__":
    main()
