#!/usr/bin/env python3
"""Compare literal BVH hierarchy drawing with anatomical semantic remapping."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

import project_0722_abx2_subject_scaled as kin
import project_joints as base


HERE = Path(__file__).resolve().parent
BASELINE = HERE / "validation_0722_2_ch308_raw_bvh"
BASELINE_SUMMARY = BASELINE / "summary.json"
ALIGNED = Path(r"C:\Users\hand\Desktop\Dataset\0722_2\0711_035935\aligned_data\aligned_30hz.csv")
CONFIG = HERE / "projection_config_0722_head_ch3_08.json"
CALIBRATION = HERE / "validation_0722_h265_fixed_time_calibration" / "calibration_fixed_time.json"
OUTPUT = HERE / "validation_0722_2_raw_bvh_semantic_remap"
CAMERAS = {"CAM_B": "module01_CAM_B", "CAM_C": "module01_CAM_C"}

# LeftShoulder/RightShoulder are BVH clavicle/scapular helper nodes and are
# deliberately omitted. LeftArm/RightArm are the anatomical shoulder centers.
SEMANTIC_EDGES = [
    ("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"),
    ("Spine2", "Neck"), ("Neck", "Neck1"), ("Neck1", "Head"),
    ("Neck", "LeftArm"), ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("Neck", "RightArm"), ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), ("LeftLeg", "LeftFoot"),
    ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"), ("RightLeg", "RightFoot"),
]
SEMANTIC_JOINTS = [
    name for name in kin.JOINT_NAMES if name not in {"LeftShoulder", "RightShoulder"}
]


def load_rows(sequences: set[int]) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    with ALIGNED.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            seq = int(row["seq"])
            if seq in sequences:
                result[seq] = row
    missing = sequences - set(result)
    if missing:
        raise RuntimeError(f"Missing aligned rows: {sorted(missing)}")
    return result


def draw_semantic(
    image: np.ndarray,
    points_world: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    model: dict[str, object],
) -> None:
    points_camera = (camera_rotation.T @ (points_world - camera_position).T).T
    uv, valid = base.omni_project(points_camera, model)
    visible = base.in_image(uv, valid, int(model["width"]), int(model["height"]))
    index = {name: i for i, name in enumerate(kin.JOINT_NAMES)}
    color = (255, 255, 0)
    for first, second in SEMANTIC_EDGES:
        a, b = index[first], index[second]
        if visible[a] and visible[b]:
            cv2.line(image, tuple(np.rint(uv[a]).astype(int)), tuple(np.rint(uv[b]).astype(int)), color, 4, cv2.LINE_AA)
    for name in SEMANTIC_JOINTS:
        i = index[name]
        if visible[i]:
            point = tuple(np.rint(uv[i]).astype(int))
            cv2.circle(image, point, 6, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(image, point, 4, color, -1, cv2.LINE_AA)

    # Mark the corrected semantic landmarks for unambiguous inspection.
    labels = {
        "Neck": "C7",
        "LeftArm": "L shoulder",
        "RightArm": "R shoulder",
    }
    for name, label in labels.items():
        i = index[name]
        if visible[i]:
            point = tuple(np.rint(uv[i]).astype(int))
            cv2.putText(image, label, (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 2, cv2.LINE_AA)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    baseline_summary = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
    sequences = [int(value) for value in baseline_summary["selected_sequences"]]
    rows = load_rows(set(sequences))
    models = base.load_camera_models(base.load_json(CONFIG))
    calibration = base.load_json(CALIBRATION)
    index = {name: i for i, name in enumerate(kin.JOINT_NAMES)}

    head_axes = np.column_stack(([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
    head_to_rigid_mm = np.asarray([-2.0, 53.8, 135.5], dtype=np.float64)
    head_joint_in_rigid_mm = -head_axes.T @ head_to_rigid_mm
    reports: dict[str, object] = {}

    for camera, camera_key in CAMERAS.items():
        model = models[camera_key]
        r_rigid_camera = np.asarray(calibration[f"R_rigid_cam_{camera[-1]}"], dtype=np.float64)
        p_rigid_camera = np.asarray(calibration[f"p_rigid_cam_{camera[-1]}_mm"], dtype=np.float64)
        head_to_camera_rigid = p_rigid_camera - head_joint_in_rigid_mm
        destination = OUTPUT / camera_key
        semantic_only_destination = OUTPUT / "semantic_only" / camera_key
        destination.mkdir(parents=True, exist_ok=True)
        semantic_only_destination.mkdir(parents=True, exist_ok=True)

        for seq in sequences:
            row = rows[seq]
            points = np.asarray([
                [float(row[f"mocap_{name}_world_{axis}"]) * 10.0 for axis in "xyz"]
                for name in kin.JOINT_NAMES
            ], dtype=np.float64)
            head_position = np.asarray([
                float(row[f"mocap_CH3_08_Rigid_K_world_{axis}"]) * 1000.0 for axis in "xyz"
            ])
            head_q = np.asarray([
                float(row[f"mocap_CH3_08_Rigid_K_world_q{axis}"]) for axis in "wxyz"
            ])
            head_q /= np.linalg.norm(head_q)
            r_world_head = kin.q_to_matrix(head_q)
            bvh_head = points[index["Head"]]
            camera_position = bvh_head + r_world_head @ head_to_camera_rigid
            camera_rotation = r_world_head @ r_rigid_camera
            source_path = BASELINE / "source_frames" / camera / f"seq_{seq:06d}.jpg"
            source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if source is None:
                raise RuntimeError(f"Missing source image {source_path}")

            literal, semantic = source.copy(), source.copy()
            kin.draw(literal, points, camera_position, camera_rotation, model, (0, 255, 255), 4)
            draw_semantic(semantic, points, camera_position, camera_rotation, model)
            for image, title, color in (
                (literal, f"Literal BVH hierarchy  seq={seq:06d}", (0, 255, 255)),
                (semantic, "Raw BVH semantic remap: Neck=C7, Arm=shoulder", (255, 255, 0)),
            ):
                cv2.rectangle(image, (0, 0), (image.shape[1], 64), (0, 0, 0), -1)
                cv2.putText(image, title, (22, 43), cv2.FONT_HERSHEY_SIMPLEX, .72, color, 2, cv2.LINE_AA)
            cv2.imwrite(str(destination / f"seq_{seq:06d}_comparison.jpg"), np.hstack((literal, semantic)), [cv2.IMWRITE_JPEG_QUALITY, 94])
            cv2.imwrite(str(semantic_only_destination / f"seq_{seq:06d}_semantic.jpg"), semantic, [cv2.IMWRITE_JPEG_QUALITY, 94])

        thumbnails = []
        for seq in sequences:
            image = cv2.imread(str(destination / f"seq_{seq:06d}_comparison.jpg"))
            thumbnails.append(cv2.resize(image, (640, 200), interpolation=cv2.INTER_AREA))
        overview = np.vstack([np.hstack(thumbnails[i:i + 4]) for i in range(0, len(thumbnails), 4)])
        cv2.imwrite(str(OUTPUT / f"overview_{camera_key}.jpg"), overview, [cv2.IMWRITE_JPEG_QUALITY, 92])
        reports[camera_key] = {"sample_count": len(sequences)}

    summary = {
        "schema": "0722_2_raw_bvh_semantic_remap.v1",
        "source": "raw BVH positions only; no bone scaling, wrist rigid constraint, or IK",
        "camera_pose": "CH3_08 with validated fixed camera extrinsics",
        "semantic_mapping": {
            "C7": "Neck",
            "left_shoulder": "LeftArm",
            "right_shoulder": "RightArm",
            "left_elbow": "LeftForeArm",
            "right_elbow": "RightForeArm",
            "left_wrist": "LeftHand",
            "right_wrist": "RightHand",
            "hidden_helpers": ["LeftShoulder", "RightShoulder"],
        },
        "selected_sequences": sequences,
        "reports": reports,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
