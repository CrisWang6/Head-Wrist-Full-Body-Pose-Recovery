#!/usr/bin/env python3
"""CAM_C hybrid overlay: BVH body, 2-D shoulders/elbows, rigid-derived wrists."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np

import project_0722_abx2_subject_scaled as kin
import project_0722_2_ch308_raw_bvh as baseline
import project_joints as base
from compare_0722_rigid_wrist_replacement import R_WRIST_RIGID, WRIST_TO_RIGID_MM


HERE = Path(__file__).resolve().parent
DATASET = Path(r"C:\Users\hand\Desktop\Dataset\0722_2")
RECORDING = DATASET / "0711_035935"
ALIGNED = RECORDING / "aligned_data" / "aligned_30hz.csv"
UPPER_2D = RECORDING / "aligned_data" / "module01_cam_c_shoulder_elbow_2d.csv"
CONFIG = HERE / "projection_config_0722_head_ch3_08.json"
CALIBRATION = HERE / "validation_0722_h265_fixed_time_calibration" / "calibration_fixed_time.json"
OUTPUT = HERE / "validation_0722_2_camc_hybrid_2d_upper_rigid_wrist"
SAMPLE_COUNT = 20
RANDOM_SEED = 20260723
MIN_2D_SCORE = 0.70
WIDTH, HEIGHT = 1920, 1200

JOINT_2D_FIELDS = {
    "LeftArm": ("left_shoulder_x", "left_shoulder_y", "left_shoulder_score"),
    "RightArm": ("right_shoulder_x", "right_shoulder_y", "right_shoulder_score"),
    "LeftForeArm": ("left_elbow_x", "left_elbow_y", "left_elbow_score"),
    "RightForeArm": ("right_elbow_x", "right_elbow_y", "right_elbow_score"),
}

# Clavicle helper joints are intentionally absent. Anatomical shoulders connect
# directly to the sternum/chest node (Spine2).
HYBRID_EDGES = [
    ("Hips", "Spine"), ("Spine", "Spine1"), ("Spine1", "Spine2"),
    ("Spine2", "Neck"), ("Neck", "Neck1"), ("Neck1", "Head"),
    ("Spine2", "LeftArm"), ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"),
    ("Spine2", "RightArm"), ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"),
    ("Hips", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), ("LeftLeg", "LeftFoot"),
    ("Hips", "RightUpLeg"), ("RightUpLeg", "RightLeg"), ("RightLeg", "RightFoot"),
]
HIDDEN = {"LeftShoulder", "RightShoulder"}


def load_aligned_rows() -> dict[int, dict[str, str]]:
    with ALIGNED.open("r", encoding="utf-8-sig", newline="") as handle:
        return {int(row["seq"]): row for row in csv.DictReader(handle)}


def load_2d_rows() -> dict[int, dict[str, str]]:
    with UPPER_2D.open("r", encoding="utf-8-sig", newline="") as handle:
        return {int(row["seq"]): row for row in csv.DictReader(handle)}


def camera_geometry(row: dict[str, str], model: dict[str, object], calibration: dict[str, object]) -> dict[str, object]:
    joint_index = {name: i for i, name in enumerate(kin.JOINT_NAMES)}
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

    head_axes = np.column_stack(([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
    head_to_rigid_mm = np.asarray([-2.0, 53.8, 135.5], dtype=np.float64)
    head_joint_in_rigid_mm = -head_axes.T @ head_to_rigid_mm
    r_rigid_camera = np.asarray(calibration["R_rigid_cam_C"], dtype=np.float64)
    p_rigid_camera = np.asarray(calibration["p_rigid_cam_C_mm"], dtype=np.float64)
    head_to_camera_rigid = p_rigid_camera - head_joint_in_rigid_mm
    bvh_head = points[joint_index["Head"]]
    camera_position = bvh_head + r_world_head @ head_to_camera_rigid
    camera_rotation = r_world_head @ r_rigid_camera
    pwr_head_joint = head_position + r_world_head @ head_joint_in_rigid_mm
    pwr_to_bvh_translation = bvh_head - pwr_head_joint

    points_camera = (camera_rotation.T @ (points - camera_position).T).T
    uv_raw, valid_raw = base.omni_project(points_camera, model)

    rigid_wrist_uv: dict[str, np.ndarray] = {}
    rigid_wrist_world: dict[str, np.ndarray] = {}
    for side, rigid_code in (("Left", "01"), ("Right", "07")):
        rigid_name = f"CH3_{rigid_code}_Rigid_K"
        rigid_position = np.asarray([
            float(row[f"mocap_{rigid_name}_world_{axis}"]) * 1000.0 for axis in "xyz"
        ])
        rigid_q = np.asarray([
            float(row[f"mocap_{rigid_name}_world_q{axis}"]) for axis in "wxyz"
        ])
        rigid_q /= np.linalg.norm(rigid_q)
        r_world_rigid = kin.q_to_matrix(rigid_q)
        r_world_wrist = r_world_rigid @ R_WRIST_RIGID[side].T
        wrist_world = (
            rigid_position
            - r_world_wrist @ WRIST_TO_RIGID_MM[side]
            + pwr_to_bvh_translation
        )
        wrist_camera = camera_rotation.T @ (wrist_world - camera_position)
        wrist_uv, wrist_valid = base.omni_project(wrist_camera[None, :], model)
        rigid_wrist_uv[side] = wrist_uv[0] if wrist_valid[0] else np.asarray([np.nan, np.nan])
        rigid_wrist_world[side] = wrist_world

    return {
        "points": points,
        "camera_position": camera_position,
        "camera_rotation": camera_rotation,
        "uv_raw": uv_raw,
        "valid_raw": valid_raw,
        "rigid_wrist_uv": rigid_wrist_uv,
        "rigid_wrist_world": rigid_wrist_world,
    }


def in_image(point: np.ndarray) -> bool:
    return bool(
        np.all(np.isfinite(point))
        and 0.0 <= point[0] < WIDTH
        and 0.0 <= point[1] < HEIGHT
    )


def candidate_rows(
    aligned: dict[int, dict[str, str]],
    upper_2d: dict[int, dict[str, str]],
    model: dict[str, object],
    calibration: dict[str, object],
) -> list[int]:
    valid: list[int] = []
    for seq, pose_row in upper_2d.items():
        if seq not in aligned or pose_row.get("status") != "ok":
            continue
        points_ok = True
        for x_field, y_field, score_field in JOINT_2D_FIELDS.values():
            uv = np.asarray([float(pose_row[x_field]), float(pose_row[y_field])])
            if float(pose_row[score_field]) < MIN_2D_SCORE or not in_image(uv):
                points_ok = False
                break
        if not points_ok:
            continue
        geometry = camera_geometry(aligned[seq], model, calibration)
        if not all(in_image(geometry["rigid_wrist_uv"][side]) for side in ("Left", "Right")):
            continue
        valid.append(seq)
    return valid


def stratified_sample(sequences: list[int]) -> list[int]:
    rng = random.Random(RANDOM_SEED)
    bins = np.array_split(np.asarray(sorted(sequences), dtype=int), SAMPLE_COUNT)
    return sorted(rng.choice(group.tolist()) for group in bins if len(group))


def draw_hybrid(
    image: np.ndarray,
    uv: np.ndarray,
    visible: np.ndarray,
    index: dict[str, int],
) -> None:
    edge_color = (0, 220, 255)
    for first, second in HYBRID_EDGES:
        a, b = index[first], index[second]
        if visible[a] and visible[b]:
            cv2.line(
                image,
                tuple(np.rint(uv[a]).astype(int)),
                tuple(np.rint(uv[b]).astype(int)),
                edge_color,
                4,
                cv2.LINE_AA,
            )
    for name in kin.JOINT_NAMES:
        if name in HIDDEN:
            continue
        i = index[name]
        if not visible[i]:
            continue
        point = tuple(np.rint(uv[i]).astype(int))
        if name in {"LeftArm", "RightArm", "LeftForeArm", "RightForeArm"}:
            color = (255, 255, 0)  # 2-D shoulder/elbow: cyan
            radius = 6
        elif name in {"LeftHand", "RightHand"}:
            color = (255, 0, 255)  # rigid wrist: magenta
            radius = 7
        elif name == "Spine2":
            color = (0, 255, 0)  # sternum/chest: green
            radius = 7
        else:
            color = (0, 220, 255)
            radius = 5
        cv2.circle(image, point, radius + 2, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, point, radius, color, -1, cv2.LINE_AA)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = base.load_json(CONFIG)
    calibration = base.load_json(CALIBRATION)
    model = base.load_camera_models(config)["module01_CAM_C"]
    aligned = load_aligned_rows()
    upper_2d = load_2d_rows()
    candidates = candidate_rows(aligned, upper_2d, model, calibration)
    selected = stratified_sample(candidates)
    if len(selected) != SAMPLE_COUNT:
        raise RuntimeError(f"Expected {SAMPLE_COUNT} selected rows, got {len(selected)}")

    # Reuse the verified H.265 decoded-index mapping and extraction path.
    baseline.OUTPUT = OUTPUT
    ordinals = baseline.load_timestamp_ordinals()
    selected_aligned_rows = [aligned[seq] for seq in selected]
    source_images, timeline_report = baseline.extract_images(
        "CAM_C", selected_aligned_rows, ordinals
    )

    comparison_dir = OUTPUT / "module01_CAM_C"
    hybrid_dir = OUTPUT / "hybrid_only" / "module01_CAM_C"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    hybrid_dir.mkdir(parents=True, exist_ok=True)
    index = {name: i for i, name in enumerate(kin.JOINT_NAMES)}
    reports: list[dict[str, object]] = []

    for seq in selected:
        geometry = camera_geometry(aligned[seq], model, calibration)
        uv_raw = geometry["uv_raw"]
        valid_raw = geometry["valid_raw"]
        uv_hybrid = uv_raw.copy()
        pose_row = upper_2d[seq]
        scores: dict[str, float] = {}
        for name, (x_field, y_field, score_field) in JOINT_2D_FIELDS.items():
            uv_hybrid[index[name]] = [float(pose_row[x_field]), float(pose_row[y_field])]
            scores[name] = float(pose_row[score_field])
        uv_hybrid[index["LeftHand"]] = geometry["rigid_wrist_uv"]["Left"]
        uv_hybrid[index["RightHand"]] = geometry["rigid_wrist_uv"]["Right"]
        visible_hybrid = np.asarray([in_image(point) for point in uv_hybrid], dtype=bool)

        source = cv2.imread(str(source_images[seq]), cv2.IMREAD_COLOR)
        if source is None:
            raise RuntimeError(f"Missing extracted image for seq={seq}")
        raw_panel, hybrid_panel = source.copy(), source.copy()
        kin.draw(
            raw_panel,
            geometry["points"],
            geometry["camera_position"],
            geometry["camera_rotation"],
            model,
            (0, 255, 255),
            3,
        )
        draw_hybrid(hybrid_panel, uv_hybrid, visible_hybrid, index)
        cv2.rectangle(raw_panel, (0, 0), (WIDTH, 68), (0, 0, 0), -1)
        cv2.rectangle(hybrid_panel, (0, 0), (WIDTH, 88), (0, 0, 0), -1)
        cv2.putText(
            raw_panel,
            f"Raw CH3_08 + BVH  seq={seq:06d}",
            (22, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            .76,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            hybrid_panel,
            "Hybrid: cyan=2D shoulder/elbow  magenta=rigid wrist  green=sternum",
            (22, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            .68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            hybrid_panel,
            "clavicle helpers removed; shoulder connected directly to Spine2",
            (22, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            .62,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(
            str(comparison_dir / f"seq_{seq:06d}_comparison.jpg"),
            np.hstack((raw_panel, hybrid_panel)),
            [cv2.IMWRITE_JPEG_QUALITY, 94],
        )
        cv2.imwrite(
            str(hybrid_dir / f"seq_{seq:06d}_hybrid.jpg"),
            hybrid_panel,
            [cv2.IMWRITE_JPEG_QUALITY, 94],
        )

        deltas = {}
        for name in ("LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand"):
            if in_image(uv_raw[index[name]]) and in_image(uv_hybrid[index[name]]):
                deltas[name] = float(np.linalg.norm(uv_hybrid[index[name]] - uv_raw[index[name]]))
            else:
                deltas[name] = None
        reports.append({"seq": seq, "scores": scores, "replacement_delta_px": deltas})

    thumbnails = []
    for seq in selected:
        image = cv2.imread(str(comparison_dir / f"seq_{seq:06d}_comparison.jpg"))
        thumbnails.append(cv2.resize(image, (640, 200), interpolation=cv2.INTER_AREA))
    overview = np.vstack([
        np.hstack(thumbnails[i:i + 4]) for i in range(0, len(thumbnails), 4)
    ])
    cv2.imwrite(str(OUTPUT / "overview_CAM_C.jpg"), overview, [cv2.IMWRITE_JPEG_QUALITY, 92])

    statistics = {}
    for name in ("LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand", "RightHand"):
        values = [
            row["replacement_delta_px"][name]
            for row in reports
            if row["replacement_delta_px"][name] is not None
        ]
        statistics[name] = {
            "delta_px_median": float(np.median(values)),
            "delta_px_p90": float(np.percentile(values, 90)),
        }
    summary = {
        "schema": "0722_2_camc_hybrid_2d_upper_rigid_wrist.v1",
        "camera": "module01_CAM_C",
        "sample_count": SAMPLE_COUNT,
        "selection": f"stratified sample from status=ok, all four 2D joints in image, each score >= {MIN_2D_SCORE}, both rigid wrists in image",
        "selected_sequences": selected,
        "sources": {
            "camera_pose": "CH3_08",
            "body_except_upper_replacements": "raw BVH projection",
            "shoulders_elbows": str(UPPER_2D),
            "wrists": "CH3_01/07 rigid-derived wrist positions",
            "sternum": "Spine2",
        },
        "hidden_joints": sorted(HIDDEN),
        "statistics": statistics,
        "timeline": timeline_report,
        "frames": reports,
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(OUTPUT),
        "candidate_count": len(candidates),
        "selected_sequences": selected,
        "statistics": statistics,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
