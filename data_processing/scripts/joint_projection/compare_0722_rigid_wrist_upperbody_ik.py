#!/usr/bin/env python3
"""Compare BVH, rigid-wrist-only, and rigid-wrist constrained upper-limb IK."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import project_0722_abx2_subject_scaled as kin
import project_joints as base
from compare_0722_rigid_wrist_replacement import (
    R_WRIST_RIGID,
    SENSOR,
    WRIST_TO_RIGID_MM,
)


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "validation_0722_rigid_wrist_upperbody_ik_comparison"
SAMPLE_COUNT = 12
IK_TOLERANCE_MM = 0.05
IK_MAX_ITERATIONS = 80


def fabrik(points: np.ndarray, lengths: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool, int, float]:
    """Solve a positional chain with a fixed root and target end effector."""
    solved = np.asarray(points, dtype=np.float64).copy()
    root = solved[0].copy()
    target = np.asarray(target, dtype=np.float64)
    total = float(np.sum(lengths))
    root_distance = float(np.linalg.norm(target - root))
    reachable = root_distance <= total + IK_TOLERANCE_MM

    if not reachable:
        direction = (target - root) / max(root_distance, 1e-12)
        for i, length in enumerate(lengths):
            solved[i + 1] = solved[i] + direction * length
        residual = float(np.linalg.norm(solved[-1] - target))
        return solved, False, 0, residual

    residual = float(np.linalg.norm(solved[-1] - target))
    iterations = 0
    while residual > IK_TOLERANCE_MM and iterations < IK_MAX_ITERATIONS:
        solved[-1] = target
        for i in range(len(solved) - 2, -1, -1):
            delta = solved[i] - solved[i + 1]
            norm = float(np.linalg.norm(delta))
            if norm < 1e-12:
                delta = points[i] - points[i + 1]
                norm = float(np.linalg.norm(delta))
            solved[i] = solved[i + 1] + delta * (lengths[i] / norm)

        solved[0] = root
        for i in range(len(solved) - 1):
            delta = solved[i + 1] - solved[i]
            norm = float(np.linalg.norm(delta))
            if norm < 1e-12:
                delta = points[i + 1] - points[i]
                norm = float(np.linalg.norm(delta))
            solved[i + 1] = solved[i] + delta * (lengths[i] / norm)
        residual = float(np.linalg.norm(solved[-1] - target))
        iterations += 1
    return solved, True, iterations, residual


def project_point(point_world: np.ndarray, camera_position: np.ndarray, camera_rotation: np.ndarray, model: dict[str, object]) -> np.ndarray:
    point_camera = camera_rotation.T @ (point_world - camera_position)
    uv, valid = base.omni_project(point_camera[None, :], model)
    return uv[0] if valid[0] else np.asarray([np.nan, np.nan])


def draw_label(image: np.ndarray, text_value: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 65), (0, 0, 0), -1)
    cv2.putText(image, text_value, (22, 43), cv2.FONT_HERSHEY_SIMPLEX, .72, color, 2, cv2.LINE_AA)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    info, abx_config = kin.read_abx2_header(kin.ABX2)
    subject_doc = json.loads(kin.SUBJECT_JSON.read_text(encoding="utf-8"))
    subject = kin.subject_values(subject_doc)
    joints, channel_count, frame_time, frame_count = kin.parse_bvh(kin.BVH)
    motion = kin.load_bvh_motion(kin.BVH, channel_count, frame_count)
    offsets = kin.exact_subject_offsets(joints, subject)
    joint_index = {joint.name: i for i, joint in enumerate(joints)}
    projected_index = {name: i for i, name in enumerate(kin.JOINT_NAMES)}

    pwrs = kin.pwr_map_for_sensors(abx_config, (301, 307, 308))
    rigid_rows = kin.extract_ch3_rigids(kin.ABX2, pwrs, float(info["ABXInfo"]["fps"]))
    mapping = base.load_json(kin.H265_MAPPING)
    calibration = base.load_json(kin.CALIBRATION)
    timing = calibration["timing"]
    models = base.load_camera_models(base.load_json(kin.CONFIG))

    head_axes = np.column_stack(([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
    head_to_rigid_mm = np.asarray([-2.0, 53.8, 135.5], dtype=np.float64)
    head_joint_in_rigid_mm = -head_axes.T @ head_to_rigid_mm
    reports: dict[str, object] = {}

    for camera, camera_key in kin.CAMERA_KEYS.items():
        records_all = sorted(mapping[camera]["mapping"], key=lambda row: int(row["seq"]))
        selected_indices = np.linspace(0, len(records_all) - 1, SAMPLE_COUNT, dtype=int)
        records = [records_all[index] for index in selected_indices]
        model = models[camera_key]
        r_rigid_camera = np.asarray(calibration[f"R_rigid_cam_{camera[-1]}"])
        p_rigid_camera = np.asarray(calibration[f"p_rigid_cam_{camera[-1]}_mm"])
        head_to_camera_rigid = p_rigid_camera - head_joint_in_rigid_mm
        destination = OUTPUT / camera_key
        overlay_destination = OUTPUT / "overlay" / camera_key
        destination.mkdir(parents=True, exist_ok=True)
        overlay_destination.mkdir(parents=True, exist_ok=True)
        camera_rows: list[dict[str, object]] = []

        for row in records:
            seq = int(row["seq"])
            sample_time = (
                float(timing["global_offset_sec"])
                + float(timing["global_scale"]) * ((float(row["device_ts_ms"]) - float(timing["origin_ms"])) / 1000.0)
                + float(timing["head_imu_delta_sec"])
                + float(timing["exposure_center_shift_sec"])
            )
            frame_float = sample_time / frame_time
            world_pos, _ = kin.forward_kinematics_at(joints, offsets, motion, frame_float)
            points_bvh = np.asarray([world_pos[joint_index[name]] for name in kin.JOINT_NAMES]) * 10.0
            points_wrist_only = points_bvh.copy()
            points_ik = points_bvh.copy()

            head_rigid_position, head_q = kin.interpolate_rigid(rigid_rows[308], frame_float)
            r_world_head_rigid = kin.q_to_matrix(head_q)
            bvh_head = points_bvh[projected_index["Head"]]
            pwr_head_joint = head_rigid_position + r_world_head_rigid @ head_joint_in_rigid_mm
            pwr_to_bvh_translation = bvh_head - pwr_head_joint
            frame_report: dict[str, object] = {}

            for side in ("Left", "Right"):
                rigid_position, rigid_q = kin.interpolate_rigid(rigid_rows[SENSOR[side]], frame_float)
                r_world_rigid = kin.q_to_matrix(rigid_q)
                r_world_wrist = r_world_rigid @ R_WRIST_RIGID[side].T
                wrist_pwr = rigid_position - r_world_wrist @ WRIST_TO_RIGID_MM[side]
                wrist_target = wrist_pwr + pwr_to_bvh_translation

                wrist_i = projected_index[f"{side}Hand"]
                points_wrist_only[wrist_i] = wrist_target
                chain_names = ["Spine2", f"{side}Shoulder", f"{side}Arm", f"{side}ForeArm", f"{side}Hand"]
                chain_indices = [projected_index[name] for name in chain_names]
                original_chain = points_bvh[chain_indices]
                chain_lengths = np.linalg.norm(np.diff(original_chain, axis=0), axis=1)
                solved, reachable, iterations, residual = fabrik(original_chain, chain_lengths, wrist_target)
                points_ik[chain_indices] = solved

                shoulder_old = points_bvh[projected_index[f"{side}Arm"]]
                elbow_old = points_bvh[projected_index[f"{side}ForeArm"]]
                shoulder_new = points_ik[projected_index[f"{side}Arm"]]
                elbow_new = points_ik[projected_index[f"{side}ForeArm"]]
                frame_report[side] = {
                    "reachable": reachable,
                    "iterations": iterations,
                    "wrist_residual_mm": residual,
                    "wrist_target_mm": wrist_target.tolist(),
                    "shoulder_delta_mm": float(np.linalg.norm(shoulder_new - shoulder_old)),
                    "elbow_delta_mm": float(np.linalg.norm(elbow_new - elbow_old)),
                    "chain_lengths_mm": chain_lengths.tolist(),
                }

            camera_position = bvh_head + r_world_head_rigid @ head_to_camera_rigid
            camera_rotation = r_world_head_rigid @ r_rigid_camera
            source = cv2.imread(str(kin.IMAGES / camera / f"seq_{seq:06d}.jpg"))
            if source is None:
                raise RuntimeError(f"Missing source image for {camera} seq={seq}")

            panel_bvh, panel_wrist, panel_ik = source.copy(), source.copy(), source.copy()
            kin.draw(panel_bvh, points_bvh, camera_position, camera_rotation, model, (255, 255, 0), 3)
            kin.draw(panel_wrist, points_wrist_only, camera_position, camera_rotation, model, (255, 0, 255), 4)
            kin.draw(panel_ik, points_ik, camera_position, camera_rotation, model, (0, 255, 0), 4)
            draw_label(panel_bvh, f"BVH original  seq={seq:06d}", (255, 255, 0))
            draw_label(panel_wrist, "Rigid wrist only", (255, 0, 255))
            draw_label(panel_ik, "Rigid wrist + full arm IK", (0, 255, 0))
            comparison = np.hstack((panel_bvh, panel_wrist, panel_ik))
            cv2.imwrite(str(destination / f"seq_{seq:06d}_comparison.jpg"), comparison, [cv2.IMWRITE_JPEG_QUALITY, 94])

            overlay = source.copy()
            kin.draw(overlay, points_bvh, camera_position, camera_rotation, model, (255, 255, 0), 2)
            kin.draw(overlay, points_ik, camera_position, camera_rotation, model, (0, 255, 0), 4)
            cv2.putText(overlay, "cyan=BVH  green=rigid-wrist IK", (22, 43), cv2.FONT_HERSHEY_SIMPLEX, .76, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(overlay_destination / f"seq_{seq:06d}_overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 94])

            for side in ("Left", "Right"):
                for joint_label, joint_name in (("shoulder", f"{side}Arm"), ("elbow", f"{side}ForeArm"), ("wrist", f"{side}Hand")):
                    ji = projected_index[joint_name]
                    uv_old = project_point(points_bvh[ji], camera_position, camera_rotation, model)
                    uv_new = project_point(points_ik[ji], camera_position, camera_rotation, model)
                    frame_report[side][f"{joint_label}_pixel_delta"] = (
                        float(np.linalg.norm(uv_new - uv_old))
                        if np.all(np.isfinite(uv_old)) and np.all(np.isfinite(uv_new)) else None
                    )
            camera_rows.append({"seq": seq, "sample_time_sec": sample_time, "arms": frame_report})
        reports[camera] = {"images": len(records), "frames": camera_rows}

    all_frames = [frame for camera_report in reports.values() for frame in camera_report["frames"]]
    statistics: dict[str, object] = {}
    for side in ("Left", "Right"):
        side_rows = [frame["arms"][side] for frame in all_frames]
        statistics[side] = {
            "reachable_count": sum(bool(row["reachable"]) for row in side_rows),
            "total_count": len(side_rows),
            "wrist_residual_mm_median": float(np.median([row["wrist_residual_mm"] for row in side_rows])),
            "shoulder_delta_mm_median": float(np.median([row["shoulder_delta_mm"] for row in side_rows])),
            "elbow_delta_mm_median": float(np.median([row["elbow_delta_mm"] for row in side_rows])),
            "shoulder_pixel_delta_median": float(np.median([row["shoulder_pixel_delta"] for row in side_rows if row["shoulder_pixel_delta"] is not None])),
            "elbow_pixel_delta_median": float(np.median([row["elbow_pixel_delta"] for row in side_rows if row["elbow_pixel_delta"] is not None])),
        }
    summary = {
        "schema": "rigid_wrist_upperbody_ik_comparison.v1",
        "method": "FABRIK positional IK; Spine2 fixed; clavicle, shoulder, elbow and wrist chain lengths fixed",
        "colors": {"BVH": "cyan", "rigid_wrist_only": "magenta", "rigid_wrist_IK": "green"},
        "sample_count_per_camera": SAMPLE_COUNT,
        "axis_mapping_determinant": {side: float(np.linalg.det(value)) for side, value in R_WRIST_RIGID.items()},
        "statistics": statistics,
        "reports": reports,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
