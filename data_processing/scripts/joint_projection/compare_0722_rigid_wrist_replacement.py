#!/usr/bin/env python3
"""Compare exact-kinematic BVH wrists with CH3 rigid-derived wrist origins."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import project_0722_abx2_subject_scaled as kin
import project_joints as base


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "validation_0722_rigid_wrist_replacement_comparison"
SAMPLE_COUNT = 12

# Position of the rigid origin expressed in its wrist frame, millimetres.
WRIST_TO_RIGID_MM = {
    "Left": np.asarray([53.5, 76.5, 2.2], dtype=np.float64),
    "Right": np.asarray([53.5, 76.5, -2.2], dtype=np.float64),
}
# Columns are rigid +x/+y/+z axes expressed in the wrist frame.
R_WRIST_RIGID = {
    "Left": np.column_stack(([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0])),
    "Right": np.column_stack(([1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])),
}
SENSOR = {"Left": 301, "Right": 307, "Head": 308}


def project_point(point_world: np.ndarray, camera_position: np.ndarray, camera_rotation: np.ndarray, model: dict[str, object]) -> np.ndarray:
    point_camera = camera_rotation.T @ (point_world - camera_position)
    uv, valid = base.omni_project(point_camera[None, :], model)
    return uv[0] if valid[0] else np.asarray([np.nan, np.nan])


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    info, abx_config = kin.read_abx2_header(kin.ABX2)
    subject_doc = json.loads(kin.SUBJECT_JSON.read_text(encoding="utf-8"))
    subject = kin.subject_values(subject_doc)
    joints, channel_count, frame_time, frame_count = kin.parse_bvh(kin.BVH)
    motion = kin.load_bvh_motion(kin.BVH, channel_count, frame_count)
    offsets = kin.exact_subject_offsets(joints, subject)
    joint_index = {joint.name: i for i, joint in enumerate(joints)}

    pwrs = kin.pwr_map_for_sensors(abx_config, (301, 307, 308))
    rigid_rows = kin.extract_ch3_rigids(
        kin.ABX2, pwrs, float(info["ABXInfo"]["fps"])
    )
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
                + float(timing["global_scale"])
                * ((float(row["device_ts_ms"]) - float(timing["origin_ms"])) / 1000.0)
                + float(timing["head_imu_delta_sec"])
                + float(timing["exposure_center_shift_sec"])
            )
            frame_float = sample_time / frame_time
            world_pos, _ = kin.forward_kinematics_at(joints, offsets, motion, frame_float)
            points_bvh = np.asarray([world_pos[joint_index[name]] for name in kin.JOINT_NAMES]) * 10.0
            points_rigid = points_bvh.copy()

            head_rigid_position, head_q = kin.interpolate_rigid(rigid_rows[308], frame_float)
            r_world_head_rigid = kin.q_to_matrix(head_q)
            bvh_head = points_bvh[kin.JOINT_NAMES.index("Head")]
            pwr_head_joint = head_rigid_position + r_world_head_rigid @ head_joint_in_rigid_mm
            pwr_to_bvh_translation = bvh_head - pwr_head_joint

            wrist_data: dict[str, object] = {}
            for side in ("Left", "Right"):
                rigid_position, rigid_q = kin.interpolate_rigid(rigid_rows[SENSOR[side]], frame_float)
                r_world_rigid = kin.q_to_matrix(rigid_q)
                r_wrist_rigid = R_WRIST_RIGID[side]
                # Literal use of the supplied axis relation. Right has det=-1;
                # position is still computable, but its orientation is improper.
                r_world_wrist = r_world_rigid @ r_wrist_rigid.T
                wrist_pwr = rigid_position - r_world_wrist @ WRIST_TO_RIGID_MM[side]
                wrist_bvh_world = wrist_pwr + pwr_to_bvh_translation
                joint_name = f"{side}Hand"
                joint_i = kin.JOINT_NAMES.index(joint_name)
                original = points_bvh[joint_i].copy()
                points_rigid[joint_i] = wrist_bvh_world
                wrist_data[side] = {
                    "bvh_wrist_mm": original.tolist(),
                    "rigid_derived_wrist_mm": wrist_bvh_world.tolist(),
                    "world_delta_mm": float(np.linalg.norm(wrist_bvh_world - original)),
                    "axis_mapping_determinant": float(np.linalg.det(r_wrist_rigid)),
                }

            camera_position = bvh_head + r_world_head_rigid @ head_to_camera_rigid
            camera_rotation = r_world_head_rigid @ r_rigid_camera
            source = cv2.imread(str(kin.IMAGES / camera / f"seq_{seq:06d}.jpg"))
            if source is None:
                raise RuntimeError(f"Missing source image for {camera} seq={seq}")

            left_panel, right_panel = source.copy(), source.copy()
            kin.draw(left_panel, points_bvh, camera_position, camera_rotation, model, (255, 255, 0), 3)
            kin.draw(right_panel, points_rigid, camera_position, camera_rotation, model, (255, 0, 255), 4)
            cv2.rectangle(left_panel, (0, 0), (left_panel.shape[1], 65), (0, 0, 0), -1)
            cv2.rectangle(right_panel, (0, 0), (right_panel.shape[1], 65), (0, 0, 0), -1)
            cv2.putText(left_panel, f"BVH wrists  seq={seq:06d}", (24, 43), cv2.FONT_HERSHEY_SIMPLEX, .85, (255,255,0), 2, cv2.LINE_AA)
            cv2.putText(right_panel, "CH3_01/07 rigid-derived wrists", (24, 43), cv2.FONT_HERSHEY_SIMPLEX, .85, (255,0,255), 2, cv2.LINE_AA)
            comparison = np.hstack((left_panel, right_panel))
            cv2.imwrite(str(destination / f"seq_{seq:06d}_comparison.jpg"), comparison, [cv2.IMWRITE_JPEG_QUALITY, 94])

            overlay = source.copy()
            kin.draw(overlay, points_bvh, camera_position, camera_rotation, model, (255,255,0), 2)
            kin.draw(overlay, points_rigid, camera_position, camera_rotation, model, (255,0,255), 4)
            cv2.putText(overlay, "cyan=BVH wrist  magenta=rigid-derived wrist", (24, 43), cv2.FONT_HERSHEY_SIMPLEX, .78, (255,255,255), 2, cv2.LINE_AA)
            cv2.imwrite(str(overlay_destination / f"seq_{seq:06d}_overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 94])

            for side in ("Left", "Right"):
                joint_i = kin.JOINT_NAMES.index(f"{side}Hand")
                uv_old = project_point(points_bvh[joint_i], camera_position, camera_rotation, model)
                uv_new = project_point(points_rigid[joint_i], camera_position, camera_rotation, model)
                wrist_data[side]["bvh_uv"] = uv_old.tolist()
                wrist_data[side]["rigid_derived_uv"] = uv_new.tolist()
                wrist_data[side]["pixel_delta"] = float(np.linalg.norm(uv_new - uv_old)) if np.all(np.isfinite(uv_old)) and np.all(np.isfinite(uv_new)) else None
            camera_rows.append({"seq": seq, "sample_time_sec": sample_time, "wrists": wrist_data})

        reports[camera] = {"images": len(records), "frames": camera_rows}

    all_rows = [frame for camera in reports.values() for frame in camera["frames"]]
    stats = {}
    for side in ("Left", "Right"):
        world = [frame["wrists"][side]["world_delta_mm"] for frame in all_rows]
        pixels = [frame["wrists"][side]["pixel_delta"] for frame in all_rows if frame["wrists"][side]["pixel_delta"] is not None]
        stats[side] = {
            "world_delta_mm_median": float(np.median(world)),
            "world_delta_mm_p90": float(np.percentile(world, 90)),
            "pixel_delta_median": float(np.median(pixels)) if pixels else None,
            "pixel_delta_p90": float(np.percentile(pixels, 90)) if pixels else None,
        }
    summary = {
        "schema": "rigid_wrist_replacement_comparison.v1",
        "left_sensor": 301, "right_sensor": 307, "head_sensor": 308,
        "translation_interpretation": "rigid origin expressed in wrist frame",
        "wrist_to_rigid_mm": {side: value.tolist() for side, value in WRIST_TO_RIGID_MM.items()},
        "R_wrist_rigid": {side: value.tolist() for side, value in R_WRIST_RIGID.items()},
        "axis_mapping_determinant": {side: float(np.linalg.det(value)) for side, value in R_WRIST_RIGID.items()},
        "warning": "The supplied right-hand axis mapping has determinant -1 (reflection). It was used literally for this positional test.",
        "statistics": stats, "reports": reports,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"axis_mapping_determinant": summary["axis_mapping_determinant"], "statistics": stats, "images_per_camera": SAMPLE_COUNT}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
