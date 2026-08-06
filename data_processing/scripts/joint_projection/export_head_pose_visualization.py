#!/usr/bin/env python3
"""Export a compact 3-D skeleton/head-camera trajectory for interactive QA."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import project_0722_head_final as final
import project_joints as base


FRAME_COUNT = 360


def rounded(values: np.ndarray, digits: int = 4) -> list:
    return np.round(values, digits).tolist()


def main() -> int:
    config = base.load_json(final.CONFIG_PATH)
    calibration = base.load_json(final.CALIBRATION_PATH)
    joint_names = list(config["skeleton_joint_names"])
    edges = [list(edge) for edge in config["skeleton_edges"]]
    motion_names = list(dict.fromkeys([*joint_names, *final.ANCHOR_JOINTS, *final.RIGID_NAMES]))
    times, positions, rotations = final.load_motion(motion_names)

    sample_indices = np.linspace(0, len(times) - 1, FRAME_COUNT).round().astype(int)
    sample_times = times[sample_indices]
    head_axes = np.column_stack(([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
    head_to_rigid_mm = np.asarray([-2.0, 53.8, 135.5], dtype=np.float64)
    head_joint_in_rigid_mm = -head_axes.T @ head_to_rigid_mm
    tag_offsets = {
        int(tag): np.asarray(offset, dtype=np.float64)
        for tag, offset in calibration["tag_offsets_rigid_mm"].items()
    }
    rotation_samples = times[::10]
    rotation_head_rigid = final.mean_rotation(
        np.einsum(
            "nij,njk->nik",
            np.transpose(rotations["Head"](rotation_samples).as_matrix(), (0, 2, 1)),
            rotations["CH3_08_Rigid_K"](rotation_samples).as_matrix(),
        )
    )

    origin = final.interpolate_position("CH3_08_Rigid_K", sample_times[0], times, positions)
    frames: list[dict[str, object]] = []
    for time_sec in sample_times:
        joints = {
            name: final.interpolate_position(name, time_sec, times, positions)
            for name in joint_names
        }
        bvh_head_position, bvh_head_rotation = final.interpolate_pose(
            "Head", time_sec, times, positions, rotations
        )
        head_position, head_rotation = final.interpolate_pose(
            "CH3_08_Rigid_K", time_sec, times, positions, rotations
        )
        left_position, left_rotation = final.interpolate_pose(
            "CH3_01_Rigid_K", time_sec, times, positions, rotations
        )
        right_position, right_rotation = final.interpolate_pose(
            "CH3_07_Rigid_K", time_sec, times, positions, rotations
        )
        target_head = head_position + head_rotation @ head_joint_in_rigid_mm
        skeleton_rotation = head_rotation @ rotation_head_rigid.T @ bvh_head_rotation.T
        transformed = {
            name: target_head
            + final.GLOBAL_SKELETON_SCALE
            * skeleton_rotation
            @ (joints[name] - bvh_head_position)
            for name in joint_names
        }
        for side, wrist_target in (
            ("Left", left_position + left_rotation @ tag_offsets[9]),
            ("Right", right_position + right_rotation @ tag_offsets[498]),
        ):
            elbow, wrist = final.solve_two_bone_ik(
                transformed[f"{side}Arm"],
                transformed[f"{side}ForeArm"],
                transformed[f"{side}Hand"],
                wrist_target,
            )
            transformed[f"{side}ForeArm"] = elbow
            transformed[f"{side}Hand"] = wrist

        camera_poses: dict[str, object] = {}
        for camera in ("B", "C"):
            rotation_rigid_camera = np.asarray(
                calibration[f"R_rigid_cam_{camera}"], dtype=np.float64
            )
            position_rigid_camera = np.asarray(
                calibration[f"p_rigid_cam_{camera}_mm"], dtype=np.float64
            )
            camera_poses[camera] = {
                "p": rounded((head_position + head_rotation @ position_rigid_camera - origin) / 1000.0),
                "r": rounded(head_rotation @ rotation_rigid_camera),
            }

        frames.append(
            {
                "t": round(float(time_sec - sample_times[0]), 3),
                "j": rounded(
                    np.asarray([transformed[name] - origin for name in joint_names]) / 1000.0
                ),
                "rigid": {
                    "p": rounded((head_position - origin) / 1000.0),
                    "r": rounded(head_rotation),
                },
                "cam": camera_poses,
            }
        )

    payload = {
        "jointNames": joint_names,
        "edges": edges,
        "frames": frames,
        "source": str(final.MOCAP_SOURCE),
        "calibration": str(final.CALIBRATION_PATH),
    }
    print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
