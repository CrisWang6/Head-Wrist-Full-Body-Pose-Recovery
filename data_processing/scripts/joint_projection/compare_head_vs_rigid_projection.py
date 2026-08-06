#!/usr/bin/env python3
"""Compare CH3_08-driven and BVH Head+fixed-extrinsic joint projections."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import project_0722_head_final as final
import project_joints as base


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = final.SOURCE_ROOT
OUTPUT_ROOT = HERE / "validation_0722_head_vs_rigid_h265_upperbody_tuned"
SUMMARY_SOURCE = final.OUTPUT_ROOT / "summary.json"
COLORS = {"rigid": (255, 0, 255), "head": (255, 255, 0)}
SHOULDER_WIDTH_SCALE = 0.84


def compact_upper_body(points_world: np.ndarray, joint_names: list[str]) -> np.ndarray:
    """Narrow the BVH shoulder girdle without changing limb lengths or the legs."""
    points = points_world.copy()
    index = {name: idx for idx, name in enumerate(joint_names)}
    left_arm = points[index["LeftArm"]].copy()
    right_arm = points[index["RightArm"]].copy()
    center = 0.5 * (left_arm + right_arm)
    for side, shoulder in (("Left", left_arm), ("Right", right_arm)):
        new_shoulder = center + SHOULDER_WIDTH_SCALE * (shoulder - center)
        delta = new_shoulder - shoulder
        # Translate the complete articulated arm, preserving both bone lengths
        # and the observed elbow bend.  This corrects body shape, not motion.
        for name in (f"{side}Arm", f"{side}ForeArm", f"{side}Hand"):
            points[index[name]] += delta
    clavicle_center = 0.5 * (
        points[index["LeftShoulder"]] + points[index["RightShoulder"]]
    )
    for side in ("Left", "Right"):
        name = f"{side}Shoulder"
        points[index[name]] = clavicle_center + SHOULDER_WIDTH_SCALE * (
            points[index[name]] - clavicle_center
        )
    return points


def project(
    points_world: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    model: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_camera = (camera_rotation.T @ (points_world - camera_position).T).T
    uv, valid = base.omni_project(points_camera, model)
    visible = base.in_image(uv, valid, int(model["width"]), int(model["height"]))
    return uv, visible, points_camera


def draw_projection(
    image: np.ndarray,
    points_world: np.ndarray,
    camera_position: np.ndarray,
    camera_rotation: np.ndarray,
    model: dict[str, object],
    joint_names: list[str],
    edges: list[list[str]],
    color: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    uv, visible, _ = project(points_world, camera_position, camera_rotation, model)
    index = {name: idx for idx, name in enumerate(joint_names)}
    for first, second in edges:
        first_idx, second_idx = index[first], index[second]
        samples = np.linspace(0.0, 1.0, 25)[:, None]
        bone_world = points_world[first_idx] * (1.0 - samples) + points_world[second_idx] * samples
        bone_uv, bone_visible, _ = project(
            bone_world, camera_position, camera_rotation, model
        )
        run: list[np.ndarray] = []
        for point, is_visible in zip(bone_uv, bone_visible):
            if is_visible:
                run.append(point)
            elif len(run) >= 2:
                cv2.polylines(
                    image,
                    [np.rint(run).astype(np.int32).reshape(-1, 1, 2)],
                    False,
                    color,
                    4,
                    cv2.LINE_AA,
                )
                run = []
        if len(run) >= 2:
            cv2.polylines(
                image,
                [np.rint(run).astype(np.int32).reshape(-1, 1, 2)],
                False,
                color,
                4,
                cv2.LINE_AA,
            )
    for point in uv[visible]:
        center = tuple(np.rint(point).astype(int))
        cv2.circle(image, center, 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, center, 4, color, -1, cv2.LINE_AA)
    return uv, visible


def add_label(image: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 70), (0, 0, 0), -1)
    cv2.putText(
        image, text, (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.86, color, 2, cv2.LINE_AA
    )


def main() -> int:
    config = base.load_json(final.CONFIG_PATH)
    calibration = base.load_json(final.CALIBRATION_PATH)
    models = base.load_camera_models(config)
    joint_names = list(config["skeleton_joint_names"])
    edges = [list(edge) for edge in config["skeleton_edges"]]
    selected = [int(value) for value in base.load_json(SUMMARY_SOURCE)["selected_sequences"]]
    target_times = final.load_fixed_camera_times()
    motion_names = list(dict.fromkeys([*joint_names, *final.ANCHOR_JOINTS, *final.RIGID_NAMES]))
    times, positions, rotations = final.load_motion(motion_names)

    rotation_sample_times = times[::10]
    rotation_head_rigid = final.mean_rotation(
        np.einsum(
            "nij,njk->nik",
            np.transpose(rotations["Head"](rotation_sample_times).as_matrix(), (0, 2, 1)),
            rotations["CH3_08_Rigid_K"](rotation_sample_times).as_matrix(),
        )
    )
    head_axes = np.column_stack(([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
    head_to_rigid_mm = np.asarray([-2.0, 53.8, 135.5], dtype=np.float64)
    head_joint_in_rigid_mm = -head_axes.T @ head_to_rigid_mm

    reports: dict[str, object] = {}
    for camera, camera_key in final.CAMERA_KEYS.items():
        model = models[camera_key]
        rotation_rigid_camera = np.asarray(
            calibration[f"R_rigid_cam_{camera[-1]}"], dtype=np.float64
        )
        position_rigid_camera = np.asarray(
            calibration[f"p_rigid_cam_{camera[-1]}_mm"], dtype=np.float64
        )
        head_to_camera_rigid = position_rigid_camera - head_joint_in_rigid_mm
        source_paths = {
            int(path.stem.split("_")[1]): path
            for path in (SOURCE_ROOT / camera).glob("seq_*.jpg")
        }
        destination = OUTPUT_ROOT / camera_key
        destination.mkdir(parents=True, exist_ok=True)
        rigid_only_destination = OUTPUT_ROOT / "ch3_08_only" / camera_key
        head_only_destination = OUTPUT_ROOT / "head_fixed_only" / camera_key
        rigid_only_destination.mkdir(parents=True, exist_ok=True)
        head_only_destination.mkdir(parents=True, exist_ok=True)
        angle_deltas: list[float] = []
        position_deltas: list[float] = []
        pixel_deltas: list[float] = []
        per_frame: list[dict[str, object]] = []

        for seq in selected:
            sample_time = target_times[camera][seq]
            head_position, head_rotation = final.interpolate_pose(
                "Head", sample_time, times, positions, rotations
            )
            _, rigid_rotation = final.interpolate_pose(
                "CH3_08_Rigid_K", sample_time, times, positions, rotations
            )
            raw_joints = np.asarray(
                [final.interpolate_position(name, sample_time, times, positions) for name in joint_names]
            )
            points_world = head_position + final.GLOBAL_SKELETON_SCALE * (raw_joints - head_position)
            points_world = compact_upper_body(points_world, joint_names)

            # Method 1: current instantaneous CH3_08 orientation, translated onto
            # the BVH Head origin so both methods share the same skeleton world.
            rigid_camera_rotation = rigid_rotation @ rotation_rigid_camera
            rigid_camera_position = head_position + rigid_rotation @ head_to_camera_rigid

            # Method 2: BVH Head pose with the mean fixed Head->CH3_08 mounting
            # rotation and the same calibrated rigid->camera transform.
            head_camera_rotation = head_rotation @ rotation_head_rigid @ rotation_rigid_camera
            head_camera_position = (
                head_position + head_rotation @ rotation_head_rigid @ head_to_camera_rigid
            )

            angle_delta = float(
                np.degrees(
                    Rotation.from_matrix(head_camera_rotation.T @ rigid_camera_rotation).magnitude()
                )
            )
            position_delta = float(np.linalg.norm(rigid_camera_position - head_camera_position))
            angle_deltas.append(angle_delta)
            position_deltas.append(position_delta)

            source = cv2.imread(str(source_paths[seq]), cv2.IMREAD_COLOR)
            if source is None:
                raise RuntimeError(f"Could not read {source_paths[seq]}")
            rigid_panel = source.copy()
            head_panel = source.copy()
            rigid_uv, rigid_visible = draw_projection(
                rigid_panel,
                points_world,
                rigid_camera_position,
                rigid_camera_rotation,
                model,
                joint_names,
                edges,
                COLORS["rigid"],
            )
            head_uv, head_visible = draw_projection(
                head_panel,
                points_world,
                head_camera_position,
                head_camera_rotation,
                model,
                joint_names,
                edges,
                COLORS["head"],
            )
            comparable = rigid_visible & head_visible
            frame_pixel_delta = float(
                np.median(np.linalg.norm(rigid_uv[comparable] - head_uv[comparable], axis=1))
            ) if np.any(comparable) else float("nan")
            if np.isfinite(frame_pixel_delta):
                pixel_deltas.append(frame_pixel_delta)
            add_label(
                rigid_panel,
                f"CH3_08 instantaneous   seq={seq:06d}",
                COLORS["rigid"],
            )
            add_label(
                head_panel,
                f"BVH Head + fixed extrinsic   dR={angle_delta:.2f} deg   dUV={frame_pixel_delta:.1f}px",
                COLORS["head"],
            )
            comparison = np.hstack((rigid_panel, head_panel))
            output_path = destination / f"seq_{seq:06d}_comparison.jpg"
            cv2.imwrite(str(output_path), comparison, [cv2.IMWRITE_JPEG_QUALITY, 93])
            cv2.imwrite(
                str(rigid_only_destination / f"seq_{seq:06d}_joints.jpg"),
                rigid_panel,
                [cv2.IMWRITE_JPEG_QUALITY, 94],
            )
            cv2.imwrite(
                str(head_only_destination / f"seq_{seq:06d}_joints.jpg"),
                head_panel,
                [cv2.IMWRITE_JPEG_QUALITY, 94],
            )
            per_frame.append(
                {
                    "seq": seq,
                    "rotation_delta_deg": angle_delta,
                    "camera_origin_delta_mm": position_delta,
                    "median_joint_delta_px": frame_pixel_delta,
                }
            )

        reports[camera_key] = {
            "sample_count": len(selected),
            "rotation_delta_deg_mean": float(np.mean(angle_deltas)),
            "rotation_delta_deg_p90": float(np.percentile(angle_deltas, 90)),
            "camera_origin_delta_mm_mean": float(np.mean(position_deltas)),
            "median_joint_delta_px_mean": float(np.mean(pixel_deltas)),
            "median_joint_delta_px_p90": float(np.percentile(pixel_deltas, 90)),
            "frames": per_frame,
        }

    summary = {
        "schema": "joint_projection.head_vs_rigid_comparison.v1",
        "left_panel": "instantaneous CH3_08 orientation + calibrated rigid-to-camera extrinsic",
        "right_panel": "BVH Head orientation + mean fixed Head-to-CH3_08 + calibrated rigid-to-camera extrinsic",
        "shared_translation_anchor": "BVH Head joint position",
        "upper_body_shape_adjustment": {
            "shoulder_width_scale": SHOULDER_WIDTH_SCALE,
            "method": "translate each complete arm chain inward; preserve upper/lower arm lengths and lower body",
        },
        "reports": reports,
    }
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote comparison images to {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
