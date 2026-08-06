#!/usr/bin/env python3
"""Render the validated 0722 head-camera skeleton overlays."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from ultralytics import YOLO

import project_joints as base


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = HERE / "validation_0722_h265_random100" / "source_frames" / "module01"
OUTPUT_ROOT = HERE / "validation_0722_h265_fixed_time_final"
CONFIG_PATH = HERE / "projection_config_0722_head_ch3_08.json"
CALIBRATION_PATH = HERE / "validation_0722_h265_fixed_time_calibration" / "calibration_fixed_time.json"
H265_MAPPING_PATH = HERE / "validation_0722_h265_random100" / "h265_frame_mapping.json"
MOCAP_SOURCE = Path(r"C:\Users\hand\Desktop\Dataset\0722\abx2_mocap_rigid_csv\mocap_rigid_20260722.csv")
RANDOM_SEED = 20260722
SAMPLE_COUNT = 100
GLOBAL_SKELETON_SCALE = 1.064

CAMERA_KEYS = {"CAM_C": "module01_CAM_C", "CAM_B": "module01_CAM_B"}
RIGID_NAMES = ("CH3_08_Rigid_K", "CH3_01_Rigid_K", "CH3_07_Rigid_K")
ANCHOR_JOINTS = ("Head", "LeftHand", "RightHand")


def refine_uv_from_image_pose(
    projected_uv: np.ndarray,
    skeleton_names: list[str],
    pose_result: object,
) -> tuple[np.ndarray, bool, float]:
    """Fuse the 3-D projection with the matching 2-D person detection.

    The rigid-body wrists identify the wearer in this severe fisheye view.  They
    also disambiguate duplicate detections of the same person.  Image keypoints
    then correct the BVH/PWR world-frame residual for the remaining body joints.
    """
    if pose_result.keypoints is None or len(pose_result.keypoints) == 0:
        return projected_uv.copy(), False, float("nan")
    xy = pose_result.keypoints.xy.cpu().numpy()
    confidence = pose_result.keypoints.conf.cpu().numpy()
    joint_index = {name: idx for idx, name in enumerate(skeleton_names)}
    left_target = projected_uv[joint_index["LeftHand"]]
    right_target = projected_uv[joint_index["RightHand"]]
    costs: list[float] = []
    for person_xy, person_conf in zip(xy, confidence):
        wrist_cost = 0.0
        used = 0
        for coco_index, target in ((9, left_target), (10, right_target)):
            if person_conf[coco_index] >= 0.12 and np.all(np.isfinite(target)):
                wrist_cost += float(np.linalg.norm(person_xy[coco_index] - target))
                used += 1
        if used == 0:
            costs.append(float("inf"))
            continue
        # A weak torso prior rejects a nearby background person when one wrist is hidden.
        if person_conf[5] >= 0.12 and person_conf[6] >= 0.12:
            shoulder_mid = 0.5 * (person_xy[5] + person_xy[6])
            wrist_cost += 0.15 * float(
                np.linalg.norm(shoulder_mid - projected_uv[joint_index["Spine2"]])
            )
        costs.append(wrist_cost / used)
    selected = int(np.argmin(costs))
    if not np.isfinite(costs[selected]):
        return projected_uv.copy(), False, float("nan")

    keypoints = xy[selected]
    conf = confidence[selected]
    # Start empty: an occluded image joint is preferable to a confidently drawn
    # but visibly false limb. The two measured wrist anchors are always retained.
    refined = np.full_like(projected_uv, np.nan)
    refined[joint_index["LeftHand"]] = left_target
    refined[joint_index["RightHand"]] = right_target

    def assign(name: str, coco_index: int, threshold: float = 0.16) -> None:
        if conf[coco_index] >= threshold:
            refined[joint_index[name]] = keypoints[coco_index]

    # Limb endpoints from the image, with rigid-body wrists retained as the
    # highest-confidence metric anchors.
    for side, indices in (("Left", (5, 7, 9, 11, 13, 15)), ("Right", (6, 8, 10, 12, 14, 16))):
        shoulder_i, elbow_i, _, hip_i, knee_i, ankle_i = indices
        assign(f"{side}Arm", shoulder_i, 0.20)
        assign(f"{side}ForeArm", elbow_i, 0.20)
        assign(f"{side}UpLeg", hip_i, 0.22)
        assign(f"{side}Leg", knee_i, 0.28)
        assign(f"{side}Foot", ankle_i, 0.30)

    if conf[5] >= 0.16 and conf[6] >= 0.16:
        shoulder_mid = 0.5 * (keypoints[5] + keypoints[6])
        refined[joint_index["LeftShoulder"]] = shoulder_mid + 0.45 * (keypoints[5] - shoulder_mid)
        refined[joint_index["RightShoulder"]] = shoulder_mid + 0.45 * (keypoints[6] - shoulder_mid)
    else:
        shoulder_mid = refined[joint_index["Spine2"]]
    if conf[11] >= 0.14 and conf[12] >= 0.14:
        hip_mid = 0.5 * (keypoints[11] + keypoints[12])
        refined[joint_index["Hips"]] = hip_mid
        if conf[5] >= 0.16 and conf[6] >= 0.16:
            # Smooth centerline between the detected pelvis and shoulder girdle.
            for name, fraction in (("Spine", 0.30), ("Spine1", 0.55), ("Spine2", 0.82), ("Neck", 1.0)):
                refined[joint_index[name]] = hip_mid + fraction * (shoulder_mid - hip_mid)
            if conf[0] >= 0.16:
                refined[joint_index["Neck1"]] = shoulder_mid + 0.18 * (keypoints[0] - shoulder_mid)

    # Bound elbow hallucinations while retaining the observed bend direction.
    for side in ("Left", "Right"):
        shoulder = refined[joint_index[f"{side}Arm"]]
        elbow = refined[joint_index[f"{side}ForeArm"]]
        wrist = refined[joint_index[f"{side}Hand"]]
        if not (np.all(np.isfinite(shoulder)) and np.all(np.isfinite(elbow)) and np.all(np.isfinite(wrist))):
            continue
        arm = wrist - shoulder
        arm_length = float(np.linalg.norm(arm))
        if arm_length < 1.0:
            continue
        axis = arm / arm_length
        along = float(np.clip(np.dot(elbow - shoulder, axis) / arm_length, 0.25, 0.75))
        perpendicular = (elbow - shoulder) - axis * float(np.dot(elbow - shoulder, axis))
        perpendicular_norm = float(np.linalg.norm(perpendicular))
        max_bend = 0.38 * arm_length
        if perpendicular_norm > max_bend:
            perpendicular *= max_bend / perpendicular_norm
        refined[joint_index[f"{side}ForeArm"]] = shoulder + axis * (along * arm_length) + perpendicular

    # Do not overwrite LeftHand/RightHand: those are projected from the measured
    # wrist rigids and were also used to select this detection.
    return refined, True, float(costs[selected])


def omni_unproject_unit(uv: np.ndarray, model: dict[str, object]) -> np.ndarray:
    """Inverse Kalibr omni+radtan pixels to unit camera rays."""
    pixels = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    camera_matrix = np.asarray(
        [[model["fx"], 0.0, model["cx"]], [0.0, model["fy"], model["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    normalized = cv2.undistortPoints(
        pixels, camera_matrix, np.asarray(model["distortion"], dtype=np.float64)
    ).reshape(-1, 2)
    x, y = normalized[:, 0], normalized[:, 1]
    radius2 = x * x + y * y
    xi = float(model["xi"])
    lam = (xi + np.sqrt(np.maximum(1.0 + (1.0 - xi * xi) * radius2, 1e-12))) / (1.0 + radius2)
    rays = np.column_stack((lam * x, lam * y, lam - xi))
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return rays


def load_target_times(path: Path, sequences: set[int]) -> dict[int, float]:
    result: dict[int, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            seq = int(row["seq"])
            if seq in sequences:
                result[seq] = float(row["mocap_time_sec_target"])
    missing = sorted(sequences - set(result))
    if missing:
        raise RuntimeError(f"Missing aligned target times for {missing[:10]}")
    return result


def load_fixed_camera_times() -> dict[str, dict[int, float]]:
    mapping = base.load_json(H265_MAPPING_PATH)
    calibration = base.load_json(CALIBRATION_PATH)
    timing = calibration["timing"]
    result: dict[str, dict[int, float]] = {}
    for camera in CAMERA_KEYS:
        result[camera] = {}
        for row in mapping[camera]["mapping"]:
            result[camera][int(row["seq"])] = (
                float(timing["global_offset_sec"])
                + float(timing["global_scale"])
                * ((float(row["device_ts_ms"]) - float(timing["origin_ms"])) / 1000.0)
                + float(timing["head_imu_delta_sec"])
                + float(timing["exposure_center_shift_sec"])
            )
    return result


def load_motion(names: list[str]) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Slerp]]:
    times: list[float] = []
    positions: dict[str, list[list[float]]] = {name: [] for name in names}
    quaternions_xyzw: dict[str, list[list[float]]] = {
        name: [] for name in (*ANCHOR_JOINTS, *RIGID_NAMES)
    }
    with MOCAP_SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            times.append(float(row["time_sec"]))
            for name in names:
                scale = 1000.0 if name.startswith("CH3_") else 10.0
                positions[name].append(
                    [float(row[f"{name}_world_{axis}"]) * scale for axis in "xyz"]
                )
            for name in quaternions_xyzw:
                qw, qx, qy, qz = [float(row[f"{name}_world_q{axis}"]) for axis in "wxyz"]
                quaternions_xyzw[name].append([qx, qy, qz, qw])
    time_array = np.asarray(times, dtype=np.float64)
    position_arrays = {
        name: np.asarray(values, dtype=np.float64) for name, values in positions.items()
    }
    rotations = {
        name: Slerp(time_array, Rotation.from_quat(np.asarray(values, dtype=np.float64)))
        for name, values in quaternions_xyzw.items()
    }
    return time_array, position_arrays, rotations


def interpolate_position(
    name: str, time_sec: float, times: np.ndarray, positions: dict[str, np.ndarray]
) -> np.ndarray:
    return np.asarray(
        [np.interp(time_sec, times, positions[name][:, axis]) for axis in range(3)],
        dtype=np.float64,
    )


def interpolate_pose(
    name: str,
    time_sec: float,
    times: np.ndarray,
    positions: dict[str, np.ndarray],
    rotations: dict[str, Slerp],
) -> tuple[np.ndarray, np.ndarray]:
    position = interpolate_position(name, time_sec, times, positions)
    rotation = rotations[name]([time_sec]).as_matrix()[0]
    return position, rotation


def mean_rotation(rotations: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(np.sum(rotations, axis=0))
    result = u @ vt
    if np.linalg.det(result) < 0.0:
        u[:, -1] *= -1.0
        result = u @ vt
    return result


def solve_two_bone_ik(
    shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray, target_wrist: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Place the wrist on its rigid-body anchor without producing a runaway elbow.

    The BVH and PWR streams are not a perfectly calibrated common world frame, so
    preserving the BVH arm lengths exactly can make the rigid wrist target
    unreachable.  In that case a conventional two-bone IK solver clamps the reach
    but still draws the final forearm to the (unclamped) wrist, which creates a very
    long, visibly wrong bone.  Retain the original bend plane instead, while
    distributing the required stretch over both arm segments and bounding the bend.
    """
    upper_length = float(np.linalg.norm(elbow - shoulder))
    lower_length = float(np.linalg.norm(wrist - elbow))
    target_vector = target_wrist - shoulder
    target_distance = float(np.linalg.norm(target_vector))
    if target_distance < 1e-6 or upper_length < 1e-6 or lower_length < 1e-6:
        return elbow, target_wrist
    axis = target_vector / target_distance
    elbow_fraction = float(np.clip(upper_length / (upper_length + lower_length), 0.38, 0.62))
    original_normal = (elbow - shoulder) - axis * float(np.dot(elbow - shoulder, axis))
    normal_norm = float(np.linalg.norm(original_normal))
    if normal_norm < 1e-6:
        fallback = np.asarray([0.0, 0.0, 1.0])
        if abs(float(np.dot(fallback, axis))) > 0.9:
            fallback = np.asarray([0.0, 1.0, 0.0])
        original_normal = fallback - axis * float(np.dot(fallback, axis))
        normal_norm = float(np.linalg.norm(original_normal))
    normal = original_normal / normal_norm
    # A modest bend preserves left/right elbow orientation but prevents calibration
    # residuals from sending an elbow hundreds of millimetres outside the body.
    bend = min(normal_norm, 0.16 * target_distance, 95.0)
    solved_elbow = shoulder + axis * (elbow_fraction * target_distance) + normal * bend
    return solved_elbow, target_wrist


def draw_curve(image: np.ndarray, curve: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    if len(curve) >= 2:
        cv2.polylines(
            image,
            [np.rint(curve).astype(np.int32).reshape(-1, 1, 2)],
            False,
            color,
            thickness,
            cv2.LINE_AA,
        )


def main() -> int:
    config = base.load_json(CONFIG_PATH)
    calibration = base.load_json(CALIBRATION_PATH)
    models = base.load_camera_models(config)
    skeleton_names = list(config["skeleton_joint_names"])
    skeleton_edges = [list(edge) for edge in config["skeleton_edges"]]

    available = {
        camera: {int(path.stem.split("_")[1]): path for path in (SOURCE_ROOT / camera).glob("seq_*.jpg")}
        for camera in CAMERA_KEYS
    }
    common_sequences = set.intersection(*(set(paths) for paths in available.values()))
    if len(common_sequences) < SAMPLE_COUNT:
        raise RuntimeError(f"Only {len(common_sequences)} common B/C images are available")
    selected = sorted(random.Random(RANDOM_SEED).sample(sorted(common_sequences), SAMPLE_COUNT))
    target_times = load_fixed_camera_times()

    motion_names = list(dict.fromkeys([*skeleton_names, *ANCHOR_JOINTS, *RIGID_NAMES]))
    times, positions, rotations = load_motion(motion_names)

    head_axes = np.column_stack(([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
    head_to_rigid_mm = np.asarray([-2.0, 53.8, 135.5], dtype=np.float64)
    head_joint_in_rigid_mm = -head_axes.T @ head_to_rigid_mm
    tag_offsets = {
        int(tag): np.asarray(offset, dtype=np.float64)
        for tag, offset in calibration["tag_offsets_rigid_mm"].items()
    }
    rotation_sample_times = times[::10]
    head_rotation_samples = rotations["Head"](rotation_sample_times).as_matrix()
    rigid_rotation_samples = rotations["CH3_08_Rigid_K"](rotation_sample_times).as_matrix()
    rotation_head_rigid = mean_rotation(
        np.einsum("nij,njk->nik", np.transpose(head_rotation_samples, (0, 2, 1)), rigid_rotation_samples)
    )
    pose_model = YOLO(str(HERE / "yolo11n-pose.pt"))

    output_rows: list[dict[str, object]] = []
    reports: dict[str, dict[str, object]] = {}
    refined_world_from_cam_c: dict[int, np.ndarray] = {}
    for camera, camera_key in CAMERA_KEYS.items():
        model = models[camera_key]
        rotation_rigid_camera = np.asarray(calibration[f"R_rigid_cam_{camera[-1]}"], dtype=np.float64)
        position_rigid_camera = np.asarray(
            calibration[f"p_rigid_cam_{camera[-1]}_mm"], dtype=np.float64
        )
        destination_dir = OUTPUT_ROOT / camera_key
        destination_dir.mkdir(parents=True, exist_ok=True)
        visible_counts: list[int] = []
        wrist_correction_distances: list[float] = []
        image_pose_frames = 0
        image_pose_costs: list[float] = []
        unrefined_sequences: list[int] = []

        for seq in selected:
            sample_time = target_times[camera][seq]
            joint_positions = {
                name: interpolate_position(name, sample_time, times, positions)
                for name in skeleton_names
            }
            bvh_head_position, bvh_head_rotation = interpolate_pose(
                "Head", sample_time, times, positions, rotations
            )
            head_position, head_rotation = interpolate_pose(
                "CH3_08_Rigid_K", sample_time, times, positions, rotations
            )
            left_position, left_rotation = interpolate_pose(
                "CH3_01_Rigid_K", sample_time, times, positions, rotations
            )
            right_position, right_rotation = interpolate_pose(
                "CH3_07_Rigid_K", sample_time, times, positions, rotations
            )
            target_head = head_position + head_rotation @ head_joint_in_rigid_mm
            target_left_wrist = left_position + left_rotation @ tag_offsets[9]
            target_right_wrist = right_position + right_rotation @ tag_offsets[498]
            skeleton_rotation = head_rotation @ rotation_head_rigid.T @ bvh_head_rotation.T
            transformed = {
                name: target_head
                + GLOBAL_SKELETON_SCALE
                * skeleton_rotation
                @ (joint_positions[name] - bvh_head_position)
                for name in skeleton_names
            }
            for prefix, target_wrist in (
                ("Left", target_left_wrist),
                ("Right", target_right_wrist),
            ):
                shoulder_name = f"{prefix}Arm"
                elbow_name = f"{prefix}ForeArm"
                wrist_name = f"{prefix}Hand"
                wrist_correction_distances.append(
                    float(np.linalg.norm(transformed[wrist_name] - target_wrist))
                )
                solved_elbow, solved_wrist = solve_two_bone_ik(
                    transformed[shoulder_name],
                    transformed[elbow_name],
                    transformed[wrist_name],
                    target_wrist,
                )
                transformed[elbow_name] = solved_elbow
                transformed[wrist_name] = solved_wrist
            points_world = np.asarray([transformed[name] for name in skeleton_names])
            if camera == "CAM_B" and seq in refined_world_from_cam_c:
                transferred = refined_world_from_cam_c[seq].copy()
                # Preserve the per-camera-time rigid wrist anchors.
                transferred[skeleton_names.index("LeftHand")] = points_world[skeleton_names.index("LeftHand")]
                transferred[skeleton_names.index("RightHand")] = points_world[skeleton_names.index("RightHand")]
                points_world = transferred
            rotation_world_camera = head_rotation @ rotation_rigid_camera
            position_world_camera = head_position + head_rotation @ position_rigid_camera
            points_camera = (
                rotation_world_camera.T @ (points_world - position_world_camera).T
            ).T
            uv, projection_valid = base.omni_project(points_camera, model)
            image = cv2.imread(str(available[camera][seq]), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not read {available[camera][seq]}")
            if camera == "CAM_C":
                pose_result = pose_model.predict(
                    image, imgsz=960, conf=0.10, iou=0.50, verbose=False
                )[0]
                uv, pose_used, pose_cost = refine_uv_from_image_pose(uv, skeleton_names, pose_result)
                if pose_used:
                    rays_camera = omni_unproject_unit(uv, model)
                    radii = np.linalg.norm(points_camera, axis=1)
                    refined_camera = rays_camera * radii[:, None]
                    refined_world = (
                        position_world_camera
                        + (rotation_world_camera @ refined_camera.T).T
                    )
                    # Metric wrist rigids are stronger than image keypoints.
                    refined_world[skeleton_names.index("LeftHand")] = points_world[skeleton_names.index("LeftHand")]
                    refined_world[skeleton_names.index("RightHand")] = points_world[skeleton_names.index("RightHand")]
                    refined_world_from_cam_c[seq] = refined_world
            else:
                pose_used = seq in refined_world_from_cam_c
                pose_cost = float("nan")
            if pose_used:
                image_pose_frames += 1
                if np.isfinite(pose_cost):
                    image_pose_costs.append(pose_cost)
            else:
                unrefined_sequences.append(seq)
            visible = np.all(np.isfinite(uv), axis=1)
            visible &= uv[:, 0] >= 0.0
            visible &= uv[:, 0] < int(model["width"])
            visible &= uv[:, 1] >= 0.0
            visible &= uv[:, 1] < int(model["height"])
            visible_counts.append(int(np.count_nonzero(visible)))

            index = {name: idx for idx, name in enumerate(skeleton_names)}
            for first, second in skeleton_edges:
                first_idx, second_idx = index[first], index[second]
                if visible[first_idx] and visible[second_idx]:
                    cv2.line(
                        image,
                        tuple(np.rint(uv[first_idx]).astype(int)),
                        tuple(np.rint(uv[second_idx]).astype(int)),
                        (0, 220, 255),
                        4,
                        cv2.LINE_AA,
                    )

            for idx in np.flatnonzero(visible):
                point = tuple(np.rint(uv[idx]).astype(int))
                cv2.circle(image, point, 7, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(image, point, 5, (0, 0, 255), -1, cv2.LINE_AA)
                output_rows.append(
                    {
                        "camera_key": camera_key,
                        "seq": seq,
                        "joint": skeleton_names[idx],
                        "u_px": round(float(uv[idx, 0]), 4),
                        "v_px": round(float(uv[idx, 1]), 4),
                        "forward_depth_mm": round(float(points_camera[idx, 2]), 4),
                    }
                )
            cv2.putText(
                image,
                f"{camera_key}  seq={seq:06d}  3D+2D refined",
                (28, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            output_path = destination_dir / f"seq_{seq:06d}_joints.jpg"
            if not cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise RuntimeError(f"Could not write {output_path}")

        reports[camera_key] = {
            "sample_count": len(selected),
            "time_source": "H265 device timestamp + head IMU fixed mapping + exposure midpoint",
            "net_additional_to_device_end_sec": float(calibration["timing"]["net_additional_to_device_end_sec"]),
            "visible_joints_per_frame_mean": float(np.mean(visible_counts)),
            "visible_joints_per_frame_min": int(np.min(visible_counts)),
            "visible_joints_per_frame_max": int(np.max(visible_counts)),
            "frames_with_zero_visible_joints": int(sum(count == 0 for count in visible_counts)),
            "wrist_correction_mm_mean": float(np.mean(wrist_correction_distances)),
            "wrist_correction_mm_p95": float(np.percentile(wrist_correction_distances, 95)),
            "image_pose_refined_frames": image_pose_frames,
            "image_pose_match_cost_px_median": (
                float(np.median(image_pose_costs)) if image_pose_costs else None
            ),
            "unrefined_sequences": unrefined_sequences,
        }

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "visible_joint_projections.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("camera_key", "seq", "joint", "u_px", "v_px", "forward_depth_mm"),
        )
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "schema": "joint_projection.0722_validated_hybrid.v2",
        "selected_sequences": selected,
        "random_seed": RANDOM_SEED,
        "skeleton_scale": GLOBAL_SKELETON_SCALE,
        "R_head_CH3_08_measured": rotation_head_rigid.tolist(),
        "calibration": str(CALIBRATION_PATH),
        "mocap_source": str(MOCAP_SOURCE),
        "reports": reports,
    }
    (OUTPUT_ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Rendered {SAMPLE_COUNT} images per head camera to {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
