#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from geosim.camera import FisheyeCamera, WristFisheyeCamera, make_default_camera_rig, make_default_wrist_camera_rig
from geosim.config import SimulationConfig, load_config
from geosim.geometry import Ray, triangulate_rays
from geosim.imu import ImuNoiseConfig, fuse_wrist_visual_imu, simulate_wrist_imu
from geosim.linalg import normalize, rigid_align, rotation_error_deg
from geosim.motion import load_motion_npz
from geosim.pose_tracks import PoseEstimate, estimate_wrist_sequence_from_tags, smooth_pose_sequence
from geosim.realistic import RealisticInputConfig, load_realistic_config
from geosim.smplx_numpy import SmplxNumpyModel, load_smplx_model
from geosim.tag_rig import make_wrist_tag_rig
from render import MARKER_IDS, _estimate_frame_poses, _project_points


FOOT_NAMES = ("left_foot", "right_foot")
FOOT_LANDMARK_NAMES = ("ankle", "toe", "inner", "outer")


class CameraView:
    def __init__(self, name: str, camera: FisheyeCamera | WristFisheyeCamera, rig_pos: np.ndarray, rig_rot: np.ndarray):
        self.name = name
        self.camera = camera
        self.rig_pos = rig_pos
        self.rig_rot = rig_rot

    def project(self, point_world: np.ndarray) -> tuple[np.ndarray, bool]:
        return self.camera.project_world(point_world, self.rig_pos, self.rig_rot)

    def ray(self, pixel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.camera.ray_world(pixel, self.rig_pos, self.rig_rot)

    def world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        return self.camera.world_pose(self.rig_pos, self.rig_rot)

    @property
    def principal_point(self) -> np.ndarray:
        return self.camera.principal_point

    @property
    def focal_px(self) -> float:
        return self.camera.focal_px


def parse_foot_landmarks_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a four-view head+wrist foot-pose pipeline test.")
    parser.add_argument("--motion", default=str(ROOT / "test_motion/HumanEva/S1/Walking_3_stageii.npz"))
    parser.add_argument("--config", default=str(ROOT / "configs/default_geometry.json"))
    parser.add_argument("--smplx-model", default=str(ROOT / "smplx_models/SMPLX_NEUTRAL_2020.npz"))
    parser.add_argument("--output", default=str(ROOT / "outputs/four_view_foot_pipeline/S1_walk_four_view_foot_tracks_realistic_imu.npz"))
    parser.add_argument("--view-set", default="all", choices=("all", "head", "wrist"))
    parser.add_argument("--foot", default="both", choices=("both", "left", "right"))
    parser.add_argument("--output-fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-output-frames", type=int, default=0)
    parser.add_argument("--pixel-noise-px", type=float, default=0.5)
    parser.add_argument("--wrist-tag-pixel-noise-px", type=float, default=0.35)
    parser.add_argument("--view-dropout-prob", type=float, default=0.0)
    parser.add_argument("--realistic-config", default=str(ROOT / "configs/realistic_camera.json"))
    parser.add_argument("--occlusion-radius-px", type=float, default=6.0)
    parser.add_argument("--occlusion-depth-margin-m", type=float, default=0.08)
    parser.add_argument("--no-wrist-imu", action="store_true", help="Use visual wrist estimates without simulated IMU fusion.")
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def foot_landmarks_main() -> int:
    args = parse_foot_landmarks_args()
    config = load_config(args.config)
    motion = load_motion_npz(args.motion, smplx_model_path=args.smplx_model)
    model = load_smplx_model(args.smplx_model)
    realistic_config = load_realistic_config(args.realistic_config) if args.realistic_config else None

    stride = max(1, int(round(motion.fps / args.output_fps)))
    output_fps = motion.fps / stride
    frame_indices = np.arange(0, motion.frames, stride, dtype=int)
    if args.max_output_frames > 0:
        frame_indices = frame_indices[: args.max_output_frames]
    foot_names = _select_foot_names(args.foot)

    result = run_foot_landmarks_pipeline(
        motion=motion,
        model=model,
        config=config,
        frame_indices=frame_indices,
        output_fps=output_fps,
        output_path=Path(args.output),
        view_set=args.view_set,
        foot_names=foot_names,
        width=args.width,
        height=args.height,
        pixel_noise_px=args.pixel_noise_px,
        wrist_tag_pixel_noise_px=args.wrist_tag_pixel_noise_px,
        view_dropout_prob=args.view_dropout_prob,
        realistic_config=realistic_config,
        occlusion_radius_px=args.occlusion_radius_px,
        occlusion_depth_margin_m=args.occlusion_depth_margin_m,
        use_wrist_imu=not args.no_wrist_imu,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))
    return 0


def run_foot_landmarks_pipeline(
    *,
    motion,
    model: SmplxNumpyModel,
    config: SimulationConfig,
    frame_indices: np.ndarray,
    output_fps: float,
    output_path: Path,
    view_set: str,
    foot_names: tuple[str, ...],
    width: int,
    height: int,
    pixel_noise_px: float,
    wrist_tag_pixel_noise_px: float,
    view_dropout_prob: float,
    realistic_config: RealisticInputConfig | None,
    occlusion_radius_px: float,
    occlusion_depth_margin_m: float,
    use_wrist_imu: bool,
    seed: int,
) -> dict[str, object]:
    if motion.source_path is None:
        raise ValueError("This pipeline expects an AMASS-backed motion.")
    source = np.load(motion.source_path, allow_pickle=True)
    poses = np.asarray(source["poses"], dtype=float)
    trans = np.asarray(source["trans"], dtype=float)
    betas = np.asarray(source["betas"], dtype=float) if "betas" in source.files else motion.betas
    rng = np.random.default_rng(seed)

    head_cameras = {
        camera.name: _resize_head_camera(camera, width, height)
        for camera in make_default_camera_rig(config.camera_rig)
        if camera.name in {"CAM_A", "CAM_B"}
    }
    wrist_cameras = {
        camera.name: _resize_wrist_camera(camera, width, height)
        for camera in make_default_wrist_camera_rig(config.camera_rig)
    }
    tag_rig = make_wrist_tag_rig(config.tag_rig)

    print("recovering wrist pose from head CAM_A/CAM_B tag observations...", flush=True)
    wrist_estimates, wrist_raw_counts = _recover_wrist_from_head_tags(
        motion=motion,
        head_cameras=head_cameras,
        tag_rig=tag_rig,
        frame_indices=frame_indices,
        marker_size_m=config.tag_rig.tag_size_m,
        pixel_noise_px=wrist_tag_pixel_noise_px,
        rng=rng,
    )
    imu_metadata: dict[str, object] = {}
    if use_wrist_imu:
        imu_noise = ImuNoiseConfig()
        wrist_imu = simulate_wrist_imu(
            motion.left_wrist_pos[frame_indices],
            motion.left_wrist_rot[frame_indices],
            output_fps,
            noise=imu_noise,
            seed=seed + 1009,
        )
        initial_idx = next((idx for idx, estimate in enumerate(wrist_estimates) if estimate is not None), 0)
        initial_estimate = wrist_estimates[initial_idx]
        initial_pos = initial_estimate.position_world if initial_estimate is not None else motion.left_wrist_pos[frame_indices[0]]
        initial_rot = initial_estimate.rotation_world if initial_estimate is not None else motion.left_wrist_rot[frame_indices[0]]
        wrist_estimates = fuse_wrist_visual_imu(
            wrist_estimates,
            wrist_imu,
            initial_position=initial_pos,
            initial_rotation=initial_rot,
        )
        imu_metadata = {
            "enabled": True,
            "gyro_noise_std_rad_s": imu_noise.gyro_noise_std_rad_s,
            "accel_noise_std_m_s2": imu_noise.accel_noise_std_m_s2,
            "gyro_bias_std_rad_s": imu_noise.gyro_bias_std_rad_s,
            "accel_bias_std_m_s2": imu_noise.accel_bias_std_m_s2,
            "simulated_gyro_bias_rad_s": wrist_imu.gyro_bias_rad_s.tolist(),
            "simulated_accel_bias_m_s2": wrist_imu.accel_bias_m_s2.tolist(),
        }
    else:
        wrist_estimates = smooth_pose_sequence(wrist_estimates)
        imu_metadata = {"enabled": False}

    foot_model_points, foot_truth_pos, foot_truth_rot, foot_truth_landmarks = _build_foot_truth(
        model=model,
        poses=poses,
        trans=trans,
        betas=betas,
        frame_indices=frame_indices,
        foot_names=foot_names,
    )

    raw_foot_estimates = {foot: [] for foot in foot_names}
    foot_view_counts = np.zeros((len(frame_indices), len(foot_names)), dtype=np.int32)
    foot_source_counts = {foot: Counter() for foot in foot_names}

    print(f"detecting untagged {','.join(foot_names)} landmarks from realistic {view_set} observations...", flush=True)
    for out_idx, frame_idx in enumerate(frame_indices):
        frame = model.forward_frame(poses[frame_idx], trans[frame_idx], betas, include_vertices=True)
        assert frame.vertices is not None
        views = []
        if view_set in {"all", "head"}:
            views.extend(
                [
                    CameraView("head_front_left", head_cameras["CAM_A"], motion.head_pos[frame_idx], motion.head_rot[frame_idx]),
                    CameraView("head_front_right", head_cameras["CAM_B"], motion.head_pos[frame_idx], motion.head_rot[frame_idx]),
                ]
            )
        wrist_estimate = wrist_estimates[out_idx]
        if view_set in {"all", "wrist"} and wrist_estimate is not None:
            for camera in wrist_cameras.values():
                views.append(CameraView(camera.name.lower(), camera, wrist_estimate.position_world, wrist_estimate.rotation_world))

        occluders = {view.name: _project_occluder_vertices(view, frame.vertices) for view in views}
        for foot_idx, foot_name in enumerate(foot_names):
            detections = _detect_foot_landmarks(
                views=views,
                landmarks_world=foot_truth_landmarks[out_idx, foot_idx],
                foot_center_world=foot_truth_pos[out_idx, foot_idx],
                foot_normal_world=foot_truth_rot[out_idx, foot_idx][:, 2],
                occluders=occluders,
                rng=rng,
                pixel_noise_px=pixel_noise_px,
                view_dropout_prob=view_dropout_prob,
                realistic_config=realistic_config,
                occlusion_radius_px=occlusion_radius_px,
                occlusion_depth_margin_m=occlusion_depth_margin_m,
            )
            foot_view_counts[out_idx, foot_idx] = len(detections)
            estimate = _estimate_foot_pose(
                views={view.name: view for view in views},
                detections=detections,
                model_points=foot_model_points[foot_idx],
            )
            raw_foot_estimates[foot_name].append(estimate)
            foot_source_counts[foot_name][estimate.source if estimate is not None else "missing"] += 1

        if (out_idx + 1) % 100 == 0 or out_idx == len(frame_indices) - 1:
            print(f"processed {out_idx + 1}/{len(frame_indices)} frames", flush=True)

    foot_estimates = {foot: smooth_pose_sequence(raw_foot_estimates[foot]) for foot in foot_names}
    saved = _save_foot_tracks(
        output_path,
        frame_indices=frame_indices,
        output_fps=output_fps,
        foot_model_points=foot_model_points,
        foot_truth_pos=foot_truth_pos,
        foot_truth_rot=foot_truth_rot,
        foot_truth_landmarks=foot_truth_landmarks,
        foot_estimates=foot_estimates,
        foot_view_counts=foot_view_counts,
        wrist_estimates=wrist_estimates,
        wrist_truth_pos=motion.left_wrist_pos[frame_indices],
        wrist_truth_rot=motion.left_wrist_rot[frame_indices],
        foot_names=foot_names,
        metadata={
            "view_set": view_set,
            "pixel_noise_px": float(pixel_noise_px),
            "wrist_tag_pixel_noise_px": float(wrist_tag_pixel_noise_px),
            "view_dropout_prob": float(view_dropout_prob),
            "wrist_imu": imu_metadata,
            "realistic_config": _realistic_metadata(realistic_config),
            "occlusion_radius_px": float(occlusion_radius_px),
            "occlusion_depth_margin_m": float(occlusion_depth_margin_m),
            "wrist_raw_counts": {tag: dict(counts) for tag, counts in wrist_raw_counts.items()},
            "foot_raw_sources": {foot: dict(counts) for foot, counts in foot_source_counts.items()},
        },
    )
    return _summarize(saved)


def _recover_wrist_from_head_tags(
    *,
    motion,
    head_cameras: dict[str, FisheyeCamera],
    tag_rig,
    frame_indices: np.ndarray,
    marker_size_m: float,
    pixel_noise_px: float,
    rng: np.random.Generator,
) -> tuple[list[PoseEstimate | None], dict[str, Counter]]:
    tag_estimates = {tag: [] for tag in MARKER_IDS}
    raw_counts = {tag: Counter() for tag in MARKER_IDS}
    for frame_idx in frame_indices:
        tag_points = tag_rig.world_points(motion.left_wrist_pos[frame_idx], motion.left_wrist_rot[frame_idx])
        detections_by_target = {}
        for label, camera_name in (("head_front_left", "CAM_A"), ("head_front_right", "CAM_B")):
            camera = head_cameras[camera_name]
            detections = {}
            for tag_name in MARKER_IDS:
                corners = np.stack([tag_points[f"{tag_name}_c{i}"] for i in range(4)], axis=0)
                pixels = []
                visible = []
                for corner in corners:
                    pixel, ok = camera.project_world(corner, motion.head_pos[frame_idx], motion.head_rot[frame_idx])
                    pixels.append(pixel)
                    visible.append(ok)
                if all(visible):
                    arr = np.stack(pixels, axis=0)
                    if pixel_noise_px > 0.0:
                        arr = arr + rng.normal(0.0, pixel_noise_px, size=arr.shape)
                    detections[tag_name] = arr
            detections_by_target[label] = detections

        pose_estimates = _estimate_frame_poses(
            cameras={"head_front_left": head_cameras["CAM_A"], "head_front_right": head_cameras["CAM_B"]},
            detections_by_target=detections_by_target,
            head_pos=motion.head_pos[frame_idx],
            head_rot=motion.head_rot[frame_idx],
            marker_size_m=marker_size_m,
        )
        for tag_name in MARKER_IDS:
            estimate = pose_estimates.get(tag_name)
            tag_estimates[tag_name].append(estimate)
            raw_counts[tag_name][estimate.source if estimate is not None else "missing"] += 1
    wrist_estimates = estimate_wrist_sequence_from_tags(tag_estimates, tag_rig, marker_size_m)
    return wrist_estimates, raw_counts


def _build_foot_truth(
    *,
    model: SmplxNumpyModel,
    poses: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray | None,
    frame_indices: np.ndarray,
    foot_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    joint_names = model.joint2num
    spec_by_name = {
        "left_foot": ("left_foot", joint_names["L_Ankle"], joint_names["L_Foot"]),
        "right_foot": ("right_foot", joint_names["R_Ankle"], joint_names["R_Foot"]),
    }
    specs = tuple(spec_by_name[name] for name in foot_names)
    foot_truth_pos = np.zeros((len(frame_indices), len(specs), 3), dtype=float)
    foot_truth_rot = np.zeros((len(frame_indices), len(specs), 3, 3), dtype=float)
    ankle_toe = {name: [] for name, _ankle_idx, _toe_idx in specs}

    for out_idx, frame_idx in enumerate(frame_indices):
        frame = model.forward_frame(poses[frame_idx], trans[frame_idx], betas, include_vertices=False)
        for foot_idx, (foot_name, ankle_idx, toe_idx) in enumerate(specs):
            ankle = frame.joints[ankle_idx]
            toe = frame.joints[toe_idx]
            rot = _foot_frame_from_ankle_toe(ankle, toe)
            pos = 0.5 * (ankle + toe)
            foot_truth_pos[out_idx, foot_idx] = pos
            foot_truth_rot[out_idx, foot_idx] = rot
            ankle_toe[foot_name].append(float(np.linalg.norm(toe - ankle)))

    foot_model_points = np.zeros((len(specs), len(FOOT_LANDMARK_NAMES), 3), dtype=float)
    for foot_idx, (foot_name, _ankle_idx, _toe_idx) in enumerate(specs):
        length = float(np.median(ankle_toe[foot_name]))
        half = 0.5 * length
        width = max(0.08, 0.42 * length)
        foot_model_points[foot_idx] = np.array(
            [
                [-half, 0.0, 0.0],
                [half, 0.0, 0.0],
                [0.20 * length, -0.5 * width, 0.0],
                [0.20 * length, 0.5 * width, 0.0],
            ],
            dtype=float,
        )

    foot_truth_landmarks = np.zeros((len(frame_indices), len(specs), len(FOOT_LANDMARK_NAMES), 3), dtype=float)
    for frame_idx in range(len(frame_indices)):
        for foot_idx in range(len(specs)):
            foot_truth_landmarks[frame_idx, foot_idx] = (
                foot_truth_rot[frame_idx, foot_idx] @ foot_model_points[foot_idx].T
            ).T + foot_truth_pos[frame_idx, foot_idx]
    return foot_model_points, foot_truth_pos, foot_truth_rot, foot_truth_landmarks


def _foot_frame_from_ankle_toe(ankle: np.ndarray, toe: np.ndarray) -> np.ndarray:
    x_axis = normalize(toe - ankle)
    up = np.array([0.0, 0.0, 1.0], dtype=float)
    z_axis = up - x_axis * float(np.dot(up, x_axis))
    if np.linalg.norm(z_axis) < 1e-8:
        z_axis = np.array([0.0, 1.0, 0.0], dtype=float)
    z_axis = normalize(z_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    z_axis = normalize(np.cross(x_axis, y_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def _detect_foot_landmarks(
    *,
    views: list[CameraView],
    landmarks_world: np.ndarray,
    foot_center_world: np.ndarray,
    foot_normal_world: np.ndarray,
    occluders: dict[str, tuple[np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    pixel_noise_px: float,
    view_dropout_prob: float,
    realistic_config: RealisticInputConfig | None,
    occlusion_radius_px: float,
    occlusion_depth_margin_m: float,
) -> dict[str, np.ndarray]:
    detections = {}
    for view in views:
        dropout_prob = _view_dropout_probability(
            view=view,
            foot_center_world=foot_center_world,
            foot_normal_world=foot_normal_world,
            base_dropout_prob=view_dropout_prob,
            realistic_config=realistic_config,
        )
        if dropout_prob > 0.0 and rng.random() < dropout_prob:
            continue
        pixels = []
        visible = []
        landmark_depths = []
        for point in landmarks_world:
            pixel, ok = view.project(point)
            pixels.append(pixel)
            visible.append(ok)
            landmark_depths.append(_point_camera_depth(view, point))
        pixels_arr = np.stack(pixels, axis=0)
        finite = np.asarray(visible, dtype=bool)
        occluded = _landmark_occlusion_mask(
            pixels=pixels_arr,
            depths=np.asarray(landmark_depths, dtype=float),
            occluder=occluders[view.name],
            radius_px=occlusion_radius_px,
            depth_margin_m=occlusion_depth_margin_m,
        )
        finite &= ~occluded
        if int(finite.sum()) < 3:
            continue
        sigma = _view_pixel_sigma(
            view=view,
            foot_center_world=foot_center_world,
            foot_normal_world=foot_normal_world,
            base_sigma_px=pixel_noise_px,
            realistic_config=realistic_config,
        )
        arr = np.full_like(pixels_arr, np.nan)
        arr[finite] = pixels_arr[finite] + rng.normal(0.0, sigma, size=(int(finite.sum()), 2))
        detections[view.name] = arr
    return detections


def _project_occluder_vertices(view: CameraView, vertices_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pixels, visible, points_cam = _project_points(view.camera, view.rig_pos, view.rig_rot, vertices_world)
    valid = visible & np.isfinite(pixels).all(axis=1) & np.isfinite(points_cam[:, 2])
    return pixels[valid], points_cam[valid, 2]


def _landmark_occlusion_mask(
    *,
    pixels: np.ndarray,
    depths: np.ndarray,
    occluder: tuple[np.ndarray, np.ndarray],
    radius_px: float,
    depth_margin_m: float,
) -> np.ndarray:
    occluded = np.zeros(len(pixels), dtype=bool)
    mesh_pixels, mesh_depths = occluder
    if len(mesh_pixels) == 0:
        return occluded
    radius_sq = float(radius_px * radius_px)
    for idx, (pixel, depth) in enumerate(zip(pixels, depths)):
        if not (np.isfinite(pixel).all() and np.isfinite(depth)):
            occluded[idx] = True
            continue
        delta = mesh_pixels - pixel
        near = np.einsum("ij,ij->i", delta, delta) <= radius_sq
        if not np.any(near):
            continue
        closest_depth = float(np.min(mesh_depths[near]))
        if closest_depth < float(depth) - float(depth_margin_m):
            occluded[idx] = True
    return occluded


def _point_camera_depth(view: CameraView, point_world: np.ndarray) -> float:
    cam_pos, cam_rot = view.world_pose()
    point_cam = cam_rot.T @ (point_world - cam_pos)
    return float(point_cam[2])


def _view_pixel_sigma(
    *,
    view: CameraView,
    foot_center_world: np.ndarray,
    foot_normal_world: np.ndarray,
    base_sigma_px: float,
    realistic_config: RealisticInputConfig | None,
) -> float:
    cam_pos, _ = view.world_pose()
    distance = float(np.linalg.norm(foot_center_world - cam_pos))
    ray_to_camera = normalize(cam_pos - foot_center_world)
    facing = abs(float(np.dot(normalize(foot_normal_world), ray_to_camera)))
    angle_factor = 1.0 + 1.8 * (1.0 - facing)
    distance_factor = 1.0 + 0.18 * max(0.0, distance - 1.0)
    realistic_sigma = 0.0
    if realistic_config is not None:
        degradation = realistic_config.degradation
        realistic_sigma += 0.35 * degradation.motion_blur_px
        realistic_sigma += 1.5 * degradation.gaussian_blur_sigma_px
        realistic_sigma += 35.0 * (degradation.shot_noise_std + degradation.read_noise_std)
        realistic_sigma += 0.6 * degradation.occluder_probability * max(1, degradation.occluder_count)
        realistic_sigma += 0.4 * degradation.glare_probability
    return float(np.clip((base_sigma_px + realistic_sigma) * angle_factor * distance_factor, 0.25, 12.0))


def _view_dropout_probability(
    *,
    view: CameraView,
    foot_center_world: np.ndarray,
    foot_normal_world: np.ndarray,
    base_dropout_prob: float,
    realistic_config: RealisticInputConfig | None,
) -> float:
    cam_pos, _ = view.world_pose()
    ray_to_camera = normalize(cam_pos - foot_center_world)
    facing = abs(float(np.dot(normalize(foot_normal_world), ray_to_camera)))
    angle_dropout = 0.32 * max(0.0, 0.35 - facing) / 0.35
    realistic_dropout = 0.0
    if realistic_config is not None:
        degradation = realistic_config.degradation
        realistic_dropout += degradation.dropout_probability
        realistic_dropout += 0.65 * degradation.occluder_probability
        realistic_dropout += 0.35 * degradation.glare_probability
        realistic_dropout += 0.015 * max(0.0, degradation.motion_blur_px - 1.0)
    return float(np.clip(base_dropout_prob + angle_dropout + realistic_dropout, 0.0, 0.95))


def _realistic_metadata(config: RealisticInputConfig | None) -> dict[str, object]:
    if config is None:
        return {}
    degradation = config.degradation
    return {
        "seed": int(config.seed),
        "motion_blur_px": float(degradation.motion_blur_px),
        "gaussian_blur_sigma_px": float(degradation.gaussian_blur_sigma_px),
        "shot_noise_std": float(degradation.shot_noise_std),
        "read_noise_std": float(degradation.read_noise_std),
        "glare_probability": float(degradation.glare_probability),
        "dropout_probability": float(degradation.dropout_probability),
        "occluder_probability": float(degradation.occluder_probability),
        "occluder_count": int(degradation.occluder_count),
    }


def _estimate_foot_pose(
    *,
    views: dict[str, CameraView],
    detections: dict[str, np.ndarray],
    model_points: np.ndarray,
) -> PoseEstimate | None:
    labels = list(detections)
    if len(labels) >= 2:
        source_points = []
        target_points = []
        for landmark_idx in range(len(model_points)):
            rays = []
            for label in labels:
                if not np.isfinite(detections[label][landmark_idx]).all():
                    continue
                try:
                    origin, direction = views[label].ray(detections[label][landmark_idx])
                except ValueError:
                    continue
                rays.append(Ray(label, origin, direction))
            if len(rays) < 2:
                continue
            source_points.append(model_points[landmark_idx])
            target_points.append(triangulate_rays(rays))
        if len(target_points) >= 3:
            rot, pos = rigid_align(np.stack(source_points, axis=0), np.stack(target_points, axis=0))
            used_views = len({label for label in labels if np.isfinite(detections[label]).any()})
            return PoseEstimate(pos, rot, f"multi:{used_views}")
    if len(labels) == 1:
        label = labels[0]
        if np.isfinite(detections[label]).all():
            return _estimate_mono_foot_pose(views[label], detections[label], model_points, source=f"mono:{label}")
    for label in labels:
        if np.isfinite(detections[label]).all():
            return _estimate_mono_foot_pose(views[label], detections[label], model_points, source=f"mono:{label}")
    return None


def _estimate_mono_foot_pose(
    view: CameraView,
    corners_px: np.ndarray,
    model_points: np.ndarray,
    source: str,
) -> PoseEstimate | None:
    directions = np.stack([_pixel_to_camera_direction(view, pixel) for pixel in corners_px], axis=0)
    if np.any(directions[:, 2] <= 1e-6):
        return None
    normalized_points = (directions[:, :2] / directions[:, 2:3]).astype(np.float32)
    ok, rvecs, tvecs, reprojection_errors = cv2.solvePnPGeneric(
        model_points.astype(np.float32),
        normalized_points,
        np.eye(3, dtype=np.float32),
        None,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok or len(rvecs) == 0:
        ok, rvec, tvec = cv2.solvePnP(
            model_points.astype(np.float32),
            normalized_points,
            np.eye(3, dtype=np.float32),
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None
        rvecs = [rvec]
        tvecs = [tvec]
        reprojection_errors = None

    best_idx = 0
    if reprojection_errors is not None:
        best_idx = int(np.argmin(np.asarray(reprojection_errors).reshape(-1)))
    r_cam_obj, _ = cv2.Rodrigues(rvecs[best_idx])
    t_cam_obj = tvecs[best_idx].reshape(3).astype(float)
    cam_pos, cam_rot = view.world_pose()
    return PoseEstimate(cam_pos + cam_rot @ t_cam_obj, cam_rot @ r_cam_obj.astype(float), source)


def _pixel_to_camera_direction(view: CameraView, pixel: np.ndarray) -> np.ndarray:
    delta = np.asarray(pixel, dtype=float) - view.principal_point
    radius = float(np.linalg.norm(delta))
    theta = radius / view.focal_px
    if radius < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    xy = delta / radius
    return normalize(np.array([np.sin(theta) * xy[0], np.sin(theta) * xy[1], np.cos(theta)], dtype=float))


def _save_foot_tracks(
    path: Path,
    *,
    frame_indices: np.ndarray,
    output_fps: float,
    foot_model_points: np.ndarray,
    foot_truth_pos: np.ndarray,
    foot_truth_rot: np.ndarray,
    foot_truth_landmarks: np.ndarray,
    foot_estimates: dict[str, list[PoseEstimate | None]],
    foot_view_counts: np.ndarray,
    wrist_estimates: list[PoseEstimate | None],
    wrist_truth_pos: np.ndarray,
    wrist_truth_rot: np.ndarray,
    foot_names: tuple[str, ...],
    metadata: dict[str, object],
) -> dict[str, np.ndarray]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = len(frame_indices)
    foot_count = len(foot_names)
    foot_est_pos = np.full((frame_count, foot_count, 3), np.nan, dtype=float)
    foot_est_rot = np.full((frame_count, foot_count, 3, 3), np.nan, dtype=float)
    foot_sources = np.full((frame_count, foot_count), "", dtype="<U96")
    foot_est_landmarks = np.full_like(foot_truth_landmarks, np.nan)

    for foot_idx, foot_name in enumerate(foot_names):
        for frame_idx, estimate in enumerate(foot_estimates[foot_name]):
            if estimate is None:
                continue
            foot_est_pos[frame_idx, foot_idx] = estimate.position_world
            foot_est_rot[frame_idx, foot_idx] = estimate.rotation_world
            foot_sources[frame_idx, foot_idx] = estimate.source
            foot_est_landmarks[frame_idx, foot_idx] = (
                estimate.rotation_world @ foot_model_points[foot_idx].T
            ).T + estimate.position_world

    wrist_est_pos = np.full((frame_count, 3), np.nan, dtype=float)
    wrist_est_rot = np.full((frame_count, 3, 3), np.nan, dtype=float)
    for frame_idx, estimate in enumerate(wrist_estimates):
        if estimate is None:
            continue
        wrist_est_pos[frame_idx] = estimate.position_world
        wrist_est_rot[frame_idx] = estimate.rotation_world

    np.savez_compressed(
        path,
        format_version=np.array([1], dtype=np.int32),
        frame_indices=frame_indices.astype(np.int64),
        output_fps=np.array([output_fps], dtype=float),
        foot_names=np.array(foot_names),
        foot_landmark_names=np.array(FOOT_LANDMARK_NAMES),
        foot_model_points=foot_model_points,
        foot_truth_pos=foot_truth_pos,
        foot_truth_rot=foot_truth_rot,
        foot_truth_landmarks=foot_truth_landmarks,
        foot_est_pos=foot_est_pos,
        foot_est_rot=foot_est_rot,
        foot_est_landmarks=foot_est_landmarks,
        foot_sources=foot_sources,
        foot_view_counts=foot_view_counts,
        wrist_est_pos=wrist_est_pos,
        wrist_est_rot=wrist_est_rot,
        wrist_truth_pos=wrist_truth_pos,
        wrist_truth_rot=wrist_truth_rot,
        metadata=np.array(json.dumps(metadata, sort_keys=True)),
    )
    return {
        "path": np.array(str(path)),
        "frame_indices": frame_indices,
        "output_fps": np.array([output_fps], dtype=float),
        "foot_truth_pos": foot_truth_pos,
        "foot_truth_rot": foot_truth_rot,
        "foot_est_pos": foot_est_pos,
        "foot_est_rot": foot_est_rot,
        "foot_sources": foot_sources,
        "foot_view_counts": foot_view_counts,
        "foot_names": np.array(foot_names),
        "metadata": np.array(json.dumps(metadata, sort_keys=True)),
    }


def _summarize(saved: dict[str, np.ndarray]) -> dict[str, object]:
    summary: dict[str, object] = {
        "path": str(saved["path"]),
        "frames": int(len(saved["frame_indices"])),
        "fps": float(saved["output_fps"][0]),
        "feet": {},
        "metadata": json.loads(str(saved["metadata"])),
    }
    foot_names = tuple(str(name) for name in saved["foot_names"])
    for foot_idx, foot_name in enumerate(foot_names):
        valid = np.isfinite(saved["foot_est_pos"][:, foot_idx]).all(axis=1)
        pos_err = np.linalg.norm(saved["foot_est_pos"][valid, foot_idx] - saved["foot_truth_pos"][valid, foot_idx], axis=1)
        rot_err = np.array(
            [
                rotation_error_deg(saved["foot_truth_rot"][idx, foot_idx], saved["foot_est_rot"][idx, foot_idx])
                for idx in np.flatnonzero(valid)
            ],
            dtype=float,
        )
        sources = Counter(str(src) for src in saved["foot_sources"][:, foot_idx] if str(src))
        view_counts = Counter(int(v) for v in saved["foot_view_counts"][:, foot_idx])
        summary["feet"][foot_name] = {
            "valid_frames": int(valid.sum()),
            "view_counts": dict(sorted(view_counts.items())),
            "sources": dict(sorted(sources.items())),
            "position_mean_m": float(pos_err.mean()) if len(pos_err) else None,
            "position_p95_m": float(np.percentile(pos_err, 95)) if len(pos_err) else None,
            "position_max_m": float(pos_err.max()) if len(pos_err) else None,
            "rotation_mean_deg": float(rot_err.mean()) if len(rot_err) else None,
            "rotation_p95_deg": float(np.percentile(rot_err, 95)) if len(rot_err) else None,
            "rotation_max_deg": float(rot_err.max()) if len(rot_err) else None,
        }
    return summary


def _select_foot_names(selection: str) -> tuple[str, ...]:
    if selection == "left":
        return ("left_foot",)
    if selection == "right":
        return ("right_foot",)
    return FOOT_NAMES


def _resize_head_camera(camera: FisheyeCamera, width: int, height: int) -> FisheyeCamera:
    return FisheyeCamera(
        name=camera.name,
        position_head=camera.position_head,
        rotation_cam_to_head=camera.rotation_cam_to_head,
        image_width=width,
        image_height=height,
        fov_deg=camera.fov_deg,
    )


def _resize_wrist_camera(camera: WristFisheyeCamera, width: int, height: int) -> WristFisheyeCamera:
    return WristFisheyeCamera(
        name=camera.name,
        position_wrist=camera.position_wrist,
        rotation_cam_to_wrist=camera.rotation_cam_to_wrist,
        image_width=width,
        image_height=height,
        fov_deg=camera.fov_deg,
    )


# Foot-dorsum tag propagation pipeline
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from geosim.camera import make_default_camera_rig, make_default_wrist_camera_rig
from geosim.config import load_config
from geosim.geometry import Ray, triangulate_rays
from geosim.imu import ImuNoiseConfig, fuse_wrist_visual_imu, simulate_wrist_imu
from geosim.linalg import rigid_align, rotation_error_deg
from geosim.motion import load_motion_npz
from geosim.pose_tracks import PoseEstimate, marker_object_points, smooth_pose_sequence
from geosim.realistic import load_realistic_config
from geosim.smplx_numpy import load_smplx_model
from geosim.tag_rig import make_wrist_tag_rig
from render import _estimate_frame_poses, _estimate_mono_pose

def parse_foot_tag_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover left-foot pose from a simulated foot-dorsum tag using wrist cameras.")
    parser.add_argument("--motion", default=str(ROOT / "test_motion/HumanEva/S1/Walking_3_stageii.npz"))
    parser.add_argument("--config", default=str(ROOT / "configs/default_geometry.json"))
    parser.add_argument("--smplx-model", default=str(ROOT / "smplx_models/SMPLX_NEUTRAL_2020.npz"))
    parser.add_argument("--realistic-config", default=str(ROOT / "configs/realistic_camera.json"))
    parser.add_argument("--output", default=str(ROOT / "outputs/foot_tag_pipeline/S1_walk_wrist_foot_tag_left_tracks.npz"))
    parser.add_argument("--wrist-set", default="both", choices=("left", "right", "both"))
    parser.add_argument("--foot-view-set", default="wrist", choices=("wrist", "head", "all"))
    parser.add_argument("--output-fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-output-frames", type=int, default=0)
    parser.add_argument("--wrist-tag-pixel-noise-px", type=float, default=0.35)
    parser.add_argument("--foot-tag-pixel-noise-px", type=float, default=0.35)
    parser.add_argument("--foot-tag-size-m", type=float, default=0.055)
    parser.add_argument("--foot-tag-forward-offset-m", type=float, default=0.035)
    parser.add_argument("--foot-tag-height-m", type=float, default=0.018)
    parser.add_argument("--occlusion-radius-px", type=float, default=6.0)
    parser.add_argument("--occlusion-depth-margin-m", type=float, default=0.08)
    parser.add_argument("--no-wrist-imu", action="store_true")
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def parse_wrist_ankle_tags_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover wrist and ankle poses from wrist/ankle tags in one pipeline.")
    parser.add_argument("--motion", default="", help="Single AMASS motion .npz.")
    parser.add_argument("--motion-dir", default="", help="Directory of AMASS motion .npz files.")
    parser.add_argument("--config", default=str(ROOT / "configs/default_geometry.json"))
    parser.add_argument("--smplx-model", default=str(ROOT / "smplx_models/SMPLX_NEUTRAL_2020.npz"))
    parser.add_argument("--realistic-config", default=str(ROOT / "configs/realistic_camera.json"))
    parser.add_argument("--output", default="", help="Output .npz for single-motion mode.")
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/wrist_ankle_recovery"))
    parser.add_argument("--output-fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-output-frames", type=int, default=0)
    parser.add_argument("--wrist-tag-pixel-noise-px", type=float, default=0.35)
    parser.add_argument("--ankle-tag-pixel-noise-px", type=float, default=0.35)
    parser.add_argument("--ankle-tag-size-m", type=float, default=0.055)
    parser.add_argument("--ankle-tag-forward-offset-m", type=float, default=0.08)
    parser.add_argument("--ankle-tag-height-m", type=float, default=0.018)
    parser.add_argument("--occlusion-radius-px", type=float, default=6.0)
    parser.add_argument("--occlusion-depth-margin-m", type=float, default=0.08)
    parser.add_argument("--no-wrist-imu", action="store_true")
    parser.add_argument("--seed", type=int, default=41)
    return parser.parse_args()


def foot_tag_main() -> int:
    args = parse_foot_tag_args()
    config = load_config(args.config)
    realistic_config = load_realistic_config(args.realistic_config) if args.realistic_config else None
    motion = load_motion_npz(args.motion, smplx_model_path=args.smplx_model)
    model = load_smplx_model(args.smplx_model)

    stride = max(1, int(round(motion.fps / args.output_fps)))
    output_fps = motion.fps / stride
    frame_indices = np.arange(0, motion.frames, stride, dtype=int)
    if args.max_output_frames > 0:
        frame_indices = frame_indices[: args.max_output_frames]

    summary = run_foot_tag_pipeline(
        motion=motion,
        model=model,
        config=config,
        realistic_config=realistic_config,
        frame_indices=frame_indices,
        output_fps=output_fps,
        output_path=Path(args.output),
        wrist_set=args.wrist_set,
        foot_view_set=args.foot_view_set,
        width=args.width,
        height=args.height,
        wrist_tag_pixel_noise_px=args.wrist_tag_pixel_noise_px,
        foot_tag_pixel_noise_px=args.foot_tag_pixel_noise_px,
        foot_tag_size_m=args.foot_tag_size_m,
        foot_tag_forward_offset_m=args.foot_tag_forward_offset_m,
        foot_tag_height_m=args.foot_tag_height_m,
        occlusion_radius_px=args.occlusion_radius_px,
        occlusion_depth_margin_m=args.occlusion_depth_margin_m,
        use_wrist_imu=not args.no_wrist_imu,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))
    return 0


def wrist_ankle_tags_main() -> int:
    args = parse_wrist_ankle_tags_args()
    config = load_config(args.config)
    realistic_config = load_realistic_config(args.realistic_config) if args.realistic_config else None
    model = load_smplx_model(args.smplx_model)

    motion_paths: list[Path] = []
    if args.motion:
        motion_paths.append(Path(args.motion))
    if args.motion_dir:
        motion_paths.extend(sorted(Path(args.motion_dir).rglob("*.npz")))
    if not motion_paths:
        motion_paths.append(ROOT / "test_motion/HumanEva/S1/Walking_3_stageii.npz")

    summaries = {}
    for motion_idx, motion_path in enumerate(motion_paths):
        try:
            motion = load_motion_npz(motion_path, smplx_model_path=args.smplx_model)
        except ValueError as exc:
            print(f"\n[{motion_idx + 1}/{len(motion_paths)}] skipping {motion_path}: {exc}", flush=True)
            continue
        stride = max(1, int(round(motion.fps / args.output_fps)))
        output_fps = motion.fps / stride
        frame_indices = np.arange(0, motion.frames, stride, dtype=int)
        if args.max_output_frames > 0:
            frame_indices = frame_indices[: args.max_output_frames]
        if args.output and len(motion_paths) == 1:
            output_path = Path(args.output)
        else:
            try:
                rel = motion_path.resolve().relative_to((ROOT / "test_motion").resolve())
            except ValueError:
                rel = Path(motion_path.name)
            safe = "_".join(rel.with_suffix("").parts) + "_wrist_ankle_recovery.npz"
            output_path = Path(args.output_dir) / safe
        print(f"\n[{motion_idx + 1}/{len(motion_paths)}] {motion_path}", flush=True)
        summary = run_wrist_ankle_tag_pipeline(
            motion=motion,
            model=model,
            config=config,
            realistic_config=realistic_config,
            frame_indices=frame_indices,
            output_fps=output_fps,
            output_path=output_path,
            width=args.width,
            height=args.height,
            wrist_tag_pixel_noise_px=args.wrist_tag_pixel_noise_px,
            ankle_tag_pixel_noise_px=args.ankle_tag_pixel_noise_px,
            ankle_tag_size_m=args.ankle_tag_size_m,
            ankle_tag_forward_offset_m=args.ankle_tag_forward_offset_m,
            ankle_tag_height_m=args.ankle_tag_height_m,
            occlusion_radius_px=args.occlusion_radius_px,
            occlusion_depth_margin_m=args.occlusion_depth_margin_m,
            use_wrist_imu=not args.no_wrist_imu,
            seed=args.seed + motion_idx * 1009,
        )
        summaries[str(motion_path)] = summary
        print(json.dumps(summary, indent=2), flush=True)
    if len(summaries) > 1:
        print(json.dumps(summaries, indent=2))
    return 0


def run_foot_tag_pipeline(
    *,
    motion,
    model,
    config,
    realistic_config,
    frame_indices: np.ndarray,
    output_fps: float,
    output_path: Path,
    wrist_set: str,
    foot_view_set: str,
    width: int,
    height: int,
    wrist_tag_pixel_noise_px: float,
    foot_tag_pixel_noise_px: float,
    foot_tag_size_m: float,
    foot_tag_forward_offset_m: float,
    foot_tag_height_m: float,
    occlusion_radius_px: float,
    occlusion_depth_margin_m: float,
    use_wrist_imu: bool,
    seed: int,
) -> dict[str, object]:
    if motion.source_path is None:
        raise ValueError("This pipeline expects an AMASS-backed motion.")
    source = np.load(motion.source_path, allow_pickle=True)
    poses = np.asarray(source["poses"], dtype=float)
    trans = np.asarray(source["trans"], dtype=float)
    betas = np.asarray(source["betas"], dtype=float) if "betas" in source.files else motion.betas
    rng = np.random.default_rng(seed)

    head_cameras = {
        camera.name: _resize_head_camera(camera, width, height)
        for camera in make_default_camera_rig(config.camera_rig)
        if camera.name in {"CAM_A", "CAM_B"}
    }
    wrist_cameras = {
        camera.name: _resize_wrist_camera(camera, width, height)
        for camera in make_default_wrist_camera_rig(config.camera_rig)
    }
    wrist_tag_rig = make_wrist_tag_rig(config.tag_rig)

    selected_sides = _select_wrist_sides(wrist_set)
    _require_right_wrist_if_needed(motion, selected_sides)
    print(f"recovering {','.join(selected_sides)} wrist pose from head CAM_A/CAM_B wrist-tag observations...", flush=True)
    wrist_estimates_by_side: dict[str, list[PoseEstimate | None]] = {}
    wrist_raw_counts_by_side = {}
    imu_metadata: dict[str, object] = {"enabled": use_wrist_imu, "sides": {}}
    for side_idx, side in enumerate(selected_sides):
        wrist_pos, wrist_rot = _wrist_truth_arrays(motion, side)
        wrist_visual_estimates, wrist_raw_counts = _recover_wrist_pose_from_head_tags(
            motion=motion,
            wrist_pos=wrist_pos,
            wrist_rot=wrist_rot,
            head_cameras=head_cameras,
            tag_rig=wrist_tag_rig,
            frame_indices=frame_indices,
            marker_size_m=config.tag_rig.tag_size_m,
            pixel_noise_px=wrist_tag_pixel_noise_px,
            rng=rng,
        )
        wrist_raw_counts_by_side[side] = {tag: dict(counts) for tag, counts in wrist_raw_counts.items()}
        if use_wrist_imu:
            imu_noise = ImuNoiseConfig()
            wrist_imu = simulate_wrist_imu(
                wrist_pos[frame_indices],
                wrist_rot[frame_indices],
                output_fps,
                noise=imu_noise,
                seed=seed + 1009 + side_idx * 97,
            )
            first_valid = next((idx for idx, estimate in enumerate(wrist_visual_estimates) if estimate is not None), 0)
            first_estimate = wrist_visual_estimates[first_valid]
            initial_pos = first_estimate.position_world if first_estimate is not None else wrist_pos[frame_indices[0]]
            initial_rot = first_estimate.rotation_world if first_estimate is not None else wrist_rot[frame_indices[0]]
            wrist_estimates_by_side[side] = fuse_wrist_visual_imu(wrist_visual_estimates, wrist_imu, initial_pos, initial_rot)
            imu_metadata["sides"][side] = {
                "gyro_noise_std_rad_s": imu_noise.gyro_noise_std_rad_s,
                "accel_noise_std_m_s2": imu_noise.accel_noise_std_m_s2,
                "gyro_bias_std_rad_s": imu_noise.gyro_bias_std_rad_s,
                "accel_bias_std_m_s2": imu_noise.accel_bias_std_m_s2,
                "simulated_gyro_bias_rad_s": wrist_imu.gyro_bias_rad_s.tolist(),
                "simulated_accel_bias_m_s2": wrist_imu.accel_bias_m_s2.tolist(),
            }
        else:
            wrist_estimates_by_side[side] = smooth_pose_sequence(wrist_visual_estimates)

    foot_model_points, foot_truth_pos, foot_truth_rot, foot_truth_landmarks = _build_foot_truth(
        model=model,
        poses=poses,
        trans=trans,
        betas=betas,
        frame_indices=frame_indices,
        foot_names=("left_foot",),
    )
    tag_rel_pos = np.array([foot_tag_forward_offset_m, 0.0, foot_tag_height_m], dtype=float)
    tag_rel_rot = np.eye(3, dtype=float)
    tag_object_points = marker_object_points(foot_tag_size_m)

    raw_foot_estimates: list[PoseEstimate | None] = []
    foot_view_counts = np.zeros((len(frame_indices), 1), dtype=np.int32)
    raw_sources: Counter[str] = Counter()

    print(f"detecting left foot-dorsum tag from {foot_view_set} cameras...", flush=True)
    for out_idx, frame_idx in enumerate(frame_indices):
        frame = model.forward_frame(poses[frame_idx], trans[frame_idx], betas, include_vertices=True)
        assert frame.vertices is not None
        views = []
        if foot_view_set in {"head", "all"}:
            views.extend(
                [
                    CameraView("head_front_left", head_cameras["CAM_A"], motion.head_pos[frame_idx], motion.head_rot[frame_idx]),
                    CameraView("head_front_right", head_cameras["CAM_B"], motion.head_pos[frame_idx], motion.head_rot[frame_idx]),
                ]
            )
        if foot_view_set in {"wrist", "all"}:
            for side in selected_sides:
                wrist_estimate = wrist_estimates_by_side[side][out_idx]
                if wrist_estimate is None:
                    continue
                for camera in wrist_cameras.values():
                    views.append(
                        CameraView(
                            f"{side}_{camera.name.lower()}",
                            camera,
                            wrist_estimate.position_world,
                            wrist_estimate.rotation_world,
                        )
                    )
        detections = {}
        if views:
            tag_rot_world = foot_truth_rot[out_idx, 0] @ tag_rel_rot
            tag_pos_world = foot_truth_pos[out_idx, 0] + foot_truth_rot[out_idx, 0] @ tag_rel_pos
            tag_corners_world = (tag_rot_world @ tag_object_points.T).T + tag_pos_world
            occluders = {view.name: _project_occluder_vertices(view, frame.vertices) for view in views}
            detections = _detect_foot_tag_corners(
                views=views,
                tag_corners_world=tag_corners_world,
                tag_center_world=tag_pos_world,
                tag_normal_world=tag_rot_world[:, 2],
                occluders=occluders,
                realistic_config=realistic_config,
                rng=rng,
                pixel_noise_px=foot_tag_pixel_noise_px,
                occlusion_radius_px=occlusion_radius_px,
                occlusion_depth_margin_m=occlusion_depth_margin_m,
            )
        foot_view_counts[out_idx, 0] = len(detections)
        tag_estimate = _estimate_tag_pose_from_wrist_views(
            views={view.name: view for view in views},
            detections=detections,
            marker_size_m=foot_tag_size_m,
        )
        if tag_estimate is None:
            raw_foot_estimates.append(None)
            raw_sources["missing"] += 1
        else:
            foot_rot = tag_estimate.rotation_world @ tag_rel_rot.T
            foot_pos = tag_estimate.position_world - foot_rot @ tag_rel_pos
            raw_foot_estimates.append(PoseEstimate(foot_pos, foot_rot, tag_estimate.source))
            raw_sources[tag_estimate.source] += 1

        if (out_idx + 1) % 100 == 0 or out_idx == len(frame_indices) - 1:
            print(f"processed {out_idx + 1}/{len(frame_indices)} frames", flush=True)

    foot_estimates = {"left_foot": smooth_pose_sequence(raw_foot_estimates)}
    saved = _save_foot_tracks(
        output_path,
        frame_indices=frame_indices,
        output_fps=output_fps,
        foot_model_points=foot_model_points,
        foot_truth_pos=foot_truth_pos,
        foot_truth_rot=foot_truth_rot,
        foot_truth_landmarks=foot_truth_landmarks,
        foot_estimates=foot_estimates,
        foot_view_counts=foot_view_counts,
        wrist_estimates=wrist_estimates_by_side[selected_sides[0]],
        wrist_truth_pos=_wrist_truth_arrays(motion, selected_sides[0])[0][frame_indices],
        wrist_truth_rot=_wrist_truth_arrays(motion, selected_sides[0])[1][frame_indices],
        foot_names=("left_foot",),
        metadata={
            "pipeline": "head_to_wrist_imu_to_wrist_cameras_to_left_foot_tag",
            "view_set": f"{foot_view_set}_foot_views",
            "wrist_sides": list(selected_sides),
            "foot_view_set": foot_view_set,
            "foot_tag_size_m": float(foot_tag_size_m),
            "foot_tag_rel_pos": tag_rel_pos.tolist(),
            "foot_tag_pixel_noise_px": float(foot_tag_pixel_noise_px),
            "wrist_tag_pixel_noise_px": float(wrist_tag_pixel_noise_px),
            "realistic_config": _realistic_metadata(realistic_config),
            "occlusion_radius_px": float(occlusion_radius_px),
            "occlusion_depth_margin_m": float(occlusion_depth_margin_m),
            "wrist_imu": imu_metadata,
            "wrist_raw_counts": wrist_raw_counts_by_side,
            "foot_tag_raw_sources": dict(raw_sources),
        },
    )
    return _summarize_left(saved)


def _detect_foot_tag_corners(
    *,
    views: list[CameraView],
    tag_corners_world: np.ndarray,
    tag_center_world: np.ndarray,
    tag_normal_world: np.ndarray,
    occluders: dict[str, tuple[np.ndarray, np.ndarray]],
    realistic_config,
    rng: np.random.Generator,
    pixel_noise_px: float,
    occlusion_radius_px: float,
    occlusion_depth_margin_m: float,
) -> dict[str, np.ndarray]:
    detections = {}
    for view in views:
        dropout = _view_dropout_probability(
            view=view,
            foot_center_world=tag_center_world,
            foot_normal_world=tag_normal_world,
            base_dropout_prob=0.0,
            realistic_config=realistic_config,
        )
        if dropout > 0.0 and rng.random() < dropout:
            continue
        pixels = []
        visible = []
        depths = []
        for corner in tag_corners_world:
            pixel, ok = view.project(corner)
            pixels.append(pixel)
            visible.append(ok)
            depths.append(_point_camera_depth(view, corner))
        pixels_arr = np.stack(pixels, axis=0)
        if not all(visible):
            continue
        occluded = _landmark_occlusion_mask(
            pixels=pixels_arr,
            depths=np.asarray(depths, dtype=float),
            occluder=occluders[view.name],
            radius_px=occlusion_radius_px,
            depth_margin_m=occlusion_depth_margin_m,
        )
        if np.any(occluded):
            continue
        sigma = _view_pixel_sigma(
            view=view,
            foot_center_world=tag_center_world,
            foot_normal_world=tag_normal_world,
            base_sigma_px=pixel_noise_px,
            realistic_config=realistic_config,
        )
        detections[view.name] = pixels_arr + rng.normal(0.0, sigma, size=pixels_arr.shape)
    return detections


def _estimate_tag_pose_from_wrist_views(
    *,
    views: dict[str, CameraView],
    detections: dict[str, np.ndarray],
    marker_size_m: float,
) -> PoseEstimate | None:
    labels = list(detections)
    if len(labels) >= 2:
        target_points = []
        for corner_idx in range(4):
            rays = []
            for label in labels:
                try:
                    origin, direction = views[label].ray(detections[label][corner_idx])
                except ValueError:
                    continue
                rays.append(Ray(label, origin, direction))
            if len(rays) < 2:
                return None
            target_points.append(triangulate_rays(rays))
        rot, pos = rigid_align(marker_object_points(marker_size_m), np.stack(target_points, axis=0))
        return PoseEstimate(pos, rot, f"multi:{len(labels)}")
    if len(labels) == 1:
        label = labels[0]
        return _estimate_mono_pose(
            camera=views[label].camera,
            corners_px=detections[label],
            head_pos=views[label].rig_pos,
            head_rot=views[label].rig_rot,
            marker_size_m=marker_size_m,
            source=f"mono:{label}",
        )
    return None


def _recover_wrist_pose_from_head_tags(
    *,
    motion,
    wrist_pos: np.ndarray,
    wrist_rot: np.ndarray,
    head_cameras,
    tag_rig,
    frame_indices: np.ndarray,
    marker_size_m: float,
    pixel_noise_px: float,
    rng: np.random.Generator,
):
    tag_estimates = {tag: [] for tag in ("tag0", "tag1")}
    raw_counts = {tag: Counter() for tag in ("tag0", "tag1")}
    for frame_idx in frame_indices:
        tag_points = tag_rig.world_points(wrist_pos[frame_idx], wrist_rot[frame_idx])
        detections_by_target = {}
        for label, camera_name in (("head_front_left", "CAM_A"), ("head_front_right", "CAM_B")):
            camera = head_cameras[camera_name]
            detections = {}
            for tag_name in ("tag0", "tag1"):
                corners = np.stack([tag_points[f"{tag_name}_c{i}"] for i in range(4)], axis=0)
                pixels = []
                visible = []
                for corner in corners:
                    pixel, ok = camera.project_world(corner, motion.head_pos[frame_idx], motion.head_rot[frame_idx])
                    pixels.append(pixel)
                    visible.append(ok)
                if all(visible):
                    arr = np.stack(pixels, axis=0)
                    if pixel_noise_px > 0.0:
                        arr = arr + rng.normal(0.0, pixel_noise_px, size=arr.shape)
                    detections[tag_name] = arr
            detections_by_target[label] = detections
        estimates = _estimate_frame_poses(
            cameras={"head_front_left": head_cameras["CAM_A"], "head_front_right": head_cameras["CAM_B"]},
            detections_by_target=detections_by_target,
            head_pos=motion.head_pos[frame_idx],
            head_rot=motion.head_rot[frame_idx],
            marker_size_m=marker_size_m,
        )
        for tag_name in ("tag0", "tag1"):
            estimate = estimates.get(tag_name)
            tag_estimates[tag_name].append(estimate)
            raw_counts[tag_name][estimate.source if estimate is not None else "missing"] += 1
    from geosim.pose_tracks import estimate_wrist_sequence_from_tags

    return estimate_wrist_sequence_from_tags(tag_estimates, tag_rig, marker_size_m), raw_counts


def _select_wrist_sides(wrist_set: str) -> tuple[str, ...]:
    if wrist_set == "left":
        return ("left",)
    if wrist_set == "right":
        return ("right",)
    return ("left", "right")


def _require_right_wrist_if_needed(motion, sides: tuple[str, ...]) -> None:
    if "right" in sides and (motion.right_wrist_pos is None or motion.right_wrist_rot is None):
        raise ValueError("Right wrist data is unavailable for this motion.")


def _wrist_truth_arrays(motion, side: str) -> tuple[np.ndarray, np.ndarray]:
    if side == "left":
        return motion.left_wrist_pos, motion.left_wrist_rot
    if motion.right_wrist_pos is None or motion.right_wrist_rot is None:
        raise ValueError("Right wrist data is unavailable for this motion.")
    return motion.right_wrist_pos, motion.right_wrist_rot


def _point_camera_depth(view: CameraView, point_world: np.ndarray) -> float:
    cam_pos, cam_rot = view.world_pose()
    return float((cam_rot.T @ (point_world - cam_pos))[2])


def _summarize_left(saved: dict[str, np.ndarray]) -> dict[str, object]:
    valid = np.isfinite(saved["foot_est_pos"][:, 0]).all(axis=1)
    pos_err = np.linalg.norm(saved["foot_est_pos"][valid, 0] - saved["foot_truth_pos"][valid, 0], axis=1)
    rot_err = np.array(
        [
            rotation_error_deg(saved["foot_truth_rot"][idx, 0], saved["foot_est_rot"][idx, 0])
            for idx in np.flatnonzero(valid)
        ],
        dtype=float,
    )
    view_counts = Counter(int(v) for v in saved["foot_view_counts"][:, 0])
    metadata = json.loads(str(saved["metadata"]))
    return {
        "path": str(saved["path"]),
        "frames": int(len(saved["frame_indices"])),
        "fps": float(saved["output_fps"][0]),
        "left_foot": {
            "valid_frames": int(valid.sum()),
            "view_counts": dict(sorted(view_counts.items())),
            "position_mean_m": float(pos_err.mean()) if len(pos_err) else None,
            "position_p95_m": float(np.percentile(pos_err, 95)) if len(pos_err) else None,
            "position_max_m": float(pos_err.max()) if len(pos_err) else None,
            "rotation_mean_deg": float(rot_err.mean()) if len(rot_err) else None,
            "rotation_p95_deg": float(np.percentile(rot_err, 95)) if len(rot_err) else None,
            "rotation_max_deg": float(rot_err.max()) if len(rot_err) else None,
        },
        "metadata": metadata,
    }


def run_wrist_ankle_tag_pipeline(
    *,
    motion,
    model,
    config,
    realistic_config,
    frame_indices: np.ndarray,
    output_fps: float,
    output_path: Path,
    width: int,
    height: int,
    wrist_tag_pixel_noise_px: float,
    ankle_tag_pixel_noise_px: float,
    ankle_tag_size_m: float,
    ankle_tag_forward_offset_m: float,
    ankle_tag_height_m: float,
    occlusion_radius_px: float,
    occlusion_depth_margin_m: float,
    use_wrist_imu: bool,
    seed: int,
) -> dict[str, object]:
    if motion.source_path is None:
        raise ValueError("This pipeline expects an AMASS-backed motion.")
    if motion.right_wrist_pos is None or motion.right_wrist_rot is None:
        raise ValueError("This pipeline expects both left and right wrist data.")
    source = np.load(motion.source_path, allow_pickle=True)
    poses = np.asarray(source["poses"], dtype=float)
    trans = np.asarray(source["trans"], dtype=float)
    betas = np.asarray(source["betas"], dtype=float) if "betas" in source.files else motion.betas
    rng = np.random.default_rng(seed)

    head_cameras = {
        camera.name: _resize_head_camera(camera, width, height)
        for camera in make_default_camera_rig(config.camera_rig)
        if camera.name in {"CAM_A", "CAM_B"}
    }
    wrist_cameras = {
        camera.name: _resize_wrist_camera(camera, width, height)
        for camera in make_default_wrist_camera_rig(config.camera_rig)
    }
    wrist_tag_rig = make_wrist_tag_rig(config.tag_rig)
    wrist_sides = ("left", "right")

    print("recovering left/right wrist poses from head-camera wrist-tag observations...", flush=True)
    wrist_estimates_by_side: dict[str, list[PoseEstimate | None]] = {}
    wrist_raw_counts_by_side = {}
    imu_metadata: dict[str, object] = {"enabled": use_wrist_imu, "sides": {}}
    for side_idx, side in enumerate(wrist_sides):
        wrist_pos, wrist_rot = _wrist_truth_arrays(motion, side)
        visual_estimates, raw_counts = _recover_wrist_pose_from_head_tags(
            motion=motion,
            wrist_pos=wrist_pos,
            wrist_rot=wrist_rot,
            head_cameras=head_cameras,
            tag_rig=wrist_tag_rig,
            frame_indices=frame_indices,
            marker_size_m=config.tag_rig.tag_size_m,
            pixel_noise_px=wrist_tag_pixel_noise_px,
            rng=rng,
        )
        wrist_raw_counts_by_side[side] = {tag: dict(counts) for tag, counts in raw_counts.items()}
        if use_wrist_imu:
            imu_noise = ImuNoiseConfig()
            wrist_imu = simulate_wrist_imu(
                wrist_pos[frame_indices],
                wrist_rot[frame_indices],
                output_fps,
                noise=imu_noise,
                seed=seed + 2003 + side_idx * 101,
            )
            first_valid = next((idx for idx, estimate in enumerate(visual_estimates) if estimate is not None), 0)
            first_estimate = visual_estimates[first_valid]
            initial_pos = first_estimate.position_world if first_estimate is not None else wrist_pos[frame_indices[0]]
            initial_rot = first_estimate.rotation_world if first_estimate is not None else wrist_rot[frame_indices[0]]
            wrist_estimates_by_side[side] = fuse_wrist_visual_imu(visual_estimates, wrist_imu, initial_pos, initial_rot)
            imu_metadata["sides"][side] = {
                "gyro_noise_std_rad_s": imu_noise.gyro_noise_std_rad_s,
                "accel_noise_std_m_s2": imu_noise.accel_noise_std_m_s2,
                "gyro_bias_std_rad_s": imu_noise.gyro_bias_std_rad_s,
                "accel_bias_std_m_s2": imu_noise.accel_bias_std_m_s2,
                "simulated_gyro_bias_rad_s": wrist_imu.gyro_bias_rad_s.tolist(),
                "simulated_accel_bias_m_s2": wrist_imu.accel_bias_m_s2.tolist(),
            }
        else:
            wrist_estimates_by_side[side] = smooth_pose_sequence(visual_estimates)

    ankle_names = ("left_ankle", "right_ankle")
    ankle_truth_pos, ankle_truth_rot = _build_ankle_truth(
        model=model,
        poses=poses,
        trans=trans,
        betas=betas,
        frame_indices=frame_indices,
    )
    tag_rel_pos = np.array([ankle_tag_forward_offset_m, 0.0, ankle_tag_height_m], dtype=float)
    tag_rel_rot = np.eye(3, dtype=float)
    tag_object_points = marker_object_points(ankle_tag_size_m)
    raw_ankle_estimates = {name: [] for name in ankle_names}
    ankle_view_counts = np.zeros((len(frame_indices), len(ankle_names)), dtype=np.int32)
    ankle_raw_sources = {name: Counter() for name in ankle_names}

    print("recovering left/right ankle poses from wrist-camera ankle-tag observations...", flush=True)
    for out_idx, frame_idx in enumerate(frame_indices):
        frame = model.forward_frame(poses[frame_idx], trans[frame_idx], betas, include_vertices=True)
        assert frame.vertices is not None
        views = []
        for side in wrist_sides:
            wrist_estimate = wrist_estimates_by_side[side][out_idx]
            if wrist_estimate is None:
                continue
            for camera in wrist_cameras.values():
                views.append(CameraView(f"{side}_{camera.name.lower()}", camera, wrist_estimate.position_world, wrist_estimate.rotation_world))
        occluders = {view.name: _project_occluder_vertices(view, frame.vertices) for view in views}
        for ankle_idx, ankle_name in enumerate(ankle_names):
            detections = {}
            if views:
                tag_rot_world = ankle_truth_rot[out_idx, ankle_idx] @ tag_rel_rot
                tag_pos_world = ankle_truth_pos[out_idx, ankle_idx] + ankle_truth_rot[out_idx, ankle_idx] @ tag_rel_pos
                tag_corners_world = (tag_rot_world @ tag_object_points.T).T + tag_pos_world
                detections = _detect_foot_tag_corners(
                    views=views,
                    tag_corners_world=tag_corners_world,
                    tag_center_world=tag_pos_world,
                    tag_normal_world=tag_rot_world[:, 2],
                    occluders=occluders,
                    realistic_config=realistic_config,
                    rng=rng,
                    pixel_noise_px=ankle_tag_pixel_noise_px,
                    occlusion_radius_px=occlusion_radius_px,
                    occlusion_depth_margin_m=occlusion_depth_margin_m,
                )
            ankle_view_counts[out_idx, ankle_idx] = len(detections)
            tag_estimate = _estimate_tag_pose_from_wrist_views(
                views={view.name: view for view in views},
                detections=detections,
                marker_size_m=ankle_tag_size_m,
            )
            if tag_estimate is None:
                raw_ankle_estimates[ankle_name].append(None)
                ankle_raw_sources[ankle_name]["missing"] += 1
            else:
                ankle_rot = tag_estimate.rotation_world @ tag_rel_rot.T
                ankle_pos = tag_estimate.position_world - ankle_rot @ tag_rel_pos
                raw_ankle_estimates[ankle_name].append(PoseEstimate(ankle_pos, ankle_rot, tag_estimate.source))
                ankle_raw_sources[ankle_name][tag_estimate.source] += 1
        if (out_idx + 1) % 100 == 0 or out_idx == len(frame_indices) - 1:
            print(f"processed {out_idx + 1}/{len(frame_indices)} frames", flush=True)

    ankle_estimates = {name: smooth_pose_sequence(raw_ankle_estimates[name]) for name in ankle_names}
    saved = _save_wrist_ankle_recovery(
        output_path,
        frame_indices=frame_indices,
        output_fps=output_fps,
        wrist_sides=wrist_sides,
        wrist_estimates_by_side=wrist_estimates_by_side,
        wrist_truth_by_side={
            "left": _wrist_truth_arrays(motion, "left"),
            "right": _wrist_truth_arrays(motion, "right"),
        },
        ankle_names=ankle_names,
        ankle_estimates=ankle_estimates,
        ankle_truth_pos=ankle_truth_pos,
        ankle_truth_rot=ankle_truth_rot,
        ankle_view_counts=ankle_view_counts,
        metadata={
            "pipeline": "head_cameras_to_wrist_tags_to_wrist_cameras_to_ankle_tags",
            "motion": str(motion.source_path),
            "wrist_tag_pixel_noise_px": float(wrist_tag_pixel_noise_px),
            "ankle_tag_pixel_noise_px": float(ankle_tag_pixel_noise_px),
            "ankle_tag_size_m": float(ankle_tag_size_m),
            "ankle_tag_rel_pos": tag_rel_pos.tolist(),
            "realistic_config": _realistic_metadata(realistic_config),
            "occlusion_radius_px": float(occlusion_radius_px),
            "occlusion_depth_margin_m": float(occlusion_depth_margin_m),
            "wrist_imu": imu_metadata,
            "wrist_raw_counts": wrist_raw_counts_by_side,
            "ankle_raw_sources": {name: dict(counts) for name, counts in ankle_raw_sources.items()},
        },
    )
    return _summarize_wrist_ankle(saved)


def _build_ankle_truth(
    *,
    model,
    poses: np.ndarray,
    trans: np.ndarray,
    betas: np.ndarray | None,
    frame_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    joint_names = model.joint2num
    specs = (
        (joint_names["L_Ankle"], joint_names["L_Foot"]),
        (joint_names["R_Ankle"], joint_names["R_Foot"]),
    )
    pos = np.zeros((len(frame_indices), len(specs), 3), dtype=float)
    rot = np.zeros((len(frame_indices), len(specs), 3, 3), dtype=float)
    for out_idx, frame_idx in enumerate(frame_indices):
        frame = model.forward_frame(poses[frame_idx], trans[frame_idx], betas, include_vertices=False)
        for ankle_idx, (ankle_joint, toe_joint) in enumerate(specs):
            ankle = frame.joints[ankle_joint]
            toe = frame.joints[toe_joint]
            pos[out_idx, ankle_idx] = ankle
            rot[out_idx, ankle_idx] = _foot_frame_from_ankle_toe(ankle, toe)
    return pos, rot


def _save_wrist_ankle_recovery(
    path: Path,
    *,
    frame_indices: np.ndarray,
    output_fps: float,
    wrist_sides: tuple[str, ...],
    wrist_estimates_by_side: dict[str, list[PoseEstimate | None]],
    wrist_truth_by_side: dict[str, tuple[np.ndarray, np.ndarray]],
    ankle_names: tuple[str, ...],
    ankle_estimates: dict[str, list[PoseEstimate | None]],
    ankle_truth_pos: np.ndarray,
    ankle_truth_rot: np.ndarray,
    ankle_view_counts: np.ndarray,
    metadata: dict[str, object],
) -> dict[str, np.ndarray]:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = len(frame_indices)
    wrist_est_pos = np.full((frame_count, len(wrist_sides), 3), np.nan, dtype=float)
    wrist_est_rot = np.full((frame_count, len(wrist_sides), 3, 3), np.nan, dtype=float)
    wrist_truth_pos = np.zeros_like(wrist_est_pos)
    wrist_truth_rot = np.zeros_like(wrist_est_rot)
    wrist_sources = np.full((frame_count, len(wrist_sides)), "", dtype="<U96")
    for wrist_idx, side in enumerate(wrist_sides):
        truth_pos, truth_rot = wrist_truth_by_side[side]
        wrist_truth_pos[:, wrist_idx] = truth_pos[frame_indices]
        wrist_truth_rot[:, wrist_idx] = truth_rot[frame_indices]
        for frame_idx, estimate in enumerate(wrist_estimates_by_side[side]):
            if estimate is None:
                continue
            wrist_est_pos[frame_idx, wrist_idx] = estimate.position_world
            wrist_est_rot[frame_idx, wrist_idx] = estimate.rotation_world
            wrist_sources[frame_idx, wrist_idx] = estimate.source

    ankle_est_pos = np.full((frame_count, len(ankle_names), 3), np.nan, dtype=float)
    ankle_est_rot = np.full((frame_count, len(ankle_names), 3, 3), np.nan, dtype=float)
    ankle_sources = np.full((frame_count, len(ankle_names)), "", dtype="<U96")
    for ankle_idx, ankle_name in enumerate(ankle_names):
        for frame_idx, estimate in enumerate(ankle_estimates[ankle_name]):
            if estimate is None:
                continue
            ankle_est_pos[frame_idx, ankle_idx] = estimate.position_world
            ankle_est_rot[frame_idx, ankle_idx] = estimate.rotation_world
            ankle_sources[frame_idx, ankle_idx] = estimate.source

    arrays = {
        "format_version": np.array([1], dtype=np.int32),
        "frame_indices": frame_indices.astype(np.int64),
        "output_fps": np.array([output_fps], dtype=float),
        "wrist_names": np.array([f"{side}_wrist" for side in wrist_sides]),
        "wrist_est_pos": wrist_est_pos,
        "wrist_est_rot": wrist_est_rot,
        "wrist_truth_pos": wrist_truth_pos,
        "wrist_truth_rot": wrist_truth_rot,
        "wrist_sources": wrist_sources,
        "ankle_names": np.array(ankle_names),
        "ankle_est_pos": ankle_est_pos,
        "ankle_est_rot": ankle_est_rot,
        "ankle_truth_pos": ankle_truth_pos,
        "ankle_truth_rot": ankle_truth_rot,
        "ankle_sources": ankle_sources,
        "ankle_view_counts": ankle_view_counts,
        "metadata": np.array(json.dumps(metadata, sort_keys=True)),
    }
    for wrist_idx, side in enumerate(wrist_sides):
        arrays[f"{side}_wrist_est_pos"] = wrist_est_pos[:, wrist_idx]
        arrays[f"{side}_wrist_est_rot"] = wrist_est_rot[:, wrist_idx]
        arrays[f"{side}_wrist_truth_pos"] = wrist_truth_pos[:, wrist_idx]
        arrays[f"{side}_wrist_truth_rot"] = wrist_truth_rot[:, wrist_idx]
    for ankle_idx, ankle_name in enumerate(ankle_names):
        arrays[f"{ankle_name}_est_pos"] = ankle_est_pos[:, ankle_idx]
        arrays[f"{ankle_name}_est_rot"] = ankle_est_rot[:, ankle_idx]
        arrays[f"{ankle_name}_truth_pos"] = ankle_truth_pos[:, ankle_idx]
        arrays[f"{ankle_name}_truth_rot"] = ankle_truth_rot[:, ankle_idx]
    np.savez_compressed(path, **arrays)
    arrays["path"] = np.array(str(path))
    return arrays


def _summarize_wrist_ankle(saved: dict[str, np.ndarray]) -> dict[str, object]:
    summary: dict[str, object] = {
        "path": str(saved["path"]),
        "frames": int(len(saved["frame_indices"])),
        "fps": float(saved["output_fps"][0]),
        "wrist": {},
        "ankle": {},
    }
    for idx, name in enumerate(saved["wrist_names"]):
        valid = np.isfinite(saved["wrist_est_pos"][:, idx]).all(axis=1)
        pos_err = np.linalg.norm(saved["wrist_est_pos"][valid, idx] - saved["wrist_truth_pos"][valid, idx], axis=1)
        rot_err = np.array([rotation_error_deg(saved["wrist_truth_rot"][i, idx], saved["wrist_est_rot"][i, idx]) for i in np.flatnonzero(valid)])
        summary["wrist"][str(name)] = _error_stats(valid, pos_err, rot_err)
    for idx, name in enumerate(saved["ankle_names"]):
        valid = np.isfinite(saved["ankle_est_pos"][:, idx]).all(axis=1)
        pos_err = np.linalg.norm(saved["ankle_est_pos"][valid, idx] - saved["ankle_truth_pos"][valid, idx], axis=1)
        rot_err = np.array([rotation_error_deg(saved["ankle_truth_rot"][i, idx], saved["ankle_est_rot"][i, idx]) for i in np.flatnonzero(valid)])
        stats = _error_stats(valid, pos_err, rot_err)
        stats["view_counts"] = dict(sorted(Counter(int(v) for v in saved["ankle_view_counts"][:, idx]).items()))
        summary["ankle"][str(name)] = stats
    return summary


def _error_stats(valid: np.ndarray, pos_err: np.ndarray, rot_err: np.ndarray) -> dict[str, object]:
    return {
        "valid_frames": int(valid.sum()),
        "position_mean_m": float(pos_err.mean()) if len(pos_err) else None,
        "position_p95_m": float(np.percentile(pos_err, 95)) if len(pos_err) else None,
        "position_max_m": float(pos_err.max()) if len(pos_err) else None,
        "rotation_mean_deg": float(rot_err.mean()) if len(rot_err) else None,
        "rotation_p95_deg": float(np.percentile(rot_err, 95)) if len(rot_err) else None,
        "rotation_max_deg": float(rot_err.max()) if len(rot_err) else None,
    }


COMMANDS = {
    'foot-landmarks': foot_landmarks_main,
    'foot-tag': foot_tag_main,
    'wrist-ankle-tags': wrist_ankle_tags_main,
}


def main() -> int:
    parser = argparse.ArgumentParser(description='Run pose-estimation pipelines.')
    parser.add_argument("command", choices=sorted(COMMANDS))
    args, rest = parser.parse_known_args()
    sys.argv = [f"{Path(sys.argv[0]).name} {args.command}", *rest]
    return int(COMMANDS[args.command]() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
