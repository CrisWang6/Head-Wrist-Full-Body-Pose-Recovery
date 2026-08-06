#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import pickle
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geosim.camera import FisheyeCamera, make_default_camera_rig
from geosim.config import SimulationConfig, load_config
from geosim.geometry import Ray, triangulate_rays
from geosim.linalg import normalize, rigid_align, rotation_error_deg
from geosim.motion import load_motion_npz
from geosim.appearance import prepare_icon_appearance_library
from geosim.pose_tracks import (
    PoseEstimate,
    estimate_wrist_sequence_from_tags,
    marker_object_points,
    save_pose_tracks,
    smooth_pose_sequence,
)
from geosim.realistic import apply_realistic_input_effects, load_realistic_config
from geosim.smplx_numpy import SmplxNumpyModel, load_smplx_model
from geosim.tag_rig import make_wrist_tag_rig


MARKER_IDS = {"tag0": 10, "tag1": 11}
MARKER_PIXELS = 180
MARKER_MARGIN_PIXELS = 28


@dataclass(frozen=True)
class RenderTarget:
    camera_name: str
    label: str


def parse_head_tags_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render stereo head-camera tag views and estimate tag poses.")
    parser.add_argument("--motion", default=str(ROOT / "test_motion/HumanEva/S1/Walking_3_stageii.npz"))
    parser.add_argument("--config", default=str(ROOT / "configs/default_geometry.json"))
    parser.add_argument("--smplx-model", default=str(ROOT / "smplx_models/SMPLX_NEUTRAL_2020.npz"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/camera_views"))
    parser.add_argument("--output-fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--mesh-stride", type=int, default=4)
    parser.add_argument("--max-output-frames", type=int, default=0)
    parser.add_argument("--no-video", action="store_true", help="Compute detections and pose errors without writing AVI.")
    parser.add_argument("--pose-output", default="", help="Output .npz file for recovered and truth wrist/tag trajectories.")
    parser.add_argument("--realistic-config", default="", help="Optional realistic camera-input degradation config JSON.")
    return parser.parse_args()


def head_tags_main() -> int:
    args = parse_head_tags_args()
    config = load_config(args.config)
    motion = load_motion_npz(args.motion, smplx_model_path=args.smplx_model)
    smplx_model = load_smplx_model(args.smplx_model)
    realistic_config = load_realistic_config(args.realistic_config) if args.realistic_config else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stride = max(1, int(round(motion.fps / args.output_fps)))
    output_fps = motion.fps / stride
    frame_indices = np.arange(0, motion.frames, stride, dtype=int)
    if args.max_output_frames > 0:
        frame_indices = frame_indices[: args.max_output_frames]
    pose_output = Path(args.pose_output) if args.pose_output else output_dir / "S1_walk_head_front_left_right_pose_tracks.npz"

    summary = render_stereo_view(
        motion=motion,
        config=config,
        smplx_model=smplx_model,
        targets=(RenderTarget("CAM_A", "head_front_left"), RenderTarget("CAM_B", "head_front_right")),
        frame_indices=frame_indices,
        output_fps=output_fps,
        output_dir=output_dir,
        width=args.width,
        height=args.height,
        mesh_stride=args.mesh_stride,
        write_video=not args.no_video,
        pose_output=pose_output,
        realistic_config=realistic_config,
    )
    print(json.dumps(summary, indent=2))
    return 0


def render_stereo_view(
    motion,
    config: SimulationConfig,
    smplx_model: SmplxNumpyModel,
    targets: tuple[RenderTarget, RenderTarget],
    frame_indices: np.ndarray,
    output_fps: float,
    output_dir: Path,
    width: int,
    height: int,
    mesh_stride: int,
    write_video: bool = True,
    pose_output: Path | None = None,
    realistic_config=None,
) -> dict[str, object]:
    if motion.source_path is None:
        raise ValueError("This renderer expects an AMASS-backed motion.")
    source = np.load(motion.source_path, allow_pickle=True)
    poses = np.asarray(source["poses"], dtype=float)
    trans = np.asarray(source["trans"], dtype=float)
    betas = np.asarray(source["betas"], dtype=float) if "betas" in source.files else motion.betas

    cameras = {camera.name: _resize_camera(camera, width, height) for camera in make_default_camera_rig(config.camera_rig)}
    view_cameras = {target.label: cameras[target.camera_name] for target in targets}
    tag_rig = make_wrist_tag_rig(config.tag_rig)
    marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    detector = cv2.aruco.ArucoDetector(marker_dict, cv2.aruco.DetectorParameters())
    marker_images = {tag: _make_marker_image(marker_dict, marker_id) for tag, marker_id in MARKER_IDS.items()}
    marker_size_m = config.tag_rig.tag_size_m * MARKER_PIXELS / (MARKER_PIXELS + 2 * MARKER_MARGIN_PIXELS)
    rng = np.random.default_rng(realistic_config.seed) if realistic_config is not None else None

    faces = smplx_model.faces[:: max(1, mesh_stride)]
    parents = smplx_model.parents
    output_path = output_dir / "S1_walk_head_front_left_right_stereo_tag_pose.avi"
    writer = None
    if write_video:
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"MJPG"), output_fps, (width * 2, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {output_path}")

    stats = {
        "stereo_front_left_right": {
            "path": str(output_path),
            "frames": int(len(frame_indices)),
            "fps": float(output_fps),
            "detections": {tag: [] for tag in MARKER_IDS},
            "raw_sources": {tag: {} for tag in MARKER_IDS},
        }
    }
    raw_sequences: dict[str, list[PoseEstimate | None]] = {tag: [] for tag in MARKER_IDS}
    truth_sequences: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {tag: [] for tag in MARKER_IDS}

    try:
        for out_idx, frame_idx in enumerate(frame_indices):
            frame = smplx_model.forward_frame(poses[frame_idx], trans[frame_idx], betas, include_vertices=True)
            assert frame.vertices is not None
            head_pos = motion.head_pos[frame_idx]
            head_rot = motion.head_rot[frame_idx]
            tag_points = tag_rig.world_points(motion.left_wrist_pos[frame_idx], motion.left_wrist_rot[frame_idx])

            canvases: dict[str, np.ndarray] = {}
            detections_by_target: dict[str, dict[str, np.ndarray]] = {}
            for target in targets:
                camera = view_cameras[target.label]
                canvas = np.full((height, width, 3), 246, dtype=np.uint8)
                depth_buffer = np.full((height, width), np.inf, dtype=float)

                _draw_mesh(canvas, depth_buffer, camera, head_pos, head_rot, frame.vertices, faces)
                _draw_tag_markers(canvas, depth_buffer, camera, head_pos, head_rot, tag_points, marker_images)
                if realistic_config is not None:
                    canvas = apply_realistic_input_effects(canvas, realistic_config, rng)
                detections_by_target[target.label] = _detect_markers(canvas, detector)
                _draw_skeleton(canvas, camera, head_pos, head_rot, frame.joints, parents)
                canvases[target.label] = canvas

            pose_estimates = _estimate_frame_poses(
                cameras=view_cameras,
                detections_by_target=detections_by_target,
                head_pos=head_pos,
                head_rot=head_rot,
                marker_size_m=marker_size_m,
            )
            for tag_name in MARKER_IDS:
                raw_sequences[tag_name].append(pose_estimates.get(tag_name))
                truth_sequences[tag_name].append(_true_marker_pose_world(tag_points, tag_name))
                if tag_name in pose_estimates:
                    source = pose_estimates[tag_name].source
                    raw_counts = stats["stereo_front_left_right"]["raw_sources"][tag_name]
                    raw_counts[source] = raw_counts.get(source, 0) + 1

            _overlay_pose_estimates(
                canvases=canvases,
                cameras=view_cameras,
                head_pos=head_pos,
                head_rot=head_rot,
                axis_len=config.tag_rig.tag_size_m * 0.65,
                pose_estimates=pose_estimates,
            )

            if write_video:
                assert writer is not None
                frames = []
                for target in targets:
                    canvas = canvases[target.label]
                    cv2.putText(
                        canvas,
                        f"S1 walk {target.label}  frame {frame_idx}/{motion.frames - 1}  {output_fps:.0f} FPS",
                        (24, 36),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.72,
                        (35, 35, 35),
                        2,
                        cv2.LINE_AA,
                    )
                    frames.append(canvas)
                writer.write(np.hstack(frames))

            if (out_idx + 1) % 50 == 0 or out_idx == len(frame_indices) - 1:
                print(f"rendered {out_idx + 1}/{len(frame_indices)} output frames", flush=True)
    finally:
        if writer is not None:
            writer.release()

    smoothed_sequences = {
        tag_name: smooth_pose_sequence(raw_sequences[tag_name]) for tag_name in MARKER_IDS
    }
    for tag_name, sequence in smoothed_sequences.items():
        for estimate, truth in zip(sequence, truth_sequences[tag_name]):
            if estimate is None:
                continue
            true_rot_world, true_pos_world = truth
            stats["stereo_front_left_right"]["detections"][tag_name].append(
                {
                    "position_error_m": float(np.linalg.norm(estimate.position_world - true_pos_world)),
                    "rotation_error_deg": float(rotation_error_deg(true_rot_world, estimate.rotation_world)),
                    "source": estimate.source,
                }
            )
    wrist_estimates = estimate_wrist_sequence_from_tags(smoothed_sequences, tag_rig, marker_size_m)
    if pose_output is not None:
        save_pose_tracks(
            pose_output,
            frame_indices=frame_indices,
            output_fps=output_fps,
            tag_names=tuple(MARKER_IDS),
            tag_estimates=smoothed_sequences,
            tag_truth=truth_sequences,
            wrist_estimates=wrist_estimates,
            wrist_truth_pos=motion.left_wrist_pos[frame_indices],
            wrist_truth_rot=motion.left_wrist_rot[frame_indices],
            raw_sources=stats["stereo_front_left_right"]["raw_sources"],
        )
        stats["stereo_front_left_right"]["pose_output"] = str(pose_output)
    return _summarize(stats)


def _resize_camera(camera: FisheyeCamera, width: int, height: int) -> FisheyeCamera:
    return FisheyeCamera(
        name=camera.name,
        position_head=camera.position_head,
        rotation_cam_to_head=camera.rotation_cam_to_head,
        image_width=width,
        image_height=height,
        fov_deg=camera.fov_deg,
    )


def _make_marker_image(
    marker_dict,
    marker_id: int,
    marker_px: int = MARKER_PIXELS,
    margin_px: int = MARKER_MARGIN_PIXELS,
) -> np.ndarray:
    marker = cv2.aruco.generateImageMarker(marker_dict, marker_id, marker_px)
    image = np.full((marker_px + 2 * margin_px, marker_px + 2 * margin_px), 255, dtype=np.uint8)
    image[margin_px : margin_px + marker_px, margin_px : margin_px + marker_px] = marker
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def _project_points(
    camera: FisheyeCamera,
    head_pos: np.ndarray,
    head_rot: np.ndarray,
    points_world: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam_pos, cam_rot = camera.world_pose(head_pos, head_rot)
    points_cam = (np.asarray(points_world, dtype=float) - cam_pos) @ cam_rot
    norms = np.linalg.norm(points_cam, axis=1)
    directions = points_cam / np.maximum(norms[:, None], 1e-12)
    theta = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
    sin_theta = np.sin(theta)
    xy = np.zeros((len(points_cam), 2), dtype=float)
    valid_sin = sin_theta > 1e-12
    xy[valid_sin] = directions[valid_sin, :2] / sin_theta[valid_sin, None]
    pixels = camera.principal_point + camera.focal_px * theta[:, None] * xy
    visible = (
        (theta <= camera.max_theta_rad)
        & np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < camera.image_width)
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < camera.image_height)
    )
    return pixels, visible, points_cam


def _draw_mesh(
    canvas: np.ndarray,
    depth_buffer: np.ndarray,
    camera: FisheyeCamera,
    head_pos: np.ndarray,
    head_rot: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> None:
    pixels, visible, points_cam = _project_points(camera, head_pos, head_rot, vertices)
    face_indices = np.flatnonzero(visible[faces].all(axis=1))
    if len(face_indices) == 0:
        return
    depths = points_cam[faces[face_indices], 2].mean(axis=1)
    light = normalize(np.array([0.25, -0.35, 0.9]))

    for face_idx in face_indices[np.argsort(depths)[::-1]]:
        face = faces[face_idx]
        pts = np.round(pixels[face]).astype(np.int32)
        min_xy = np.maximum(pts.min(axis=0), 0)
        max_xy = np.minimum(pts.max(axis=0), np.array([canvas.shape[1] - 1, canvas.shape[0] - 1]))
        if np.any(max_xy <= min_xy):
            continue
        x0, y0 = min_xy
        x1, y1 = max_xy
        roi_shape = (int(y1 - y0 + 1), int(x1 - x0 + 1))
        mask = np.zeros(roi_shape, dtype=np.uint8)
        local_pts = pts - np.array([x0, y0])
        cv2.fillConvexPoly(mask, local_pts, 255, cv2.LINE_AA)
        if not np.any(mask):
            continue

        v0, v1, v2 = vertices[face]
        normal = np.cross(v1 - v0, v2 - v0)
        normal_norm = np.linalg.norm(normal)
        shade = 0.62 if normal_norm < 1e-12 else 0.52 + 0.26 * abs(float(np.dot(normal / normal_norm, light)))
        gray = int(np.clip(205 * shade + 40, 120, 222))
        face_depth = float(points_cam[face, 2].mean())
        roi_depth = depth_buffer[y0 : y1 + 1, x0 : x1 + 1]
        update = (mask > 0) & (face_depth < roi_depth)
        if not np.any(update):
            continue
        roi = canvas[y0 : y1 + 1, x0 : x1 + 1]
        roi[update] = (gray, gray, gray)
        roi_depth[update] = face_depth


def _draw_skeleton(
    canvas: np.ndarray,
    camera: FisheyeCamera,
    head_pos: np.ndarray,
    head_rot: np.ndarray,
    joints: np.ndarray,
    parents: np.ndarray,
) -> None:
    pixels, visible, _ = _project_points(camera, head_pos, head_rot, joints)
    points = np.round(pixels).astype(np.int32)
    for joint_idx, parent in enumerate(parents):
        if parent < 0 or not (visible[joint_idx] and visible[parent]):
            continue
        cv2.line(canvas, tuple(points[joint_idx]), tuple(points[parent]), (65, 65, 65), 2, cv2.LINE_AA)


def _draw_tag_markers(
    canvas: np.ndarray,
    depth_buffer: np.ndarray,
    camera: FisheyeCamera,
    head_pos: np.ndarray,
    head_rot: np.ndarray,
    tag_points: dict[str, np.ndarray],
    marker_images: dict[str, np.ndarray],
) -> None:
    cam_pos, cam_rot = camera.world_pose(head_pos, head_rot)
    for tag_name, marker_image in marker_images.items():
        world_corners = np.stack([tag_points[f"{tag_name}_c{i}"] for i in range(4)], axis=0)
        pixels, visible, _ = _project_points(camera, head_pos, head_rot, world_corners)
        if not visible.all():
            continue
        center = world_corners.mean(axis=0)
        x_axis = normalize(world_corners[1] - world_corners[0])
        y_axis = normalize(world_corners[3] - world_corners[0])
        normal = normalize(np.cross(x_axis, y_axis))
        side = float(np.linalg.norm(world_corners[1] - world_corners[0]))
        half = side * 0.5

        min_xy = np.floor(pixels.min(axis=0) - 8.0).astype(int)
        max_xy = np.ceil(pixels.max(axis=0) + 8.0).astype(int)
        x0 = max(0, int(min_xy[0]))
        y0 = max(0, int(min_xy[1]))
        x1 = min(canvas.shape[1] - 1, int(max_xy[0]))
        y1 = min(canvas.shape[0] - 1, int(max_xy[1]))
        if x1 <= x0 or y1 <= y0:
            continue

        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        pixel_grid = np.column_stack([xs.reshape(-1), ys.reshape(-1)])
        directions_cam = np.stack([_pixel_to_camera_direction(camera, pixel) for pixel in pixel_grid], axis=0)
        directions_world = directions_cam @ cam_rot.T
        denom = directions_world @ normal
        valid = np.abs(denom) > 1e-9
        distances = np.zeros(len(pixel_grid), dtype=float)
        distances[valid] = ((center - cam_pos) @ normal) / denom[valid]
        valid &= distances > 0.0
        if not np.any(valid):
            continue

        hit_points = cam_pos + directions_world[valid] * distances[valid, None]
        rel = hit_points - center
        u = rel @ x_axis
        v = rel @ y_axis
        inside = (np.abs(u) <= half) & (np.abs(v) <= half)
        if not np.any(inside):
            continue

        valid_indices = np.flatnonzero(valid)[inside]
        tag_depth = directions_cam[valid_indices, 2] * distances[valid][inside]
        canvas_y = pixel_grid[valid_indices, 1]
        canvas_x = pixel_grid[valid_indices, 0]
        front = np.ones_like(tag_depth, dtype=bool)
        if not np.any(front):
            continue

        tex_x = np.clip(((u[inside] / side) + 0.5) * (marker_image.shape[1] - 1), 0, marker_image.shape[1] - 1).astype(int)
        tex_y = np.clip(((v[inside] / side) + 0.5) * (marker_image.shape[0] - 1), 0, marker_image.shape[0] - 1).astype(int)
        canvas[canvas_y[front], canvas_x[front]] = marker_image[tex_y[front], tex_x[front]]
        depth_buffer[canvas_y[front], canvas_x[front]] = tag_depth[front]


def _detect_markers(canvas: np.ndarray, detector) -> dict[str, np.ndarray]:
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return {}
    cv2.aruco.drawDetectedMarkers(canvas, corners, ids)
    marker_id_to_tag = {marker_id: tag_name for tag_name, marker_id in MARKER_IDS.items()}
    detections = {}
    for marker_corners, marker_id in zip(corners, ids.reshape(-1)):
        tag_name = marker_id_to_tag.get(int(marker_id))
        if tag_name is not None:
            detections[tag_name] = marker_corners.reshape(4, 2).astype(float)
    return detections


def _estimate_frame_poses(
    cameras: dict[str, FisheyeCamera],
    detections_by_target: dict[str, dict[str, np.ndarray]],
    head_pos: np.ndarray,
    head_rot: np.ndarray,
    marker_size_m: float,
) -> dict[str, PoseEstimate]:
    estimates = {}
    for tag_name in MARKER_IDS:
        visible_labels = [label for label, detections in detections_by_target.items() if tag_name in detections]
        if len(visible_labels) >= 2:
            estimate = _estimate_stereo_pose(
                cameras=cameras,
                detections_by_target=detections_by_target,
                labels=visible_labels[:2],
                tag_name=tag_name,
                head_pos=head_pos,
                head_rot=head_rot,
                marker_size_m=marker_size_m,
            )
            if estimate is not None:
                estimates[tag_name] = estimate
                continue
            visible_labels = visible_labels[:1]
        if len(visible_labels) == 1:
            label = visible_labels[0]
            estimate = _estimate_mono_pose(
                camera=cameras[label],
                corners_px=detections_by_target[label][tag_name],
                head_pos=head_pos,
                head_rot=head_rot,
                marker_size_m=marker_size_m,
                source=f"mono:{label}",
            )
            if estimate is not None:
                estimates[tag_name] = estimate
    return estimates


def _estimate_stereo_pose(
    cameras: dict[str, FisheyeCamera],
    detections_by_target: dict[str, dict[str, np.ndarray]],
    labels: list[str],
    tag_name: str,
    head_pos: np.ndarray,
    head_rot: np.ndarray,
    marker_size_m: float,
) -> PoseEstimate | None:
    triangulated_corners = []
    for corner_idx in range(4):
        rays = []
        for label in labels:
            pixel = detections_by_target[label][tag_name][corner_idx]
            try:
                origin, direction = cameras[label].ray_world(pixel, head_pos, head_rot)
            except ValueError:
                continue
            rays.append(Ray(label, origin, direction))
        if len(rays) < 2:
            return None
        triangulated_corners.append(triangulate_rays(rays))
    target_points = np.stack(triangulated_corners, axis=0)
    est_rot_world, est_pos_world = rigid_align(marker_object_points(marker_size_m), target_points)
    return PoseEstimate(est_pos_world, est_rot_world, "stereo")


def _estimate_mono_pose(
    camera: FisheyeCamera,
    corners_px: np.ndarray,
    head_pos: np.ndarray,
    head_rot: np.ndarray,
    marker_size_m: float,
    source: str,
) -> PoseEstimate | None:
    directions = np.stack([_pixel_to_camera_direction(camera, pixel) for pixel in corners_px], axis=0)
    if np.any(directions[:, 2] <= 1e-6):
        return None
    normalized_points = (directions[:, :2] / directions[:, 2:3]).astype(np.float32)
    object_points = marker_object_points(marker_size_m).astype(np.float32)
    ok, rvecs, tvecs, reprojection_errors = cv2.solvePnPGeneric(
        object_points,
        normalized_points,
        np.eye(3, dtype=np.float32),
        None,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not ok or len(rvecs) == 0:
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
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
    cam_pos, cam_rot = camera.world_pose(head_pos, head_rot)
    return PoseEstimate(
        position_world=cam_pos + cam_rot @ t_cam_obj,
        rotation_world=cam_rot @ r_cam_obj.astype(float),
        source=source,
    )


def _overlay_pose_estimates(
    canvases: dict[str, np.ndarray],
    cameras: dict[str, FisheyeCamera],
    head_pos: np.ndarray,
    head_rot: np.ndarray,
    axis_len: float,
    pose_estimates: dict[str, PoseEstimate],
) -> None:
    for tag_name, estimate in pose_estimates.items():
        for label, camera in cameras.items():
            cam_pos, cam_rot = camera.world_pose(head_pos, head_rot)
            r_cam_obj = cam_rot.T @ estimate.rotation_world
            t_cam_obj = cam_rot.T @ (estimate.position_world - cam_pos)
            _draw_pose_axes(canvases[label], camera, r_cam_obj, t_cam_obj, axis_len=axis_len)


def _pixel_to_camera_direction(camera: FisheyeCamera, pixel: np.ndarray) -> np.ndarray:
    delta = np.asarray(pixel, dtype=float) - camera.principal_point
    radius = float(np.linalg.norm(delta))
    theta = radius / camera.focal_px
    if radius < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    xy = delta / radius
    return normalize(np.array([np.sin(theta) * xy[0], np.sin(theta) * xy[1], np.cos(theta)]))


def _true_marker_pose_world(tag_points: dict[str, np.ndarray], tag_name: str) -> tuple[np.ndarray, np.ndarray]:
    corners = np.stack([tag_points[f"{tag_name}_c{i}"] for i in range(4)], axis=0)
    center = corners.mean(axis=0)
    x_axis = normalize(corners[1] - corners[0])
    y_axis = normalize(corners[3] - corners[0])
    z_axis = normalize(np.cross(x_axis, y_axis))
    return np.column_stack([x_axis, y_axis, z_axis]), center


def _draw_pose_axes(
    canvas: np.ndarray,
    camera: FisheyeCamera,
    r_cam_obj: np.ndarray,
    t_cam_obj: np.ndarray,
    axis_len: float,
) -> None:
    points_obj = np.array(
        [[0.0, 0.0, 0.0], [axis_len, 0.0, 0.0], [0.0, axis_len, 0.0], [0.0, 0.0, axis_len]],
        dtype=float,
    )
    points_cam = (r_cam_obj @ points_obj.T).T + t_cam_obj
    projected = []
    for point_cam in points_cam:
        pixel, visible = camera.project_camera(point_cam)
        if not visible:
            return
        projected.append(np.round(pixel).astype(int))
    origin, x_end, y_end, z_end = projected
    cv2.arrowedLine(canvas, tuple(origin), tuple(x_end), (0, 0, 230), 2, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(canvas, tuple(origin), tuple(y_end), (0, 170, 0), 2, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(canvas, tuple(origin), tuple(z_end), (230, 0, 0), 2, cv2.LINE_AA, tipLength=0.25)


def _summarize(stats: dict[str, object]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for label, data in stats.items():
        detections = data["detections"]
        label_summary = {
            "path": data["path"],
            "pose_output": data.get("pose_output", ""),
            "frames": data["frames"],
            "fps": data["fps"],
            "tags": {},
        }
        for tag_name, rows in detections.items():
            if not rows:
                label_summary["tags"][tag_name] = {"detections": 0}
                continue
            pos = np.array([row["position_error_m"] for row in rows], dtype=float)
            rot = np.array([row["rotation_error_deg"] for row in rows], dtype=float)
            source_counts: dict[str, int] = {}
            for row in rows:
                source = row.get("source", "unknown")
                source_counts[source] = source_counts.get(source, 0) + 1
            label_summary["tags"][tag_name] = {
                "detections": int(len(rows)),
                "raw_sources": data.get("raw_sources", {}).get(tag_name, {}),
                "estimate_sources": source_counts,
                "position_mean_m": float(pos.mean()),
                "position_p95_m": float(np.percentile(pos, 95)),
                "position_max_m": float(pos.max()),
                "rotation_mean_deg": float(rot.mean()),
                "rotation_p95_deg": float(np.percentile(rot, 95)),
                "rotation_max_deg": float(rot.max()),
            }
        summary[label] = label_summary
    return summary


# Wrist-camera view rendering
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from geosim.camera import WristFisheyeCamera, make_default_wrist_camera_rig
from geosim.config import load_config
from geosim.linalg import normalize
from geosim.motion import load_motion_npz
from geosim.realistic import apply_realistic_input_effects, load_realistic_config
from geosim.smplx_numpy import load_smplx_model
from geosim.tag_rig import make_wrist_tag_rig

CAMERA_CHOICES = ("BOTH", "WRIST_FORWARD", "WRIST_PALM_NORMAL")


def parse_wrist_views_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render wrist-mounted fisheye camera views for an AMASS motion.")
    parser.add_argument("--motion", default=str(ROOT / "test_motion/HumanEva/S1/Walking_3_stageii.npz"))
    parser.add_argument("--config", default=str(ROOT / "configs/default_geometry.json"))
    parser.add_argument("--smplx-model", default=str(ROOT / "smplx_models/SMPLX_NEUTRAL_2020.npz"))
    parser.add_argument("--camera", default="BOTH", choices=CAMERA_CHOICES)
    parser.add_argument("--output-dir", default=str(ROOT / "outputs/wrist_camera_views"))
    parser.add_argument("--output-fps", type=float, default=60.0)
    parser.add_argument("--width", type=int, default=1920, help="Final output width. BOTH mode splits this across two views.")
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--mesh-stride", type=int, default=1)
    parser.add_argument("--max-output-frames", type=int, default=0)
    parser.add_argument("--realistic-config", default="", help="Optional realistic camera-input degradation config JSON.")
    return parser.parse_args()


def wrist_views_main() -> int:
    args = parse_wrist_views_args()
    config = load_config(args.config)
    motion = load_motion_npz(args.motion, smplx_model_path=args.smplx_model)
    smplx_model = load_smplx_model(args.smplx_model)
    realistic_config = load_realistic_config(args.realistic_config) if args.realistic_config else None

    stride = max(1, int(round(motion.fps / args.output_fps)))
    output_fps = motion.fps / stride
    frame_indices = np.arange(0, motion.frames, stride, dtype=int)
    if args.max_output_frames > 0:
        frame_indices = frame_indices[: args.max_output_frames]

    camera_names = ("WRIST_FORWARD", "WRIST_PALM_NORMAL") if args.camera == "BOTH" else (args.camera,)
    view_width = args.width // len(camera_names)
    cameras = {
        camera.name: _resize_wrist_camera(camera, view_width, args.height)
        for camera in make_default_wrist_camera_rig(config.camera_rig)
    }
    selected_cameras = [cameras[name] for name in camera_names]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_slug = "wrist_forward_normal" if args.camera == "BOTH" else args.camera.lower()
    output_path = output_dir / f"S1_walk_{camera_slug}_view.avi"

    summary = render_wrist_views(
        motion=motion,
        smplx_model=smplx_model,
        cameras=selected_cameras,
        tag_rig=make_wrist_tag_rig(config.tag_rig),
        frame_indices=frame_indices,
        output_fps=output_fps,
        output_path=output_path,
        width=view_width,
        height=args.height,
        mesh_stride=args.mesh_stride,
        realistic_config=realistic_config,
    )
    print(json.dumps(summary, indent=2))
    return 0


def render_wrist_views(
    *,
    motion,
    smplx_model,
    cameras: list[WristFisheyeCamera],
    tag_rig,
    frame_indices: np.ndarray,
    output_fps: float,
    output_path: Path,
    width: int,
    height: int,
    mesh_stride: int,
    realistic_config=None,
) -> dict[str, object]:
    if motion.source_path is None:
        raise ValueError("This renderer expects an AMASS-backed motion.")
    source = np.load(motion.source_path, allow_pickle=True)
    poses = np.asarray(source["poses"], dtype=float)
    trans = np.asarray(source["trans"], dtype=float)
    betas = np.asarray(source["betas"], dtype=float) if "betas" in source.files else motion.betas

    marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker_images = {tag: _make_marker_image(marker_dict, marker_id) for tag, marker_id in MARKER_IDS.items()}
    faces = smplx_model.faces[:: max(1, mesh_stride)]
    parents = smplx_model.parents
    rng = np.random.default_rng(realistic_config.seed) if realistic_config is not None else None

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"MJPG"), output_fps, (width * len(cameras), height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    try:
        for out_idx, frame_idx in enumerate(frame_indices):
            frame = smplx_model.forward_frame(poses[frame_idx], trans[frame_idx], betas, include_vertices=True)
            assert frame.vertices is not None
            wrist_pos = motion.left_wrist_pos[frame_idx]
            wrist_rot = motion.left_wrist_rot[frame_idx]
            tag_points = tag_rig.world_points(wrist_pos, wrist_rot)

            panels = []
            for camera in cameras:
                canvas = np.full((height, width, 3), 246, dtype=np.uint8)
                depth_buffer = np.full((height, width), np.inf, dtype=float)
                _draw_dense_mesh(canvas, camera, wrist_pos, wrist_rot, frame.vertices, faces)
                _draw_tag_markers(canvas, depth_buffer, camera, wrist_pos, wrist_rot, tag_points, marker_images)
                if realistic_config is not None:
                    canvas = apply_realistic_input_effects(canvas, realistic_config, rng)
                _draw_skeleton(canvas, camera, wrist_pos, wrist_rot, frame.joints, parents)
                cv2.putText(
                    canvas,
                    f"S1 walk {camera.name.lower()}  frame {frame_idx}/{motion.frames - 1}  {output_fps:.0f} FPS",
                    (24, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (35, 35, 35),
                    2,
                    cv2.LINE_AA,
                )
                panels.append(canvas)
            writer.write(np.hstack(panels))
            if (out_idx + 1) % 50 == 0 or out_idx == len(frame_indices) - 1:
                print(f"rendered {out_idx + 1}/{len(frame_indices)} output frames", flush=True)
    finally:
        writer.release()

    return {
        "path": str(output_path),
        "cameras": [camera.name for camera in cameras],
        "frames": int(len(frame_indices)),
        "fps": float(output_fps),
        "resolution": [int(width * len(cameras)), int(height)],
        "view_resolution": [int(width), int(height)],
        "mesh_faces": int(len(faces)),
    }


def _draw_dense_mesh(
    canvas: np.ndarray,
    camera: WristFisheyeCamera,
    wrist_pos: np.ndarray,
    wrist_rot: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> None:
    pixels, visible, points_cam = _project_points(camera, wrist_pos, wrist_rot, vertices)
    face_indices = np.flatnonzero(visible[faces].all(axis=1))
    if len(face_indices) == 0:
        return
    visible_faces = faces[face_indices]
    depths = points_cam[visible_faces, 2].mean(axis=1)
    light = normalize(np.array([0.2, -0.4, 0.88], dtype=float))

    pts = np.round(pixels[visible_faces]).astype(np.int32)
    in_bounds = (
        (pts[:, :, 0].min(axis=1) >= -20)
        & (pts[:, :, 0].max(axis=1) < canvas.shape[1] + 20)
        & (pts[:, :, 1].min(axis=1) >= -20)
        & (pts[:, :, 1].max(axis=1) < canvas.shape[0] + 20)
    )
    if not np.any(in_bounds):
        return

    pts = pts[in_bounds]
    depths = depths[in_bounds]
    mesh_faces = visible_faces[in_bounds]
    v0 = vertices[mesh_faces[:, 0]]
    v1 = vertices[mesh_faces[:, 1]]
    v2 = vertices[mesh_faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    normal_norms = np.linalg.norm(normals, axis=1)
    unit_normals = normals / np.maximum(normal_norms[:, None], 1e-12)
    shades = 0.55 + 0.25 * np.abs(unit_normals @ light)
    shades = np.where(normal_norms > 1e-12, shades, 0.64)
    grays = np.clip(210 * shades + 38, 120, 225)

    depth_edges = np.linspace(float(depths.min()), float(depths.max()) + 1e-9, 9)
    depth_bins = np.clip(np.digitize(depths, depth_edges) - 1, 0, 7)
    shade_bins = np.clip(((grays - 120) / (225 - 120) * 5).astype(int), 0, 5)
    shade_values = np.linspace(132, 218, 6).astype(int)

    for depth_bin in range(7, -1, -1):
        depth_mask = depth_bins == depth_bin
        if not np.any(depth_mask):
            continue
        for shade_bin, gray in enumerate(shade_values):
            group = pts[depth_mask & (shade_bins == shade_bin)]
            if len(group) == 0:
                continue
            cv2.fillPoly(canvas, list(group), (int(gray), int(gray), int(gray)), cv2.LINE_AA)


def _resize_wrist_camera(camera: WristFisheyeCamera, width: int, height: int) -> WristFisheyeCamera:
    return WristFisheyeCamera(
        name=camera.name,
        position_wrist=camera.position_wrist,
        rotation_cam_to_wrist=camera.rotation_cam_to_wrist,
        image_width=width,
        image_height=height,
        fov_deg=camera.fov_deg,
    )


def parse_blenderproc_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an AMASS/SMPL-X motion with BlenderProc cameras and wrist tags.")
    parser.add_argument("--motion", default=str(ROOT / "test_motion/HumanEva/S1/Walking_3_stageii.npz"))
    parser.add_argument("--config", default=str(ROOT / "configs/default_geometry.json"))
    parser.add_argument("--smplx-model", default=str(ROOT / "smplx_models/SMPLX_NEUTRAL_2020.npz"))
    parser.add_argument("--output-dir", default="", help="Defaults to outputs/blenderproc/<motion_stem>.")
    parser.add_argument("--output-fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--max-output-frames", type=int, default=0, help="0 renders the whole 30 Hz trajectory.")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--engine", default="CYCLES", choices=("CYCLES", "BLENDER_EEVEE_NEXT"))
    parser.add_argument("--device", default="GPU", choices=("GPU", "CPU"))
    parser.add_argument("--denoise", action="store_true", help="Enable Cycles denoising. Disabled by default for sharper camera data.")
    parser.add_argument("--video-format", default="mp4", choices=("mp4", "avi"))
    parser.add_argument("--video-crf", type=int, default=16, help="Final MP4 CRF. Lower is higher quality; 16 is visually high quality.")
    parser.add_argument("--camera-name", default="", help="Render only one cached camera, e.g. head_front_left. Empty renders all cameras.")
    parser.add_argument(
        "--head-frame",
        default="smplx_relative",
        choices=("smplx_relative", "smplx", "shoulders"),
        help="Head rig orientation for BlenderProc. 'smplx_relative' calibrates the first frame to the rig axes, then follows SMPL-X Head rotation.",
    )
    parser.add_argument("--blenderproc-bin", default="", help="Path to blenderproc inside camtest if auto-detection fails.")
    parser.add_argument(
        "--appearance-root",
        default=str(ROOT / "smplx_models/icon_appearances"),
        help="Prepared appearance folders containing mesh_smplx.obj and smplx_param.pkl.",
    )
    parser.add_argument("--appearance-subjects", default="", help="Comma-separated subject ids under --appearance-root.")
    parser.add_argument("--appearance-index-offset", type=int, default=0)
    parser.add_argument("--appearance-count", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=20260612)
    parser.add_argument(
        "--parallel-cameras",
        type=int,
        default=0,
        help="Number of BlenderProc camera jobs to run at once. 0 auto-uses visible GPUs for GPU rendering.",
    )
    parser.add_argument("--gpu-ids", default="", help="Comma-separated physical GPU ids for parallel camera jobs, e.g. 0,1,2,3.")
    parser.add_argument("--prepare-only", action="store_true", help="Only build the BlenderProc cache; do not launch Blender.")
    return parser.parse_args()


def blenderproc_main() -> int:
    args = parse_blenderproc_args()
    from geosim.blenderproc_cache import build_blenderproc_motion_cache_from_amass

    motion_path = Path(args.motion)
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "outputs/blenderproc" / motion_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    smplx_model = load_smplx_model(args.smplx_model)
    blenderproc_bin = _resolve_blenderproc_bin(args.blenderproc_bin)
    runner = ROOT / "src/geosim/blenderproc_runner.py"
    gpu_ids = _resolve_gpu_ids(args.gpu_ids) if args.device == "GPU" else []
    appearance_subjects = _select_appearance_subjects(
        args.appearance_root,
        args.appearance_count,
        args.random_seed,
        args.appearance_subjects,
    )
    if len(appearance_subjects) > 1 or appearance_subjects[0] is not None:
        print(
            f"Selected appearances: {[subject.name if subject is not None else 'default' for subject in appearance_subjects]}",
            flush=True,
        )

    for local_appearance_idx, appearance_subject in enumerate(appearance_subjects):
        appearance_idx = int(args.appearance_index_offset) + int(local_appearance_idx)
        rng = np.random.default_rng(int(args.random_seed) + 7919 * appearance_idx)
        run_output_dir = output_dir
        if len(appearance_subjects) > 1 or appearance_subject is not None:
            subject_name = appearance_subject.name if appearance_subject is not None else "default"
            run_output_dir = output_dir / f"appearance_{appearance_idx:02d}_{subject_name}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        body_betas = _load_thuman_betas(appearance_subject) if appearance_subject is not None else None
        appearance_manifest = _load_appearance_asset_manifest(appearance_subject) if appearance_subject is not None else {}
        body_face_groups = _make_body_face_groups(smplx_model)
        body_group_colors = _make_body_group_colors(rng)
        scene_config = _make_random_scene_config(rng)

        cache_result = build_blenderproc_motion_cache_from_amass(
            motion_path=motion_path,
            model=smplx_model,
            config=config,
            tag_rig=make_wrist_tag_rig(config.tag_rig),
            output_dir=run_output_dir,
            output_fps=args.output_fps,
            max_output_frames=args.max_output_frames,
            video_width=args.width,
            video_height=args.height,
            head_frame_mode=args.head_frame,
            body_betas=body_betas,
            body_face_groups=body_face_groups,
            body_group_colors=body_group_colors,
            appearance_subject=appearance_subject.name if appearance_subject is not None else "",
            appearance_render_status=str(appearance_manifest.get("render_status", "")),
            scene_config=scene_config,
        )
        camera_names = tuple(name for name in cache_result.camera_names if not args.camera_name or name == args.camera_name)
        if args.camera_name and not camera_names:
            raise ValueError(f"Unknown camera {args.camera_name!r}. Available cameras: {', '.join(cache_result.camera_names)}")
        summary = {
            "backend": "blenderproc",
            "cache": str(cache_result.cache_path),
            "metadata": str(cache_result.metadata_path),
            "output_dir": str(run_output_dir),
            "frames": cache_result.frame_count,
            "fps": cache_result.output_fps,
            "cameras": list(camera_names),
            "appearance_subject": appearance_subject.name if appearance_subject is not None else "",
            "appearance_render_status": appearance_manifest.get("render_status", ""),
            "body_group_colors": body_group_colors.tolist(),
            "scene": scene_config,
        }
        print(json.dumps(summary, indent=2), flush=True)
        if args.prepare_only:
            continue

        parallel_cameras = _resolve_parallel_camera_count(args.parallel_cameras, args.device, gpu_ids, len(camera_names))
        print(
            f"Launching BlenderProc camera jobs: parallel={parallel_cameras}, gpu_ids={gpu_ids or ['shared/default']}",
            flush=True,
        )
        _run_blenderproc_camera_jobs(
            blenderproc_bin=blenderproc_bin,
            runner=runner,
            cache_path=cache_result.cache_path,
            output_dir=run_output_dir,
            samples=args.samples,
            engine=args.engine,
            device=args.device,
            denoise=args.denoise,
            video_format=args.video_format,
            video_crf=args.video_crf,
            motion_label=motion_path.stem,
            camera_names=camera_names,
            parallel_cameras=parallel_cameras,
            gpu_ids=gpu_ids,
        )
    return 0


def parse_isaacsim_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an AMASS/SMPL-X motion with Isaac Sim cameras.")
    parser.add_argument("--motion", default=str(ROOT / "test_motion/HumanEva/S1/Walking_3_stageii.npz"))
    parser.add_argument("--config", default=str(ROOT / "configs/default_geometry.json"))
    parser.add_argument("--smplx-model", default=str(ROOT / "smplx_models/SMPLX_NEUTRAL_2020.npz"))
    parser.add_argument("--output-dir", default="", help="Defaults to outputs/isaacsim/<motion_stem>.")
    parser.add_argument("--output-fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1920, help="Final cropped video width.")
    parser.add_argument("--height", type=int, default=1080, help="Final cropped video height.")
    parser.add_argument("--max-output-frames", type=int, default=0, help="0 renders the whole 30 Hz trajectory.")
    parser.add_argument(
        "--head-frame",
        default="smplx_relative",
        choices=("smplx_relative", "smplx", "shoulders"),
        help="Head rig orientation. 'smplx_relative' calibrates the first frame to the rig axes, then follows SMPL-X Head rotation.",
    )
    parser.add_argument("--isaacsim-python", default=os.environ.get("ISAACSIM_PYTHON", ""))
    parser.add_argument("--renderer", default="RayTracedLighting", choices=("RayTracedLighting", "PathTracing"))
    parser.add_argument("--rt-subframes", type=int, default=1)
    parser.add_argument("--warmup-frames", type=int, default=12)
    parser.add_argument("--video-format", default="mp4", choices=("mp4", "avi"))
    parser.add_argument("--camera-name", default="", help="Render only one cached camera, e.g. head_front_left. Empty renders all cameras.")
    parser.add_argument("--hide-wrist-tags", action="store_true", help="Keep wrist tag geometry out of rendered RGB videos.")
    parser.add_argument(
        "--appearance-root",
        default=str(ROOT / "smplx_models/icon_appearances"),
        help="Prepared appearance folders containing mesh_smplx.obj and smplx_param.pkl.",
    )
    parser.add_argument("--appearance-subjects", default="", help="Comma-separated subject ids under --appearance-root, e.g. 0147,1353.")
    parser.add_argument("--appearance-index-offset", type=int, default=0, help="Index offset used in appearance output directory names.")
    parser.add_argument("--appearance-count", type=int, default=1, help="Number of random appearance variants to render.")
    parser.add_argument("--random-seed", type=int, default=20260609)
    parser.add_argument(
        "--parallel-cameras",
        type=int,
        default=0,
        help="Number of Isaac Sim camera jobs to run at once. 0 auto-uses visible GPUs.",
    )
    parser.add_argument("--gpu-ids", default="", help="Comma-separated physical GPU ids for parallel camera jobs, e.g. 0,1,2,3.")
    parser.add_argument("--prepare-only", action="store_true", help="Only build the motion cache; do not launch Isaac Sim.")
    return parser.parse_args()


def parse_prepare_icon_appearances_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a reusable ICON/THuman appearance pool for later Isaac Sim rendering."
    )
    parser.add_argument("--smplx-root", default=str(ROOT / "smplx_models/smplx"))
    parser.add_argument("--output-root", default=str(ROOT / "smplx_models/icon_appearances"))
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=20260609)
    parser.add_argument("--subjects", default="", help="Comma-separated fixed subject ids. Overrides random selection.")
    parser.add_argument("--thuman-scans-root", default="", help="Optional original THuman scans root with textured OBJ/MTL/images.")
    parser.add_argument("--icon-root", default="", help="Optional local ICON checkout. Needed only with --run-icon-render.")
    parser.add_argument("--icon-python", default="", help="Python executable for ICON render_batch. Defaults to this interpreter.")
    parser.add_argument("--run-icon-render", action="store_true", help="Run ICON scripts.render_batch before building the asset manifest.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--strict-textures",
        action="store_true",
        help="Fail if a selected subject has no textured scan/ICON mesh. Useful after original THuman scans are available.",
    )
    return parser.parse_args()


def prepare_icon_appearances_main() -> int:
    args = parse_prepare_icon_appearances_args()
    subjects = tuple(part.strip() for part in args.subjects.split(",") if part.strip())
    thuman_scans_root = Path(args.thuman_scans_root) if args.thuman_scans_root else None
    icon_root = Path(args.icon_root) if args.icon_root else None
    manifest = prepare_icon_appearance_library(
        smplx_root=Path(args.smplx_root),
        output_root=Path(args.output_root),
        count=args.count,
        seed=args.random_seed,
        subjects=subjects,
        thuman_scans_root=thuman_scans_root,
        icon_root=icon_root,
        icon_python=args.icon_python,
        run_icon_render=args.run_icon_render,
        overwrite=args.overwrite,
        strict_textures=args.strict_textures,
    )
    summary = {
        "output_root": manifest["output_root"],
        "manifest": str(Path(manifest["output_root"]) / "appearances_manifest.json"),
        "subjects": manifest["selected_subjects"],
        "status_counts": {},
    }
    for asset in manifest["assets"]:
        status = str(asset["render_status"])
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def isaacsim_main() -> int:
    args = parse_isaacsim_args()
    from geosim.blenderproc_cache import build_blenderproc_motion_cache_from_amass

    motion_path = Path(args.motion)
    output_dir = Path(args.output_dir) if args.output_dir else ROOT / "outputs/isaacsim" / motion_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    smplx_model = load_smplx_model(args.smplx_model)
    isaacsim_python = _resolve_isaacsim_python(args.isaacsim_python)
    runner = ROOT / "src/geosim/isaacsim_runner.py"
    gpu_ids = _resolve_gpu_ids(args.gpu_ids)
    appearance_subjects = _select_appearance_subjects(
        args.appearance_root,
        args.appearance_count,
        args.random_seed,
        args.appearance_subjects,
    )
    if len(appearance_subjects) > 1 or appearance_subjects[0] is not None:
        print(
            f"Selected appearances: {[subject.name if subject is not None else 'default' for subject in appearance_subjects]}",
            flush=True,
        )

    for local_appearance_idx, appearance_subject in enumerate(appearance_subjects):
        appearance_idx = int(args.appearance_index_offset) + int(local_appearance_idx)
        rng = np.random.default_rng(int(args.random_seed) + 7919 * appearance_idx)
        run_output_dir = output_dir
        if len(appearance_subjects) > 1 or appearance_subject is not None:
            subject_name = appearance_subject.name if appearance_subject is not None else "default"
            run_output_dir = output_dir / f"appearance_{appearance_idx:02d}_{subject_name}"
        run_output_dir.mkdir(parents=True, exist_ok=True)
        body_betas = _load_thuman_betas(appearance_subject) if appearance_subject is not None else None
        appearance_manifest = _load_appearance_asset_manifest(appearance_subject) if appearance_subject is not None else {}
        body_face_groups = _make_body_face_groups(smplx_model)
        body_group_colors = _make_body_group_colors(rng)
        scene_config = _make_random_scene_config(rng)

        cache_result = build_blenderproc_motion_cache_from_amass(
            motion_path=motion_path,
            model=smplx_model,
            config=config,
            tag_rig=make_wrist_tag_rig(config.tag_rig),
            output_dir=run_output_dir,
            output_fps=args.output_fps,
            max_output_frames=args.max_output_frames,
            video_width=args.width,
            video_height=args.height,
            head_frame_mode=args.head_frame,
            body_betas=body_betas,
            body_face_groups=body_face_groups,
            body_group_colors=body_group_colors,
            appearance_subject=appearance_subject.name if appearance_subject is not None else "",
            appearance_render_status=str(appearance_manifest.get("render_status", "")),
            scene_config=scene_config,
        )
        summary = {
            "backend": "isaacsim",
            "cache": str(cache_result.cache_path),
            "metadata": str(cache_result.metadata_path),
            "output_dir": str(run_output_dir),
            "frames": cache_result.frame_count,
            "fps": cache_result.output_fps,
            "cameras": list(cache_result.camera_names),
            "appearance_subject": appearance_subject.name if appearance_subject is not None else "",
            "appearance_render_status": appearance_manifest.get("render_status", ""),
            "appearance_asset_manifest": str(appearance_subject / "asset_manifest.json")
            if appearance_subject is not None and (appearance_subject / "asset_manifest.json").exists()
            else "",
            "body_group_colors": body_group_colors.tolist(),
            "scene": scene_config,
        }
        print(json.dumps(summary, indent=2), flush=True)
        if args.prepare_only:
            continue

        camera_names = tuple(name for name in cache_result.camera_names if not args.camera_name or name == args.camera_name)
        if args.camera_name and not camera_names:
            raise ValueError(f"Unknown camera {args.camera_name!r}. Available cameras: {', '.join(cache_result.camera_names)}")
        parallel_cameras = _resolve_parallel_camera_count(args.parallel_cameras, "GPU", gpu_ids, len(camera_names))
        print(
            f"Launching Isaac Sim camera jobs: parallel={parallel_cameras}, gpu_ids={gpu_ids or ['shared/default']}",
            flush=True,
        )
        _run_isaacsim_camera_jobs(
            isaacsim_python=isaacsim_python,
            runner=runner,
            cache_path=cache_result.cache_path,
            output_dir=run_output_dir,
            renderer=args.renderer,
            rt_subframes=args.rt_subframes,
            warmup_frames=args.warmup_frames,
            video_format=args.video_format,
            hide_wrist_tags=args.hide_wrist_tags,
            motion_label=motion_path.stem,
            camera_names=camera_names,
            parallel_cameras=parallel_cameras,
            gpu_ids=gpu_ids,
        )
    return 0


def _resolve_blenderproc_bin(explicit: str) -> str:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"blenderproc binary does not exist: {path}")
        return str(path)
    env_bin = Path(sys.executable).resolve().parent / "blenderproc"
    if env_bin.exists():
        return str(env_bin)
    found = shutil.which("blenderproc")
    if found:
        return found
    raise FileNotFoundError("Could not find blenderproc. Install it in camtest or pass --blenderproc-bin.")


def _select_appearance_subjects(root: str, count: int, seed: int, explicit_subjects: str = "") -> list[Path | None]:
    count = max(1, int(count))
    if not root:
        return [None] * count
    root_path = Path(root).expanduser()
    if not root_path.exists():
        raise FileNotFoundError(f"Appearance root does not exist: {root_path}")
    if explicit_subjects:
        selected = []
        for subject in [part.strip() for part in explicit_subjects.split(",") if part.strip()]:
            subject_path = root_path / subject
            if not subject_path.exists():
                raise FileNotFoundError(f"Appearance subject does not exist: {subject_path}")
            if not (subject_path / "mesh_smplx.obj").exists() or not (subject_path / "smplx_param.pkl").exists():
                raise FileNotFoundError(f"Appearance subject is missing mesh_smplx.obj or smplx_param.pkl: {subject_path}")
            selected.append(subject_path)
        return selected
    subjects = sorted(
        path
        for path in root_path.iterdir()
        if path.is_dir() and (path / "mesh_smplx.obj").exists() and (path / "smplx_param.pkl").exists()
    )
    if not subjects:
        raise FileNotFoundError(f"No appearance folders found under {root_path}")
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(len(subjects), size=min(count, len(subjects)), replace=False)
    return [subjects[int(idx)] for idx in chosen]


def _load_thuman_betas(subject_dir: Path) -> np.ndarray:
    with (subject_dir / "smplx_param.pkl").open("rb") as file:
        params = pickle.load(file)
    if "betas" not in params:
        raise KeyError(f"{subject_dir / 'smplx_param.pkl'} does not contain betas")
    return np.asarray(params["betas"], dtype=float).reshape(-1)


def _load_appearance_asset_manifest(subject_dir: Path | None) -> dict[str, object]:
    if subject_dir is None:
        return {}
    manifest_path = subject_dir / "asset_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _make_body_face_groups(smplx_model: SmplxNumpyModel) -> np.ndarray:
    vertices = np.asarray(smplx_model.v_template, dtype=float)
    faces = np.asarray(smplx_model.faces, dtype=np.int32)
    y = vertices[faces].mean(axis=1)[:, 1]
    groups = np.ones(len(faces), dtype=np.int32)
    groups[y > 0.12] = 0
    groups[(y <= 0.12) & (y > -0.35)] = 1
    groups[(y <= -0.35) & (y > -1.03)] = 2
    groups[y <= -1.03] = 3
    return groups


def _make_body_group_colors(rng: np.random.Generator) -> np.ndarray:
    palettes = (
        np.asarray([[0.72, 0.56, 0.44], [0.58, 0.42, 0.32], [0.80, 0.66, 0.54], [0.45, 0.31, 0.24]], dtype=np.float32),
        np.asarray([[0.10, 0.25, 0.36], [0.54, 0.12, 0.16], [0.18, 0.38, 0.22], [0.72, 0.68, 0.54], [0.17, 0.16, 0.18]], dtype=np.float32),
        np.asarray([[0.05, 0.08, 0.12], [0.12, 0.18, 0.28], [0.33, 0.30, 0.25], [0.20, 0.22, 0.20]], dtype=np.float32),
        np.asarray([[0.03, 0.03, 0.03], [0.82, 0.82, 0.78], [0.30, 0.28, 0.26]], dtype=np.float32),
    )
    return np.asarray([palette[int(rng.integers(0, len(palette)))] for palette in palettes], dtype=np.float32)


def _make_random_scene_config(rng: np.random.Generator) -> dict[str, object]:
    styles = ("tile", "carpet", "grass", "concrete")
    style = styles[int(rng.integers(0, len(styles)))]
    floor_palettes = {
        "tile": ([0.62, 0.60, 0.55], [0.22, 0.22, 0.20]),
        "carpet": ([0.34, 0.18, 0.22], [0.46, 0.28, 0.33]),
        "grass": ([0.18, 0.34, 0.16], [0.08, 0.24, 0.08]),
        "concrete": ([0.50, 0.49, 0.45], [0.34, 0.34, 0.32]),
    }
    floor_color, floor_accent = floor_palettes[style]
    return {
        "floor_style": style,
        "floor_color": floor_color,
        "floor_accent": floor_accent,
        "sun_rotation": [
            float(rng.uniform(25.0, 65.0)),
            float(rng.uniform(-10.0, 10.0)),
            float(rng.uniform(-55.0, 55.0)),
        ],
        "sun_intensity": float(rng.uniform(1800.0, 4200.0)),
    }


def _resolve_isaacsim_python(explicit: str) -> str:
    if explicit:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Isaac Sim python does not exist: {path}")
        return str(path)
    candidates = (
        Path.home() / "isaacsim" / "python.sh",
        Path.home() / "isaacsim" / "kit" / "python.sh",
        Path("/opt/isaacsim/python.sh"),
        Path("/opt/isaacsim/kit/python.sh"),
    )
    for path in candidates:
        if path.exists():
            return str(path)
    found = shutil.which("python.sh")
    if found:
        return found
    raise FileNotFoundError("Could not find Isaac Sim python.sh. Pass --isaacsim-python.")


def _resolve_gpu_ids(explicit: str) -> list[str]:
    if explicit:
        return [part.strip() for part in explicit.split(",") if part.strip()]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [part.strip() for part in visible.split(",") if part.strip()]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _resolve_parallel_camera_count(requested: int, device: str, gpu_ids: list[str], camera_count: int) -> int:
    if requested > 0:
        return max(1, min(int(requested), int(camera_count)))
    if device == "GPU" and gpu_ids:
        return max(1, min(len(gpu_ids), int(camera_count)))
    return 1


def _run_blenderproc_camera_jobs(
    *,
    blenderproc_bin: str,
    runner: Path,
    cache_path: Path,
    output_dir: Path,
    samples: int,
    engine: str,
    device: str,
    denoise: bool,
    video_format: str,
    video_crf: int,
    motion_label: str,
    camera_names: tuple[str, ...],
    parallel_cameras: int,
    gpu_ids: list[str],
) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for job_idx, camera_name in enumerate(camera_names):
        command = [
            blenderproc_bin,
            "run",
            str(runner),
            "--cache",
            str(cache_path),
            "--output-dir",
            str(output_dir),
            "--camera-name",
            camera_name,
            "--samples",
            str(samples),
            "--engine",
            engine,
            "--device",
            device,
            "--video-format",
            video_format,
            "--video-crf",
            str(video_crf),
            "--motion-label",
            motion_label,
        ]
        if denoise:
            command.append("--denoise")
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        gpu_id = ""
        if device == "GPU" and gpu_ids:
            gpu_id = gpu_ids[job_idx % len(gpu_ids)]
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
        jobs.append((camera_name, gpu_id, command, env, log_dir / f"{camera_name}.log"))

    if parallel_cameras <= 1:
        for camera_name, gpu_id, command, env, log_path in jobs:
            print(f"[BlenderProc] {camera_name} gpu={gpu_id or 'default'} log={log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as log_file:
                subprocess.run(command, check=True, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        return

    pending = list(jobs)
    running: list[tuple[str, str, subprocess.Popen, object, Path]] = []
    failures: list[tuple[str, int, Path]] = []
    try:
        while pending or running:
            while pending and len(running) < parallel_cameras:
                camera_name, gpu_id, command, env, log_path = pending.pop(0)
                log_file = log_path.open("w", encoding="utf-8")
                print(f"[BlenderProc] start {camera_name} gpu={gpu_id or 'default'} log={log_path}", flush=True)
                proc = subprocess.Popen(command, env=env, stdout=log_file, stderr=subprocess.STDOUT)
                running.append((camera_name, gpu_id, proc, log_file, log_path))

            still_running = []
            for camera_name, gpu_id, proc, log_file, log_path in running:
                return_code = proc.poll()
                if return_code is None:
                    still_running.append((camera_name, gpu_id, proc, log_file, log_path))
                    continue
                log_file.close()
                if return_code == 0:
                    print(f"[BlenderProc] done  {camera_name} gpu={gpu_id or 'default'}", flush=True)
                else:
                    print(f"[BlenderProc] fail  {camera_name} rc={return_code} log={log_path}", flush=True)
                    failures.append((camera_name, int(return_code), log_path))
            running = still_running
            if running:
                time.sleep(0.5)
    except KeyboardInterrupt:
        for _, _, proc, log_file, _ in running:
            proc.terminate()
            log_file.close()
        raise

    if failures:
        details = ", ".join(f"{name}:rc={code}:log={path}" for name, code, path in failures)
        raise RuntimeError(f"BlenderProc camera jobs failed: {details}")


def _run_isaacsim_camera_jobs(
    *,
    isaacsim_python: str,
    runner: Path,
    cache_path: Path,
    output_dir: Path,
    renderer: str,
    rt_subframes: int,
    warmup_frames: int,
    video_format: str,
    hide_wrist_tags: bool,
    motion_label: str,
    camera_names: tuple[str, ...],
    parallel_cameras: int,
    gpu_ids: list[str],
) -> None:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for job_idx, camera_name in enumerate(camera_names):
        command = [
            isaacsim_python,
            str(runner),
            "--cache",
            str(cache_path),
            "--output-dir",
            str(output_dir),
            "--camera-name",
            camera_name,
            "--renderer",
            renderer,
            "--rt-subframes",
            str(rt_subframes),
            "--warmup-frames",
            str(warmup_frames),
            "--video-format",
            video_format,
            "--motion-label",
            motion_label,
        ]
        if hide_wrist_tags:
            command.append("--hide-wrist-tags")
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        gpu_id = ""
        if gpu_ids:
            gpu_id = gpu_ids[job_idx % len(gpu_ids)]
            env["CUDA_VISIBLE_DEVICES"] = gpu_id
        jobs.append((camera_name, gpu_id, command, env, log_dir / f"{camera_name}.log"))

    _run_camera_jobs(jobs, parallel_cameras, backend_label="IsaacSim")
    _validate_rendered_camera_videos(
        output_dir=output_dir,
        motion_label=motion_label,
        camera_names=camera_names,
        video_format=video_format,
        expected_frames=_read_expected_frame_count(cache_path),
    )


def _read_expected_frame_count(cache_path: Path) -> int:
    data = np.load(cache_path, allow_pickle=True)
    return int(np.asarray(data["vertices"]).shape[0])


def _validate_rendered_camera_videos(
    *,
    output_dir: Path,
    motion_label: str,
    camera_names: tuple[str, ...],
    video_format: str,
    expected_frames: int,
) -> None:
    failures = []
    for camera_name in camera_names:
        video_path = output_dir / f"{motion_label}_{camera_name}.{video_format}"
        stats_path = output_dir / f"{motion_label}_{camera_name}_isaacsim_stats.json"
        if not stats_path.exists():
            failures.append(f"{camera_name}: missing stats {stats_path}")
            continue
        cap = cv2.VideoCapture(str(video_path))
        ok = cap.isOpened()
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if ok else -1
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if ok else -1
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if ok else -1
        cap.release()
        if not ok or frame_count < int(expected_frames):
            failures.append(
                f"{camera_name}: invalid video {video_path} "
                f"(opened={ok}, frames={frame_count}/{expected_frames}, size={width}x{height})"
            )
    if failures:
        raise RuntimeError("IsaacSim output validation failed:\n" + "\n".join(failures))


def _run_camera_jobs(
    jobs: list[tuple[str, str, list[str], dict[str, str], Path]],
    parallel_cameras: int,
    *,
    backend_label: str,
) -> None:
    if parallel_cameras <= 1:
        for camera_name, gpu_id, command, env, log_path in jobs:
            print(f"[{backend_label}] {camera_name} gpu={gpu_id or 'default'} log={log_path}", flush=True)
            with log_path.open("w", encoding="utf-8") as log_file:
                subprocess.run(command, check=True, env=env, stdout=log_file, stderr=subprocess.STDOUT)
        return

    pending = list(jobs)
    running: list[tuple[str, str, subprocess.Popen, object, Path]] = []
    failures: list[tuple[str, int, Path]] = []
    try:
        while pending or running:
            while pending and len(running) < parallel_cameras:
                camera_name, gpu_id, command, env, log_path = pending.pop(0)
                log_file = log_path.open("w", encoding="utf-8")
                print(f"[{backend_label}] start {camera_name} gpu={gpu_id or 'default'} log={log_path}", flush=True)
                proc = subprocess.Popen(command, env=env, stdout=log_file, stderr=subprocess.STDOUT)
                running.append((camera_name, gpu_id, proc, log_file, log_path))

            still_running = []
            for camera_name, gpu_id, proc, log_file, log_path in running:
                return_code = proc.poll()
                if return_code is None:
                    still_running.append((camera_name, gpu_id, proc, log_file, log_path))
                    continue
                log_file.close()
                if return_code == 0:
                    print(f"[{backend_label}] done  {camera_name} gpu={gpu_id or 'default'}", flush=True)
                else:
                    print(f"[{backend_label}] fail  {camera_name} rc={return_code} log={log_path}", flush=True)
                    failures.append((camera_name, int(return_code), log_path))
            running = still_running
            if running:
                time.sleep(0.5)
    except KeyboardInterrupt:
        for _, _, proc, log_file, _ in running:
            proc.terminate()
            log_file.close()
        raise

    if failures:
        details = ", ".join(f"{name}:rc={code}:log={path}" for name, code, path in failures)
        raise RuntimeError(f"{backend_label} camera jobs failed: {details}")


COMMANDS = {
    'blenderproc': blenderproc_main,
    'head-tags': head_tags_main,
    'isaacsim': isaacsim_main,
    'prepare-icon-appearances': prepare_icon_appearances_main,
    'wrist-views': wrist_views_main,
}


def main() -> int:
    parser = argparse.ArgumentParser(description='Render camera simulations.')
    parser.add_argument("command", choices=sorted(COMMANDS))
    args, rest = parser.parse_known_args()
    sys.argv = [f"{Path(sys.argv[0]).name} {args.command}", *rest]
    return int(COMMANDS[args.command]() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
