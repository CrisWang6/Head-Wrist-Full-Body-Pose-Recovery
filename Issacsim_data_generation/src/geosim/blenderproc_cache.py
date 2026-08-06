from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path

import cv2
import numpy as np

from geosim.camera import FisheyeCamera, WristFisheyeCamera, make_default_camera_rig, make_default_wrist_camera_rig
from geosim.config import SimulationConfig
from geosim.linalg import frame_from_forearm, frame_from_shoulders
from geosim.motion import MotionSequence
from geosim.smplx_numpy import SmplxNumpyModel
from geosim.tag_rig import WristTagRig


MARKER_IDS = {"tag0": 10, "tag1": 11}
MARKER_PIXELS = 512
MARKER_MARGIN_PIXELS = 80


@dataclass(frozen=True)
class BlenderProcCacheResult:
    cache_path: Path
    metadata_path: Path
    marker_paths: dict[str, Path]
    frame_count: int
    output_fps: float
    camera_names: tuple[str, ...]


def build_blenderproc_motion_cache(
    *,
    motion: MotionSequence,
    model: SmplxNumpyModel,
    config: SimulationConfig,
    tag_rig: WristTagRig,
    output_dir: Path,
    output_fps: float,
    max_output_frames: int,
    width: int,
    height: int,
) -> BlenderProcCacheResult:
    if motion.source_path is None:
        raise ValueError("BlenderProc rendering expects an AMASS-backed motion.")
    if motion.right_wrist_pos is None or motion.right_wrist_rot is None:
        raise ValueError("BlenderProc wrist-camera rendering needs right wrist pose arrays.")

    source = np.load(motion.source_path, allow_pickle=True)
    if not {"poses", "trans"}.issubset(source.files):
        raise ValueError(f"{motion.source_path} does not contain AMASS poses/trans.")
    poses = np.asarray(source["poses"], dtype=float)
    trans = np.asarray(source["trans"], dtype=float)
    betas = np.asarray(source["betas"], dtype=float) if "betas" in source.files else motion.betas

    stride = max(1, int(round(float(motion.fps) / float(output_fps))))
    actual_fps = float(motion.fps) / stride
    frame_indices = np.arange(0, motion.frames, stride, dtype=np.int32)
    if max_output_frames > 0:
        frame_indices = frame_indices[: int(max_output_frames)]
    if len(frame_indices) == 0:
        frame_indices = np.array([0], dtype=np.int32)

    output_dir.mkdir(parents=True, exist_ok=True)
    marker_paths = _write_marker_images(output_dir)

    vertices = np.empty((len(frame_indices), len(model.v_template), 3), dtype=np.float32)
    for out_idx, frame_idx in enumerate(frame_indices):
        frame = model.forward_frame(poses[int(frame_idx)], trans[int(frame_idx)], betas, include_vertices=True)
        assert frame.vertices is not None
        vertices[out_idx] = frame.vertices.astype(np.float32)

    camera_names, camera_positions, camera_rotations = _camera_pose_arrays(
        motion=motion,
        config=config,
        frame_indices=frame_indices,
        width=width,
        height=height,
    )
    tag_names, tag_corners = _tag_corner_arrays(motion=motion, tag_rig=tag_rig, frame_indices=frame_indices)
    cache_path = output_dir / "blenderproc_motion_cache.npz"
    np.savez_compressed(
        cache_path,
        vertices=vertices,
        faces=model.faces.astype(np.int32),
        frame_indices=frame_indices,
        output_fps=np.array([actual_fps], dtype=np.float32),
        image_size=np.array([int(width), int(height)], dtype=np.int32),
        fisheye_fov_deg=np.array([float(config.camera_rig.fisheye_fov_deg)], dtype=np.float32),
        camera_names=np.asarray(camera_names),
        camera_positions=camera_positions.astype(np.float32),
        camera_rotations=camera_rotations.astype(np.float32),
        tag_names=np.asarray(tag_names),
        tag_corners=tag_corners.astype(np.float32),
    )

    metadata = {
        "motion": str(motion.source_path),
        "source_fps": float(motion.fps),
        "output_fps": actual_fps,
        "frame_count": int(len(frame_indices)),
        "frame_indices": [int(idx) for idx in frame_indices],
        "resolution": [int(width), int(height)],
        "fisheye_fov_deg": float(config.camera_rig.fisheye_fov_deg),
        "camera_names": list(camera_names),
        "tag_names": list(tag_names),
        "cache_path": str(cache_path),
        "marker_paths": {name: str(path) for name, path in marker_paths.items()},
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return BlenderProcCacheResult(
        cache_path=cache_path,
        metadata_path=metadata_path,
        marker_paths=marker_paths,
        frame_count=int(len(frame_indices)),
        output_fps=actual_fps,
        camera_names=tuple(camera_names),
    )


def build_blenderproc_motion_cache_from_amass(
    *,
    motion_path: Path,
    model: SmplxNumpyModel,
    config: SimulationConfig,
    tag_rig: WristTagRig,
    output_dir: Path,
    output_fps: float,
    max_output_frames: int,
    video_width: int,
    video_height: int,
    head_frame_mode: str = "smplx_relative",
    body_betas: np.ndarray | None = None,
    body_face_groups: np.ndarray | None = None,
    body_group_colors: np.ndarray | None = None,
    appearance_subject: str = "",
    appearance_render_status: str = "",
    scene_config: dict[str, object] | None = None,
) -> BlenderProcCacheResult:
    source = np.load(motion_path, allow_pickle=True)
    if not {"poses", "trans"}.issubset(source.files):
        raise ValueError(f"{motion_path} does not contain AMASS poses/trans.")
    poses = np.asarray(source["poses"], dtype=float)
    trans = np.asarray(source["trans"], dtype=float)
    betas = np.asarray(body_betas, dtype=float).reshape(-1) if body_betas is not None else (
        np.asarray(source["betas"], dtype=float) if "betas" in source.files else None
    )
    source_fps = float(np.asarray(source["mocap_frame_rate"]).reshape(-1)[0]) if "mocap_frame_rate" in source.files else 30.0

    stride = max(1, int(round(source_fps / float(output_fps))))
    actual_fps = source_fps / stride
    frame_indices = np.arange(0, poses.shape[0], stride, dtype=np.int32)
    if max_output_frames > 0:
        frame_indices = frame_indices[: int(max_output_frames)]
    if len(frame_indices) == 0:
        frame_indices = np.array([0], dtype=np.int32)

    output_dir.mkdir(parents=True, exist_ok=True)
    marker_paths = _write_marker_images(output_dir)

    joint_names = model.joint2num
    head_idx = joint_names["Head"]
    left_shoulder_idx = joint_names["L_Shoulder"]
    right_shoulder_idx = joint_names["R_Shoulder"]
    left_elbow_idx = joint_names["L_Elbow"]
    left_wrist_idx = joint_names["L_Wrist"]
    right_elbow_idx = joint_names["R_Elbow"]
    right_wrist_idx = joint_names["R_Wrist"]

    frame_count = len(frame_indices)
    vertices = np.empty((frame_count, len(model.v_template), 3), dtype=np.float32)
    head_pos = np.empty((frame_count, 3), dtype=float)
    head_rot = np.empty((frame_count, 3, 3), dtype=float)
    raw_head_rot = np.empty((frame_count, 3, 3), dtype=float)
    shoulder_head_rot = np.empty((frame_count, 3, 3), dtype=float)
    left_elbow_pos = np.empty((frame_count, 3), dtype=float)
    left_wrist_pos = np.empty((frame_count, 3), dtype=float)
    left_wrist_rot = np.empty((frame_count, 3, 3), dtype=float)
    right_elbow_pos = np.empty((frame_count, 3), dtype=float)
    right_wrist_pos = np.empty((frame_count, 3), dtype=float)
    right_wrist_rot = np.empty((frame_count, 3, 3), dtype=float)

    for out_idx, frame_idx in enumerate(frame_indices):
        frame = model.forward_frame(poses[int(frame_idx)], trans[int(frame_idx)], betas, include_vertices=True)
        assert frame.vertices is not None
        vertices[out_idx] = frame.vertices.astype(np.float32)
        joints = frame.joints
        head_pos[out_idx] = joints[head_idx]
        raw_head_rot[out_idx] = frame.joint_rotations[head_idx]
        shoulder_head_rot[out_idx] = frame_from_shoulders(joints[left_shoulder_idx], joints[right_shoulder_idx])
        left_elbow_pos[out_idx] = joints[left_elbow_idx]
        left_wrist_pos[out_idx] = joints[left_wrist_idx]
        right_elbow_pos[out_idx] = joints[right_elbow_idx]
        right_wrist_pos[out_idx] = joints[right_wrist_idx]
        left_wrist_rot[out_idx] = frame_from_forearm(left_elbow_pos[out_idx], left_wrist_pos[out_idx])
        right_wrist_rot[out_idx] = frame_from_forearm(right_elbow_pos[out_idx], right_wrist_pos[out_idx])

    if head_frame_mode == "smplx_relative":
        head_rot = raw_head_rot @ raw_head_rot[0].T @ shoulder_head_rot[0]
    elif head_frame_mode == "smplx":
        head_rot = raw_head_rot
    elif head_frame_mode == "shoulders":
        head_rot = shoulder_head_rot
    else:
        raise ValueError(f"Unsupported head_frame_mode: {head_frame_mode}")

    sensor_width, sensor_height = _square_sensor_size(video_width, video_height)
    camera_names, camera_positions, camera_rotations = _camera_pose_arrays_from_sampled(
        config=config,
        head_pos=head_pos,
        head_rot=head_rot,
        left_wrist_pos=left_wrist_pos,
        left_wrist_rot=left_wrist_rot,
        right_wrist_pos=right_wrist_pos,
        right_wrist_rot=right_wrist_rot,
        width=sensor_width,
        height=sensor_height,
    )
    tag_names, tag_corners = _tag_corner_arrays_from_sampled(
        tag_rig=tag_rig,
        left_wrist_pos=left_wrist_pos,
        left_wrist_rot=left_wrist_rot,
        right_wrist_pos=right_wrist_pos,
        right_wrist_rot=right_wrist_rot,
    )

    cache_path = output_dir / "blenderproc_motion_cache.npz"
    np.savez_compressed(
        cache_path,
        vertices=vertices,
        faces=model.faces.astype(np.int32),
        frame_indices=frame_indices,
        output_fps=np.array([actual_fps], dtype=np.float32),
        image_size=np.array([sensor_width, sensor_height], dtype=np.int32),
        video_size=np.array([int(video_width), int(video_height)], dtype=np.int32),
        fisheye_fov_deg=np.array([float(config.camera_rig.fisheye_fov_deg)], dtype=np.float32),
        camera_names=np.asarray(camera_names),
        camera_positions=camera_positions.astype(np.float32),
        camera_rotations=camera_rotations.astype(np.float32),
        tag_names=np.asarray(tag_names),
        tag_corners=tag_corners.astype(np.float32),
        body_face_groups=np.asarray(body_face_groups, dtype=np.int32)
        if body_face_groups is not None
        else np.zeros(len(model.faces), dtype=np.int32),
        body_group_colors=np.asarray(body_group_colors, dtype=np.float32)
        if body_group_colors is not None
        else np.asarray([[0.72, 0.58, 0.49]], dtype=np.float32),
        appearance_subject=np.asarray([appearance_subject]),
        appearance_render_status=np.asarray([appearance_render_status]),
        scene_floor_style=np.asarray([str((scene_config or {}).get("floor_style", "concrete"))]),
        scene_floor_color=np.asarray((scene_config or {}).get("floor_color", [0.50, 0.49, 0.45]), dtype=np.float32),
        scene_floor_accent=np.asarray((scene_config or {}).get("floor_accent", [0.22, 0.22, 0.22]), dtype=np.float32),
        scene_sun_rotation=np.asarray((scene_config or {}).get("sun_rotation", [42.0, 0.0, 28.0]), dtype=np.float32),
        scene_sun_intensity=np.array([float((scene_config or {}).get("sun_intensity", 2600.0))], dtype=np.float32),
    )

    metadata = {
        "motion": str(motion_path),
        "source_fps": float(source_fps),
        "output_fps": float(actual_fps),
        "source_frames": int(poses.shape[0]),
        "frame_count": int(frame_count),
        "frame_stride": int(stride),
        "frame_indices": [int(idx) for idx in frame_indices],
        "video_resolution": [int(video_width), int(video_height)],
        "sensor_resolution": [int(sensor_width), int(sensor_height)],
        "head_frame_mode": head_frame_mode,
        "fisheye_fov_deg": float(config.camera_rig.fisheye_fov_deg),
        "camera_names": list(camera_names),
        "tag_names": list(tag_names),
        "cache_path": str(cache_path),
        "marker_paths": {name: str(path) for name, path in marker_paths.items()},
        "appearance_subject": appearance_subject,
        "appearance_render_status": appearance_render_status,
        "scene": scene_config or {},
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return BlenderProcCacheResult(
        cache_path=cache_path,
        metadata_path=metadata_path,
        marker_paths=marker_paths,
        frame_count=int(frame_count),
        output_fps=float(actual_fps),
        camera_names=tuple(camera_names),
    )


def _write_marker_images(output_dir: Path) -> dict[str, Path]:
    marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    paths: dict[str, Path] = {}
    for tag_name, marker_id in MARKER_IDS.items():
        marker = cv2.aruco.generateImageMarker(marker_dict, marker_id, MARKER_PIXELS)
        image = np.full(
            (MARKER_PIXELS + 2 * MARKER_MARGIN_PIXELS, MARKER_PIXELS + 2 * MARKER_MARGIN_PIXELS),
            255,
            dtype=np.uint8,
        )
        image[
            MARKER_MARGIN_PIXELS : MARKER_MARGIN_PIXELS + MARKER_PIXELS,
            MARKER_MARGIN_PIXELS : MARKER_MARGIN_PIXELS + MARKER_PIXELS,
        ] = marker
        path = output_dir / f"{tag_name}_aruco.png"
        cv2.imwrite(str(path), image)
        paths[tag_name] = path
    return paths


def _camera_pose_arrays(
    *,
    motion: MotionSequence,
    config: SimulationConfig,
    frame_indices: np.ndarray,
    width: int,
    height: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    head_labels = {
        "CAM_A": "head_front_left",
        "CAM_B": "head_front_right",
        "CAM_C": "head_back_left",
        "CAM_D": "head_back_right",
    }
    head_cameras = [_resize_head_camera(camera, width, height) for camera in make_default_camera_rig(config.camera_rig)]
    wrist_cameras = [_resize_wrist_camera(camera, width, height) for camera in make_default_wrist_camera_rig(config.camera_rig)]

    entries: list[tuple[str, np.ndarray, np.ndarray]] = []
    for camera in head_cameras:
        label = head_labels.get(camera.name, camera.name.lower())
        positions = []
        rotations = []
        for frame_idx in frame_indices:
            pos, rot = camera.world_pose(motion.head_pos[int(frame_idx)], motion.head_rot[int(frame_idx)])
            positions.append(pos)
            rotations.append(rot)
        entries.append((label, np.stack(positions), np.stack(rotations)))

    wrist_specs = (
        ("left", motion.left_wrist_pos, motion.left_wrist_rot),
        ("right", motion.right_wrist_pos, motion.right_wrist_rot),
    )
    for side, wrist_pos, wrist_rot in wrist_specs:
        assert wrist_pos is not None and wrist_rot is not None
        for camera in wrist_cameras:
            label = f"{side}_{camera.name.lower()}"
            positions = []
            rotations = []
            for frame_idx in frame_indices:
                pos, rot = camera.world_pose(wrist_pos[int(frame_idx)], wrist_rot[int(frame_idx)])
                positions.append(pos)
                rotations.append(rot)
            entries.append((label, np.stack(positions), np.stack(rotations)))

    names = [entry[0] for entry in entries]
    positions = np.stack([entry[1] for entry in entries], axis=0)
    rotations = np.stack([entry[2] for entry in entries], axis=0)
    return names, positions, rotations


def _camera_pose_arrays_from_sampled(
    *,
    config: SimulationConfig,
    head_pos: np.ndarray,
    head_rot: np.ndarray,
    left_wrist_pos: np.ndarray,
    left_wrist_rot: np.ndarray,
    right_wrist_pos: np.ndarray,
    right_wrist_rot: np.ndarray,
    width: int,
    height: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    head_labels = {
        "CAM_A": "head_front_left",
        "CAM_B": "head_front_right",
        "CAM_C": "head_back_left",
        "CAM_D": "head_back_right",
    }
    head_cameras = [_resize_head_camera(camera, width, height) for camera in make_default_camera_rig(config.camera_rig)]
    wrist_cameras = [_resize_wrist_camera(camera, width, height) for camera in make_default_wrist_camera_rig(config.camera_rig)]
    entries: list[tuple[str, np.ndarray, np.ndarray]] = []
    for camera in head_cameras:
        label = head_labels.get(camera.name, camera.name.lower())
        positions = []
        rotations = []
        for frame_idx in range(len(head_pos)):
            pos, rot = camera.world_pose(head_pos[frame_idx], head_rot[frame_idx])
            positions.append(pos)
            rotations.append(rot)
        entries.append((label, np.stack(positions), np.stack(rotations)))

    wrist_specs = (
        ("left", left_wrist_pos, left_wrist_rot),
        ("right", right_wrist_pos, right_wrist_rot),
    )
    for side, wrist_pos, wrist_rot in wrist_specs:
        for camera in wrist_cameras:
            label = f"{side}_{camera.name.lower()}"
            positions = []
            rotations = []
            for frame_idx in range(len(wrist_pos)):
                pos, rot = camera.world_pose(wrist_pos[frame_idx], wrist_rot[frame_idx])
                positions.append(pos)
                rotations.append(rot)
            entries.append((label, np.stack(positions), np.stack(rotations)))
    names = [entry[0] for entry in entries]
    positions = np.stack([entry[1] for entry in entries], axis=0)
    rotations = np.stack([entry[2] for entry in entries], axis=0)
    return names, positions, rotations


def _tag_corner_arrays(
    *,
    motion: MotionSequence,
    tag_rig: WristTagRig,
    frame_indices: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    tag_names = ["left_tag0", "left_tag1", "right_tag0", "right_tag1"]
    corners = np.zeros((len(tag_names), len(frame_indices), 4, 3), dtype=np.float32)
    wrist_specs = (
        ("left", motion.left_wrist_pos, motion.left_wrist_rot),
        ("right", motion.right_wrist_pos, motion.right_wrist_rot),
    )
    for side_idx, (side, wrist_pos, wrist_rot) in enumerate(wrist_specs):
        assert wrist_pos is not None and wrist_rot is not None
        for out_idx, frame_idx in enumerate(frame_indices):
            points = tag_rig.world_points(wrist_pos[int(frame_idx)], wrist_rot[int(frame_idx)])
            for tag_idx, tag_name in enumerate(("tag0", "tag1")):
                global_idx = side_idx * 2 + tag_idx
                corners[global_idx, out_idx] = np.stack(
                    [points[f"{tag_name}_c{corner_idx}"] for corner_idx in range(4)],
                    axis=0,
                )
    return tag_names, corners


def _tag_corner_arrays_from_sampled(
    *,
    tag_rig: WristTagRig,
    left_wrist_pos: np.ndarray,
    left_wrist_rot: np.ndarray,
    right_wrist_pos: np.ndarray,
    right_wrist_rot: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    tag_names = ["left_tag0", "left_tag1", "right_tag0", "right_tag1"]
    corners = np.zeros((len(tag_names), len(left_wrist_pos), 4, 3), dtype=np.float32)
    wrist_specs = (
        ("left", left_wrist_pos, left_wrist_rot),
        ("right", right_wrist_pos, right_wrist_rot),
    )
    for side_idx, (_, wrist_pos, wrist_rot) in enumerate(wrist_specs):
        for frame_idx in range(len(wrist_pos)):
            points = tag_rig.world_points(wrist_pos[frame_idx], wrist_rot[frame_idx])
            for tag_idx, tag_name in enumerate(("tag0", "tag1")):
                global_idx = side_idx * 2 + tag_idx
                corners[global_idx, frame_idx] = np.stack(
                    [points[f"{tag_name}_c{corner_idx}"] for corner_idx in range(4)],
                    axis=0,
                )
    return tag_names, corners


def _square_sensor_size(video_width: int, video_height: int) -> tuple[int, int]:
    sensor_side = int(video_width)
    if sensor_side <= 0 or int(video_height) <= 0:
        raise ValueError("Video width and height must be positive.")
    return sensor_side, sensor_side


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


def geosim_camera_to_blender_matrix(position: np.ndarray, rotation_cam_to_world: np.ndarray) -> np.ndarray:
    """Convert geosim/OpenCV-style +Z-forward camera pose to Blender/OpenGL."""
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.asarray(rotation_cam_to_world, dtype=float) @ np.diag([1.0, -1.0, -1.0])
    transform[:3, 3] = np.asarray(position, dtype=float)
    return transform
