from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import sys

import numpy as np


LABEL_SCHEMA_VERSION = "egorear_head16_wrist7_fisheye_v1"

HEAD_CAMERA_JOINTS_16 = (
    "Head",
    "Neck",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
    "LeftUpLeg",
    "RightUpLeg",
    "LeftLeg",
    "RightLeg",
    "LeftFoot",
    "RightFoot",
    "LeftToeBase",
    "RightToeBase",
)

WRIST_CAMERA_JOINTS_7 = (
    "L_Ankle",
    "R_Ankle",
    "L_Knee",
    "R_Knee",
    "L_Hip",
    "R_Hip",
    "Spine1",
)

HEAD_TO_SMPLX_JOINT = {
    "Head": "Head",
    "Neck": "Neck",
    "LeftArm": "L_Shoulder",
    "RightArm": "R_Shoulder",
    "LeftForeArm": "L_Elbow",
    "RightForeArm": "R_Elbow",
    "LeftHand": "L_Wrist",
    "RightHand": "R_Wrist",
    "LeftUpLeg": "L_Hip",
    "RightUpLeg": "R_Hip",
    "LeftLeg": "L_Knee",
    "RightLeg": "R_Knee",
    "LeftFoot": "L_Ankle",
    "RightFoot": "R_Ankle",
    "LeftToeBase": "L_Foot",
    "RightToeBase": "R_Foot",
}


@dataclass(frozen=True)
class LabelBuildResult:
    path: Path
    frames: int
    cameras: tuple[str, ...]
    valid_points: int


def add_simulation_to_path(simulation_root: Path) -> None:
    src = simulation_root.expanduser().resolve() / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def build_labels_for_render_dir(
    *,
    render_dir: Path,
    simulation_root: Path,
    smplx_model_path: Path,
    output_path: Path | None = None,
    heatmap_size: tuple[int, int] = (114, 64),
    sigma: float = 1.5,
    projection_model: str = "auto",
    fisheye_fov_deg: float = 220.0,
) -> LabelBuildResult:
    add_simulation_to_path(simulation_root)
    from geosim.smplx_numpy import load_smplx_model

    render_dir = render_dir.expanduser().resolve()
    cache_path = render_dir / "blenderproc_motion_cache.npz"
    metadata_path = render_dir / "metadata.json"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing render cache: {cache_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata: {metadata_path}")

    model = load_smplx_model(smplx_model_path)
    data = np.load(cache_path, allow_pickle=True)
    vertices = np.asarray(data["vertices"], dtype=np.float32)
    camera_names = tuple(str(name) for name in data["camera_names"])
    camera_positions = np.asarray(data["camera_positions"], dtype=np.float32)
    camera_rotations = np.asarray(data["camera_rotations"], dtype=np.float32)
    sensor_size = tuple(int(v) for v in np.asarray(data["image_size"]).reshape(-1))
    video_size = tuple(int(v) for v in np.asarray(data["video_size"]).reshape(-1)) if "video_size" in data.files else sensor_size
    cached_fov_deg = float(np.asarray(data["fisheye_fov_deg"]).reshape(-1)[0])
    fisheye_fov_deg = float(fisheye_fov_deg) if fisheye_fov_deg > 0 else cached_fov_deg
    frame_indices = np.asarray(data["frame_indices"], dtype=np.int32)
    projection_model = resolve_projection_model(render_dir, projection_model)

    joints_world = joints_from_vertices(vertices, model)
    head_source_joints = tuple(HEAD_TO_SMPLX_JOINT[name] for name in HEAD_CAMERA_JOINTS_16)
    head_indices = [model.joint2num[name] for name in head_source_joints]
    wrist_indices = [model.joint2num[name] for name in WRIST_CAMERA_JOINTS_7]
    head_keypoints, head_visible = project_to_video_pixels(
        points_world=joints_world[:, head_indices],
        camera_positions=camera_positions,
        camera_rotations=camera_rotations,
        sensor_size=sensor_size,
        video_size=video_size,
        fisheye_fov_deg=fisheye_fov_deg,
        projection_model=projection_model,
    )
    wrist_keypoints, wrist_visible = project_to_video_pixels(
        points_world=joints_world[:, wrist_indices],
        camera_positions=camera_positions,
        camera_rotations=camera_rotations,
        sensor_size=sensor_size,
        video_size=video_size,
        fisheye_fov_deg=fisheye_fov_deg,
        projection_model=projection_model,
    )
    head_camera_mask, wrist_camera_mask = camera_type_masks(camera_names)
    head_joint_mask = np.repeat(head_camera_mask[:, None], len(HEAD_CAMERA_JOINTS_16), axis=1)
    wrist_joint_mask = np.repeat(wrist_camera_mask[:, None], len(WRIST_CAMERA_JOINTS_7), axis=1)
    head_visible = head_visible & head_joint_mask[None, :, :]
    wrist_visible = wrist_visible & wrist_joint_mask[None, :, :]

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    motion_stem = Path(metadata["motion"]).stem
    video_paths = []
    for camera_name in camera_names:
        path = render_dir / f"{motion_stem}_{camera_name}.mp4"
        if not path.exists():
            matches = sorted(render_dir.glob(f"*_{camera_name}.mp4"))
            if not matches:
                raise FileNotFoundError(f"Missing video for camera {camera_name} under {render_dir}")
            path = matches[0]
        video_paths.append(path)

    if output_path is None:
        output_path = render_dir / f"heatmap_labels_{heatmap_size[0]}x{heatmap_size[1]}.npz"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray([LABEL_SCHEMA_VERSION]),
        keypoints=head_keypoints.astype(np.float32),
        visible=head_visible.astype(np.bool_),
        joint_mask=head_joint_mask.astype(np.bool_),
        head_keypoints=head_keypoints.astype(np.float32),
        head_visible=head_visible.astype(np.bool_),
        head_joint_mask=head_joint_mask.astype(np.bool_),
        wrist_keypoints=wrist_keypoints.astype(np.float32),
        wrist_visible=wrist_visible.astype(np.bool_),
        wrist_joint_mask=wrist_joint_mask.astype(np.bool_),
        camera_is_head=head_camera_mask.astype(np.bool_),
        camera_is_wrist=wrist_camera_mask.astype(np.bool_),
        camera_names=np.asarray(camera_names),
        joints=np.asarray(HEAD_CAMERA_JOINTS_16),
        head_camera_joints=np.asarray(HEAD_CAMERA_JOINTS_16),
        head_source_smplx_joints=np.asarray(head_source_joints),
        wrist_camera_joints=np.asarray(WRIST_CAMERA_JOINTS_7),
        video_paths=np.asarray([str(path) for path in video_paths]),
        frame_indices=frame_indices,
        video_size=np.asarray(video_size, dtype=np.int32),
        heatmap_size=np.asarray(heatmap_size, dtype=np.int32),
        sigma=np.asarray([float(sigma)], dtype=np.float32),
        projection_model=np.asarray([projection_model]),
        fisheye_fov_deg=np.asarray([float(fisheye_fov_deg)], dtype=np.float32),
        source_cache=np.asarray([str(cache_path)]),
        source_render_dir=np.asarray([str(render_dir)]),
    )
    return LabelBuildResult(output_path, int(vertices.shape[0]), camera_names, int(head_visible.sum() + wrist_visible.sum()))


def resolve_projection_model(render_dir: Path, projection_model: str = "auto") -> str:
    if projection_model != "auto":
        return str(projection_model)
    return "fisheye_equidistant"


def joints_from_vertices(vertices: np.ndarray, smplx_model) -> np.ndarray:
    regressor = np.asarray(smplx_model.j_regressor, dtype=np.float32)
    return np.einsum("jv,fvc->fjc", regressor, np.asarray(vertices, dtype=np.float32))


def project_to_video_pixels(
    *,
    points_world: np.ndarray,
    camera_positions: np.ndarray,
    camera_rotations: np.ndarray,
    sensor_size: tuple[int, int],
    video_size: tuple[int, int],
    fisheye_fov_deg: float,
    projection_model: str = "fisheye_equidistant",
) -> tuple[np.ndarray, np.ndarray]:
    frame_count, joint_count = points_world.shape[:2]
    camera_count = camera_positions.shape[0]
    keypoints = np.full((frame_count, camera_count, joint_count, 2), np.nan, dtype=np.float32)
    visible = np.zeros((frame_count, camera_count, joint_count), dtype=bool)

    sensor_width, sensor_height = sensor_size
    video_width, video_height = video_size
    crop = np.asarray([(sensor_width - video_width) * 0.5, (sensor_height - video_height) * 0.5], dtype=np.float32)
    principal = np.asarray([sensor_width * 0.5, sensor_height * 0.5], dtype=np.float32)
    max_theta = math.radians(float(fisheye_fov_deg)) * 0.5
    focal_px = min(sensor_width, sensor_height) * 0.5 / max_theta
    usd_focal_px = sensor_width * 5.0 / 20.955

    for camera_idx in range(camera_count):
        for frame_idx in range(frame_count):
            points_cam = (points_world[frame_idx] - camera_positions[camera_idx, frame_idx]) @ camera_rotations[
                camera_idx, frame_idx
            ]
            if projection_model == "perspective_usd":
                z = points_cam[:, 2]
                pixels_sensor = np.full((joint_count, 2), np.nan, dtype=np.float32)
                in_front = z > 1e-6
                pixels_sensor[in_front, 0] = principal[0] + usd_focal_px * points_cam[in_front, 0] / z[in_front]
                pixels_sensor[in_front, 1] = principal[1] + usd_focal_px * points_cam[in_front, 1] / z[in_front]
                projection_valid = in_front
            else:
                norms = np.linalg.norm(points_cam, axis=1)
                directions = points_cam / np.maximum(norms[:, None], 1e-8)
                theta = np.arccos(np.clip(directions[:, 2], -1.0, 1.0))
                sin_theta = np.sin(theta)
                xy = np.zeros((joint_count, 2), dtype=np.float32)
                valid_sin = sin_theta > 1e-8
                xy[valid_sin] = directions[valid_sin, :2] / sin_theta[valid_sin, None]
                pixels_sensor = principal + focal_px * theta[:, None] * xy
                projection_valid = (norms > 1e-8) & (theta <= max_theta)
            pixels_video = pixels_sensor - crop
            in_view = (
                projection_valid
                & np.isfinite(pixels_video).all(axis=1)
                & (pixels_video[:, 0] >= 0.0)
                & (pixels_video[:, 0] < video_width)
                & (pixels_video[:, 1] >= 0.0)
                & (pixels_video[:, 1] < video_height)
            )
            keypoints[frame_idx, camera_idx] = pixels_video
            visible[frame_idx, camera_idx] = in_view
    return keypoints, visible




# Per-joint Gaussian blob radius in source video pixels (3*sigma convention in generate_heatmaps).
DEFAULT_JOINT_HEATMAP_RADIUS_PX: dict[str, float] = {
    # head branch (EgoRear naming)
    "Head": 10.0,          # nose
    "Neck": 10.0,
    "LeftArm": 20.0,       # shoulder
    "RightArm": 20.0,
    "LeftForeArm": 30.0,   # elbow
    "RightForeArm": 30.0,
    "LeftHand": 50.0,      # wrist
    "RightHand": 50.0,
    "LeftUpLeg": 10.0,     # hip
    "RightUpLeg": 10.0,
    "LeftLeg": 20.0,       # knee
    "RightLeg": 20.0,
    "LeftFoot": 10.0,      # ankle
    "RightFoot": 10.0,
    "LeftToeBase": 10.0,   # toe tip
    "RightToeBase": 10.0,
    # wrist-camera branch
    "L_Hip": 10.0,
    "R_Hip": 10.0,
    "L_Knee": 20.0,
    "R_Knee": 20.0,
    "L_Ankle": 10.0,
    "R_Ankle": 10.0,
    "Spine1": 10.0,
}


def resolve_joint_radii_px(
    joint_names: tuple[str, ...] | list[str],
    radius_map: dict[str, float] | None = None,
    *,
    default_radius_px: float = 10.0,
) -> np.ndarray:
    mapping = DEFAULT_JOINT_HEATMAP_RADIUS_PX if radius_map is None else radius_map
    return np.asarray(
        [float(mapping.get(str(name), default_radius_px)) for name in joint_names],
        dtype=np.float32,
    )


def sigma_heatmap_from_radius_px(
    radius_px: float,
    *,
    stride_x: float,
    stride_y: float,
) -> float:
    """Convert video-space blob radius (≈3σ) to heatmap σ using average stride."""
    if radius_px <= 0:
        return 0.0
    stride = 0.5 * (float(stride_x) + float(stride_y))
    return float(radius_px) / (3.0 * max(stride, 1e-8))


def camera_type_masks(camera_names: tuple[str, ...] | list[str]) -> tuple[np.ndarray, np.ndarray]:
    wrist_mask = np.asarray(["wrist" in str(camera_name) for camera_name in camera_names], dtype=bool)
    head_mask = ~wrist_mask
    return head_mask, wrist_mask


def generate_heatmaps(
    keypoints: np.ndarray,
    visible: np.ndarray,
    *,
    video_size: tuple[int, int],
    heatmap_size: tuple[int, int],
    sigma: float | None = None,
    joint_radii_px: np.ndarray | float | None = None,
) -> np.ndarray:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    visible = np.asarray(visible, dtype=bool)
    prefix_shape = keypoints.shape[:-2]
    joint_count = keypoints.shape[-2]
    heatmap_width, heatmap_height = heatmap_size
    video_width, video_height = video_size
    heatmaps = np.zeros((*prefix_shape, joint_count, heatmap_height, heatmap_width), dtype=np.float32)

    stride_x = float(video_width) / float(heatmap_width)
    stride_y = float(video_height) / float(heatmap_height)

    if joint_radii_px is None:
        legacy_sigma = 1.5 if sigma is None else float(sigma)
        radii = np.full((joint_count,), legacy_sigma * 3.0, dtype=np.float32)
    elif np.ndim(joint_radii_px) == 0:
        radii = np.full((joint_count,), float(joint_radii_px), dtype=np.float32)
    else:
        radii = np.asarray(joint_radii_px, dtype=np.float32).reshape(-1)
        if radii.shape[0] != joint_count:
            raise ValueError(
                f"joint_radii_px length {radii.shape[0]} != joint_count {joint_count}"
            )

    flat_points = keypoints.reshape(-1, joint_count, 2)
    flat_visible = visible.reshape(-1, joint_count)
    flat_heatmaps = heatmaps.reshape(-1, joint_count, heatmap_height, heatmap_width)
    for sample_idx in range(flat_points.shape[0]):
        for joint_idx in range(joint_count):
            if not flat_visible[sample_idx, joint_idx]:
                continue
            radius_px = float(radii[joint_idx])
            if radius_px <= 0:
                continue
            sigma_h = sigma_heatmap_from_radius_px(
                radius_px, stride_x=stride_x, stride_y=stride_y
            )
            tmp_size = max(1, int(round(sigma_h * 3.0)))
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, None]
            x0 = y0 = size // 2
            gaussian = np.exp(
                -((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma_h ** 2)
            )
            mu_x = int(flat_points[sample_idx, joint_idx, 0] / stride_x + 0.5)
            mu_y = int(flat_points[sample_idx, joint_idx, 1] / stride_y + 0.5)
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= heatmap_width or ul[1] >= heatmap_height or br[0] < 0 or br[1] < 0:
                continue
            g_x = max(0, -ul[0]), min(br[0], heatmap_width) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], heatmap_height) - ul[1]
            img_x = max(0, ul[0]), min(br[0], heatmap_width)
            img_y = max(0, ul[1]), min(br[1], heatmap_height)
            flat_heatmaps[sample_idx, joint_idx, img_y[0] : img_y[1], img_x[0] : img_x[1]] = gaussian[
                g_y[0] : g_y[1], g_x[0] : g_x[1]
            ]
    return heatmaps
