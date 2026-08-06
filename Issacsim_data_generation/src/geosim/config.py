from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraRigConfig:
    image_width: int = 1920
    image_height: int = 1080
    fps: float = 30.0
    fisheye_fov_deg: float = 220.0
    rectangle_length_m: float = 0.44
    rectangle_width_m: float = 0.20
    camera_names: tuple[str, ...] = ("CAM_A", "CAM_B", "CAM_C", "CAM_D")


@dataclass(frozen=True)
class TagRigConfig:
    tag_size_m: float = 0.08
    dihedral_angle_deg: float = 45.0
    wrist_to_hinge_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class NoiseConfig:
    pixel_noise_std: float = 0.0


@dataclass(frozen=True)
class SimulationConfig:
    camera_rig: CameraRigConfig = CameraRigConfig()
    tag_rig: TagRigConfig = TagRigConfig()
    noise: NoiseConfig = NoiseConfig()
    min_views_per_corner: int = 2
    max_frames: int = 0


def load_config(path: str | Path = "configs/default_geometry.json") -> SimulationConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    camera_data = data.get("camera_rig", {})
    tag_data = data.get("tag_rig", {})
    sim_data = data.get("simulation", {})
    return SimulationConfig(
        camera_rig=CameraRigConfig(
            image_width=int(camera_data.get("image_width", 1920)),
            image_height=int(camera_data.get("image_height", 1080)),
            fps=float(camera_data.get("fps", 30.0)),
            fisheye_fov_deg=float(camera_data.get("fisheye_fov_deg", 220.0)),
            rectangle_length_m=float(camera_data.get("rectangle_length_m", 0.44)),
            rectangle_width_m=float(camera_data.get("rectangle_width_m", 0.20)),
            camera_names=tuple(camera_data.get("camera_names", ["CAM_A", "CAM_B", "CAM_C", "CAM_D"])),
        ),
        tag_rig=TagRigConfig(
            tag_size_m=float(tag_data.get("tag_size_m", 0.08)),
            dihedral_angle_deg=float(tag_data.get("dihedral_angle_deg", 45.0)),
            wrist_to_hinge_offset_m=tuple(np.asarray(tag_data.get("wrist_to_hinge_offset_m", [0.0, 0.0, 0.0]), dtype=float)),
        ),
        noise=NoiseConfig(pixel_noise_std=float(sim_data.get("pixel_noise_std", 0.0))),
        min_views_per_corner=int(sim_data.get("min_views_per_corner", 2)),
        max_frames=int(sim_data.get("max_frames", 0)),
    )


def override_config(
    config: SimulationConfig,
    tag_size_m: float | None = None,
    pixel_noise_std: float | None = None,
    max_frames: int | None = None,
) -> SimulationConfig:
    tag_rig = config.tag_rig
    noise = config.noise
    if tag_size_m is not None:
        tag_rig = TagRigConfig(
            tag_size_m=tag_size_m,
            dihedral_angle_deg=tag_rig.dihedral_angle_deg,
            wrist_to_hinge_offset_m=tag_rig.wrist_to_hinge_offset_m,
        )
    if pixel_noise_std is not None:
        noise = NoiseConfig(pixel_noise_std=pixel_noise_std)
    return SimulationConfig(
        camera_rig=config.camera_rig,
        tag_rig=tag_rig,
        noise=noise,
        min_views_per_corner=config.min_views_per_corner,
        max_frames=config.max_frames if max_frames is None else max_frames,
    )
