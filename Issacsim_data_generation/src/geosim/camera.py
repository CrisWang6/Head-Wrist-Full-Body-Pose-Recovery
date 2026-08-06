from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from geosim.config import CameraRigConfig
from geosim.linalg import normalize


# ---------------------------------------------------------------------------
# Shared fisheye projection model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FisheyeCamera:
    name: str
    position_head: np.ndarray
    rotation_cam_to_head: np.ndarray
    image_width: int
    image_height: int
    fov_deg: float

    @property
    def focal_px(self) -> float:
        half_fov = math.radians(self.fov_deg) * 0.5
        return min(self.image_width, self.image_height) * 0.5 / half_fov

    @property
    def principal_point(self) -> np.ndarray:
        return np.array([self.image_width * 0.5, self.image_height * 0.5], dtype=float)

    @property
    def max_theta_rad(self) -> float:
        return math.radians(self.fov_deg) * 0.5

    def world_pose(self, head_pos: np.ndarray, head_rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cam_pos_world = head_pos + head_rot @ self.position_head
        cam_rot_world = head_rot @ self.rotation_cam_to_head
        return cam_pos_world, cam_rot_world

    def project_world(
        self,
        point_world: np.ndarray,
        head_pos: np.ndarray,
        head_rot: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        cam_pos_world, cam_rot_world = self.world_pose(head_pos, head_rot)
        point_cam = cam_rot_world.T @ (point_world - cam_pos_world)
        pixel, in_fov = self.project_camera(point_cam)
        return pixel, in_fov and self.pixel_in_image(pixel)

    def project_camera(self, point_cam: np.ndarray) -> tuple[np.ndarray, bool]:
        direction = normalize(point_cam)
        theta = math.acos(float(np.clip(direction[2], -1.0, 1.0)))
        if theta > self.max_theta_rad:
            return np.array([np.nan, np.nan]), False
        sin_theta = math.sin(theta)
        if sin_theta < 1e-12:
            xy = np.array([0.0, 0.0])
        else:
            xy = direction[:2] / sin_theta
        pixel = self.principal_point + self.focal_px * theta * xy
        return pixel, True

    def ray_world(
        self,
        pixel: np.ndarray,
        head_pos: np.ndarray,
        head_rot: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cam_pos_world, cam_rot_world = self.world_pose(head_pos, head_rot)
        delta = np.asarray(pixel, dtype=float) - self.principal_point
        radius = float(np.linalg.norm(delta))
        theta = radius / self.focal_px
        if theta > self.max_theta_rad + 1e-9:
            raise ValueError("Pixel lies outside the configured fisheye FOV.")
        if radius < 1e-12:
            direction_cam = np.array([0.0, 0.0, 1.0])
        else:
            xy = delta / radius
            direction_cam = np.array([math.sin(theta) * xy[0], math.sin(theta) * xy[1], math.cos(theta)])
        return cam_pos_world, normalize(cam_rot_world @ direction_cam)

    def pixel_in_image(self, pixel: np.ndarray) -> bool:
        return (
            np.isfinite(pixel).all()
            and 0.0 <= pixel[0] < self.image_width
            and 0.0 <= pixel[1] < self.image_height
        )


# ---------------------------------------------------------------------------
# Head-mounted camera rig
# ---------------------------------------------------------------------------


def make_default_camera_rig(config: CameraRigConfig) -> list[FisheyeCamera]:
    half_width = config.rectangle_width_m * 0.5
    half_length = config.rectangle_length_m * 0.5
    positions = {
        "CAM_A": np.array([-half_width, half_length, 0.0], dtype=float),
        "CAM_B": np.array([half_width, half_length, 0.0], dtype=float),
        "CAM_C": np.array([-half_width, -half_length, 0.0], dtype=float),
        "CAM_D": np.array([half_width, -half_length, 0.0], dtype=float),
    }
    rotation_cam_to_head = np.diag([1.0, -1.0, -1.0])
    cameras = []
    for name in config.camera_names:
        if name not in positions:
            raise ValueError(f"Unsupported default camera name: {name}")
        cameras.append(
            FisheyeCamera(
                name=name,
                position_head=positions[name],
                rotation_cam_to_head=rotation_cam_to_head,
                image_width=config.image_width,
                image_height=config.image_height,
                fov_deg=config.fisheye_fov_deg,
            )
        )
    return cameras


# ---------------------------------------------------------------------------
# Wrist-mounted virtual camera rig
#
# The wrist frame used by geosim.motion has:
#   +x: elbow -> wrist, along the forearm
#   +y/+z: the plane perpendicular to the forearm
#
# The wrist tags also live in the local yz plane. The two virtual wrist-camera
# optical axes are therefore chosen in that same yz plane:
#   WRIST_PALM_NORMAL: local -y, approximately perpendicular to the palm /
#                      inner forearm side under the current wrist-frame model.
#   WRIST_FORWARD:     local -z, perpendicular to the first axis and pointing
#                      forward when the arm hangs naturally at the trouser seam.
#
# Each camera center is placed on its optical-axis side of the wrist, so the
# infinite optical-axis line passes through the wrist joint while the +z optical
# ray points away from the wrist.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WristFisheyeCamera:
    name: str
    position_wrist: np.ndarray
    rotation_cam_to_wrist: np.ndarray
    image_width: int
    image_height: int
    fov_deg: float

    @property
    def focal_px(self) -> float:
        half_fov = math.radians(self.fov_deg) * 0.5
        return min(self.image_width, self.image_height) * 0.5 / half_fov

    @property
    def principal_point(self) -> np.ndarray:
        return np.array([self.image_width * 0.5, self.image_height * 0.5], dtype=float)

    @property
    def max_theta_rad(self) -> float:
        return math.radians(self.fov_deg) * 0.5

    def world_pose(self, wrist_pos: np.ndarray, wrist_rot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cam_pos_world = wrist_pos + wrist_rot @ self.position_wrist
        cam_rot_world = wrist_rot @ self.rotation_cam_to_wrist
        return cam_pos_world, cam_rot_world

    def project_world(
        self,
        point_world: np.ndarray,
        wrist_pos: np.ndarray,
        wrist_rot: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        cam_pos_world, cam_rot_world = self.world_pose(wrist_pos, wrist_rot)
        point_cam = cam_rot_world.T @ (point_world - cam_pos_world)
        pixel, in_fov = self.project_camera(point_cam)
        return pixel, in_fov and self.pixel_in_image(pixel)

    def project_camera(self, point_cam: np.ndarray) -> tuple[np.ndarray, bool]:
        direction = normalize(point_cam)
        theta = math.acos(float(np.clip(direction[2], -1.0, 1.0)))
        if theta > self.max_theta_rad:
            return np.array([np.nan, np.nan]), False
        sin_theta = math.sin(theta)
        if sin_theta < 1e-12:
            xy = np.array([0.0, 0.0])
        else:
            xy = direction[:2] / sin_theta
        pixel = self.principal_point + self.focal_px * theta * xy
        return pixel, True

    def ray_world(
        self,
        pixel: np.ndarray,
        wrist_pos: np.ndarray,
        wrist_rot: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        cam_pos_world, cam_rot_world = self.world_pose(wrist_pos, wrist_rot)
        delta = np.asarray(pixel, dtype=float) - self.principal_point
        radius = float(np.linalg.norm(delta))
        theta = radius / self.focal_px
        if theta > self.max_theta_rad + 1e-9:
            raise ValueError("Pixel lies outside the configured fisheye FOV.")
        if radius < 1e-12:
            direction_cam = np.array([0.0, 0.0, 1.0])
        else:
            xy = delta / radius
            direction_cam = np.array([math.sin(theta) * xy[0], math.sin(theta) * xy[1], math.cos(theta)])
        return cam_pos_world, normalize(cam_rot_world @ direction_cam)

    def pixel_in_image(self, pixel: np.ndarray) -> bool:
        return (
            np.isfinite(pixel).all()
            and 0.0 <= pixel[0] < self.image_width
            and 0.0 <= pixel[1] < self.image_height
        )


def make_default_wrist_camera_rig(
    config: CameraRigConfig,
    camera_distance_m: float = 0.04,
) -> list[WristFisheyeCamera]:
    optical_axes = {
        "WRIST_PALM_NORMAL": np.array([0.0, -1.0, 0.0], dtype=float),
        "WRIST_FORWARD": np.array([0.0, 0.0, -1.0], dtype=float),
    }
    x_hints = {
        "WRIST_PALM_NORMAL": np.array([0.0, 0.0, 1.0], dtype=float),
        "WRIST_FORWARD": np.array([0.0, 1.0, 0.0], dtype=float),
    }

    cameras = []
    for name, optical_axis in optical_axes.items():
        axis = normalize(optical_axis)
        cameras.append(
            WristFisheyeCamera(
                name=name,
                position_wrist=float(camera_distance_m) * axis,
                rotation_cam_to_wrist=_camera_rotation_from_optical_axis(axis, x_hints[name]),
                image_width=config.image_width,
                image_height=config.image_height,
                fov_deg=config.fisheye_fov_deg,
            )
        )
    return cameras


def _camera_rotation_from_optical_axis(optical_axis: np.ndarray, x_hint: np.ndarray) -> np.ndarray:
    z_axis = normalize(optical_axis)
    x_axis = np.asarray(x_hint, dtype=float)
    x_axis = x_axis - z_axis * float(np.dot(x_axis, z_axis))
    if np.linalg.norm(x_axis) < 1e-8:
        fallback = np.array([1.0, 0.0, 0.0], dtype=float)
        x_axis = fallback - z_axis * float(np.dot(fallback, z_axis))
    x_axis = normalize(x_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.column_stack([x_axis, y_axis, z_axis])
