#!/usr/bin/env python3
"""Shared geometry for mocap-anchored omni-camera triangulation."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
from scipy.optimize import least_squares


EPS = 1e-12


def load_json(path: Path | str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def homogeneous(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return homogeneous(rotation.T, -rotation.T @ translation)


def quaternion_wxyz_to_rotation(quaternion: Sequence[float]) -> np.ndarray:
    """Return the active rotation mapping rigid-frame vectors into world."""
    w, x, y, z = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm < EPS:
        raise ValueError("Cannot convert a zero quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def rigid_world_transform(row: Mapping[str, str | float], prefix: str) -> np.ndarray:
    translation = np.array([float(row[f"{prefix}_{axis}"]) for axis in "xyz"])
    quaternion = [float(row[f"{prefix}_q{axis}"]) for axis in "wxyz"]
    return homogeneous(quaternion_wxyz_to_rotation(quaternion), translation)


def camera_a_mount_transform(rigid_extrinsics: Mapping) -> np.ndarray:
    """Return T_rigid_cameraA in metres from the mechanical truth file."""
    left = rigid_extrinsics["cameras"]["left"]
    rotation = np.asarray(left["R_rigid_camera"], dtype=np.float64)
    translation = np.asarray(left["p_rigid_camera_mm"], dtype=np.float64) / 1000.0
    return homogeneous(rotation, translation)


def stereo_transform_d_a(calibration: Mapping) -> np.ndarray:
    """Return Kalibr T_CAM_D_CAM_A, mapping CAM_A coordinates into CAM_D."""
    stereo = calibration.get("stereo_extrinsics", {})
    matrix = stereo.get("T_CAM_D_CAM_A")
    if matrix is None:
        raise KeyError("Calibration has no stereo_extrinsics.T_CAM_D_CAM_A")
    return np.asarray(matrix, dtype=np.float64).reshape(4, 4)


def module_camera_world_transforms(
    rigid_world: np.ndarray,
    rigid_camera_a: np.ndarray,
    camera_d_camera_a: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build T_world_CAM_A and T_world_CAM_D.

    The mechanical file is authoritative for CAM_A. Kalibr stores
    p_D = T_D_A p_A, therefore T_world_D = T_world_A inverse(T_D_A).
    """
    world_a = np.asarray(rigid_world, dtype=np.float64) @ np.asarray(
        rigid_camera_a, dtype=np.float64
    )
    world_d = world_a @ invert_transform(camera_d_camera_a)
    return world_a, world_d


@dataclass(frozen=True)
class OmniCamera:
    name: str
    xi: float
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: np.ndarray
    width: int
    height: int

    @classmethod
    def from_calibration(
        cls, calibration: Mapping, camera_name: str, *, name: str | None = None
    ) -> "OmniCamera":
        camera = calibration["cameras"][camera_name]
        xi, fx, fy, cx, cy = map(float, camera["intrinsics"])
        width, height = map(int, camera.get("resolution", calibration["resolution"]))
        return cls(
            name=name or camera_name,
            xi=xi,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion=np.asarray(camera["distortion_coeffs"], dtype=np.float64),
            width=width,
            height=height,
        )

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def ray(self, uv: Sequence[float]) -> np.ndarray:
        point = np.asarray(uv, dtype=np.float64).reshape(1, 1, 2)
        xy = cv2.undistortPoints(
            point, self.intrinsic_matrix, self.distortion
        ).reshape(2)
        radius_sq = float(xy @ xy)
        root = max(0.0, 1.0 + (1.0 - self.xi * self.xi) * radius_sq)
        scale = (self.xi + math.sqrt(root)) / (1.0 + radius_sq)
        ray = np.array([scale * xy[0], scale * xy[1], scale - self.xi])
        return ray / max(float(np.linalg.norm(ray)), EPS)

    def project(self, point_camera: Sequence[float]) -> np.ndarray | None:
        x, y, z = map(float, point_camera)
        radius = math.sqrt(x * x + y * y + z * z)
        denominator = z + self.xi * radius
        if radius < EPS or denominator <= EPS:
            return None
        xn, yn = x / denominator, y / denominator
        k1, k2, p1, p2 = map(float, self.distortion)
        r2 = xn * xn + yn * yn
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
        yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
        return np.array([self.fx * xd + self.cx, self.fy * yd + self.cy])


@dataclass(frozen=True)
class CameraPose:
    camera: OmniCamera
    world_camera: np.ndarray

    @property
    def center_world(self) -> np.ndarray:
        return np.asarray(self.world_camera, dtype=np.float64)[:3, 3]

    @property
    def rotation_world_camera(self) -> np.ndarray:
        return np.asarray(self.world_camera, dtype=np.float64)[:3, :3]

    def world_ray(self, uv: Sequence[float]) -> np.ndarray:
        direction = self.rotation_world_camera @ self.camera.ray(uv)
        return direction / max(float(np.linalg.norm(direction)), EPS)

    def point_camera(self, point_world: Sequence[float]) -> np.ndarray:
        point = np.asarray(point_world, dtype=np.float64)
        return self.rotation_world_camera.T @ (point - self.center_world)

    def project_world(self, point_world: Sequence[float]) -> np.ndarray | None:
        return self.camera.project(self.point_camera(point_world))


@dataclass(frozen=True)
class Observation:
    camera_name: str
    uv: np.ndarray
    confidence: float
    pose: CameraPose


@dataclass(frozen=True)
class RayIntersection:
    point_world: np.ndarray
    condition_number: float
    depths: np.ndarray
    misses_m: np.ndarray


@dataclass(frozen=True)
class TriangulationResult:
    point_world: np.ndarray
    used_cameras: tuple[str, ...]
    rejected_cameras: tuple[str, ...]
    reprojection_errors_px: dict[str, float]
    ray_misses_m: dict[str, float]
    condition_number: float
    maximum_ray_angle_deg: float


def weighted_ray_intersection(
    origins: Sequence[np.ndarray],
    directions: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
) -> RayIntersection | None:
    if len(origins) < 2 or len(origins) != len(directions):
        return None
    if weights is None:
        weights = np.ones(len(origins), dtype=np.float64)
    matrix = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)
    normalized_directions: list[np.ndarray] = []
    for origin, direction, weight in zip(origins, directions, weights):
        direction = np.asarray(direction, dtype=np.float64)
        direction /= max(float(np.linalg.norm(direction)), EPS)
        normalized_directions.append(direction)
        projector = identity - np.outer(direction, direction)
        matrix += max(float(weight), EPS) * projector
        rhs += max(float(weight), EPS) * projector @ np.asarray(origin, dtype=np.float64)
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular[-1] < 1e-10:
        return None
    point = np.linalg.solve(matrix, rhs)
    depths = np.array(
        [float(direction @ (point - origin)) for origin, direction in zip(origins, normalized_directions)]
    )
    misses = np.array(
        [
            float(np.linalg.norm(np.cross(point - origin, direction)))
            for origin, direction in zip(origins, normalized_directions)
        ]
    )
    return RayIntersection(
        point_world=point,
        condition_number=float(singular[0] / singular[-1]),
        depths=depths,
        misses_m=misses,
    )


def maximum_ray_angle_deg(directions: Sequence[np.ndarray]) -> float:
    maximum = 0.0
    for first, second in itertools.combinations(directions, 2):
        cosine = float(
            np.clip(
                np.dot(first, second)
                / max(float(np.linalg.norm(first) * np.linalg.norm(second)), EPS),
                -1.0,
                1.0,
            )
        )
        maximum = max(maximum, math.degrees(math.acos(cosine)))
    return maximum


def _reprojection_error(observation: Observation, point_world: np.ndarray) -> float:
    projected = observation.pose.project_world(point_world)
    if projected is None or not np.all(np.isfinite(projected)):
        return math.inf
    return float(np.linalg.norm(projected - observation.uv))


def _refine_reprojection(
    initial: np.ndarray,
    observations: Sequence[Observation],
    loss_scale_px: float,
) -> np.ndarray:
    def residual(point: np.ndarray) -> np.ndarray:
        values: list[float] = []
        for observation in observations:
            projected = observation.pose.project_world(point)
            if projected is None or not np.all(np.isfinite(projected)):
                delta = np.array([1000.0, 1000.0])
            else:
                delta = projected - observation.uv
            values.extend((math.sqrt(max(observation.confidence, 0.01)) * delta).tolist())
        return np.asarray(values)

    answer = least_squares(
        residual,
        np.asarray(initial, dtype=np.float64),
        method="trf",
        loss="huber",
        f_scale=float(loss_scale_px),
        max_nfev=80,
    )
    return answer.x if answer.success and np.all(np.isfinite(answer.x)) else initial


def triangulate_observations(
    observations: Iterable[Observation],
    *,
    minimum_confidence: float = 0.05,
    minimum_ray_angle_deg: float = 0.5,
    maximum_reprojection_error_px: float = 35.0,
    robust_loss_scale_px: float = 8.0,
) -> TriangulationResult | None:
    usable = [
        observation
        for observation in observations
        if observation.confidence >= minimum_confidence
        and np.all(np.isfinite(observation.uv))
    ]
    if len(usable) < 2:
        return None

    origins = [item.pose.center_world for item in usable]
    directions = [item.pose.world_ray(item.uv) for item in usable]
    if maximum_ray_angle_deg(directions) < minimum_ray_angle_deg:
        return None

    hypotheses: list[np.ndarray] = []
    for first, second in itertools.combinations(range(len(usable)), 2):
        if maximum_ray_angle_deg([directions[first], directions[second]]) < minimum_ray_angle_deg:
            continue
        answer = weighted_ray_intersection(
            [origins[first], origins[second]],
            [directions[first], directions[second]],
            [usable[first].confidence, usable[second].confidence],
        )
        if answer is not None and np.all(answer.depths > 0.0):
            hypotheses.append(answer.point_world)
    all_answer = weighted_ray_intersection(
        origins, directions, [item.confidence for item in usable]
    )
    if all_answer is not None:
        hypotheses.append(all_answer.point_world)
    if not hypotheses:
        return None

    best_indices: list[int] = []
    best_score: tuple[int, float, float] | None = None
    for point in hypotheses:
        indices = []
        errors = []
        for index, (item, origin, direction) in enumerate(zip(usable, origins, directions)):
            forward = float(direction @ (point - origin)) > 0.0
            error = _reprojection_error(item, point)
            if forward and error <= maximum_reprojection_error_px:
                indices.append(index)
                errors.append(error)
        if len(indices) < 2:
            continue
        score = (
            len(indices),
            float(sum(usable[index].confidence for index in indices)),
            -float(np.median(errors)),
        )
        if best_score is None or score > best_score:
            best_indices, best_score = indices, score
    if len(best_indices) < 2:
        return None

    chosen = [usable[index] for index in best_indices]
    chosen_origins = [origins[index] for index in best_indices]
    chosen_directions = [directions[index] for index in best_indices]
    if maximum_ray_angle_deg(chosen_directions) < minimum_ray_angle_deg:
        return None
    linear = weighted_ray_intersection(
        chosen_origins,
        chosen_directions,
        [item.confidence for item in chosen],
    )
    if linear is None or np.any(linear.depths <= 0.0):
        return None
    refined = _refine_reprojection(linear.point_world, chosen, robust_loss_scale_px)

    final_indices = [
        index
        for index, (item, origin, direction) in enumerate(zip(usable, origins, directions))
        if float(direction @ (refined - origin)) > 0.0
        and _reprojection_error(item, refined) <= maximum_reprojection_error_px
    ]
    if len(final_indices) < 2:
        return None
    final = [usable[index] for index in final_indices]
    final_origins = [origins[index] for index in final_indices]
    final_directions = [directions[index] for index in final_indices]
    if maximum_ray_angle_deg(final_directions) < minimum_ray_angle_deg:
        return None
    final_linear = weighted_ray_intersection(
        final_origins,
        final_directions,
        [item.confidence for item in final],
    )
    if final_linear is None or np.any(final_linear.depths <= 0.0):
        return None
    point = _refine_reprojection(
        final_linear.point_world, final, robust_loss_scale_px
    )
    errors = {item.camera_name: _reprojection_error(item, point) for item in final}
    misses = {
        item.camera_name: float(
            np.linalg.norm(np.cross(point - origin, direction))
        )
        for item, origin, direction in zip(final, final_origins, final_directions)
    }
    used_names = tuple(item.camera_name for item in final)
    rejected_names = tuple(
        item.camera_name for item in usable if item.camera_name not in used_names
    )
    return TriangulationResult(
        point_world=point,
        used_cameras=used_names,
        rejected_cameras=rejected_names,
        reprojection_errors_px=errors,
        ray_misses_m=misses,
        condition_number=final_linear.condition_number,
        maximum_ray_angle_deg=maximum_ray_angle_deg(final_directions),
    )
