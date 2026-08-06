from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geosim.linalg import normalize
from geosim.pose_tracks import PoseEstimate, matrix_to_quat, quat_to_matrix, slerp


@dataclass(frozen=True)
class ImuNoiseConfig:
    gyro_noise_std_rad_s: float = 0.015
    accel_noise_std_m_s2: float = 0.12
    gyro_bias_std_rad_s: float = 0.004
    accel_bias_std_m_s2: float = 0.04


@dataclass(frozen=True)
class SimulatedImu:
    fps: float
    gyro_rad_s: np.ndarray
    accel_m_s2: np.ndarray
    gyro_bias_rad_s: np.ndarray
    accel_bias_m_s2: np.ndarray


def simulate_wrist_imu(
    wrist_pos: np.ndarray,
    wrist_rot: np.ndarray,
    fps: float,
    noise: ImuNoiseConfig = ImuNoiseConfig(),
    seed: int = 0,
) -> SimulatedImu:
    """Generate local-frame 6-axis IMU samples from a wrist pose sequence."""
    rng = np.random.default_rng(seed)
    positions = np.asarray(wrist_pos, dtype=float)
    rotations = np.asarray(wrist_rot, dtype=float)
    dt = 1.0 / float(fps)
    frames = len(positions)
    gravity = np.array([0.0, 0.0, -9.80665], dtype=float)

    gyro = np.zeros((frames, 3), dtype=float)
    for idx in range(frames - 1):
        delta_local = rotations[idx].T @ rotations[idx + 1]
        gyro[idx] = rotation_log(delta_local) / dt
    if frames > 1:
        gyro[-1] = gyro[-2]

    velocity = np.gradient(positions, dt, axis=0, edge_order=1)
    acceleration_world = np.gradient(velocity, dt, axis=0, edge_order=1)
    accel = np.einsum("fji,fj->fi", rotations, acceleration_world - gravity)

    gyro_bias = rng.normal(0.0, noise.gyro_bias_std_rad_s, size=3)
    accel_bias = rng.normal(0.0, noise.accel_bias_std_m_s2, size=3)
    gyro = gyro + gyro_bias + rng.normal(0.0, noise.gyro_noise_std_rad_s, size=gyro.shape)
    accel = accel + accel_bias + rng.normal(0.0, noise.accel_noise_std_m_s2, size=accel.shape)
    return SimulatedImu(float(fps), gyro, accel, gyro_bias, accel_bias)


def fuse_wrist_visual_imu(
    visual_estimates: list[PoseEstimate | None],
    imu: SimulatedImu,
    initial_position: np.ndarray,
    initial_rotation: np.ndarray,
    gravity_world: np.ndarray | None = None,
    visual_position_weight: float = 0.95,
    visual_rotation_weight: float = 0.68,
) -> list[PoseEstimate | None]:
    """Complementary pose fusion using visual corrections and IMU propagation."""
    if not visual_estimates:
        return []
    gravity = np.array([0.0, 0.0, -9.80665], dtype=float) if gravity_world is None else np.asarray(gravity_world, dtype=float)
    dt = 1.0 / imu.fps
    pos = np.asarray(initial_position, dtype=float).copy()
    rot = np.asarray(initial_rotation, dtype=float).copy()
    velocity = np.zeros(3, dtype=float)
    fused: list[PoseEstimate | None] = []

    for frame_idx, visual in enumerate(visual_estimates):
        if frame_idx > 0:
            rot = rot @ rotation_exp(imu.gyro_rad_s[frame_idx - 1] * dt)
            acc_world = rot @ imu.accel_m_s2[frame_idx - 1] + gravity
            acc_world = np.clip(acc_world, -35.0, 35.0)
            pos = pos + velocity * dt + 0.15 * acc_world * dt * dt
            velocity = 0.92 * velocity + 0.15 * acc_world * dt

        source = "imu"
        if visual is not None:
            pred_pos = pos.copy()
            pos = (1.0 - visual_position_weight) * pos + visual_position_weight * visual.position_world
            velocity = 0.25 * velocity + 0.75 * (pos - pred_pos) / dt
            rot = blend_rotations(rot, visual.rotation_world, visual_rotation_weight)
            source = f"imu+visual:{visual.source}"
        fused.append(PoseEstimate(pos.copy(), project_rotation(rot), source))
    return fused


def rotation_exp(vector: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3)
    axis = np.asarray(vector, dtype=float) / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def rotation_log(matrix: np.ndarray) -> np.ndarray:
    m = project_rotation(np.asarray(matrix, dtype=float))
    cos_angle = float(np.clip((np.trace(m) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cos_angle))
    if angle < 1e-12:
        return np.zeros(3, dtype=float)
    axis = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]], dtype=float)
    axis = normalize(axis)
    return axis * angle


def blend_rotations(a: np.ndarray, b: np.ndarray, weight_b: float) -> np.ndarray:
    quat = slerp(matrix_to_quat(a), matrix_to_quat(b), float(np.clip(weight_b, 0.0, 1.0)))
    return quat_to_matrix(quat)


def project_rotation(matrix: np.ndarray) -> np.ndarray:
    u, _, vt = np.linalg.svd(matrix)
    rot = u @ vt
    if np.linalg.det(rot) < 0.0:
        u[:, -1] *= -1.0
        rot = u @ vt
    return rot
