#!/usr/bin/env python3
"""Minimal Mahony AHRS + quaternion helpers for offline IMU orientation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def q_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return q / n


def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def q_inv(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[..., 1:] *= -1.0
    return q_normalize(out)


def q_to_matrix(q: np.ndarray) -> np.ndarray:
    q = q_normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def matrix_to_q(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)
    batch = m.shape[:-2]
    flat = m.reshape(-1, 3, 3)
    out = np.zeros((flat.shape[0], 4), dtype=np.float64)
    for i, mat in enumerate(flat):
        tr = float(np.trace(mat))
        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            out[i] = (
                0.25 * s,
                (mat[2, 1] - mat[1, 2]) / s,
                (mat[0, 2] - mat[2, 0]) / s,
                (mat[1, 0] - mat[0, 1]) / s,
            )
        elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
            s = math.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
            out[i] = (
                (mat[2, 1] - mat[1, 2]) / s,
                0.25 * s,
                (mat[0, 1] + mat[1, 0]) / s,
                (mat[0, 2] + mat[2, 0]) / s,
            )
        elif mat[1, 1] > mat[2, 2]:
            s = math.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
            out[i] = (
                (mat[0, 2] - mat[2, 0]) / s,
                (mat[0, 1] + mat[1, 0]) / s,
                0.25 * s,
                (mat[1, 2] + mat[2, 1]) / s,
            )
        else:
            s = math.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
            out[i] = (
                (mat[1, 0] - mat[0, 1]) / s,
                (mat[0, 2] + mat[2, 0]) / s,
                (mat[1, 2] + mat[2, 1]) / s,
                0.25 * s,
            )
    return q_normalize(out.reshape(*batch, 4))


def rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    vq = np.zeros(q.shape[:-1] + (4,), dtype=np.float64)
    vq[..., 1:] = v
    return q_mul(q_mul(q, vq), q_inv(q))[..., 1:]


def angular_velocity_from_quat(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    n = len(q)
    out = np.zeros((n, 3), dtype=np.float64)
    if n < 3:
        return out
    prev_i = np.arange(n) - 1
    next_i = np.arange(n) + 1
    prev_i[0] = 0
    next_i[-1] = n - 1
    dt = t[next_i] - t[prev_i]
    valid = dt > 1e-12
    dq = q_mul(q[next_i], q_inv(q[prev_i]))
    dq = q_normalize(dq)
    neg = dq[:, 0] < 0
    dq[neg] *= -1
    w = np.clip(dq[:, 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(np.maximum(0.0, 1.0 - w * w))
    ok = valid & (s > 1e-9) & (angle > 1e-12)
    out[ok] = dq[ok, 1:] / s[ok, None] * (angle[ok] / dt[ok])[:, None]
    return out


def fit_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source - np.mean(source, axis=0)
    target = target - np.mean(target, axis=0)
    u, _s, vt = np.linalg.svd(source.T @ target)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1.0
        rot = u @ vt
    return rot


def average_rotation_matrices(mats: np.ndarray) -> np.ndarray:
    """Average rotation matrices via quaternion mean with sign unification."""
    qs = matrix_to_q(mats)
    ref = qs[0]
    for i in range(1, len(qs)):
        if float(np.dot(qs[i], ref)) < 0.0:
            qs[i] *= -1.0
    mean_q = q_normalize(np.mean(qs, axis=0))
    mat = q_to_matrix(mean_q)
    if mat.shape == (3, 3):
        return mat
    return mat[0]


@dataclass
class MahonyEstimator:
    kp: float = 1.2
    ki: float = 0.03
    calibration_samples: int = 150

    def __post_init__(self) -> None:
        self.quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.last_timestamp_ms: float | None = None
        self.gyro_bias = np.zeros(3, dtype=np.float64)
        self.integral_error = np.zeros(3, dtype=np.float64)
        self._calib_count = 0
        self._gyro_sum = np.zeros(3, dtype=np.float64)
        self._accel_sum = np.zeros(3, dtype=np.float64)

    @property
    def is_calibrating(self) -> bool:
        return self._calib_count < self.calibration_samples

    def update(self, accel: np.ndarray, gyro: np.ndarray, timestamp_ms: float) -> np.ndarray:
        accel = np.asarray(accel, dtype=np.float64)
        gyro = np.asarray(gyro, dtype=np.float64)
        if self.is_calibrating:
            self._gyro_sum += gyro
            self._accel_sum += accel
            self._calib_count += 1
            self.last_timestamp_ms = timestamp_ms
            if self._calib_count >= self.calibration_samples:
                self.gyro_bias = self._gyro_sum / max(1, self._calib_count)
                mean_accel = self._accel_sum / max(1, self._calib_count)
                self.quaternion = self._accel_initial_quaternion(mean_accel)
            return self.quaternion.copy()

        if self.last_timestamp_ms is None:
            self.quaternion = self._accel_initial_quaternion(accel)
            self.last_timestamp_ms = timestamp_ms
            return self.quaternion.copy()

        dt = max(0.0, min(0.1, (timestamp_ms - self.last_timestamp_ms) / 1000.0))
        self.last_timestamp_ms = timestamp_ms
        if dt <= 0.0:
            return self.quaternion.copy()

        g = gyro - self.gyro_bias
        norm = float(np.linalg.norm(accel))
        if norm > 1e-6:
            a = accel / norm
            vx, vy, vz = self._estimated_gravity()
            ex = a[1] * vz - a[2] * vy
            ey = a[2] * vx - a[0] * vz
            ez = a[0] * vy - a[1] * vx
            self.integral_error += self.ki * np.array([ex, ey, ez]) * dt
            g = g + self.kp * np.array([ex, ey, ez]) + self.integral_error

        q = self.quaternion
        wx, wy, wz = g
        qx, qy, qz, qw = q[1], q[2], q[3], q[0]
        half_dt = 0.5 * dt
        self.quaternion = q_normalize(
            np.array(
                [
                    qw + (-qx * wx - qy * wy - qz * wz) * half_dt,
                    qx + (qw * wx + qy * wz - qz * wy) * half_dt,
                    qy + (qw * wy - qx * wz + qz * wx) * half_dt,
                    qz + (qw * wz + qx * wy - qy * wx) * half_dt,
                ],
                dtype=np.float64,
            )
        )
        return self.quaternion.copy()

    def _estimated_gravity(self) -> tuple[float, float, float]:
        qx, qy, qz, qw = self.quaternion[1], self.quaternion[2], self.quaternion[3], self.quaternion[0]
        return (
            2.0 * (qx * qz - qw * qy),
            2.0 * (qw * qx + qy * qz),
            qw * qw - qx * qx - qy * qy + qz * qz,
        )

    @staticmethod
    def _accel_initial_quaternion(accel: np.ndarray) -> np.ndarray:
        ax, ay, az = accel
        roll = math.degrees(math.atan2(ay, az))
        pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))
        cr = math.cos(math.radians(roll * 0.5))
        sr = math.sin(math.radians(roll * 0.5))
        cp = math.cos(math.radians(pitch * 0.5))
        sp = math.sin(math.radians(pitch * 0.5))
        return q_normalize(np.array([cr * cp, sr * cp, sp * cr, -sr * sp], dtype=np.float64))
