from __future__ import annotations

import math

import numpy as np


def normalize(value: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(value)
    if norm < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return value / norm


def rotx(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def roty(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotz(angle_rad: float) -> np.ndarray:
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotation_error_deg(reference: np.ndarray, estimate: np.ndarray) -> float:
    delta = reference.T @ estimate
    cos_angle = (np.trace(delta) - 1.0) * 0.5
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def rigid_align(source_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R, t such that target ~= R @ source + t."""
    if source_points.shape != target_points.shape:
        raise ValueError("source_points and target_points must have the same shape.")
    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("Expected point arrays with shape (N, 3).")
    if len(source_points) < 3:
        raise ValueError("At least three points are required for rigid alignment.")

    source_centroid = source_points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)
    src = source_points - source_centroid
    dst = target_points - target_centroid
    cov = src.T @ dst
    u, _, vt = np.linalg.svd(cov)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0.0:
        vt[-1, :] *= -1.0
        r = vt.T @ u.T
    t = target_centroid - r @ source_centroid
    return r, t


def frame_from_forearm(elbow: np.ndarray, wrist: np.ndarray, up_hint: np.ndarray | None = None) -> np.ndarray:
    """Build an approximate wrist frame from the elbow-to-wrist direction.

    The wrist-frame +x axis follows the forearm from elbow to wrist. The +z axis
    is chosen close to up_hint while remaining orthogonal to +x.
    """
    x_axis = normalize(wrist - elbow)
    hint = np.array([0.0, 0.0, 1.0]) if up_hint is None else normalize(up_hint)
    y_axis = np.cross(hint, x_axis)
    if np.linalg.norm(y_axis) < 1e-8:
        hint = np.array([0.0, 1.0, 0.0])
        y_axis = np.cross(hint, x_axis)
    y_axis = normalize(y_axis)
    z_axis = normalize(np.cross(x_axis, y_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def frame_from_shoulders(
    left_shoulder: np.ndarray,
    right_shoulder: np.ndarray,
    up_hint: np.ndarray | None = None,
) -> np.ndarray:
    """Build a head-mounted rig frame with a stable vertical axis.

    AMASS/SMPL-X joint rotations are skeletal rotations, not device frames. This
    frame keeps head-mounted cameras level: +z is world up, +x points to the
    subject's right side from the shoulder line, and +y completes the frame.
    """
    up = np.array([0.0, 0.0, 1.0]) if up_hint is None else normalize(up_hint)
    right_axis = np.asarray(right_shoulder, dtype=float) - np.asarray(left_shoulder, dtype=float)
    right_axis = right_axis - up * float(np.dot(right_axis, up))
    if np.linalg.norm(right_axis) < 1e-8:
        right_axis = np.array([1.0, 0.0, 0.0])
    x_axis = normalize(right_axis)
    y_axis = normalize(np.cross(up, x_axis))
    z_axis = normalize(np.cross(x_axis, y_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def ensure_rotation_stack(value: np.ndarray, frames: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == (3, 3):
        arr = np.repeat(arr[None, :, :], frames, axis=0)
    if arr.shape != (frames, 3, 3):
        raise ValueError(f"{name} must have shape ({frames}, 3, 3).")
    return arr


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    vectors = np.asarray(axis_angle, dtype=float)
    original_shape = vectors.shape[:-1]
    flat = vectors.reshape(-1, 3)
    angles = np.linalg.norm(flat, axis=1)
    matrices = np.repeat(np.eye(3)[None, :, :], len(flat), axis=0)

    valid = angles > 1e-12
    if np.any(valid):
        axes = flat[valid] / angles[valid, None]
        x = axes[:, 0]
        y = axes[:, 1]
        z = axes[:, 2]
        c = np.cos(angles[valid])
        s = np.sin(angles[valid])
        one_c = 1.0 - c
        matrices[valid] = np.stack(
            [
                c + x * x * one_c,
                x * y * one_c - z * s,
                x * z * one_c + y * s,
                y * x * one_c + z * s,
                c + y * y * one_c,
                y * z * one_c - x * s,
                z * x * one_c - y * s,
                z * y * one_c + x * s,
                c + z * z * one_c,
            ],
            axis=1,
        ).reshape(-1, 3, 3)
    return matrices.reshape(*original_shape, 3, 3)
