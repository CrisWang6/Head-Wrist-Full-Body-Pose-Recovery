from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from geosim.linalg import normalize, rigid_align


@dataclass(frozen=True)
class PoseEstimate:
    position_world: np.ndarray
    rotation_world: np.ndarray
    source: str


def marker_object_points(marker_size_m: float) -> np.ndarray:
    half = marker_size_m * 0.5
    return np.array(
        [[-half, -half, 0.0], [half, -half, 0.0], [half, half, 0.0], [-half, half, 0.0]],
        dtype=float,
    )


def marker_model_points_wrist(tag_rig, tag_name: str, marker_size_m: float) -> np.ndarray:
    full = np.stack([tag_rig.model_points[f"{tag_name}_c{i}"] for i in range(4)], axis=0)
    center = full.mean(axis=0)
    x_axis = normalize(full[1] - full[0])
    y_axis = normalize(full[3] - full[0])
    object_points = marker_object_points(marker_size_m)
    return center + object_points[:, 0:1] * x_axis + object_points[:, 1:2] * y_axis


def estimate_wrist_sequence_from_tags(
    tag_estimates: dict[str, list[PoseEstimate | None]],
    tag_rig,
    marker_size_m: float,
) -> list[PoseEstimate | None]:
    frame_count = max((len(sequence) for sequence in tag_estimates.values()), default=0)
    object_points = marker_object_points(marker_size_m)
    wrist_points = {tag_name: marker_model_points_wrist(tag_rig, tag_name, marker_size_m) for tag_name in tag_estimates}
    wrist_estimates = []
    for frame_idx in range(frame_count):
        source_points = []
        target_points = []
        sources = []
        for tag_name, sequence in tag_estimates.items():
            estimate = sequence[frame_idx]
            if estimate is None:
                continue
            source_points.append(wrist_points[tag_name])
            target_points.append((estimate.rotation_world @ object_points.T).T + estimate.position_world)
            sources.append(f"{tag_name}:{estimate.source}")
        if not source_points:
            wrist_estimates.append(None)
            continue
        rot, pos = rigid_align(np.vstack(source_points), np.vstack(target_points))
        wrist_estimates.append(PoseEstimate(pos, rot, ",".join(sources)))
    return wrist_estimates


def smooth_pose_sequence(sequence: list[PoseEstimate | None]) -> list[PoseEstimate | None]:
    valid_indices = np.array([idx for idx, estimate in enumerate(sequence) if estimate is not None], dtype=int)
    if len(valid_indices) == 0:
        return [None for _ in sequence]

    all_indices = np.arange(len(sequence), dtype=float)
    positions_valid = np.stack([sequence[idx].position_world for idx in valid_indices], axis=0)
    positions = np.column_stack(
        [np.interp(all_indices, valid_indices.astype(float), positions_valid[:, axis]) for axis in range(3)]
    )
    quats_valid = np.stack([matrix_to_quat(sequence[idx].rotation_world) for idx in valid_indices], axis=0)
    quats_valid = make_quaternion_sequence_continuous(quats_valid)
    quats = interpolate_quaternions(valid_indices, quats_valid, len(sequence))
    positions[valid_indices] = positions_valid
    quats[valid_indices] = quats_valid

    valid_set = set(int(idx) for idx in valid_indices)
    smoothed = []
    for idx in range(len(sequence)):
        source = sequence[idx].source if idx in valid_set else "interpolated"
        smoothed.append(PoseEstimate(positions[idx], quat_to_matrix(quats[idx]), source))
    return smoothed


def save_pose_tracks(
    path: str | Path,
    frame_indices: np.ndarray,
    output_fps: float,
    tag_names: tuple[str, ...],
    tag_estimates: dict[str, list[PoseEstimate | None]],
    tag_truth: dict[str, list[tuple[np.ndarray, np.ndarray]]],
    wrist_estimates: list[PoseEstimate | None],
    wrist_truth_pos: np.ndarray,
    wrist_truth_rot: np.ndarray,
    raw_sources: dict[str, dict[str, int]],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_count = len(frame_indices)
    tag_count = len(tag_names)
    tag_est_pos = np.full((frame_count, tag_count, 3), np.nan, dtype=float)
    tag_est_rot = np.full((frame_count, tag_count, 3, 3), np.nan, dtype=float)
    tag_truth_pos = np.full((frame_count, tag_count, 3), np.nan, dtype=float)
    tag_truth_rot = np.full((frame_count, tag_count, 3, 3), np.nan, dtype=float)
    tag_sources = np.full((frame_count, tag_count), "", dtype="<U64")

    for tag_idx, tag_name in enumerate(tag_names):
        for frame_idx, estimate in enumerate(tag_estimates[tag_name]):
            if estimate is not None:
                tag_est_pos[frame_idx, tag_idx] = estimate.position_world
                tag_est_rot[frame_idx, tag_idx] = estimate.rotation_world
                tag_sources[frame_idx, tag_idx] = estimate.source
        for frame_idx, (rot, pos) in enumerate(tag_truth[tag_name]):
            tag_truth_pos[frame_idx, tag_idx] = pos
            tag_truth_rot[frame_idx, tag_idx] = rot

    wrist_est_pos = np.full((frame_count, 3), np.nan, dtype=float)
    wrist_est_rot = np.full((frame_count, 3, 3), np.nan, dtype=float)
    wrist_sources = np.full(frame_count, "", dtype="<U256")
    for frame_idx, estimate in enumerate(wrist_estimates):
        if estimate is None:
            continue
        wrist_est_pos[frame_idx] = estimate.position_world
        wrist_est_rot[frame_idx] = estimate.rotation_world
        wrist_sources[frame_idx] = estimate.source

    np.savez_compressed(
        path,
        format_version=np.array([1], dtype=np.int32),
        frame_indices=frame_indices.astype(np.int64),
        output_fps=np.array([output_fps], dtype=float),
        tag_names=np.array(tag_names),
        tag_est_pos=tag_est_pos,
        tag_est_rot=tag_est_rot,
        tag_truth_pos=tag_truth_pos,
        tag_truth_rot=tag_truth_rot,
        tag_sources=tag_sources,
        wrist_est_pos=wrist_est_pos,
        wrist_est_rot=wrist_est_rot,
        wrist_truth_pos=np.asarray(wrist_truth_pos, dtype=float),
        wrist_truth_rot=np.asarray(wrist_truth_rot, dtype=float),
        wrist_sources=wrist_sources,
        raw_source_summary=np.array(json.dumps(raw_sources, sort_keys=True)),
    )


def load_pose_tracks(path: str | Path) -> np.lib.npyio.NpzFile:
    return np.load(path, allow_pickle=False)


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    else:
        axis = int(np.argmax(np.diag(m)))
        if axis == 0:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            quat = np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
        elif axis == 1:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            quat = np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            quat = np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])
    return quat / np.linalg.norm(quat)


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=float) / np.linalg.norm(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def make_quaternion_sequence_continuous(quats: np.ndarray) -> np.ndarray:
    result = quats.copy()
    for idx in range(1, len(result)):
        if np.dot(result[idx - 1], result[idx]) < 0.0:
            result[idx] *= -1.0
    return result


def interpolate_quaternions(valid_indices: np.ndarray, quats_valid: np.ndarray, length: int) -> np.ndarray:
    result = np.zeros((length, 4), dtype=float)
    for idx in range(length):
        right_pos = int(np.searchsorted(valid_indices, idx, side="left"))
        if right_pos == 0:
            quat = quats_valid[0]
        elif right_pos >= len(valid_indices):
            quat = quats_valid[-1]
        elif valid_indices[right_pos] == idx:
            quat = quats_valid[right_pos]
        else:
            left_pos = right_pos - 1
            span = float(valid_indices[right_pos] - valid_indices[left_pos])
            alpha = float(idx - valid_indices[left_pos]) / span
            quat = slerp(quats_valid[left_pos], quats_valid[right_pos], alpha)
        result[idx] = quat / np.linalg.norm(quat)
    return result


def slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        quat = q0 + alpha * (q1 - q0)
        return quat / np.linalg.norm(quat)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * alpha
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    return s0 * q0 + s1 * q1
