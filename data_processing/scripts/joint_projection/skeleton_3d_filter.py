#!/usr/bin/env python3
"""Temporal + kinematic filtering for multiview 3D skeleton playback records."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from delivery_keypoints import DELIVERY_EDGES, DELIVERY_JOINTS, resolve_joint_xyz

# Per-frame displacement caps (m @ 30 Hz) for obvious spikes.
MAX_SPEED_M_PER_FRAME: dict[str, float] = {
    "nose": 0.18,
    "left_shoulder": 0.22,
    "right_shoulder": 0.22,
    "left_elbow": 0.30,
    "right_elbow": 0.30,
    "left_wrist": 0.40,
    "right_wrist": 0.40,
    "left_hip": 0.22,
    "right_hip": 0.22,
    "left_knee": 0.35,
    "right_knee": 0.35,
    "left_ankle": 0.45,
    "right_ankle": 0.45,
    "left_big_toe": 0.50,
    "right_big_toe": 0.50,
}
DEFAULT_MAX_SPEED_M = 0.35


def records_to_arrays(
    records: Sequence[dict],
    joint_names: Sequence[str],
    scores_by_seq: Mapping[int, Mapping[str, float]] | None = None,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    seqs = [int(r["seq"]) for r in records]
    traj = np.full((len(records), len(joint_names), 3), np.nan, dtype=np.float64)
    scores = np.full((len(records), len(joint_names)), np.nan, dtype=np.float64)
    for fi, record in enumerate(records):
        joints = record["methods"]["filtered"]["multiview"]
        seq = int(record["seq"])
        frame_scores = scores_by_seq.get(seq, {}) if scores_by_seq else {}
        for ji, name in enumerate(joint_names):
            xyz = resolve_joint_xyz(joints, name)
            if xyz is not None:
                traj[fi, ji] = xyz
            if name in frame_scores:
                scores[fi, ji] = float(frame_scores[name])
            elif xyz is not None:
                scores[fi, ji] = 1.0
    return seqs, traj, scores


def arrays_to_records(
    seqs: Sequence[int],
    traj: np.ndarray,
    joint_names: Sequence[str],
) -> list[dict]:
    records: list[dict] = []
    for fi, seq in enumerate(seqs):
        payload = {}
        for ji, name in enumerate(joint_names):
            point = traj[fi, ji]
            if not np.isfinite(point).all():
                continue
            payload[name] = {"xyz_world_m": point.tolist()}
        records.append({"seq": int(seq), "methods": {"filtered": {"multiview": payload}}})
    return records


def _reject_low_scores(traj: np.ndarray, scores: np.ndarray, min_score: float) -> int:
    mask = np.isfinite(scores) & (scores < min_score)
    traj[mask] = np.nan
    return int(mask.sum())


def _reject_temporal_spikes(
    traj: np.ndarray,
    scores: np.ndarray,
    joint_names: Sequence[str],
    *,
    speed_mad_factor: float,
    min_speed_m: float,
) -> int:
    removed = 0
    for ji, name in enumerate(joint_names):
        cap = MAX_SPEED_M_PER_FRAME.get(name, DEFAULT_MAX_SPEED_M)
        points = traj[:, ji]
        valid = np.isfinite(points).all(axis=1)
        if valid.sum() < 4:
            continue
        speeds = np.full(len(points), np.nan, dtype=np.float64)
        for t in range(1, len(points)):
            if valid[t] and valid[t - 1]:
                speeds[t] = float(np.linalg.norm(points[t] - points[t - 1]))
        ref = speeds[np.isfinite(speeds)]
        if ref.size < 3:
            continue
        med = float(np.median(ref))
        mad = float(np.median(np.abs(ref - med)))
        threshold = max(min_speed_m, med + speed_mad_factor * 1.4826 * mad, cap * 0.55)
        hard_cap = cap
        for t in range(len(points)):
            if not valid[t]:
                continue
            step_prev = speeds[t] if np.isfinite(speeds[t]) else 0.0
            step_next = (
                float(np.linalg.norm(points[t + 1] - points[t]))
                if t + 1 < len(points) and valid[t + 1]
                else 0.0
            )
            spike = step_prev > threshold or step_next > threshold or max(step_prev, step_next) > hard_cap
            if not spike:
                continue
            score = scores[t, ji]
            score_cut = float(np.nanpercentile(scores[:, ji], 35)) if np.isfinite(scores[:, ji]).any() else 0.0
            if max(step_prev, step_next) > hard_cap or (
                np.isfinite(score) and score <= max(score_cut, 0.15)
            ):
                traj[t, ji] = np.nan
                removed += 1
    return removed


def _reject_bone_length_outliers(
    traj: np.ndarray,
    scores: np.ndarray,
    joint_names: Sequence[str],
    *,
    deviation_ratio: float,
) -> int:
    index = {name: i for i, name in enumerate(joint_names)}
    removed = 0
    for parent, child in DELIVERY_EDGES:
        pi = index.get(parent)
        ci = index.get(child)
        if pi is None or ci is None:
            continue
        valid = np.isfinite(traj[:, pi]).all(axis=1) & np.isfinite(traj[:, ci]).all(axis=1)
        if valid.sum() < 8:
            continue
        lengths = np.linalg.norm(traj[valid, ci] - traj[valid, pi], axis=1)
        target = float(np.median(lengths))
        if target < 0.05:
            continue
        lo = target * (1.0 - deviation_ratio)
        hi = target * (1.0 + deviation_ratio)
        for t in np.where(valid)[0]:
            length = float(np.linalg.norm(traj[t, ci] - traj[t, pi]))
            if lo <= length <= hi:
                continue
            parent_score = scores[t, pi]
            child_score = scores[t, ci]
            if np.isfinite(parent_score) and np.isfinite(child_score):
                drop = ci if child_score <= parent_score else pi
            else:
                drop = ci
            if np.isfinite(traj[t, drop]).all():
                traj[t, drop] = np.nan
                removed += 1
    return removed


def _interpolate_gaps(traj: np.ndarray, max_gap: int) -> int:
    filled = 0
    n_frames, n_joints, _ = traj.shape
    for ji in range(n_joints):
        valid = np.isfinite(traj[:, ji, 0])
        t = 0
        while t < n_frames:
            if valid[t]:
                t += 1
                continue
            gap_start = t
            while t < n_frames and not valid[t]:
                t += 1
            gap_end = t
            gap_len = gap_end - gap_start
            if gap_len > max_gap:
                continue
            left = gap_start - 1
            right = gap_end
            if left < 0 or right >= n_frames or not valid[left] or not valid[right]:
                continue
            for k, frame in enumerate(range(gap_start, gap_end)):
                alpha = (k + 1) / (gap_len + 1)
                traj[frame, ji] = (1.0 - alpha) * traj[left, ji] + alpha * traj[right, ji]
                filled += 1
            valid = np.isfinite(traj[:, ji, 0])
    return filled


def _median_filter_traj(traj: np.ndarray, window: int) -> None:
    if window < 3:
        return
    radius = window // 2
    n_frames = traj.shape[0]
    for ji in range(traj.shape[1]):
        for axis in range(3):
            col = traj[:, ji, axis].copy()
            valid = np.isfinite(col)
            if valid.sum() < window:
                continue
            out = col.copy()
            for t in range(n_frames):
                if not valid[t]:
                    continue
                lo = max(0, t - radius)
                hi = min(n_frames, t + radius + 1)
                sample = col[lo:hi]
                sample = sample[np.isfinite(sample)]
                if sample.size >= max(3, window // 2):
                    out[t] = float(np.median(sample))
            traj[:, ji, axis] = np.where(valid, out, np.nan)


def _gaussian_smooth_traj(traj: np.ndarray, sigma: float) -> None:
    if sigma <= 0:
        return
    radius = max(1, int(round(3 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    for ji in range(traj.shape[1]):
        for axis in range(3):
            col = traj[:, ji, axis]
            valid = np.isfinite(col)
            if valid.sum() < 5:
                continue
            filled = col.copy()
            idx = np.where(valid)[0]
            for t in range(len(filled)):
                if not np.isfinite(filled[t]):
                    j = idx[np.argmin(np.abs(idx - t))]
                    filled[t] = col[j]
            smooth = np.convolve(filled, kernel, mode="same")
            traj[:, ji, axis] = np.where(valid, smooth, np.nan)


def filter_skeleton_playback_records(
    records: list[dict],
    joint_names: Sequence[str],
    scores_by_seq: Mapping[int, Mapping[str, float]] | None = None,
    *,
    min_volume_score: float = 0.12,
    speed_mad_factor: float = 4.0,
    min_speed_m: float = 0.08,
    bone_length_deviation: float = 0.42,
    gap_interp_max: int = 4,
    median_window: int = 5,
    temporal_sigma: float = 1.0,
) -> tuple[list[dict], dict]:
    """Remove bad 3D points, interpolate short gaps, then temporally smooth."""
    seqs, traj, scores = records_to_arrays(records, joint_names, scores_by_seq)
    raw_valid = int(np.isfinite(traj).all(axis=2).sum())
    report: dict[str, int | float | dict] = {"raw_valid_joint_frames": raw_valid}

    report["removed_low_score"] = _reject_low_scores(traj, scores, min_volume_score)
    report["removed_temporal_spike"] = _reject_temporal_spikes(
        traj,
        scores,
        joint_names,
        speed_mad_factor=speed_mad_factor,
        min_speed_m=min_speed_m,
    )
    report["removed_bone_length"] = _reject_bone_length_outliers(
        traj,
        scores,
        joint_names,
        deviation_ratio=bone_length_deviation,
    )
    report["interpolated_joint_frames"] = _interpolate_gaps(traj, gap_interp_max)
    _median_filter_traj(traj, median_window)
    _gaussian_smooth_traj(traj, temporal_sigma)

    filtered_valid = int(np.isfinite(traj).all(axis=2).sum())
    report["filtered_valid_joint_frames"] = filtered_valid
    report["filter_params"] = {
        "min_volume_score": min_volume_score,
        "speed_mad_factor": speed_mad_factor,
        "min_speed_m": min_speed_m,
        "bone_length_deviation": bone_length_deviation,
        "gap_interp_max": gap_interp_max,
        "median_window": median_window,
        "temporal_sigma": temporal_sigma,
    }
    return arrays_to_records(seqs, traj, joint_names), report
