#!/usr/bin/env python3
"""Compare CAM_B wrist pose estimates with CH3_08/LeftHand mocap ground truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


R_WRIST_RIGID_LEFT = np.column_stack(
    ([1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0])
)
WRIST_TO_RIGID_LEFT_M = (
    np.array([53.5, 76.5, 2.2], dtype=np.float64) / 1000.0
)
# The rigid/BVH wrist convention differs from the wrist-ring convention used
# by the supplied tag-to-wrist transforms by 180 degrees about wrist +Y.
R_ANATOMICAL_WRIST_RING_WRIST = np.diag([-1.0, 1.0, -1.0])


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--pose-csv", type=Path, required=True)
    p.add_argument("--aligned-csv", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def quat_to_R(q):
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def R_to_quat(R):
    trace = np.trace(R)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = [0.25*s, (R[2,1]-R[1,2])/s,
             (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s]
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2
            q = [(R[2,1]-R[1,2])/s, .25*s,
                 (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s]
        elif i == 1:
            s = math.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2
            q = [(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s,
                 .25*s, (R[1,2]+R[2,1])/s]
        else:
            s = math.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2
            q = [(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s,
                 (R[1,2]+R[2,1])/s, .25*s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def make_T(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def angle_deg(R_a, R_b):
    value = np.clip((np.trace(R_a.T @ R_b) - 1.0) / 2.0, -1.0, 1.0)
    return math.degrees(math.acos(value))


def mocap_gt(row, R_head_cam, p_head_cam):
    if row.get("mocap_valid") != "1":
        return None
    try:
        p_anchor = np.array([
            float(row[f"mocap_CH3_08_Rigid_K_world_{a}"]) for a in "xyz"
        ])
        q_anchor = [
            float(row[f"mocap_CH3_08_Rigid_K_world_q{a}"]) for a in "wxyz"
        ]
        p_rigid = np.array([
            float(row[f"mocap_CH3_01_Rigid_K_world_{a}"]) for a in "xyz"
        ])
        q_rigid = [
            float(row[f"mocap_CH3_01_Rigid_K_world_q{a}"]) for a in "wxyz"
        ]
    except (ValueError, KeyError):
        return None
    R_world_anchor = quat_to_R(q_anchor)
    T_world_cam = make_T(
        R_world_anchor @ R_head_cam,
        p_anchor + R_world_anchor @ p_head_cam,
    )
    R_world_rigid = quat_to_R(q_rigid)
    R_world_wrist_anatomical = R_world_rigid @ R_WRIST_RIGID_LEFT.T
    p_world_wrist = (
        p_rigid - R_world_wrist_anatomical @ WRIST_TO_RIGID_LEFT_M
    )
    R_world_wrist = (
        R_world_wrist_anatomical @ R_ANATOMICAL_WRIST_RING_WRIST
    )
    T_world_wrist = make_T(R_world_wrist, p_world_wrist)
    return np.linalg.inv(T_world_cam) @ T_world_wrist


def main():
    a = args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(a.calibration.read_text(encoding="utf-8"))
    R_head_cam = np.asarray(calibration["R_rigid_cam_B"], dtype=np.float64)
    p_head_cam = (
        np.asarray(calibration["p_rigid_cam_B_mm"], dtype=np.float64) / 1000.0
    )
    with a.aligned_csv.open(newline="", encoding="utf-8-sig") as f:
        aligned = [
            row for row in csv.DictReader(f)
            if row.get("module01_CAM_B_device_ts_ms", "").strip()
        ]
    aligned_times = np.asarray(
        [float(row["module01_CAM_B_device_ts_ms"]) for row in aligned]
    )
    with a.pose_csv.open(newline="", encoding="utf-8-sig") as f:
        poses = list(csv.DictReader(f))

    records = []
    for source_row_index, pose in enumerate(poses):
        if not pose.get("wrist_CAM_B_x_m"):
            continue
        device_ts = float(pose["CAM_B_device_ts_ms"])
        at = int(np.searchsorted(aligned_times, device_ts))
        choices = [i for i in (at - 1, at) if 0 <= i < len(aligned)]
        idx = min(choices, key=lambda i: abs(aligned_times[i] - device_ts))
        dt_ms = aligned_times[idx] - device_ts
        gt = mocap_gt(aligned[idx], R_head_cam, p_head_cam)
        if gt is None:
            continue
        p_est = np.array([
            float(pose[f"wrist_CAM_B_{a}_m"]) for a in "xyz"
        ])
        q_est = [float(pose[f"wrist_CAM_B_q{a}"]) for a in "wxyz"]
        R_est = quat_to_R(q_est)
        q_gt = R_to_quat(gt[:3, :3])
        p_gt = gt[:3, 3]
        records.append({
            "source_pose_row": source_row_index,
            "timestamp_ms": pose["timestamp_ms"],
            "CAM_B_device_ts_ms": pose["CAM_B_device_ts_ms"],
            "aligned_seq": aligned[idx]["seq"],
            "mocap_frame_index": aligned[idx]["mocap_frame_index"],
            "timestamp_match_dt_ms": dt_ms,
            "detected_tag_ids": pose["detected_tag_ids"],
            "observation_sources": pose["observation_sources"],
            "estimated_x_m": p_est[0], "estimated_y_m": p_est[1],
            "estimated_z_m": p_est[2],
            "gt_x_m": p_gt[0], "gt_y_m": p_gt[1], "gt_z_m": p_gt[2],
            "position_error_m": float(np.linalg.norm(p_est - p_gt)),
            "estimated_qw": q_est[0], "estimated_qx": q_est[1],
            "estimated_qy": q_est[2], "estimated_qz": q_est[3],
            "gt_qw": q_gt[0], "gt_qx": q_gt[1],
            "gt_qy": q_gt[2], "gt_qz": q_gt[3],
            "orientation_error_deg": angle_deg(R_est, gt[:3, :3]),
            "reprojection_error_px": pose["mean_reprojection_error_px"],
        })

    output_csv = a.output_dir / "wrist_pose_error_per_frame.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    position = np.asarray([r["position_error_m"] for r in records])
    orientation = np.asarray([r["orientation_error_deg"] for r in records])
    x = np.arange(len(records))
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    axes[0].plot(x, position * 1000.0, linewidth=0.8, color="#1769aa")
    axes[0].set_ylabel("Position error (mm)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(x, orientation, linewidth=0.8, color="#c62828")
    axes[1].set_ylabel("Orientation error (deg)")
    axes[1].set_xlabel("Successful detection frame")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Stereo AprilTag wrist pose error vs mocap ground truth")
    fig.tight_layout()
    chart = a.output_dir / "wrist_pose_error_lines.png"
    fig.savefig(chart, dpi=160)
    plt.close(fig)

    def stats(v):
        return {
            "count": int(len(v)), "mean": float(np.mean(v)),
            "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "max": float(np.max(v)),
        }
    summary = {
        "matched_frames": len(records),
        "position_error_m": stats(position),
        "orientation_error_deg": stats(orientation),
        "truth_definition": {
            "head_rigid_for_camera": "mocap_CH3_08_Rigid_K_world",
            "wrist_rigid": "mocap_CH3_01_Rigid_K_world",
            "camera": "module01_CAM_B",
            "camera_offset_in_CH3_08_m": p_head_cam.tolist(),
            "R_CH3_08_CAM_B": R_head_cam.tolist(),
            "wrist_to_rigid_left_m": WRIST_TO_RIGID_LEFT_M.tolist(),
            "R_wrist_rigid_left": R_WRIST_RIGID_LEFT.tolist(),
            "R_anatomical_wrist_ring_wrist": (
                R_ANATOMICAL_WRIST_RING_WRIST.tolist()
            ),
        },
        "outputs": {"csv": str(output_csv), "chart": str(chart)},
    }
    (a.output_dir / "wrist_pose_error_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
