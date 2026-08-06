#!/usr/bin/env python3
"""Create a wrist-associated evaluation subset with auditable quality gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-error-csv", type=Path, required=True)
    parser.add_argument("--raw-pose-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-gt-range-m", type=float, default=2.0)
    parser.add_argument("--max-position-error-m", type=float, default=0.2)
    parser.add_argument("--max-orientation-error-deg", type=float, default=30.0)
    return parser.parse_args()


def stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    error = pd.read_csv(args.raw_error_csv)
    pose = pd.read_csv(args.raw_pose_csv)
    error["gt_range_m"] = np.linalg.norm(
        error[["gt_x_m", "gt_y_m", "gt_z_m"]].to_numpy(), axis=1
    )
    id9_only = error["detected_tag_ids"].astype(str).eq("9")
    gt_valid = error["gt_range_m"].le(args.max_gt_range_m)
    position_valid = error["position_error_m"].le(args.max_position_error_m)
    orientation_valid = error["orientation_error_deg"].le(
        args.max_orientation_error_deg
    )
    keep = id9_only & gt_valid & position_valid & orientation_valid
    accepted = error.loc[keep].copy()
    accepted.insert(0, "associated_success_index", np.arange(len(accepted)))
    accepted_path = args.output_dir / "wrist_pose_error_associated_success.csv"
    accepted.to_csv(accepted_path, index=False)

    accepted_device_ts = set(
        np.round(accepted["CAM_B_device_ts_ms"].to_numpy(dtype=float), 6)
    )
    pose_keep = np.round(
        pose["CAM_B_device_ts_ms"].to_numpy(dtype=float), 6
    )
    pose_accepted = pose[np.isin(pose_keep, list(accepted_device_ts))].copy()
    pose_path = args.output_dir / "wrist_pose_CAM_B_associated_success.csv"
    pose_accepted.to_csv(pose_path, index=False)

    x = np.arange(len(accepted))
    fig, axes = plt.subplots(2, 1, figsize=(17, 8.5), sharex=True)
    axes[0].plot(
        x, accepted["position_error_m"] * 1000.0,
        linewidth=0.75, color="#1769aa"
    )
    axes[0].set_ylabel("Euclidean position error (mm)")
    axes[0].grid(alpha=0.25)
    axes[1].plot(
        x, accepted["orientation_error_deg"],
        linewidth=0.75, color="#c62828"
    )
    axes[1].set_ylabel("Coordinate-frame angle error (deg)")
    axes[1].set_xlabel("Wrist-associated successful detection frame")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Stereo AprilTag left-wrist pose error vs CH03-01")
    fig.tight_layout()
    chart_path = args.output_dir / "wrist_pose_error_associated_lines.png"
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)

    summary = {
        "schema": "wrist_pose_associated_evaluation.v1",
        "raw_successfully_decoded_frames": int(len(error)),
        "associated_success_frames": int(len(accepted)),
        "quality_gates": {
            "detected_tag_ids_exactly": "9",
            "max_gt_wrist_range_from_CAM_B_m": args.max_gt_range_m,
            "max_position_error_m": args.max_position_error_m,
            "max_orientation_error_deg": args.max_orientation_error_deg,
        },
        "exclusion_diagnostics": {
            "frames_containing_id22": int((~id9_only).sum()),
            "frames_with_invalid_mocap_range": int((~gt_valid).sum()),
            "frames_failing_position_gate": int((~position_valid).sum()),
            "frames_failing_orientation_gate": int((~orientation_valid).sum()),
            "total_excluded_union": int((~keep).sum()),
        },
        "position_error_m": stats(accepted["position_error_m"]),
        "orientation_error_deg": stats(accepted["orientation_error_deg"]),
        "outputs": {
            "error_csv": str(accepted_path),
            "pose_csv": str(pose_path),
            "chart": str(chart_path),
        },
    }
    summary_path = args.output_dir / "wrist_pose_associated_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
