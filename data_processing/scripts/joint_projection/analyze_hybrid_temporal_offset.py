#!/usr/bin/env python3
"""Estimate the temporal offset between video pose and projected mocap pose."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import savgol_filter


JOINT_FIELDS = {
    "LeftArm": ("left_shoulder_x", "left_shoulder_y"),
    "RightArm": ("right_shoulder_x", "right_shoulder_y"),
    "LeftForeArm": ("left_elbow_x", "left_elbow_y"),
    "RightForeArm": ("right_elbow_x", "right_elbow_y"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--max-shift", type=int, default=15)
    return parser.parse_args()


def correlation(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    values = []
    for column in range(a.shape[1]):
        valid = mask & np.isfinite(a[:, column]) & np.isfinite(b[:, column])
        if np.count_nonzero(valid) < 100:
            continue
        x = a[valid, column]
        y = b[valid, column]
        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            continue
        values.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.mean(values)) if values else float("nan")


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.project_dir))
    import project_0722_abx2_subject_scaled as kin
    import project_joints as base
    from render_camc_hybrid_skeleton_video import camera_geometry, load_csv

    aligned = load_csv(args.recording / "aligned_data" / "aligned_30hz.csv")
    pose = load_csv(
        args.recording
        / "aligned_data"
        / "module01_cam_c_shoulder_elbow_2d.csv"
    )
    if len(aligned) != len(pose):
        raise RuntimeError(f"Row mismatch: aligned={len(aligned)}, pose={len(pose)}")

    config = base.load_json(
        args.project_dir / "projection_config_0722_head_ch3_08.json"
    )
    calibration = base.load_json(
        args.project_dir
        / "validation_0722_h265_fixed_time_calibration"
        / "calibration_fixed_time.json"
    )
    model = base.load_camera_models(config)["module01_CAM_C"]
    joint_index = {name: index for index, name in enumerate(kin.JOINT_NAMES)}

    sapiens = np.full((len(pose), len(JOINT_FIELDS), 2), np.nan)
    mocap = np.full_like(sapiens, np.nan)
    for seq, (aligned_row, pose_row) in enumerate(zip(aligned, pose)):
        if pose_row["status"] == "ok":
            for joint_number, fields in enumerate(JOINT_FIELDS.values()):
                if pose_row[fields[0]].strip() and pose_row[fields[1]].strip():
                    sapiens[seq, joint_number] = [
                        float(pose_row[fields[0]]),
                        float(pose_row[fields[1]]),
                    ]
        try:
            geometry = camera_geometry(aligned_row, model, calibration)
        except (KeyError, ValueError, ZeroDivisionError):
            continue
        uv = np.asarray(geometry["uv_raw"])
        valid = np.asarray(geometry["valid_raw"], dtype=bool)
        for joint_number, name in enumerate(JOINT_FIELDS):
            index = joint_index[name]
            if valid[index] and np.all(np.isfinite(uv[index])):
                mocap[seq, joint_number] = uv[index]

    # A centered filter suppresses detector jitter without introducing phase delay.
    for values in (sapiens, mocap):
        for joint in range(values.shape[1]):
            for axis in range(2):
                series = values[:, joint, axis]
                valid = np.isfinite(series)
                if np.count_nonzero(valid) < 31:
                    continue
                filled = np.interp(
                    np.arange(len(series)), np.flatnonzero(valid), series[valid]
                )
                values[:, joint, axis] = savgol_filter(filled, 15, 2)

    sapiens_velocity = np.diff(sapiens, axis=0).reshape(len(sapiens) - 1, -1)
    mocap_velocity = np.diff(mocap, axis=0).reshape(len(mocap) - 1, -1)
    motion = np.linalg.norm(sapiens_velocity, axis=1)
    high_motion_threshold = float(np.nanpercentile(motion, 65))

    results = []
    for shift in range(-args.max_shift, args.max_shift + 1):
        if shift >= 0:
            video = sapiens_velocity[: len(sapiens_velocity) - shift or None]
            projected = mocap_velocity[shift:]
            motion_slice = motion[: len(motion) - shift or None]
        else:
            video = sapiens_velocity[-shift:]
            projected = mocap_velocity[: len(mocap_velocity) + shift]
            motion_slice = motion[-shift:]
        finite = np.all(np.isfinite(video) & np.isfinite(projected), axis=1)
        results.append(
            {
                "shift_frames": shift,
                "shift_ms_at_30hz": shift * 1000.0 / 30.0,
                "correlation_all": correlation(video, projected, finite),
                "correlation_high_motion": correlation(
                    video,
                    projected,
                    finite & (motion_slice >= high_motion_threshold),
                ),
                "samples": int(np.count_nonzero(finite)),
            }
        )

    best = max(
        results,
        key=lambda item: (
            item["correlation_high_motion"],
            item["correlation_all"],
        ),
    )
    report = {
        "convention": (
            "Positive shift means use a later mocap/BVH pose for the current video frame."
        ),
        "best": best,
        "results": results,
    }
    output = (
        args.recording
        / "aligned_data"
        / "hybrid_skeleton_temporal_offset_report.json"
    )
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
