#!/usr/bin/env python3
"""Correlate camera/tag geometry with errors and calibrate T_tag_wrist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation


R_OLD = np.diag([-1.0, 1.0, -1.0])
T_OLD_M = np.array([0.016, -0.008, -0.053], dtype=float)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--error-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    return parser.parse_args()


def quat_matrix(frame, prefix):
    # CSV quaternion order is w, x, y, z; scipy expects x, y, z, w.
    q = frame[
        [f"{prefix}_qx", f"{prefix}_qy", f"{prefix}_qz", f"{prefix}_qw"]
    ].to_numpy(dtype=float)
    return Rotation.from_quat(q).as_matrix()


def rotation_errors_deg(a, b):
    return np.degrees(
        Rotation.from_matrix(np.transpose(a, (0, 2, 1)) @ b).magnitude()
    )


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "rmse": float(np.sqrt(np.mean(values * values))),
    }


def estimate_extrinsic(r_tag, p_tag, r_gt, p_gt, mask):
    implied_r = np.transpose(r_tag[mask], (0, 2, 1)) @ r_gt[mask]
    implied_t = np.einsum(
        "nij,nj->ni",
        np.transpose(r_tag[mask], (0, 2, 1)),
        p_gt[mask] - p_tag[mask],
    )
    # The associated-success input has already removed false tags and invalid
    # mocap. The Frechet rotation mean and arithmetic translation mean minimize
    # squared geodesic/Euclidean residuals over the retained calibration set.
    r_tag_wrist = Rotation.from_matrix(implied_r).mean().as_matrix()
    t_tag_wrist = implied_t.mean(axis=0)
    return r_tag_wrist, t_tag_wrist, implied_r, implied_t


def evaluate(r_tag, p_tag, r_gt, p_gt, r_x, t_x):
    r_est = r_tag @ r_x
    p_est = p_tag + np.einsum("nij,j->ni", r_tag, t_x)
    position = np.linalg.norm(p_est - p_gt, axis=1)
    orientation = rotation_errors_deg(r_est, r_gt)
    return r_est, p_est, position, orientation


def transform_dict(r, t):
    quat = Rotation.from_matrix(r).as_quat()
    return {
        "translation_m": t.tolist(),
        "translation_mm": (t * 1000.0).tolist(),
        "rotation_matrix": r.tolist(),
        "quaternion_wxyz": [quat[3], quat[0], quat[1], quat[2]],
        "rotvec_deg": np.degrees(Rotation.from_matrix(r).as_rotvec()).tolist(),
        "euler_xyz_deg": Rotation.from_matrix(r).as_euler(
            "xyz", degrees=True
        ).tolist(),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.error_csv).sort_values(
        "CAM_B_device_ts_ms"
    ).reset_index(drop=True)
    r_wrist_old = quat_matrix(data, "estimated")
    r_gt = quat_matrix(data, "gt")
    p_wrist_old = data[
        ["estimated_x_m", "estimated_y_m", "estimated_z_m"]
    ].to_numpy(dtype=float)
    p_gt = data[["gt_x_m", "gt_y_m", "gt_z_m"]].to_numpy(dtype=float)

    # T_cam_tag = T_cam_wrist * inverse(T_tag_wrist_old).
    r_tag = r_wrist_old @ R_OLD.T
    p_tag = p_wrist_old - np.einsum("nij,j->ni", r_tag, T_OLD_M)

    distance = np.linalg.norm(p_tag, axis=1)
    bearing = np.degrees(
        np.arctan2(np.linalg.norm(p_tag[:, :2], axis=1), p_tag[:, 2])
    )
    view_to_camera = -p_tag / distance[:, None]
    tag_normal = r_tag[:, :, 2]
    normal_view = np.degrees(
        np.arccos(
            np.clip(np.abs(np.sum(tag_normal * view_to_camera, axis=1)), 0, 1)
        )
    )
    euler = Rotation.from_matrix(r_tag).as_euler("xyz", degrees=True)
    features = pd.DataFrame({
        "camera_tag_x_m": p_tag[:, 0],
        "camera_tag_y_m": p_tag[:, 1],
        "camera_tag_z_m": p_tag[:, 2],
        "camera_tag_distance_m": distance,
        "camera_tag_bearing_deg": bearing,
        "tag_normal_view_angle_deg": normal_view,
        "camera_tag_roll_deg": euler[:, 0],
        "camera_tag_pitch_deg": euler[:, 1],
        "camera_tag_yaw_deg": euler[:, 2],
        "position_error_mm": data["position_error_m"].to_numpy() * 1000.0,
        "orientation_error_deg": data["orientation_error_deg"].to_numpy(),
    })
    feature_names = list(features.columns[:9])
    error_names = ["position_error_mm", "orientation_error_deg"]
    correlation_rows = []
    for feature in feature_names:
        for error in error_names:
            correlation_rows.append({
                "geometry_feature": feature,
                "error_metric": error,
                "pearson_r": features[feature].corr(features[error], method="pearson"),
                "spearman_rho": features[feature].corr(
                    features[error], method="spearman"
                ),
            })
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(
        args.output_dir / "camera_tag_error_correlations.csv", index=False
    )

    split = int(round(len(data) * args.train_fraction))
    train = np.arange(len(data)) < split
    validation = ~train
    r_train, t_train, _, _ = estimate_extrinsic(
        r_tag, p_tag, r_gt, p_gt, train
    )
    r_full, t_full, implied_r, implied_t = estimate_extrinsic(
        r_tag, p_tag, r_gt, p_gt, np.ones(len(data), dtype=bool)
    )
    _, _, pos_train_x, ang_train_x = evaluate(
        r_tag, p_tag, r_gt, p_gt, r_train, t_train
    )
    r_full_est, p_full_est, pos_full, ang_full = evaluate(
        r_tag, p_tag, r_gt, p_gt, r_full, t_full
    )
    pos_old = data["position_error_m"].to_numpy(dtype=float)
    ang_old = data["orientation_error_deg"].to_numpy(dtype=float)

    output = data.copy()
    output["camera_tag_x_m"] = p_tag[:, 0]
    output["camera_tag_y_m"] = p_tag[:, 1]
    output["camera_tag_z_m"] = p_tag[:, 2]
    output["camera_tag_distance_m"] = distance
    output["camera_tag_bearing_deg"] = bearing
    output["tag_normal_view_angle_deg"] = normal_view
    output["calibrated_x_m"] = p_full_est[:, 0]
    output["calibrated_y_m"] = p_full_est[:, 1]
    output["calibrated_z_m"] = p_full_est[:, 2]
    calibrated_q = Rotation.from_matrix(r_full_est).as_quat()
    output["calibrated_qw"] = calibrated_q[:, 3]
    output["calibrated_qx"] = calibrated_q[:, 0]
    output["calibrated_qy"] = calibrated_q[:, 1]
    output["calibrated_qz"] = calibrated_q[:, 2]
    output["calibrated_position_error_m"] = pos_full
    output["calibrated_orientation_error_deg"] = ang_full
    output.to_csv(
        args.output_dir / "per_frame_calibrated_errors.csv", index=False
    )

    fig, axes = plt.subplots(2, 1, figsize=(17, 8.5), sharex=True)
    x = np.arange(len(data))
    axes[0].plot(x, pos_old * 1000, lw=0.65, alpha=0.65, label="Original")
    axes[0].plot(x, pos_full * 1000, lw=0.7, label="Optimized")
    axes[0].set_ylabel("Position error (mm)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(x, ang_old, lw=0.65, alpha=0.65, label="Original")
    axes[1].plot(x, ang_full, lw=0.7, label="Optimized")
    axes[1].set_ylabel("Orientation error (deg)")
    axes[1].set_xlabel("Wrist-associated successful frame")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.suptitle("Tag-to-wrist extrinsic calibration: before vs after")
    fig.tight_layout()
    fig.savefig(
        args.output_dir / "tag_wrist_calibration_before_after.png", dpi=180
    )
    plt.close(fig)

    pivot_p = correlations.pivot(
        index="geometry_feature", columns="error_metric", values="spearman_rho"
    )
    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    image = ax.imshow(pivot_p.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(pivot_p.columns)), pivot_p.columns, rotation=15)
    ax.set_yticks(range(len(pivot_p.index)), pivot_p.index)
    for i in range(len(pivot_p.index)):
        for j in range(len(pivot_p.columns)):
            ax.text(
                j, i, f"{pivot_p.iloc[i, j]:.3f}",
                ha="center", va="center", fontsize=9
            )
    fig.colorbar(image, ax=ax, label="Spearman rho")
    ax.set_title("Camera–Tag geometry vs wrist-pose error")
    fig.tight_layout()
    fig.savefig(args.output_dir / "correlation_heatmap.png", dpi=180)
    plt.close(fig)

    validation_summary = {
        "original_position_error_m": summarize(pos_old[validation]),
        "optimized_position_error_m": summarize(pos_train_x[validation]),
        "original_orientation_error_deg": summarize(ang_old[validation]),
        "optimized_orientation_error_deg": summarize(ang_train_x[validation]),
    }
    strongest = {}
    for error in error_names:
        group = correlations[correlations["error_metric"] == error].copy()
        group["abs_spearman"] = group["spearman_rho"].abs()
        strongest[error] = group.nlargest(5, "abs_spearman")[
            ["geometry_feature", "pearson_r", "spearman_rho"]
        ].to_dict("records")
    summary = {
        "schema": "tag_wrist_extrinsic_calibration.v1",
        "input_frames": int(len(data)),
        "relative_pose_definition": (
            "T_CAM_B_TAG = inverse(T_WORLD_CAM_B from CH03-08) * "
            "T_WORLD_TAG; algebraically identical to reconstructed detected "
            "T_CAM_B_TAG"
        ),
        "original_T_tag_wrist": transform_dict(R_OLD, T_OLD_M),
        "optimized_T_tag_wrist_all_frames": transform_dict(r_full, t_full),
        "delta_translation_mm": ((t_full - T_OLD_M) * 1000.0).tolist(),
        "delta_rotation_deg": float(
            np.degrees(Rotation.from_matrix(R_OLD.T @ r_full).magnitude())
        ),
        "all_frames_before": {
            "position_error_m": summarize(pos_old),
            "orientation_error_deg": summarize(ang_old),
        },
        "all_frames_after": {
            "position_error_m": summarize(pos_full),
            "orientation_error_deg": summarize(ang_full),
        },
        "temporal_holdout": {
            "train_fraction": args.train_fraction,
            "train_frames": int(train.sum()),
            "validation_frames": int(validation.sum()),
            "train_only_transform": transform_dict(r_train, t_train),
            **validation_summary,
        },
        "strongest_correlations": strongest,
        "implied_translation_std_mm": (
            np.std(implied_t, axis=0) * 1000.0
        ).tolist(),
        "implied_rotation_deviation_deg": summarize(
            np.degrees(
                Rotation.from_matrix(
                    np.transpose(r_full[None, :, :], (0, 2, 1)) @ implied_r
                ).magnitude()
            )
        ),
        "outputs": {
            "correlations_csv": "camera_tag_error_correlations.csv",
            "per_frame_csv": "per_frame_calibrated_errors.csv",
            "before_after_chart": "tag_wrist_calibration_before_after.png",
            "correlation_heatmap": "correlation_heatmap.png",
        },
    }
    (args.output_dir / "tag_wrist_calibration_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
