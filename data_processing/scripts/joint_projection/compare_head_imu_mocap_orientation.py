#!/usr/bin/env python3
"""Compare head module01 Mahony IMU orientation vs CH3_08 mocap rigid (GT)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from imu_mahony import (
    MahonyEstimator,
    average_rotation_matrices,
    matrix_to_q,
    q_inv,
    q_mul,
    q_normalize,
    q_to_matrix,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--preset",
        choices=("none", "0810-line1", "0810-line2"),
        default="0810-line1",
    )
    p.add_argument(
        "--aligned-csv",
        type=Path,
        default=Path(r"C:\Users\hand\Desktop\双外部双目\0810\1\aligned_data\aligned_30hz.csv"),
    )
    p.add_argument(
        "--aligned-report",
        type=Path,
        default=Path(r"C:\Users\hand\Desktop\双外部双目\0810\1\aligned_data\aligned_30hz_report.json"),
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--gt-prefix", default="mocap_CH3_08")
    p.add_argument("--calibration-frames", type=int, default=300)
    p.add_argument(
        "--eval-start-frame",
        type=int,
        default=None,
        help="Start eval after calibration (default: calibration_frames).",
    )
    return p.parse_args()


def apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "0810-line1":
        root = Path(r"C:\Users\hand\Desktop\双外部双目\0810\1")
    elif args.preset == "0810-line2":
        root = Path(r"C:\Users\hand\Desktop\双外部双目\0810\2")
    else:
        return
    args.aligned_csv = root / "aligned_data" / "aligned_30hz.csv"
    args.aligned_report = root / "aligned_data" / "aligned_30hz_report.json"
    if args.output_dir is None:
        args.output_dir = (
            Path(__file__).resolve().parent
            / "output"
            / f"head_imu_vs_ch08_{args.preset.replace('0810-', '0810_')}"
        )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def row_valid(row: dict[str, str], prefix: str) -> bool:
    if row.get(f"{prefix}_status") != "1" or row.get(f"{prefix}_raw_tick_valid") != "1":
        return False
    for axis in ("qw", "qx", "qy", "qz"):
        if row.get(f"{prefix}_{axis}") in (None, ""):
            return False
    return True


def row_quat(row: dict[str, str], prefix: str) -> np.ndarray:
    return np.array(
        [
            float(row[f"{prefix}_qw"]),
            float(row[f"{prefix}_qx"]),
            float(row[f"{prefix}_qy"]),
            float(row[f"{prefix}_qz"]),
        ],
        dtype=np.float64,
    )


def imu_timestamp_ms(row: dict[str, str]) -> float:
    accel = row.get("head_imu_accel_device_ts_ms")
    gyro = row.get("head_imu_gyro_device_ts_ms")
    if accel in (None, "") and gyro in (None, ""):
        return float("nan")
    if accel in (None, ""):
        return float(gyro)
    if gyro in (None, ""):
        return float(accel)
    return 0.5 * (float(accel) + float(gyro))


def run_mahony_rows(rows: list[dict[str, str]]) -> np.ndarray:
    est = MahonyEstimator()
    quats = np.zeros((len(rows), 4), dtype=np.float64)
    for index, row in enumerate(rows):
        try:
            ts = imu_timestamp_ms(row)
            accel = np.array(
                [
                    float(row["head_imu_ax_m_s2"]),
                    float(row["head_imu_ay_m_s2"]),
                    float(row["head_imu_az_m_s2"]),
                ],
                dtype=np.float64,
            )
            gyro = np.array(
                [
                    float(row["head_imu_gx_rad_s"]),
                    float(row["head_imu_gy_rad_s"]),
                    float(row["head_imu_gz_rad_s"]),
                ],
                dtype=np.float64,
            )
        except (KeyError, TypeError, ValueError):
            quats[index] = quats[index - 1] if index else np.array([1.0, 0.0, 0.0, 0.0])
            continue
        if not np.all(np.isfinite(accel)) or not np.all(np.isfinite(gyro)) or not np.isfinite(ts):
            quats[index] = quats[index - 1] if index else np.array([1.0, 0.0, 0.0, 0.0])
            continue
        quats[index] = est.update(accel, gyro, ts)
    return quats


def fit_sensor_to_world(
    gt_q: np.ndarray,
    sensor_q: np.ndarray,
) -> np.ndarray:
    rel = q_to_matrix(q_mul(gt_q, q_inv(sensor_q)))
    if rel.ndim == 2:
        rel = rel.reshape(1, 3, 3)
    return average_rotation_matrices(rel)


def apply_world(
    sensor_q: np.ndarray,
    r_sensor_to_world: np.ndarray,
) -> np.ndarray:
    r_sensor = q_to_matrix(sensor_q)
    if r_sensor.ndim == 2:
        r_world = r_sensor_to_world @ r_sensor
        return matrix_to_q(r_world.reshape(1, 3, 3))[0]
    r_world = r_sensor_to_world @ r_sensor
    return matrix_to_q(r_world)


def geodesic_deg(q_gt: np.ndarray, q_est: np.ndarray) -> float:
    dq = q_mul(q_gt.reshape(1, 4), q_inv(q_est.reshape(1, 4)))[0]
    if dq[0] < 0.0:
        dq *= -1.0
    w = float(np.clip(dq[0], -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(w)))


def summarize(errors_deg: np.ndarray) -> dict[str, float]:
    if len(errors_deg) == 0:
        return {"count": 0.0}
    return {
        "count": float(len(errors_deg)),
        "min_deg": float(np.min(errors_deg)),
        "p5_deg": float(np.percentile(errors_deg, 5)),
        "median_deg": float(np.median(errors_deg)),
        "mean_deg": float(np.mean(errors_deg)),
        "p90_deg": float(np.percentile(errors_deg, 90)),
        "p95_deg": float(np.percentile(errors_deg, 95)),
        "max_deg": float(np.max(errors_deg)),
        "std_deg": float(np.std(errors_deg)),
    }


def main() -> None:
    args = parse_args()
    apply_preset(args)
    assert args.output_dir is not None

    rows = read_rows(args.aligned_csv)
    report = json.loads(args.aligned_report.read_text(encoding="utf-8"))
    eval_start = args.eval_start_frame
    if eval_start is None:
        eval_start = args.calibration_frames

    mahony_q = run_mahony_rows(rows)
    gt_valid = np.array([row_valid(row, args.gt_prefix) for row in rows], dtype=bool)
    calib_mask = gt_valid.copy()
    calib_mask[args.calibration_frames :] = False
    eval_mask = gt_valid.copy()
    eval_mask[:eval_start] = False

    calib_idx = np.flatnonzero(calib_mask)
    if len(calib_idx) < 30:
        raise RuntimeError(f"Too few calibration frames: {len(calib_idx)}")

    gt_calib = np.stack([row_quat(rows[i], args.gt_prefix) for i in calib_idx], axis=0)
    sensor_calib = mahony_q[calib_idx]
    r_sensor_to_world = fit_sensor_to_world(gt_calib, sensor_calib)

    errors_all: list[float] = []
    errors_eval: list[float] = []
    eval_gyro_norms: list[float] = []

    for index, row in enumerate(rows):
        if not gt_valid[index]:
            continue
        q_gt = row_quat(row, args.gt_prefix)
        q_est = q_normalize(apply_world(mahony_q[index], r_sensor_to_world))
        err = geodesic_deg(q_gt, q_est)
        gyro = np.array(
            [
                float(row["head_imu_gx_rad_s"]),
                float(row["head_imu_gy_rad_s"]),
                float(row["head_imu_gz_rad_s"]),
            ],
            dtype=np.float64,
        )
        gyro_norm = float(np.linalg.norm(gyro))
        errors_all.append(err)
        if eval_mask[index]:
            errors_eval.append(err)
            eval_gyro_norms.append(gyro_norm)

    if eval_gyro_norms:
        active_threshold = float(np.percentile(eval_gyro_norms, 45.0))
        errors_active = [
            err
            for err, norm in zip(errors_eval, eval_gyro_norms, strict=True)
            if norm >= active_threshold
        ]
    else:
        errors_active = []

    errors_all_arr = np.asarray(errors_all, dtype=np.float64)
    errors_eval_arr = np.asarray(errors_eval, dtype=np.float64)
    errors_active_arr = np.asarray(errors_active, dtype=np.float64)

    valid_indices = [index for index, ok in enumerate(gt_valid) if ok]
    calib_only_errors: list[float] = []
    rolling_errors: list[float] = []
    segment_stats: list[dict[str, float | int]] = []
    segment_len = 300
    segment_calib = 30
    for start in range(0, len(valid_indices) - segment_len, segment_len):
        chunk = valid_indices[start : start + segment_len]
        calib_chunk = chunk[:segment_calib]
        eval_chunk = chunk[segment_calib:]
        r_local = fit_sensor_to_world(
            np.stack([row_quat(rows[i], args.gt_prefix) for i in calib_chunk], axis=0),
            mahony_q[calib_chunk],
        )
        seg_errs = [
            geodesic_deg(row_quat(rows[i], args.gt_prefix), q_normalize(apply_world(mahony_q[i], r_local)))
            for i in eval_chunk
        ]
        rolling_errors.extend(seg_errs)
        segment_stats.append(
            {
                "start_seq": int(rows[chunk[0]]["seq"]),
                "eval_count": len(seg_errs),
                "median_deg": float(np.median(seg_errs)),
                "p90_deg": float(np.percentile(seg_errs, 90)),
            }
        )
    for index in valid_indices[: args.calibration_frames]:
        calib_only_errors.append(
            geodesic_deg(
                row_quat(rows[index], args.gt_prefix),
                q_normalize(apply_world(mahony_q[index], r_sensor_to_world)),
            )
        )
    calib_only_arr = np.asarray(calib_only_errors, dtype=np.float64)
    rolling_arr = np.asarray(rolling_errors, dtype=np.float64)

    curve_seq: list[int] = []
    curve_time: list[float] = []
    curve_global: list[float] = []
    curve_local: list[float] = []
    for start in range(0, len(valid_indices) - segment_len, segment_len):
        chunk = valid_indices[start : start + segment_len]
        calib_chunk = chunk[:segment_calib]
        eval_chunk = chunk[segment_calib:]
        r_local = fit_sensor_to_world(
            np.stack([row_quat(rows[i], args.gt_prefix) for i in calib_chunk], axis=0),
            mahony_q[calib_chunk],
        )
        for index in eval_chunk:
            curve_seq.append(int(rows[index]["seq"]))
            curve_time.append(float(rows[index]["camera_elapsed_sec"]))
            curve_global.append(
                geodesic_deg(
                    row_quat(rows[index], args.gt_prefix),
                    q_normalize(apply_world(mahony_q[index], r_sensor_to_world)),
                )
            )
            curve_local.append(
                geodesic_deg(
                    row_quat(rows[index], args.gt_prefix),
                    q_normalize(apply_world(mahony_q[index], r_local)),
                )
            )

    def moving_average(values: np.ndarray, width: int) -> np.ndarray:
        if len(values) == 0:
            return values
        width = max(1, width)
        kernel = np.ones(width, dtype=np.float64) / width
        pad = width // 2
        padded = np.pad(values, (pad, width - 1 - pad), mode="edge")
        return np.convolve(padded, kernel, mode="valid")

    curve_global_arr = np.asarray(curve_global, dtype=np.float64)
    curve_local_arr = np.asarray(curve_local, dtype=np.float64)
    curve_time_arr = np.asarray(curve_time, dtype=np.float64)
    smooth_w = max(15, segment_len // 20)
    curve_global_smooth = moving_average(curve_global_arr, smooth_w)
    curve_local_smooth = moving_average(curve_local_arr, smooth_w)

    payload = {
        "schema": "joint_projection.head_imu_orientation_eval.v1",
        "preset": args.preset,
        "aligned_csv": str(args.aligned_csv),
        "gt_rigid": args.gt_prefix,
        "imu_source": "aligned_30hz head_imu (module01 Mahony AHRS)",
        "calibration_frames": args.calibration_frames,
        "eval_start_frame": eval_start,
        "calibration_samples": int(len(calib_idx)),
        "gt_valid_frames": int(np.count_nonzero(gt_valid)),
        "head_imu_mocap_gyro_score": report.get("head_imu_mocap_alignment", {}).get("score"),
        "error_deg": {
            "calibration_window_only": summarize(calib_only_arr),
            "all_valid_gt_frames_global_extrinsic": summarize(errors_all_arr),
            "eval_after_calibration_global_extrinsic": summarize(errors_eval_arr),
            "eval_active_motion_p45_gyro": summarize(errors_active_arr),
            "rolling_10s_segments_local_extrinsic": summarize(rolling_arr),
        },
        "rolling_segment_len_frames": segment_len,
        "rolling_segment_calib_frames": segment_calib,
        "rolling_segment_medians_deg": [float(s["median_deg"]) for s in segment_stats],
        "notes": (
            "GT = mocap CH3_08 quaternion in aligned_30hz. "
            "IMU = Mahony on module01 accel+gyro. "
            "Global extrinsic is one constant sensor->world rotation fit on the first "
            "calibration window; it drifts over long sequences without magnetometer aiding. "
            "Rolling 10s segments re-fit extrinsic every 10 s (first 1 s calib) and better "
            "reflect instantaneous orientation tracking error."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "head_imu_vs_ch08_report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    if len(errors_eval_arr):
        ax.hist(errors_eval_arr, bins=60, color="#2563eb", alpha=0.85, edgecolor="white")
        ax.axvline(float(np.median(errors_eval_arr)), color="#ef4444", linestyle="--", linewidth=1.5, label="median")
        ax.axvline(float(np.percentile(errors_eval_arr, 90)), color="#f97316", linestyle=":", linewidth=1.5, label="p90")
    ax.set_xlabel("Orientation error vs CH3_08 (deg)")
    ax.set_ylabel("Frame count")
    ax.set_title(f"0810 head IMU vs mocap rigid ({args.preset})")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    png_path = args.output_dir / "head_imu_vs_ch08_error_hist.png"
    fig.savefig(png_path)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), dpi=150, sharex=True)
    if len(curve_time_arr):
        axes[0].plot(curve_time_arr, curve_global_arr, color="#93c5fd", linewidth=0.6, alpha=0.45, label="per frame")
        axes[0].plot(
            curve_time_arr,
            curve_global_smooth,
            color="#1d4ed8",
            linewidth=1.8,
            label=f"global extrinsic (smooth {smooth_w}f)",
        )
        axes[0].axhline(float(np.median(curve_global_arr)), color="#ef4444", linestyle="--", linewidth=1.2, label="median")
        axes[0].set_ylabel("Error (deg)")
        axes[0].set_title(f"Head IMU vs CH3_08 — global extrinsic ({args.preset})")
        axes[0].grid(alpha=0.25)
        axes[0].legend(loc="upper left", fontsize=8)
        axes[0].set_ylim(0, min(180, float(np.percentile(curve_global_arr, 99.5)) * 1.15 + 5))

        axes[1].plot(curve_time_arr, curve_local_arr, color="#86efac", linewidth=0.6, alpha=0.45, label="per frame")
        axes[1].plot(
            curve_time_arr,
            curve_local_smooth,
            color="#15803d",
            linewidth=1.8,
            label=f"rolling 10s local extrinsic (smooth {smooth_w}f)",
        )
        axes[1].axhline(float(np.median(curve_local_arr)), color="#ef4444", linestyle="--", linewidth=1.2, label="median")
        axes[1].set_xlabel("Camera elapsed (s)")
        axes[1].set_ylabel("Error (deg)")
        axes[1].set_title("Rolling 10 s segment — local extrinsic re-fit")
        axes[1].grid(alpha=0.25)
        axes[1].legend(loc="upper left", fontsize=8)
        axes[1].set_ylim(0, min(180, float(np.percentile(curve_local_arr, 99.5)) * 1.15 + 5))
    fig.tight_layout()
    curve_path = args.output_dir / "head_imu_vs_ch08_error_curve.png"
    fig.savefig(curve_path)
    plt.close(fig)

    print(json.dumps(payload["error_deg"], ensure_ascii=False, indent=2))
    print(f"wrote {json_path}")
    print(f"wrote {png_path}")
    print(f"wrote {curve_path}")


if __name__ == "__main__":
    main()
