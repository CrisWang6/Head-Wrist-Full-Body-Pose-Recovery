#!/usr/bin/env python3
"""Estimate wrist IMU orientation (Mahony AHRS) and export overlay for skeleton playback."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from imu_mahony import (
    MahonyEstimator,
    angular_velocity_from_quat,
    average_rotation_matrices,
    fit_rotation,
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
        choices=("none", "0810-line1"),
        default="none",
        help="Optional path bundle for common datasets.",
    )
    p.add_argument(
        "--aligned-csv",
        type=Path,
        default=Path(
            r"C:\Users\hand\Desktop\双外部双目\0806\无\aligned_data\aligned_30hz.csv"
        ),
    )
    p.add_argument(
        "--aligned-report",
        type=Path,
        default=Path(
            r"C:\Users\hand\Desktop\双外部双目\0806\无\aligned_data\aligned_30hz_report.json"
        ),
    )
    p.add_argument(
        "--playback",
        type=Path,
        default=Path(
            r"C:\Users\hand\Desktop\双外部双目\0806\无\multiview_3d_results\full"
            r"\skeleton_playback.json"
        ),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--left-imu-csv",
        type=Path,
        default=None,
        help="Optional wrist IMU CSV (e.g. module02). Uses Mahony on full stream.",
    )
    p.add_argument(
        "--right-imu-csv",
        type=Path,
        default=None,
        help="Optional wrist IMU CSV (e.g. module03).",
    )
    p.add_argument(
        "--sync-field",
        default="head_trigger_time_device_ts_ms",
        help="Aligned CSV timestamp field for nearest IMU lookup.",
    )
    p.add_argument(
        "--sync-max-dt-ms",
        type=float,
        default=20.0,
        help="Reject IMU samples farther than this from the sync timestamp.",
    )
    p.add_argument(
        "--calibration",
        choices=("mocap_rigid", "gravity_y_up"),
        default="mocap_rigid",
        help="How to map Mahony sensor frame to mocap Y-up world.",
    )
    p.add_argument(
        "--left-target",
        default="mocap_CH3_04",
        help="Mocap rigid prefix for left-side calibration (mocap_rigid mode).",
    )
    p.add_argument(
        "--right-target",
        default="",
        help="Optional second target prefix; empty = mirror left estimate to right wrist.",
    )
    p.add_argument(
        "--calibration-frames",
        type=int,
        default=300,
        help="Frames used to fit IMU sensor -> mocap world rotation.",
    )
    args = p.parse_args()
    if args.preset == "0810-line1":
        root = Path(r"C:\Users\hand\Desktop\双外部双目\0810\1")
        session = root / "0712_035226"
        args.aligned_csv = root / "aligned_data" / "aligned_30hz.csv"
        args.aligned_report = root / "aligned_data" / "aligned_30hz_report.json"
        args.playback = root / "multiview_3d_results" / "full" / "skeleton_playback.json"
        args.left_imu_csv = next(session.glob("module02_*_imu.csv"))
        args.right_imu_csv = next(session.glob("module03_*_imu.csv"))
        args.calibration = "gravity_y_up"
        args.sync_field = "head_trigger_time_device_ts_ms"
    return args


def read_aligned_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def row_valid(row: dict[str, str], prefix: str) -> bool:
    status = row.get(f"{prefix}_status")
    tick = row.get(f"{prefix}_raw_tick_valid")
    if status != "1" or tick != "1":
        return False
    for axis in ("qw", "qx", "qy", "qz"):
        value = row.get(f"{prefix}_{axis}")
        if value in (None, ""):
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


def read_imu_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: list[float] = []
    accels: list[list[float]] = []
    gyros: list[list[float]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            accel_ts = row.get("accel_device_ts_ms")
            gyro_ts = row.get("gyro_device_ts_ms")
            if accel_ts in (None, "") and gyro_ts in (None, ""):
                continue
            if accel_ts in (None, ""):
                ts = float(gyro_ts)
            elif gyro_ts in (None, ""):
                ts = float(accel_ts)
            else:
                ts = 0.5 * (float(accel_ts) + float(gyro_ts))
            times.append(ts)
            accels.append(
                [float(row["ax_m_s2"]), float(row["ay_m_s2"]), float(row["az_m_s2"])]
            )
            gyros.append(
                [float(row["gx_rad_s"]), float(row["gy_rad_s"]), float(row["gz_rad_s"])]
            )
    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(accels, dtype=np.float64),
        np.asarray(gyros, dtype=np.float64),
    )


def run_mahony_stream(
    times_ms: np.ndarray,
    accels: np.ndarray,
    gyros: np.ndarray,
) -> np.ndarray:
    est = MahonyEstimator()
    quats = np.zeros((len(times_ms), 4), dtype=np.float64)
    for index in range(len(times_ms)):
        quats[index] = est.update(accels[index], gyros[index], float(times_ms[index]))
    return quats


def nearest_imu_sample(
    times_ms: np.ndarray,
    accels: np.ndarray,
    gyros: np.ndarray,
    quats: np.ndarray,
    target_ms: float,
    max_dt_ms: float,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    if len(times_ms) == 0 or not np.isfinite(target_ms):
        return None, None, None, None
    pos = int(np.searchsorted(times_ms, target_ms))
    candidates: list[int] = []
    if pos < len(times_ms):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)
    best = min(candidates, key=lambda idx: abs(float(times_ms[idx]) - target_ms))
    if abs(float(times_ms[best]) - target_ms) > max_dt_ms:
        return None, None, None, None
    return accels[best].copy(), gyros[best].copy(), quats[best].copy(), np.array([best])


def sample_imu_for_rows(
    rows: list[dict[str, str]],
    times_ms: np.ndarray,
    accels: np.ndarray,
    gyros: np.ndarray,
    stream_quats: np.ndarray,
    sync_field: str,
    max_dt_ms: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sampled_q = np.zeros((len(rows), 4), dtype=np.float64)
    sampled_a = np.zeros((len(rows), 3), dtype=np.float64)
    sampled_g = np.zeros((len(rows), 3), dtype=np.float64)
    valid = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        try:
            target_ms = float(row[sync_field])
        except (KeyError, TypeError, ValueError):
            continue
        accel, gyro, quat, _ = nearest_imu_sample(
            times_ms, accels, gyros, stream_quats, target_ms, max_dt_ms
        )
        if quat is None or accel is None or gyro is None:
            continue
        sampled_q[index] = quat
        sampled_a[index] = accel
        sampled_g[index] = gyro
        valid[index] = True
    return sampled_q, sampled_a, sampled_g, valid


def rotation_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    a = source / max(np.linalg.norm(source), 1e-12)
    b = target / max(np.linalg.norm(target), 1e-12)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        v = np.cross(a, axis)
        v /= max(np.linalg.norm(v), 1e-12)
        vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
        return np.eye(3) + 2.0 * (vx @ vx)
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + (vx @ vx) * (1.0 / (1.0 + c))


def fit_gravity_alignment(
    mahony_q: np.ndarray,
    accels: np.ndarray,
    gyros: np.ndarray,
    valid_mask: np.ndarray,
    calibration_frames: int,
) -> tuple[np.ndarray, dict]:
    calib = np.zeros(len(mahony_q), dtype=bool)
    calib[: min(calibration_frames, len(mahony_q))] = True
    calib &= valid_mask
    gyro_norm = np.linalg.norm(gyros, axis=1)
    if np.any(calib):
        threshold = float(np.percentile(gyro_norm[calib], 35.0))
        still = calib & (gyro_norm <= threshold)
    else:
        still = np.zeros(len(mahony_q), dtype=bool)
    if np.count_nonzero(still) < 20:
        still = calib.copy()
    if np.count_nonzero(still) < 10:
        raise RuntimeError("Not enough still IMU samples for gravity alignment")

    gravity_sensor = np.mean(accels[still], axis=0)
    gravity_sensor /= max(np.linalg.norm(gravity_sensor), 1e-12)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    r_sensor_to_world = rotation_from_vectors(gravity_sensor, world_up)
    meta = {
        "method": "gravity_y_up",
        "calibration_frames": calibration_frames,
        "still_samples": int(np.count_nonzero(still)),
        "gravity_sensor_unit": gravity_sensor.tolist(),
        "world_up": world_up.tolist(),
    }
    return r_sensor_to_world, meta


def quats_to_world(
    mahony_q: np.ndarray,
    r_sensor_to_world: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[list[list[float] | None], list[bool]]:
    out: list[list[float] | None] = []
    valid_flags: list[bool] = []
    for q_sensor, ok in zip(mahony_q, valid_mask):
        if not ok:
            out.append(None)
            valid_flags.append(False)
            continue
        r_sensor = q_to_matrix(np.asarray(q_sensor, dtype=np.float64))
        q_world = matrix_to_q((r_sensor_to_world @ r_sensor).reshape(1, 3, 3))[0]
        out.append([float(x) for x in q_normalize(q_world)])
        valid_flags.append(True)
    return out, valid_flags


def run_mahony(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    est = MahonyEstimator()
    quats = np.zeros((len(rows), 4), dtype=np.float64)
    times = np.zeros(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        ts = imu_timestamp_ms(row)
        times[index] = ts
        try:
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
        except (KeyError, ValueError, TypeError):
            quats[index] = np.array([1.0, 0.0, 0.0, 0.0])
            continue
        if not np.all(np.isfinite(accel)) or not np.all(np.isfinite(gyro)) or not np.isfinite(ts):
            quats[index] = quats[index - 1] if index else np.array([1.0, 0.0, 0.0, 0.0])
            continue
        quats[index] = est.update(accel, gyro, ts)
    return quats, times


def fit_imu_to_target(
    rows: list[dict[str, str]],
    mahony_q: np.ndarray,
    times: np.ndarray,
    prefix: str,
    report: dict,
    calibration_frames: int,
) -> tuple[np.ndarray, dict]:
    valid_idx = [i for i, row in enumerate(rows[:calibration_frames]) if row_valid(row, prefix)]
    if len(valid_idx) < 30:
        raise RuntimeError(f"Not enough valid calibration frames for {prefix}: {len(valid_idx)}")

    target_q = np.stack([row_quat(rows[i], prefix) for i in valid_idx], axis=0)
    sensor_q = mahony_q[valid_idx]
    rel_mats = q_to_matrix(q_mul(target_q, q_inv(sensor_q)))
    r_sensor_to_world = average_rotation_matrices(rel_mats)

    target_t = np.array([float(rows[i]["mocap_time_sec"]) for i in valid_idx], dtype=np.float64)
    target_gyro = angular_velocity_from_quat(target_q, target_t)
    sensor_gyro = np.stack(
        [
            np.array(
                [
                    float(rows[i]["head_imu_gx_rad_s"]),
                    float(rows[i]["head_imu_gy_rad_s"]),
                    float(rows[i]["head_imu_gz_rad_s"]),
                ],
                dtype=np.float64,
            )
            for i in valid_idx
        ],
        axis=0,
    )
    head_align = report.get("head_imu_mocap_alignment", {})
    r_gyro = np.asarray(head_align.get("rotation_mocap_body_gyro_to_head_imu"), dtype=np.float64)
    if r_gyro.shape != (3, 3):
        r_gyro = fit_rotation(target_gyro, sensor_gyro)
    else:
        r_gyro = fit_rotation(target_gyro, sensor_gyro @ r_gyro.T)

    meta = {
        "target_prefix": prefix,
        "calibration_frames": calibration_frames,
        "calibration_samples": len(valid_idx),
        "gyro_vector_corr_after_fit": float(
            np.corrcoef(
                (target_gyro @ r_gyro).reshape(-1),
                sensor_gyro.reshape(-1),
            )[0, 1]
        )
        if len(valid_idx) > 2
        else 0.0,
        "head_imu_mocap_score": head_align.get("score"),
    }
    return r_sensor_to_world, meta


def estimate_world_quat(
    rows: list[dict[str, str]],
    mahony_q: np.ndarray,
    r_sensor_to_world: np.ndarray,
    prefix: str,
) -> tuple[list[list[float] | None], list[bool]]:
    out: list[list[float] | None] = []
    valid_flags: list[bool] = []
    for row, q_sensor in zip(rows, mahony_q):
        if not row_valid(row, prefix):
            out.append(None)
            valid_flags.append(False)
            continue
        r_sensor = q_to_matrix(np.asarray(q_sensor, dtype=np.float64))
        if r_sensor.shape == (3, 3):
            r_world = r_sensor_to_world @ r_sensor
        else:
            r_world = r_sensor_to_world @ r_sensor[0]
        q_world = matrix_to_q(r_world.reshape(1, 3, 3))[0]
        out.append([float(x) for x in q_normalize(q_world)])
        valid_flags.append(True)
    return out, valid_flags


def main() -> None:
    args = parse_args()
    rows = read_aligned_rows(args.aligned_csv)
    report = json.loads(args.aligned_report.read_text(encoding="utf-8"))
    playback = json.loads(args.playback.read_text(encoding="utf-8"))
    seqs = playback["seqs"]
    frame_count = playback["frame_count"]

    if args.left_imu_csv is not None:
        left_times, left_accels, left_gyros = read_imu_csv(args.left_imu_csv)
        left_stream_q = run_mahony_stream(left_times, left_accels, left_gyros)
        left_q, left_row_accels, left_row_gyros, left_valid_mask = sample_imu_for_rows(
            rows,
            left_times,
            left_accels,
            left_gyros,
            left_stream_q,
            args.sync_field,
            args.sync_max_dt_ms,
        )
        if args.calibration == "gravity_y_up":
            r_left, left_meta = fit_gravity_alignment(
                left_q,
                left_row_accels,
                left_row_gyros,
                left_valid_mask,
                args.calibration_frames,
            )
        else:
            raise RuntimeError("Wrist IMU CSV mode currently supports gravity_y_up only")
        left_all, _ = quats_to_world(left_q, r_left, left_valid_mask)
        left_meta["imu_csv"] = str(args.left_imu_csv)

        if args.right_imu_csv is not None:
            right_times, right_accels, right_gyros = read_imu_csv(args.right_imu_csv)
            right_stream_q = run_mahony_stream(right_times, right_accels, right_gyros)
            right_q, right_row_accels, right_row_gyros, right_valid_mask = sample_imu_for_rows(
                rows,
                right_times,
                right_accels,
                right_gyros,
                right_stream_q,
                args.sync_field,
                args.sync_max_dt_ms,
            )
            r_right, right_meta = fit_gravity_alignment(
                right_q,
                right_row_accels,
                right_row_gyros,
                right_valid_mask,
                args.calibration_frames,
            )
            right_all, _ = quats_to_world(right_q, r_right, right_valid_mask)
            right_meta["imu_csv"] = str(args.right_imu_csv)
        else:
            right_all = left_all
            right_meta = dict(left_meta)
            right_meta["mirrored_from"] = "left"

        method = "mahony_wrist_imu_gravity_y_up"
        imu_source = f"{args.left_imu_csv.name} + {args.right_imu_csv.name if args.right_imu_csv else 'mirror'}"
        notes = (
            "0810 line1 wrist modules (module02/module03) Mahony AHRS, synced to aligned frames "
            f"via {args.sync_field}, gravity-aligned to mocap Y-up world."
        )
    else:
        mahony_q, _times = run_mahony(rows)
        right_target = args.right_target.strip() or args.left_target
        r_left, left_meta = fit_imu_to_target(
            rows, mahony_q, _times, args.left_target, report, args.calibration_frames
        )
        if right_target == args.left_target:
            r_right = r_left
            right_meta = dict(left_meta)
            right_meta["mirrored_from"] = args.left_target
        else:
            r_right, right_meta = fit_imu_to_target(
                rows, mahony_q, _times, right_target, report, args.calibration_frames
            )
        left_all, _ = estimate_world_quat(rows, mahony_q, r_left, args.left_target)
        right_all, _ = estimate_world_quat(rows, mahony_q, r_right, right_target)
        method = "mahony_head_imu_aligned_to_mocap_rigid"
        imu_source = "aligned_30hz head_imu (module01 recording)"
        notes = (
            "Head IMU only; wrist overlay calibrates Mahony orientation to the chosen mocap rigid "
            f"({args.left_target}) and draws IMU axes at skeleton wrist joints."
        )

    left_by_seq = {int(row["seq"]): left_all[i] for i, row in enumerate(rows)}
    right_by_seq = {int(row["seq"]): right_all[i] for i, row in enumerate(rows)}
    left_valid_by_seq = {int(row["seq"]): left_all[i] is not None for i, row in enumerate(rows)}
    right_valid_by_seq = {int(row["seq"]): right_all[i] is not None for i, row in enumerate(rows)}

    left_playback: list[list[float] | None] = []
    right_playback: list[list[float] | None] = []
    left_valid: list[bool] = []
    right_valid: list[bool] = []
    for seq in seqs:
        seq_i = int(seq)
        left_playback.append(left_by_seq.get(seq_i))
        right_playback.append(right_by_seq.get(seq_i))
        left_valid.append(left_valid_by_seq.get(seq_i, False))
        right_valid.append(right_valid_by_seq.get(seq_i, False))

    payload = {
        "schema": "joint_projection.imu_wrist_overlay.v1",
        "method": method,
        "imu_source": imu_source,
        "axis_length_m": 0.08,
        "frame_count": frame_count,
        "seqs": seqs,
        "quat_wxyz": {
            "left": left_playback,
            "right": right_playback,
        },
        "valid": {
            "left": left_valid,
            "right": right_valid,
        },
        "calibration": {
            "left": left_meta,
            "right": right_meta,
        },
        "notes": notes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "frames": frame_count,
                "left_valid": sum(left_valid),
                "right_valid": sum(right_valid),
                "calibration": payload["calibration"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
