#!/usr/bin/env python3
"""Independently validate module01 head IMU timing against CH3_08 and BVH Head."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATASET = Path(r"C:\Users\hand\Desktop\Dataset\0722\record_9cam_0711_021044")
MOCAP = Path(r"C:\Users\hand\Desktop\Dataset\0722\abx2_mocap_rigid_csv\mocap_rigid_20260722.csv")
ALIGN_REPORT = DATASET / "aligned_data" / "aligned_30hz_report.json"
PREPROCESS = Path(r"C:\Users\hand\Desktop\Dataset\tools\preprocess_9cam_imu_mocap.py")
GLOBAL_ALIGN = Path(r"C:\Users\hand\Desktop\Dataset\tools\global_imu_mocap_alignment.py")
OUTPUT = Path(__file__).resolve().parent / "validation_0722_head_imu_sync"
TARGETS = ("CH3_08_Rigid_K", "Head")


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_camera_origin_ms() -> tuple[float, dict[str, float]]:
    first: dict[str, float] = {}
    with (DATASET / "timestamps.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["module"] == "1" and row["camera"] not in first:
                first[row["camera"]] = float(row["device_ts_ms"])
    return float(np.median(list(first.values()))), first


def normalized_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def evaluate_lag(
    delta: float,
    imu_t: np.ndarray,
    imu_gyro: np.ndarray,
    mocap_t: np.ndarray,
    mocap_gyro: np.ndarray,
) -> float:
    target_t = imu_t + delta
    valid = (target_t >= mocap_t[0]) & (target_t <= mocap_t[-1])
    sampled = np.stack(
        [np.interp(target_t[valid], mocap_t, mocap_gyro[:, axis]) for axis in range(3)], axis=1
    )
    imu_speed = np.linalg.norm(imu_gyro[valid], axis=1)
    mocap_speed = np.linalg.norm(sampled, axis=1)
    active = np.maximum(imu_speed, mocap_speed) >= np.percentile(
        np.maximum(imu_speed, mocap_speed), 45.0
    )
    return normalized_corr(imu_speed[active], mocap_speed[active])


def main() -> int:
    prep = load_module(PREPROCESS, "head_sync_prep")
    align = load_module(GLOBAL_ALIGN, "head_sync_align")
    report = json.loads(ALIGN_REPORT.read_text(encoding="utf-8"))
    scale = float(report["global_scale"])
    global_offset = float(report["alignment_parameters"]["global_mocap_offset_sec"])
    origin_ms, first_exposures = first_camera_origin_ms()
    imu = prep.read_imu(DATASET / "module01_D45D2E00_imu.csv")
    elapsed = (imu["gyro_device_ts_ms"] - origin_ms) / 1000.0
    imu_mocap_t = global_offset + scale * elapsed
    imu_gyro = np.stack((imu["gx_rad_s"], imu["gy_rad_s"], imu["gz_rad_s"]), axis=1)

    usecols = ["time_sec"]
    for target in TARGETS:
        usecols.extend(f"{target}_world_q{axis}" for axis in "wxyz")
    columns: dict[str, list[float]] = {name: [] for name in usecols}
    with MOCAP.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            for name in usecols:
                columns[name].append(float(row[name]))
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in columns.items()}
    mocap_t = arrays["time_sec"]
    deltas = np.arange(-0.250, 0.2501, 0.001)
    results: dict[str, object] = {}
    curves: dict[str, np.ndarray] = {}

    for target in TARGETS:
        quaternion = np.column_stack(
            [arrays[f"{target}_world_q{axis}"] for axis in "wxyz"]
        )
        mocap_gyro = align.body_angular_velocity(quaternion, mocap_t)
        scores = np.asarray(
            [evaluate_lag(delta, imu_mocap_t, imu_gyro, mocap_t, mocap_gyro) for delta in deltas]
        )
        best_delta = float(deltas[int(np.argmax(scores))])
        curves[target] = scores

        mapped = imu_mocap_t + best_delta
        valid = (mapped >= mocap_t[0]) & (mapped <= mocap_t[-1])
        sampled = np.stack(
            [np.interp(mapped[valid], mocap_t, mocap_gyro[:, axis]) for axis in range(3)], axis=1
        )
        measured = imu_gyro[valid]
        active = np.linalg.norm(measured, axis=1) >= np.percentile(
            np.linalg.norm(measured, axis=1), 45.0
        )
        rotation = align.fit_rotation(sampled[active], measured[active])
        predicted = sampled @ rotation
        vector_corr = align.flat_correlation(predicted[active], measured[active])
        axis_corr = [
            float(np.corrcoef(predicted[active, axis], measured[active, axis])[0, 1])
            for axis in range(3)
        ]

        local_lags: list[float] = []
        window_sec = 40.0
        for start in np.arange(max(mapped[valid][0], mocap_t[0]), min(mapped[valid][-1], mocap_t[-1]) - window_sec, 40.0):
            mask = valid & (imu_mocap_t + best_delta >= start) & (imu_mocap_t + best_delta < start + window_sec)
            if np.count_nonzero(mask) < 500:
                continue
            local_grid = np.arange(best_delta - 0.030, best_delta + 0.0301, 0.001)
            local_scores = [
                evaluate_lag(delta, imu_mocap_t[mask], imu_gyro[mask], mocap_t, mocap_gyro)
                for delta in local_grid
            ]
            local_lags.append(float(local_grid[int(np.argmax(local_scores))]))

        results[target] = {
            "additional_delta_sec": best_delta,
            "effective_global_offset_sec": global_offset + best_delta,
            "magnitude_corr": float(np.max(scores)),
            "vector_corr_after_axis_fit": float(vector_corr),
            "axis_corr": axis_corr,
            "local_lag_count": len(local_lags),
            "local_additional_delta_ms_median": float(np.median(local_lags) * 1000.0),
            "local_additional_delta_ms_p90_abs_from_global": float(
                np.percentile(np.abs(np.asarray(local_lags) - best_delta), 90) * 1000.0
            ),
            "imu_to_mocap_rotation": rotation.tolist(),
        }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    out = {
        "mapping_before_head_validation": "mocap_time = global_offset + global_scale * ((module01_imu_ts - module01_camera_origin) / 1000)",
        "global_scale": scale,
        "global_offset_sec": global_offset,
        "module01_camera_origin_ms": origin_ms,
        "module01_first_camera_device_ts_ms": first_exposures,
        "results": results,
    }
    (OUTPUT / "head_imu_mocap_sync_report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    fig, axis = plt.subplots(figsize=(10, 5.2), dpi=150)
    for target, scores in curves.items():
        axis.plot(deltas * 1000.0, scores, label=target)
        best = int(np.argmax(scores))
        axis.scatter([deltas[best] * 1000.0], [scores[best]], s=28)
    axis.axvline(0.0, color="0.45", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Additional offset after wrist-derived global mapping (ms)")
    axis.set_ylabel("Angular-speed correlation")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "head_imu_mocap_lag_curve.png")
    plt.close(fig)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
