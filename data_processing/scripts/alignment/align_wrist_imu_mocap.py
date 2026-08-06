"""
Match a wrist IMU gyro segment against mocap wrist angular velocity.

Default example:
  - Dataset: C:\\Users\\hand\\Desktop\\Dataset\\0714\\002
  - IMU segment: module02/module03 rows 700..1600
  - Mocap: fbx_mocap_csv\\wrist_mocap_derivatives_wide.csv

The primary matching signal is gyro norm, because it is invariant to different
IMU/mocap coordinate frames. After the best time offset is found, the script
also estimates signed axis correspondence from the per-axis gyro correlation.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np


DEFAULT_ROOT = Path(r"C:\Users\hand\Desktop\Dataset\0714\002")
DEFAULT_MOCAP = DEFAULT_ROOT / "fbx_mocap_csv" / "wrist_mocap_derivatives_wide.csv"
DEFAULT_OUTDIR = DEFAULT_ROOT / "sync_analysis"
MODULES = {
    "module02": DEFAULT_ROOT / "module02_13652E00_imu.csv",
    "module03": DEFAULT_ROOT / "module03_41782E00_imu.csv",
}
WRISTS = ("LeftHand", "RightHand")
AXES = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--mocap", default=str(DEFAULT_MOCAP))
    parser.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    parser.add_argument("--start-row", type=int, default=700)
    parser.add_argument("--end-row", type=int, default=1600)
    parser.add_argument("--smooth-sec", type=float, default=0.12)
    parser.add_argument("--candidate-step", type=int, default=1)
    return parser.parse_args()


def read_imu(path: Path) -> dict[str, np.ndarray]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "t": float(row["gyro_device_ts_ms"]) / 1000.0,
                    "gx": float(row["gx_rad_s"]),
                    "gy": float(row["gy_rad_s"]),
                    "gz": float(row["gz_rad_s"]),
                }
            )
    t0 = rows[0]["t"]
    arr = {k: np.array([r[k] for r in rows], dtype=float) for k in ("t", "gx", "gy", "gz")}
    arr["t"] = arr["t"] - t0
    arr["norm"] = np.sqrt(arr["gx"] ** 2 + arr["gy"] ** 2 + arr["gz"] ** 2)
    return arr


def read_mocap(path: Path) -> dict[str, dict[str, np.ndarray]]:
    data = {w: {} for w in WRISTS}
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    t = np.array([float(r["time_sec"]) for r in rows], dtype=float)
    for wrist in WRISTS:
        data[wrist]["t"] = t
        for axis in AXES:
            data[wrist][f"g{axis}"] = np.array(
                [float(r[f"{wrist}_gyro_{axis}_rad_s"]) for r in rows], dtype=float
            )
        data[wrist]["norm"] = np.array(
            [float(r[f"{wrist}_gyro_norm_rad_s"]) for r in rows], dtype=float
        )
    return data


def moving_average(x: np.ndarray, samples: int) -> np.ndarray:
    if samples <= 1:
        return x.copy()
    kernel = np.ones(samples, dtype=float) / samples
    pad = samples // 2
    padded = np.pad(x, (pad, samples - 1 - pad), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    std = float(np.std(x))
    if std < 1e-12:
        return x * 0.0
    return (x - float(np.mean(x))) / std


def corr(a: np.ndarray, b: np.ndarray) -> float:
    za = zscore(a)
    zb = zscore(b)
    return float(np.mean(za * zb))


def segment_imu(imu: dict[str, np.ndarray], start: int, end: int) -> dict[str, np.ndarray]:
    end = min(end, len(imu["t"]) - 1)
    sl = slice(start, end + 1)
    out = {k: v[sl].copy() for k, v in imu.items()}
    out["t"] = out["t"] - out["t"][0]
    return out


def match_norm(
    imu_seg: dict[str, np.ndarray],
    mocap_wrist: dict[str, np.ndarray],
    smooth_sec: float,
    candidate_step: int,
) -> dict:
    imu_t = imu_seg["t"]
    duration = float(imu_t[-1])
    imu_dt = float(np.median(np.diff(imu_t)))
    mocap_t = mocap_wrist["t"]
    mocap_dt = float(np.median(np.diff(mocap_t)))
    imu_smooth_n = max(1, int(round(smooth_sec / imu_dt)))
    mocap_smooth_n = max(1, int(round(smooth_sec / mocap_dt)))
    imu_norm = moving_average(imu_seg["norm"], imu_smooth_n)
    mocap_norm_full = moving_average(mocap_wrist["norm"], mocap_smooth_n)

    max_start = mocap_t[-1] - duration
    candidates = np.arange(0, np.searchsorted(mocap_t, max_start), max(1, candidate_step))
    best = {"corr": -999.0, "mocap_start_index": 0, "mocap_start_time": 0.0, "mocap_values": None}
    for idx in candidates:
        start_t = mocap_t[int(idx)]
        sample_t = start_t + imu_t
        values = np.interp(sample_t, mocap_t, mocap_norm_full)
        c = corr(imu_norm, values)
        if c > best["corr"]:
            best = {
                "corr": c,
                "mocap_start_index": int(idx),
                "mocap_start_time": float(start_t),
                "mocap_values": values,
            }
    best["duration_sec"] = duration
    best["imu_dt_sec"] = imu_dt
    best["mocap_dt_sec"] = mocap_dt
    best["imu_samples"] = len(imu_t)
    best["smooth_sec"] = smooth_sec
    best["imu_norm_smooth"] = imu_norm
    return best


def axis_corr_matrix(imu_seg: dict[str, np.ndarray], mocap_wrist: dict[str, np.ndarray], start_t: float) -> np.ndarray:
    imu_t = imu_seg["t"]
    mocap_t = mocap_wrist["t"]
    mat = np.zeros((3, 3), dtype=float)
    for i, ia in enumerate(("gx", "gy", "gz")):
        iv = imu_seg[ia]
        for j, ma in enumerate(("gx", "gy", "gz")):
            mv = np.interp(start_t + imu_t, mocap_t, mocap_wrist[ma])
            mat[i, j] = corr(iv, mv)
    return mat


def best_signed_permutation(mat: np.ndarray) -> dict:
    best = {"score": -999.0, "mapping": []}
    for perm in itertools.permutations(range(3)):
        total = 0.0
        mapping = []
        for i, j in enumerate(perm):
            c = mat[i, j]
            sign = 1 if c >= 0 else -1
            total += abs(c)
            mapping.append((AXES[i], AXES[j], sign, c))
        score = total / 3.0
        if score > best["score"]:
            best = {"score": float(score), "mapping": mapping}
    return best


def write_overlay_csv(path: Path, module: str, wrist: str, imu_seg, mocap_wrist, best) -> None:
    fields = [
        "module",
        "wrist",
        "imu_segment_sample",
        "imu_t_sec",
        "mocap_t_sec",
        "imu_gyro_norm_rad_s",
        "mocap_gyro_norm_rad_s",
        "imu_gx_rad_s",
        "imu_gy_rad_s",
        "imu_gz_rad_s",
        "mocap_gx_rad_s",
        "mocap_gy_rad_s",
        "mocap_gz_rad_s",
    ]
    mocap_t = mocap_wrist["t"]
    sample_t = best["mocap_start_time"] + imu_seg["t"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, t in enumerate(imu_seg["t"]):
            row = {
                "module": module,
                "wrist": wrist,
                "imu_segment_sample": i,
                "imu_t_sec": f"{t:.9f}",
                "mocap_t_sec": f"{sample_t[i]:.9f}",
                "imu_gyro_norm_rad_s": f"{imu_seg['norm'][i]:.9f}",
                "mocap_gyro_norm_rad_s": f"{np.interp(sample_t[i], mocap_t, mocap_wrist['norm']):.9f}",
                "imu_gx_rad_s": f"{imu_seg['gx'][i]:.9f}",
                "imu_gy_rad_s": f"{imu_seg['gy'][i]:.9f}",
                "imu_gz_rad_s": f"{imu_seg['gz'][i]:.9f}",
                "mocap_gx_rad_s": f"{np.interp(sample_t[i], mocap_t, mocap_wrist['gx']):.9f}",
                "mocap_gy_rad_s": f"{np.interp(sample_t[i], mocap_t, mocap_wrist['gy']):.9f}",
                "mocap_gz_rad_s": f"{np.interp(sample_t[i], mocap_t, mocap_wrist['gz']):.9f}",
            }
            w.writerow(row)


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    modules = {
        "module02": root / "module02_13652E00_imu.csv",
        "module03": root / "module03_41782E00_imu.csv",
    }
    mocap = read_mocap(Path(args.mocap))

    report_rows = []
    json_report = []
    best_by_module = {}
    for module, imu_path in modules.items():
        imu = read_imu(imu_path)
        imu_seg = segment_imu(imu, args.start_row, args.end_row)
        for wrist in WRISTS:
            best = match_norm(imu_seg, mocap[wrist], args.smooth_sec, args.candidate_step)
            mat = axis_corr_matrix(imu_seg, mocap[wrist], best["mocap_start_time"])
            perm = best_signed_permutation(mat)
            row = {
                "module": module,
                "imu_file": str(imu_path),
                "imu_start_row": args.start_row,
                "imu_end_row": min(args.end_row, len(imu["t"]) - 1),
                "imu_start_device_t_sec_rel": float(imu["t"][args.start_row]),
                "imu_segment_duration_sec": best["duration_sec"],
                "imu_samples": best["imu_samples"],
                "wrist": wrist,
                "mocap_start_frame": best["mocap_start_index"],
                "mocap_start_time_sec": best["mocap_start_time"],
                "norm_corr": best["corr"],
                "axis_mapping_score_mean_abs_corr": perm["score"],
                "axis_mapping": "; ".join(
                    f"imu_g{ia} ~= {'+' if sign > 0 else '-'}mocap_g{ma} corr={c:.3f}"
                    for ia, ma, sign, c in perm["mapping"]
                ),
                "axis_corr_matrix_rows_imu_xyz_cols_mocap_xyz": json.dumps(mat.tolist()),
                "imu_dt_ms_median": best["imu_dt_sec"] * 1000.0,
                "mocap_dt_ms_median": best["mocap_dt_sec"] * 1000.0,
            }
            report_rows.append(row)
            json_report.append(row.copy())
            if module not in best_by_module or row["norm_corr"] > best_by_module[module]["row"]["norm_corr"]:
                best_by_module[module] = {"row": row, "imu_seg": imu_seg, "mocap_wrist": mocap[wrist], "best": best}

    report_path = outdir / f"imu_mocap_wrist_match_rows_{args.start_row}_{args.end_row}.csv"
    with report_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)

    overlays = []
    for module, payload in best_by_module.items():
        row = payload["row"]
        overlay_path = outdir / f"{module}_{row['wrist']}_best_overlay_rows_{args.start_row}_{args.end_row}.csv"
        write_overlay_csv(
            overlay_path,
            module,
            row["wrist"],
            payload["imu_seg"],
            payload["mocap_wrist"],
            payload["best"],
        )
        overlays.append(str(overlay_path))

    json_path = outdir / f"imu_mocap_wrist_match_rows_{args.start_row}_{args.end_row}.json"
    json_path.write_text(
        json.dumps(
            {
                "report_csv": str(report_path),
                "overlay_csv": overlays,
                "method": "gyro_norm sliding-window Pearson correlation; per-axis correlation after best norm match",
                "rows": json_report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"report_csv={report_path}")
    print(f"json={json_path}")
    for module, payload in best_by_module.items():
        row = payload["row"]
        print(
            f"{module}: best {row['wrist']} corr={row['norm_corr']:.4f} "
            f"mocap_start={row['mocap_start_time_sec']:.3f}s "
            f"duration={row['imu_segment_duration_sec']:.3f}s "
            f"axis_score={row['axis_mapping_score_mean_abs_corr']:.3f}"
        )


if __name__ == "__main__":
    main()
