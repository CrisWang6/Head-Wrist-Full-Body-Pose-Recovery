import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


WRIST_MODULES = ("module02", "module03")
WRISTS = ("LeftHand", "RightHand")
CAMERA_ALIGNMENT_TS_FIELD = "device_ts_ms"


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return values.astype(np.float64, copy=True)
    width = min(width, len(values))
    kernel = np.ones(width, dtype=np.float64) / width
    left = width // 2
    right = width - 1 - left
    return np.convolve(np.pad(values, (left, right), mode="edge"), kernel, mode="valid")


def normalize_quaternions(q: np.ndarray) -> np.ndarray:
    q = q.astype(np.float64, copy=True)
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
    signs = np.ones(len(q), dtype=np.float64)
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            signs[i] = -1.0
            q[i] *= -1.0
    return q


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def quat_inverse(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[:, 1:] *= -1.0
    return out


def body_angular_velocity(q: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return angular velocity in the moving wrist/body coordinate frame."""
    q = normalize_quaternions(q)
    rel = quat_multiply(quat_inverse(q[:-1]), q[1:])
    rel /= np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-12)
    rel[rel[:, 0] < 0] *= -1.0
    w = np.clip(rel[:, 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(np.maximum(0.0, 1.0 - w * w))
    dt = np.diff(t)
    omega = np.zeros((len(rel), 3), dtype=np.float64)
    valid = (dt > 1e-12) & (s > 1e-10) & (angle > 1e-12)
    omega[valid] = rel[valid, 1:] / s[valid, None] * (angle[valid] / dt[valid])[:, None]
    out = np.empty((len(q), 3), dtype=np.float64)
    out[0] = omega[0]
    out[-1] = omega[-1]
    out[1:-1] = 0.5 * (omega[:-1] + omega[1:])
    return out


def load_preprocess_module(script: Path):
    spec = importlib.util.spec_from_file_location("preprocess_alignment", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_camera_anchored_imus(dataset: Path, preprocess_script: Path) -> dict:
    prep = load_preprocess_module(preprocess_script)
    raw_imus = {
        module: prep.read_imu(next(dataset.glob(f"{module}_*_imu.csv")))
        for module in WRIST_MODULES
    }
    first_exposures = {module: {} for module in WRIST_MODULES}
    with (dataset / "timestamps.csv").open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            module = f"module{int(row['module']):02d}"
            camera = row["camera"]
            if module in first_exposures and camera not in first_exposures[module]:
                first_exposures[module][camera] = float(row[CAMERA_ALIGNMENT_TS_FIELD])
    origins = {}
    for module in WRIST_MODULES:
        values = [first_exposures[module][camera] for camera in ("CAM_A", "CAM_B", "CAM_C")]
        origins[module] = float(np.median(values))

    out = {
        "camera_time_origins_ms": origins,
        "first_camera_device_ts_ms": first_exposures,
        "valid_rows": {},
    }
    for module in WRIST_MODULES:
        imu = raw_imus[module]
        vec = np.stack(
            (imu["gx_rad_s"], imu["gy_rad_s"], imu["gz_rad_s"]),
            axis=1,
        )
        elapsed = (imu["gyro_device_ts_ms"] - origins[module]) / 1000.0
        valid = np.isfinite(elapsed) & np.all(np.isfinite(vec), axis=1)
        valid &= np.r_[True, np.diff(elapsed) > 0]
        valid &= elapsed >= -1.0
        out[f"{module}_time_sec"] = elapsed[valid]
        out[module] = vec[valid]
        out["valid_rows"][module] = int(np.sum(valid))
    return out


def load_mocap_body_gyro(mocap_wide: Path, cache_csv: Path) -> dict:
    frame = None
    if cache_csv.exists():
        cached = pd.read_csv(cache_csv)
        required = {"time_sec"}
        for wrist in WRISTS:
            required.update(f"{wrist}_body_g{axis}_rad_s" for axis in ("x", "y", "z"))
        if required.issubset(cached.columns):
            frame = cached
    if frame is None:
        usecols = ["frame_index", "time_sec"]
        for wrist in WRISTS:
            usecols += [f"{wrist}_world_q{s}" for s in ("w", "x", "y", "z")]
        frame = pd.read_csv(mocap_wide, usecols=usecols)
        result = {
            "frame_index": frame["frame_index"].to_numpy(dtype=np.int64),
            "time_sec": frame["time_sec"].to_numpy(dtype=np.float64),
        }
        t = result["time_sec"]
        for wrist in WRISTS:
            q = frame[[f"{wrist}_world_q{s}" for s in ("w", "x", "y", "z")]].to_numpy(dtype=np.float64)
            gyro = body_angular_velocity(q, t)
            result[f"{wrist}_body_gx_rad_s"] = gyro[:, 0]
            result[f"{wrist}_body_gy_rad_s"] = gyro[:, 1]
            result[f"{wrist}_body_gz_rad_s"] = gyro[:, 2]
        frame = pd.DataFrame(result)
        frame.to_csv(cache_csv, index=False, encoding="utf-8-sig")

    out = {"time_sec": frame["time_sec"].to_numpy(dtype=np.float64)}
    for wrist in WRISTS:
        out[wrist] = frame[
            [f"{wrist}_body_gx_rad_s", f"{wrist}_body_gy_rad_s", f"{wrist}_body_gz_rad_s"]
        ].to_numpy(dtype=np.float64)
    return out


def interpolate_vectors(source_t: np.ndarray, source_v: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(targets, source_t, source_v[:, i]) for i in range(3)], axis=1)


def fit_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source - np.mean(source, axis=0)
    target = target - np.mean(target, axis=0)
    u, _s, vt = np.linalg.svd(source.T @ target)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def flat_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = a - np.mean(a, axis=0)
    b = b - np.mean(b, axis=0)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.sum(a * b) / denom) if denom > 1e-12 else 0.0


def smooth_vectors(vectors: np.ndarray, width: int) -> np.ndarray:
    return np.stack([moving_average(vectors[:, i], width) for i in range(3)], axis=1)


def evaluate_vector_mapping(
    scale: float,
    offset: float,
    camera: dict,
    mocap: dict,
    pairing: dict[str, str],
    stride: int = 2,
    rotations: dict[str, np.ndarray] | None = None,
    fit_parity: int | None = None,
    score_parity: int | None = None,
) -> dict:
    results = {}
    scores = []
    fitted_rotations = {}
    valid_samples = []
    for module in WRIST_MODULES:
        t = camera[f"{module}_time_sec"][::stride]
        mapped_t = offset + scale * t
        valid = (mapped_t >= mocap["time_sec"][0]) & (mapped_t <= mocap["time_sec"][-1])
        if np.mean(valid) < 0.8:
            return {"score": -1.0}
        t = t[valid]
        mapped_t = mapped_t[valid]
        imu = smooth_vectors(camera[module][::stride][valid], 3)
        wrist = pairing[module]
        mocap_vec = interpolate_vectors(mocap["time_sec"], mocap[wrist], mapped_t)
        mocap_vec = smooth_vectors(mocap_vec, 3)
        active = np.linalg.norm(imu - np.median(imu, axis=0), axis=1)
        threshold = float(np.percentile(active, 45.0))
        active_mask = active >= threshold
        window_parity = np.floor(t / 60.0).astype(np.int64) % 2
        if fit_parity is not None:
            local_fit = (window_parity == fit_parity) & active_mask
        else:
            local_fit = active_mask
        if score_parity is not None:
            local_score = (window_parity == score_parity) & active_mask
        else:
            local_score = active_mask
        if np.sum(local_fit) < 100:
            return {"score": -1.0}
        if np.sum(local_score) < 100:
            return {"score": -1.0}
        rotation = rotations[module] if rotations is not None else fit_rotation(mocap_vec[local_fit], imu[local_fit])
        prediction = mocap_vec @ rotation
        score = flat_correlation(prediction[local_score], imu[local_score])
        axis_corr = []
        for axis in range(3):
            x = prediction[local_score, axis]
            y = imu[local_score, axis]
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                axis_corr.append(0.0)
            else:
                axis_corr.append(float(np.corrcoef(x, y)[0, 1]))
        fitted_rotations[module] = rotation
        results[module] = {
            "wrist": wrist,
            "vector_corr": score,
            "axis_corr": axis_corr,
            "active_samples": int(np.sum(local_score)),
        }
        scores.append(score)
        valid_samples.append(len(t))
    return {
        "score": float(np.mean(scores)),
        "scale": float(scale),
        "offset_sec": float(offset),
        "pairing": pairing,
        "valid_samples": {module: int(count) for module, count in zip(WRIST_MODULES, valid_samples)},
        "rotations": fitted_rotations,
        "modules": results,
    }


def high_pass_speed(vectors: np.ndarray, sample_rate: float) -> np.ndarray:
    speed = np.linalg.norm(vectors, axis=1)
    smooth = moving_average(speed, max(3, int(round(sample_rate * 0.12))))
    baseline = moving_average(smooth, max(5, int(round(sample_rate * 2.0))))
    signal = smooth - baseline
    limit = float(np.percentile(np.abs(signal), 99.5))
    return np.clip(signal, -limit, limit)


def scalar_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - np.mean(a)
    b = b - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


def coarse_search(camera: dict, mocap: dict, scale_center: float) -> list[dict]:
    stride = 6
    imu_signals = {
        module: high_pass_speed(
            camera[module],
            (len(camera[f"{module}_time_sec"]) - 1)
            / (camera[f"{module}_time_sec"][-1] - camera[f"{module}_time_sec"][0]),
        )[::stride]
        for module in WRIST_MODULES
    }
    mocap_rate = 1.0 / float(np.median(np.diff(mocap["time_sec"])))
    mocap_signals = {wrist: high_pass_speed(mocap[wrist], mocap_rate) for wrist in WRISTS}
    scale_min = max(0.90, scale_center - 0.025)
    scale_max = min(1.10, scale_center + 0.025)
    scales = np.arange(scale_min, scale_max + 0.0005, 0.001)
    offsets = np.arange(-20.0, 20.0001, 0.2)
    pairings = (
        {"module02": WRISTS[0], "module03": WRISTS[1]},
        {"module02": WRISTS[1], "module03": WRISTS[0]},
    )
    candidates = []
    for pairing in pairings:
        for scale in scales:
            for offset in offsets:
                scores = []
                for module in WRIST_MODULES:
                    camera_t = camera[f"{module}_time_sec"][::stride]
                    target = offset + scale * camera_t
                    valid = (target >= mocap["time_sec"][0]) & (target <= mocap["time_sec"][-1])
                    if np.mean(valid) < 0.85:
                        scores = []
                        break
                    sampled = np.interp(target[valid], mocap["time_sec"], mocap_signals[pairing[module]])
                    scores.append(scalar_corr(imu_signals[module][valid], sampled))
                if not scores:
                    continue
                candidates.append(
                    {
                        "score": float(np.mean(scores)),
                        "scale": float(scale),
                        "offset_sec": float(offset),
                        "pairing": pairing,
                        "module_scores": scores,
                    }
                )
    candidates.sort(key=lambda row: row["score"], reverse=True)
    selected = []
    for row in candidates:
        if all(
            row["pairing"] != old["pairing"]
            or abs(row["scale"] - old["scale"]) > 0.002
            or abs(row["offset_sec"] - old["offset_sec"]) > 0.8
            for old in selected
        ):
            selected.append(row)
        if len(selected) >= 24:
            break
    return selected


def refine_search(camera: dict, mocap: dict, coarse: list[dict]) -> dict:
    evaluated = []
    for seed in coarse:
        for scale in np.arange(seed["scale"] - 0.002, seed["scale"] + 0.0021, 0.001):
            for offset in np.arange(seed["offset_sec"] - 0.4, seed["offset_sec"] + 0.4001, 0.1):
                evaluated.append(evaluate_vector_mapping(scale, offset, camera, mocap, seed["pairing"], stride=3))
    best = max(evaluated, key=lambda row: row["score"])
    for scale_step, offset_step in ((0.0002, 0.02), (0.00005, 0.005)):
        local = []
        for ds in range(-5, 6):
            for do in range(-10, 11):
                local.append(
                    evaluate_vector_mapping(
                        best["scale"] + ds * scale_step,
                        best["offset_sec"] + do * offset_step,
                        camera,
                        mocap,
                        best["pairing"],
                        stride=2,
                    )
                )
        best = max(local, key=lambda row: row["score"])
    return best


def cross_validate(best: dict, camera: dict, mocap: dict) -> dict:
    train_even = evaluate_vector_mapping(
        best["scale"], best["offset_sec"], camera, mocap, best["pairing"], stride=1, fit_parity=0
    )
    train_odd = evaluate_vector_mapping(
        best["scale"], best["offset_sec"], camera, mocap, best["pairing"], stride=1, fit_parity=1
    )
    eval_odd = evaluate_vector_mapping(
        best["scale"],
        best["offset_sec"],
        camera,
        mocap,
        best["pairing"],
        stride=1,
        rotations=train_even["rotations"],
        score_parity=1,
    )
    eval_even = evaluate_vector_mapping(
        best["scale"],
        best["offset_sec"],
        camera,
        mocap,
        best["pairing"],
        stride=1,
        rotations=train_odd["rotations"],
        score_parity=0,
    )
    return {
        "fit_even_windows_eval_odd_windows": serialize_evaluation(eval_odd),
        "fit_odd_windows_eval_even_windows": serialize_evaluation(eval_even),
        "mean_held_rotation_score": float(np.mean([eval_odd["score"], eval_even["score"]])),
    }


def local_lag_validation(
    best: dict,
    camera: dict,
    mocap: dict,
    delta_radius_sec: float = 0.6,
    delta_step_sec: float = 0.02,
) -> list[dict]:
    full = evaluate_vector_mapping(
        best["scale"], best["offset_sec"], camera, mocap, best["pairing"], stride=1
    )
    rotations = full["rotations"]
    rows = []
    end_time = min(camera[f"{module}_time_sec"][-1] for module in WRIST_MODULES)
    for start in np.arange(0.0, end_time - 60.0, 60.0):
        camera_window = {}
        enough = True
        for module in WRIST_MODULES:
            t = camera[f"{module}_time_sec"]
            mask = (t >= start) & (t < start + 60.0)
            if np.sum(mask) < 1200:
                enough = False
                break
            camera_window[f"{module}_time_sec"] = t[mask]
            camera_window[module] = camera[module][mask]
        if not enough:
            continue
        motion = float(
            np.mean(
                [
                    np.percentile(np.linalg.norm(camera_window[module], axis=1), 90)
                    for module in WRIST_MODULES
                ]
            )
        )
        if motion < 0.25:
            continue
        best_local = None
        for delta in np.arange(
            -delta_radius_sec,
            delta_radius_sec + delta_step_sec * 0.5,
            delta_step_sec,
        ):
            result = evaluate_vector_mapping(
                best["scale"],
                best["offset_sec"] + delta,
                camera_window,
                mocap,
                best["pairing"],
                stride=1,
                rotations=rotations,
            )
            if best_local is None or result["score"] > best_local["score"]:
                best_local = result
                best_local["delta_sec"] = float(delta)
        if best_local is not None:
            rows.append(
                {
                    "window_start_sec": float(start),
                    "window_end_sec": float(start + 60.0),
                    "motion_p90_rad_s": motion,
                    "best_delta_sec": best_local["delta_sec"],
                    "vector_corr": best_local["score"],
                }
            )
    return rows


def refine_from_local_lag_trend(
    best: dict,
    camera: dict,
    mocap: dict,
    local_rows: list[dict],
) -> tuple[dict, dict]:
    if len(local_rows) < 4:
        return best, {"applied": False, "reason": "fewer than four valid local windows"}

    centers = np.array(
        [(row["window_start_sec"] + row["window_end_sec"]) * 0.5 for row in local_rows],
        dtype=np.float64,
    )
    deltas = np.array([row["best_delta_sec"] for row in local_rows], dtype=np.float64)
    slope, intercept = np.polyfit(centers, deltas, 1)
    seed_scale = float(best["scale"] + slope)
    seed_offset = float(best["offset_sec"] + intercept)

    stages = (
        (2e-5, 0.004, 5, 8),
        (5e-6, 0.001, 4, 5),
        (1e-6, 0.00025, 3, 4),
    )
    stage_rows = []
    refined = None
    for scale_step, offset_step, scale_radius, offset_radius in stages:
        candidates = []
        for ds in range(-scale_radius, scale_radius + 1):
            for do in range(-offset_radius, offset_radius + 1):
                candidates.append(
                    evaluate_vector_mapping(
                        seed_scale + ds * scale_step,
                        seed_offset + do * offset_step,
                        camera,
                        mocap,
                        best["pairing"],
                        stride=1,
                    )
                )
        refined = max(candidates, key=lambda row: row["score"])
        seed_scale = refined["scale"]
        seed_offset = refined["offset_sec"]
        stage_rows.append(
            {
                "scale_step": scale_step,
                "offset_step_sec": offset_step,
                "best_scale": seed_scale,
                "best_offset_sec": seed_offset,
                "best_score": refined["score"],
            }
        )

    return refined, {
        "applied": True,
        "initial_scale": best["scale"],
        "initial_offset_sec": best["offset_sec"],
        "initial_score_stride2": best["score"],
        "local_delta_linear_slope_sec_per_sec": float(slope),
        "local_delta_linear_intercept_sec": float(intercept),
        "trend_seed_scale": float(best["scale"] + slope),
        "trend_seed_offset_sec": float(best["offset_sec"] + intercept),
        "final_scale": refined["scale"],
        "final_offset_sec": refined["offset_sec"],
        "final_score_stride1": refined["score"],
        "stages": stage_rows,
    }


def serialize_evaluation(result: dict) -> dict:
    out = {key: value for key, value in result.items() if key != "rotations"}
    if "rotations" in result:
        out["rotations"] = {module: matrix.tolist() for module, matrix in result["rotations"].items()}
    return out


def main() -> None:
    global WRISTS
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument(
        "--preprocess-script",
        default=r"C:\Users\hand\Desktop\Dataset\tools\preprocess_9cam_imu_mocap.py",
    )
    parser.add_argument(
        "--mocap-wide",
        default=None,
        help="Existing mocap wide CSV. Defaults to aligned_data/mocap_source_csv/mocap_joints_wide.csv.",
    )
    parser.add_argument(
        "--mocap-targets",
        nargs=2,
        default=None,
        metavar=("TARGET_A", "TARGET_B"),
        help="Two mocap quaternion prefixes to test in both module02/module03 pairings.",
    )
    args = parser.parse_args()
    if args.mocap_targets:
        WRISTS = tuple(args.mocap_targets)
    dataset = Path(args.dataset)
    aligned = dataset / "aligned_data"
    mocap_dir = aligned / "mocap_source_csv"
    mocap_dir.mkdir(parents=True, exist_ok=True)
    cache_csv = mocap_dir / "wrist_mocap_body_gyro.csv"
    mocap_wide = Path(args.mocap_wide) if args.mocap_wide else mocap_dir / "mocap_joints_wide.csv"

    print("Loading raw wrist IMUs and anchoring their device clocks to the first camera exposure...")
    camera = load_camera_anchored_imus(dataset, Path(args.preprocess_script))
    durations = {module: float(camera[f"{module}_time_sec"][-1]) for module in WRIST_MODULES}
    print(f"raw IMU rows={camera['valid_rows']} durations={durations}")
    print("Loading wrist quaternions and computing body-frame angular velocity...")
    mocap = load_mocap_body_gyro(mocap_wide, cache_csv)
    print(f"mocap rows={len(mocap['time_sec'])} duration={mocap['time_sec'][-1]:.3f}s")

    duration_ratio = float(mocap["time_sec"][-1] / np.mean(list(durations.values())))
    print(f"duration ratio={duration_ratio:.6f}")
    print("Coarse global search...")
    coarse = coarse_search(camera, mocap, duration_ratio)
    print(json.dumps(coarse[:5], indent=2))
    print("Three-axis refinement...")
    best = refine_search(camera, mocap, coarse)
    initial_best = best
    print(
        f"initial best scale={best['scale']:.8f} "
        f"offset={best['offset_sec']:.4f}s score={best['score']:.5f}"
    )

    print("Estimating residual lag trend across 60-second windows...")
    initial_local_rows = local_lag_validation(best, camera, mocap)
    best, lag_trend_refinement = refine_from_local_lag_trend(
        best,
        camera,
        mocap,
        initial_local_rows,
    )
    print(
        f"final best scale={best['scale']:.9f} "
        f"offset={best['offset_sec']:.6f}s score={best['score']:.6f}"
    )

    validation = cross_validate(best, camera, mocap)
    local_rows = local_lag_validation(
        best,
        camera,
        mocap,
        delta_radius_sec=0.08,
        delta_step_sec=0.002,
    )
    deltas = np.array([row["best_delta_sec"] for row in local_rows], dtype=np.float64)
    local_summary = {
        "validated_windows": len(local_rows),
        "median_delta_ms": float(np.median(deltas) * 1000.0) if len(deltas) else None,
        "median_abs_delta_ms": float(np.median(np.abs(deltas)) * 1000.0) if len(deltas) else None,
        "p90_abs_delta_ms": float(np.percentile(np.abs(deltas), 90) * 1000.0) if len(deltas) else None,
        "max_abs_delta_ms": float(np.max(np.abs(deltas)) * 1000.0) if len(deltas) else None,
        "median_window_corr": float(np.median([row["vector_corr"] for row in local_rows])) if local_rows else None,
    }

    report = {
        "method": "global two-wrist three-axis body-frame angular-velocity alignment",
        "mapping": "mocap_time_sec = global_offset_sec + global_scale * camera_elapsed_sec",
        "global_scale": best["scale"],
        "global_offset_sec": best["offset_sec"],
        "pairing": best["pairing"],
        "global_vector_corr": best["score"],
        "modules": best["modules"],
        "imu_to_mocap_body_rotations": {module: matrix.tolist() for module, matrix in best["rotations"].items()},
        "duration_ratio": duration_ratio,
        "camera_time_origins_ms": camera["camera_time_origins_ms"],
        "camera_alignment_timestamp_field": CAMERA_ALIGNMENT_TS_FIELD,
        "raw_imu_rows": camera["valid_rows"],
        "raw_imu_durations_sec": durations,
        "first_camera_device_ts_ms": camera["first_camera_device_ts_ms"],
        "source_mocap_wide": str(mocap_wide),
        "mocap_alignment_targets": list(WRISTS),
        "coarse_top_candidates": coarse[:12],
        "initial_vector_refinement": serialize_evaluation(initial_best),
        "lag_trend_refinement": lag_trend_refinement,
        "cross_validation": validation,
        "local_lag_validation": local_summary,
        "notes": [
            "Both wrists share exactly one scale and one offset.",
            "Mocap angular velocity is expressed in the moving wrist/body frame.",
            "A fixed proper 3D rotation is fitted separately for each rigidly mounted wrist IMU.",
            "Speed-norm correlation is used only to seed the search; final selection uses full three-axis vectors.",
            "Residual lag versus time is fitted before a full-resolution joint scale/offset search.",
        ],
    }
    report_path = aligned / "global_imu_mocap_alignment_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    local_path = aligned / "global_imu_mocap_local_lag_validation.csv"
    with local_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(local_rows[0].keys()) if local_rows else ["window_start_sec"])
        writer.writeheader()
        writer.writerows(local_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
