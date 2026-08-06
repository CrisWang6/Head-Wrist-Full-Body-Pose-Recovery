from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


CAMERAS = ("CAM_A", "CAM_D")
MODULES = (1, 2, 3)
CAMERA_TS_FIELD = "device_ts_ms"
DEFAULT_SENSOR_ID = 307
IMU_NEAREST_LIMIT_MS = 20.0
MOCAP_NEAREST_LIMIT_MS = 20.0
TRIGGER_CLUSTER_TOLERANCE_MS = 4.0


def load_exporter(script: Path):
    spec = importlib.util.spec_from_file_location("abx2_rigid_export", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import ABX2 exporter: {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ffloat(value: str | float | int | None, default: float = float("nan")) -> float:
    if value is None or value == "":
        return default
    return float(value)


def nearest_index(sorted_values: np.ndarray, target: float) -> int:
    idx = int(np.searchsorted(sorted_values, target))
    if idx <= 0:
        return 0
    if idx >= len(sorted_values):
        return len(sorted_values) - 1
    return idx - 1 if abs(sorted_values[idx - 1] - target) <= abs(sorted_values[idx] - target) else idx


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
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
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
    """Angular velocity in the CH07 moving/body coordinate system."""
    q = normalize_quaternions(q)
    rel = quat_multiply(quat_inverse(q[:-1]), q[1:])
    rel /= np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-12)
    rel[rel[:, 0] < 0] *= -1.0
    w = np.clip(rel[:, 0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(np.maximum(0.0, 1.0 - w * w))
    dt = np.diff(t)
    interval = np.zeros((len(rel), 3), dtype=np.float64)
    valid = (dt > 1e-12) & (s > 1e-10) & (angle > 1e-12)
    interval[valid] = rel[valid, 1:] / s[valid, None] * (angle[valid] / dt[valid])[:, None]
    out = np.empty((len(q), 3), dtype=np.float64)
    out[0] = interval[0]
    out[-1] = interval[-1]
    out[1:-1] = 0.5 * (interval[:-1] + interval[1:])
    return out


def fit_rotation(mocap: np.ndarray, imu: np.ndarray) -> np.ndarray:
    """Fit a proper rotation so mocap_body_gyro @ R predicts IMU gyro."""
    source = mocap - np.mean(mocap, axis=0)
    target = imu - np.mean(imu, axis=0)
    u, _s, vt = np.linalg.svd(source.T @ target)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def vector_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a - np.mean(a, axis=0)
    bb = b - np.mean(b, axis=0)
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.sum(aa * bb) / denom) if denom > 1e-12 else 0.0


def scalar_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a - np.mean(a)
    bb = b - np.mean(b)
    denom = np.linalg.norm(aa) * np.linalg.norm(bb)
    return float(np.dot(aa, bb) / denom) if denom > 1e-12 else 0.0


def smooth_vectors(vectors: np.ndarray, width: int = 3) -> np.ndarray:
    return np.stack([moving_average(vectors[:, axis], width) for axis in range(3)], axis=1)


def high_pass_speed(vectors: np.ndarray, sample_rate: float) -> np.ndarray:
    speed = np.linalg.norm(vectors, axis=1)
    smoothed = moving_average(speed, max(3, int(round(sample_rate * 0.10))))
    baseline = moving_average(smoothed, max(5, int(round(sample_rate * 1.5))))
    signal = smoothed - baseline
    limit = float(np.percentile(np.abs(signal), 99.5))
    return np.clip(signal, -limit, limit)


def recover_shared_trigger_ordinals(
    streams: dict[str, list[dict]],
    timestamp_ms,
    sequence_field: str,
) -> dict:
    """Recover one trigger timeline jointly for all cameras sharing a device clock."""
    normalized_periods = []
    for rows in streams.values():
        rows.sort(key=timestamp_ms)
        times = np.asarray([timestamp_ms(row) for row in rows], dtype=np.float64)
        delta = np.diff(times)
        steps = np.maximum(1, np.rint(delta / 20.0).astype(np.int64))
        normalized_periods.extend((delta / steps).tolist())
    period_ms = float(np.median(normalized_periods))

    tagged = sorted(
        (timestamp_ms(row), camera, row)
        for camera, rows in streams.items()
        for row in rows
    )
    events: list[list[tuple[float, str, dict]]] = []
    for item in tagged:
        if not events or item[0] - float(np.median([x[0] for x in events[-1]])) > TRIGGER_CLUSTER_TOLERANCE_MS:
            events.append([item])
        else:
            if any(existing[1] == item[1] for existing in events[-1]):
                raise RuntimeError(f"Two {item[1]} frames fell in one trigger cluster")
            events[-1].append(item)

    event_time = np.asarray([np.median([item[0] for item in event]) for event in events], dtype=np.float64)
    anchor_row = streams["CAM_A"][0]
    anchor_event = next(
        index for index, event in enumerate(events)
        if any(row is anchor_row for _time, _camera, row in event)
    )
    anchor_ordinal = int(anchor_row[sequence_field])
    event_ordinal = anchor_ordinal + np.rint(
        (event_time - event_time[anchor_event]) / period_ms
    ).astype(np.int64)

    pair_delta = []
    for index, event in enumerate(events):
        for time_ms, _camera, row in event:
            row["packet_seq"] = row.get(sequence_field, "")
            row["seq"] = str(int(event_ordinal[index]))
            row["_trigger_event_time_ms"] = float(event_time[index])
        if len(event) == 2:
            pair_delta.append(abs(event[0][0] - event[1][0]))

    return {
        "period_ms": period_ms,
        "events": len(events),
        "first_ordinal": int(event_ordinal[0]),
        "last_ordinal": int(event_ordinal[-1]),
        "paired_events": len(pair_delta),
        "pair_p90_abs_ms": float(np.percentile(pair_delta, 90)) if pair_delta else None,
        "pair_max_abs_ms": float(np.max(pair_delta)) if pair_delta else None,
    }


def read_camera_rows(dataset: Path) -> dict[tuple[int, str], list[dict]]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    with (dataset / "timestamps.csv").open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {"module", "camera", "seq", CAMERA_TS_FIELD}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"timestamps.csv missing fields: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            key = (int(row["module"]), row["camera"])
            grouped.setdefault(key, []).append(row)
    expected = {(module, camera) for module in MODULES for camera in CAMERAS}
    missing = sorted(expected - set(grouped))
    if missing:
        raise ValueError(f"Missing camera streams: {missing}")
    # Recover one shared trigger event stream per module. Clustering CAM_A and
    # CAM_D before assigning ordinals prevents half-period rounding from pairing
    # adjacent triggers after asymmetric drops.
    for module in MODULES:
        recover_shared_trigger_ordinals(
            {camera: grouped[(module, camera)] for camera in CAMERAS},
            lambda row: float(row[CAMERA_TS_FIELD]),
            "seq",
        )
        for camera in CAMERAS:
            grouped[(module, camera)].sort(key=lambda row: int(row["seq"]))
    return grouped


def read_external_camera_rows(folder: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {camera: [] for camera in CAMERAS}
    with (folder / "timestamps.csv").open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {
            "camera",
            "exposure_end_device_timestamp_us",
            "timestamp_reference",
            "jpeg_valid",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"external timestamps.csv missing fields: {sorted(required - set(reader.fieldnames or []))}")
        for row in reader:
            camera = row["camera"]
            if camera not in grouped or row["jpeg_valid"] != "1":
                continue
            if row["timestamp_reference"] != "exposure_end":
                raise ValueError(f"External {camera} timestamp_reference is not exposure_end")
            grouped[camera].append(row)
    if any(not rows for rows in grouped.values()):
        raise ValueError("External camera timestamps must contain valid CAM_A and CAM_D rows")
    recover_shared_trigger_ordinals(
        grouped,
        lambda row: float(row["exposure_end_device_timestamp_us"]) / 1000.0,
        "sequence",
    )
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["seq"]))
    return grouped


def fit_external_exposure_end_mapping(
    external: dict[str, list[dict]],
    main_grouped: dict[tuple[int, str], list[dict]],
) -> tuple[dict[tuple[str, int], dict], dict]:
    """Anchor clocks at the first trigger, then fit drift by nearest exposure-end pulses."""
    reference_extra_ms = float(external["CAM_A"][0]["exposure_end_device_timestamp_us"]) / 1000.0
    reference_main_ms = float(main_grouped[(1, "CAM_A")][0][CAMERA_TS_FIELD])
    scale = 1.0
    main_at_reference_ms = reference_main_ms
    for _ in range(4):
        pairs = []
        for camera in CAMERAS:
            main_rows = main_grouped[(1, camera)]
            main_t = np.asarray([float(row[CAMERA_TS_FIELD]) for row in main_rows], dtype=np.float64)
            for row in external[camera]:
                ext_ms = float(row["exposure_end_device_timestamp_us"]) / 1000.0
                mapped = main_at_reference_ms + scale * (ext_ms - reference_extra_ms)
                index = nearest_index(main_t, mapped)
                if abs(float(main_t[index] - mapped)) <= 8.0:
                    pairs.append((ext_ms, float(main_t[index])))
        if len(pairs) < 100:
            raise RuntimeError(f"Too few shared exposure-end matches: {len(pairs)}")
        x = np.asarray([pair[0] - reference_extra_ms for pair in pairs], dtype=np.float64)
        y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        scale, main_at_reference_ms = np.polyfit(x, y, 1)
        residual = y - (main_at_reference_ms + scale * x)
        center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - center)))
        keep = np.abs(residual - center) <= max(0.5, 6.0 * 1.4826 * mad)
        if np.sum(keep) >= 100:
            scale, main_at_reference_ms = np.polyfit(x[keep], y[keep], 1)

    lookup: dict[tuple[str, int], dict] = {}
    residuals = []
    matched_counts = {camera: 0 for camera in CAMERAS}
    unmatched_counts = {camera: 0 for camera in CAMERAS}
    for camera in CAMERAS:
        main_rows = main_grouped[(1, camera)]
        main_t = np.asarray([float(row[CAMERA_TS_FIELD]) for row in main_rows], dtype=np.float64)
        for row in external[camera]:
            ext_ms = float(row["exposure_end_device_timestamp_us"]) / 1000.0
            mapped = main_at_reference_ms + scale * (ext_ms - reference_extra_ms)
            index = nearest_index(main_t, mapped)
            match_residual = float(main_t[index] - mapped)
            if abs(match_residual) > 8.0:
                unmatched_counts[camera] += 1
                continue
            key = (camera, int(main_rows[index]["seq"]))
            if key not in lookup or abs(match_residual) < abs(lookup[key]["_match_residual_ms"]):
                lookup[key] = {**row, "_match_residual_ms": match_residual}
            matched_counts[camera] += 1
            residuals.append(match_residual)
    stats = {
        "method": "first-trigger clock anchor plus nearest exposure-end affine drift fit",
        "first_trigger_ordinal": int(external["CAM_A"][0]["seq"]),
        "external_reference_exposure_end_ms": reference_extra_ms,
        "module01_reference_exposure_end_ms": reference_main_ms,
        "first_trigger_clock_offset_ms": reference_main_ms - reference_extra_ms,
        "module01_time_at_external_reference_ms": float(main_at_reference_ms),
        "external_to_module01_clock_scale": float(scale),
        "matched_rows": matched_counts,
        "unmatched_rows": unmatched_counts,
        "match_residual_ms": {
            "median": float(np.median(residuals)),
            "p90_abs": float(np.percentile(np.abs(residuals), 90)),
            "p99_abs": float(np.percentile(np.abs(residuals), 99)),
            "max_abs": float(np.max(np.abs(residuals))),
        },
    }
    return lookup, stats


def robust_clock_fit(rows: list[dict]) -> dict:
    seq = np.array([int(row["seq"]) for row in rows], dtype=np.float64)
    ts = np.array([float(row[CAMERA_TS_FIELD]) for row in rows], dtype=np.float64)
    keep = np.ones(len(seq), dtype=bool)
    slope = intercept = 0.0
    for _ in range(3):
        slope, intercept = np.polyfit(seq[keep], ts[keep], 1)
        residual = ts - (intercept + slope * seq)
        center = float(np.median(residual[keep]))
        mad = float(np.median(np.abs(residual[keep] - center)))
        limit = max(0.2, 8.0 * 1.4826 * mad)
        keep = np.abs(residual - center) <= limit
    residual = ts - (intercept + slope * seq)
    return {
        "slope_ms_per_seq": float(slope),
        "intercept_ms_at_seq0": float(intercept),
        "residual_median_ms": float(np.median(residual)),
        "residual_p90_abs_ms": float(np.percentile(np.abs(residual), 90)),
        "residual_max_abs_ms": float(np.max(np.abs(residual))),
        "rows": len(rows),
    }


def build_camera_clock_model(grouped: dict[tuple[int, str], list[dict]]) -> tuple[dict[int, dict], dict]:
    stream_fits = {key: robust_clock_fit(rows) for key, rows in grouped.items()}
    module_models = {}
    for module in MODULES:
        fits = [stream_fits[(module, camera)] for camera in CAMERAS]
        module_models[module] = {
            "slope_ms_per_seq": float(np.median([fit["slope_ms_per_seq"] for fit in fits])),
            "intercept_ms_at_seq0": float(np.median([fit["intercept_ms_at_seq0"] for fit in fits])),
        }
    stats = {f"module{m:02d}_{c}": stream_fits[(m, c)] for m in MODULES for c in CAMERAS}
    return module_models, stats


def read_imu(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        rows = list(reader)
    required = {"gyro_device_ts_ms", "gx_rad_s", "gy_rad_s", "gz_rad_s"}
    if not required.issubset(fields):
        raise ValueError(f"{path.name} missing fields: {sorted(required - set(fields))}")
    gyro_t = np.array([ffloat(row["gyro_device_ts_ms"]) for row in rows], dtype=np.float64)
    gyro = np.array(
        [[ffloat(row["gx_rad_s"]), ffloat(row["gy_rad_s"]), ffloat(row["gz_rad_s"])] for row in rows],
        dtype=np.float64,
    )
    valid = np.isfinite(gyro_t) & np.all(np.isfinite(gyro), axis=1)
    valid &= np.r_[True, np.diff(gyro_t) > 0]
    return {
        "path": path,
        "fields": fields,
        "rows": rows,
        "gyro_t_ms": gyro_t,
        "gyro": gyro,
        "valid": valid,
    }


def extract_ch07(abx2: Path, exporter_script: Path, sensor_id: int) -> dict:
    exporter = load_exporter(exporter_script)
    info, config = exporter.read_abx2_header(abx2)
    abx_info = info.get("ABXInfo", {})
    nominal_fps = float(abx_info.get("fps") or 60.0)
    pwrs = exporter.pwr_map_for_sensors(config, (sensor_id,))
    rows = exporter.extract_ch3_rigids(abx2, pwrs, nominal_fps)[sensor_id]
    if len(rows) < 10:
        raise ValueError(f"CH sensor {sensor_id} has too few rows: {len(rows)}")

    frame_index = np.arange(len(rows), dtype=np.int64)
    raw_tick = np.array([int(row["raw_tick"]) for row in rows], dtype=np.int64)
    tick_valid = (raw_tick >= 0) & (raw_tick < 0xFFFFFFFE)
    tick_valid &= np.r_[True, np.diff(raw_tick, prepend=raw_tick[0] - 1)[1:] >= 0]
    valid_idx = frame_index[tick_valid]
    valid_tick = raw_tick[tick_valid].astype(np.float64)
    per_frame_tick = np.diff(valid_tick) / np.maximum(np.diff(valid_idx), 1)
    tick_rate_hz = float(np.median(per_frame_tick) * nominal_fps)
    if not (nominal_fps <= tick_rate_hz <= nominal_fps * 4.0):
        tick_rate_hz = nominal_fps * 2.0
    interp_tick = np.interp(frame_index, valid_idx, valid_tick)
    time_sec = (interp_tick - interp_tick[0]) / tick_rate_hz

    q = np.array([[row["qw"], row["qx"], row["qy"], row["qz"]] for row in rows], dtype=np.float64)
    gyro = body_angular_velocity(q, time_sec)
    return {
        "rows": rows,
        "time_sec": time_sec,
        "gyro": gyro,
        "nominal_fps": nominal_fps,
        "effective_rate_hz": float(1.0 / np.median(np.diff(time_sec))),
        "tick_rate_hz": tick_rate_hz,
        "invalid_tick_rows": int(np.sum(~tick_valid)),
        "pwr": pwrs[0],
    }


def interpolate_vectors(source_t: np.ndarray, source_v: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(target_t, source_t, source_v[:, axis]) for axis in range(3)], axis=1)


def evaluate_mapping(
    scale: float,
    offset: float,
    imu_t: np.ndarray,
    imu_vec: np.ndarray,
    mocap_t: np.ndarray,
    mocap_vec: np.ndarray,
    rotation: np.ndarray | None = None,
    fit_mask_extra: np.ndarray | None = None,
    score_mask_extra: np.ndarray | None = None,
    stride: int = 1,
) -> dict:
    t = imu_t[::stride]
    imu = smooth_vectors(imu_vec[::stride], 3)
    mapped = offset + scale * t
    valid = (mapped >= mocap_t[0]) & (mapped <= mocap_t[-1])
    if np.sum(valid) < 300 or np.mean(valid) < 0.75:
        return {"score": -1.0}
    t = t[valid]
    imu = imu[valid]
    mapped = mapped[valid]
    mocap = smooth_vectors(interpolate_vectors(mocap_t, mocap_vec, mapped), 3)
    activity = np.linalg.norm(imu - np.median(imu, axis=0), axis=1)
    active = activity >= np.percentile(activity, 40.0)
    fit_mask = active.copy()
    score_mask = active.copy()
    if fit_mask_extra is not None:
        fit_mask &= fit_mask_extra(t)
    if score_mask_extra is not None:
        score_mask &= score_mask_extra(t)
    if np.sum(fit_mask) < 100 or np.sum(score_mask) < 100:
        return {"score": -1.0}
    fitted = fit_rotation(mocap[fit_mask], imu[fit_mask]) if rotation is None else rotation
    prediction = mocap @ fitted
    score = vector_corr(prediction[score_mask], imu[score_mask])
    axis_corr = []
    for axis in range(3):
        x = prediction[score_mask, axis]
        y = imu[score_mask, axis]
        axis_corr.append(float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 1e-12 and np.std(y) > 1e-12 else 0.0)
    return {
        "score": score,
        "scale": float(scale),
        "offset_sec": float(offset),
        "rotation_mocap_to_imu": fitted,
        "axis_corr": axis_corr,
        "valid_samples": int(len(t)),
        "active_samples": int(np.sum(score_mask)),
    }


def search_alignment(imu_t: np.ndarray, imu_vec: np.ndarray, mocap_t: np.ndarray, mocap_vec: np.ndarray) -> dict:
    imu_rate = (len(imu_t) - 1) / (imu_t[-1] - imu_t[0])
    mocap_rate = (len(mocap_t) - 1) / (mocap_t[-1] - mocap_t[0])
    imu_signal = high_pass_speed(imu_vec, imu_rate)
    mocap_signal = high_pass_speed(mocap_vec, mocap_rate)
    scale_center = float((mocap_t[-1] - mocap_t[0]) / (imu_t[-1] - imu_t[0]))
    coarse = []
    stride = 3
    for scale in np.arange(max(0.94, scale_center - 0.035), min(1.06, scale_center + 0.035) + 0.0005, 0.001):
        for offset in np.arange(-8.0, 8.0001, 0.05):
            mapped = offset + scale * imu_t[::stride]
            valid = (mapped >= mocap_t[0]) & (mapped <= mocap_t[-1])
            if np.mean(valid) < 0.75:
                continue
            sampled = np.interp(mapped[valid], mocap_t, mocap_signal)
            coarse.append((scalar_corr(imu_signal[::stride][valid], sampled), float(scale), float(offset)))
    coarse.sort(reverse=True)
    seeds = []
    for candidate in coarse:
        if all(abs(candidate[1] - old[1]) > 0.002 or abs(candidate[2] - old[2]) > 0.4 for old in seeds):
            seeds.append(candidate)
        if len(seeds) >= 12:
            break

    evaluated = []
    for _score, seed_scale, seed_offset in seeds:
        for ds in range(-3, 4):
            for do in range(-10, 11):
                evaluated.append(
                    evaluate_mapping(
                        seed_scale + ds * 0.0005,
                        seed_offset + do * 0.01,
                        imu_t,
                        imu_vec,
                        mocap_t,
                        mocap_vec,
                        stride=2,
                    )
                )
    best = max(evaluated, key=lambda row: row["score"])
    for scale_step, offset_step, scale_radius, offset_radius in (
        (0.0001, 0.002, 5, 8),
        (0.00002, 0.0005, 5, 6),
        (0.000004, 0.0001, 4, 5),
    ):
        candidates = []
        for ds in range(-scale_radius, scale_radius + 1):
            for do in range(-offset_radius, offset_radius + 1):
                candidates.append(
                    evaluate_mapping(
                        best["scale"] + ds * scale_step,
                        best["offset_sec"] + do * offset_step,
                        imu_t,
                        imu_vec,
                        mocap_t,
                        mocap_vec,
                        stride=1,
                    )
                )
        best = max(candidates, key=lambda row: row["score"])
    best["coarse_top"] = [
        {"speed_corr": score, "scale": scale, "offset_sec": offset} for score, scale, offset in seeds
    ]
    best["scale_center_from_duration"] = scale_center
    return best


def detect_rigid_coupling_start(
    initial: dict,
    imu_t: np.ndarray,
    imu_vec: np.ndarray,
    mocap_t: np.ndarray,
    mocap_vec: np.ndarray,
) -> tuple[float, dict]:
    """Find when CH07 becomes rigidly attached to the module01 IMU."""
    if initial["score"] >= 0.90:
        return 0.0, {"method": "full recording already rigidly coupled", "window_scores": []}
    mapped = initial["offset_sec"] + initial["scale"] * imu_t
    valid = (mapped >= mocap_t[0]) & (mapped <= mocap_t[-1])
    t = imu_t[valid]
    imu = smooth_vectors(imu_vec[valid], 3)
    mocap = smooth_vectors(interpolate_vectors(mocap_t, mocap_vec, mapped[valid]), 3)
    reference_start = max(float(t[0]), float(t[-1] - min(180.0, 0.55 * (t[-1] - t[0]))))
    reference = t >= reference_start
    activity = np.linalg.norm(imu - np.median(imu[reference], axis=0), axis=1)
    reference &= activity >= np.percentile(activity[reference], 40.0)
    rotation = fit_rotation(mocap[reference], imu[reference])
    prediction = mocap @ rotation
    windows = []
    for start in np.arange(max(0.0, math.floor(t[0] / 2.0) * 2.0), t[-1] - 6.0, 2.0):
        mask = (t >= start) & (t < start + 6.0)
        if np.sum(mask) < 150:
            continue
        local_activity = np.linalg.norm(imu[mask] - np.median(imu[mask], axis=0), axis=1)
        active = local_activity >= np.percentile(local_activity, 40.0)
        score = vector_corr(prediction[mask][active], imu[mask][active])
        windows.append({"start_sec": float(start), "end_sec": float(start + 6.0), "vector_corr": float(score)})
    for index in range(max(0, len(windows) - 4)):
        block = windows[index:index + 5]
        if len(block) < 5 or block[-1]["start_sec"] - block[0]["start_sec"] > 8.1:
            continue
        scores = np.asarray([row["vector_corr"] for row in block], dtype=np.float64)
        start = float(block[0]["start_sec"])
        if t[-1] - start < 60.0 or np.min(scores) < 0.80 or np.median(scores) < 0.90:
            continue
        mask_after = lambda values, threshold=start: values >= threshold
        suffix = evaluate_mapping(
            initial["scale"], initial["offset_sec"], imu_t, imu_vec, mocap_t, mocap_vec,
            fit_mask_extra=mask_after, score_mask_extra=mask_after,
        )
        if suffix["score"] >= 0.90:
            return start, {
                "method": "first sustained fixed-rotation 6 s window block",
                "reference_rotation_fit_start_sec": reference_start,
                "thresholds": {"block_windows": 5, "minimum_each_corr": 0.80, "minimum_median_corr": 0.90},
                "window_scores": windows,
                "initial_suffix_score": suffix["score"],
            }
    raise RuntimeError("No sustained CH07-module01 rigid-coupling interval exceeded correlation 0.90")


def refine_alignment_for_interval(
    initial: dict,
    start_sec: float,
    imu_t: np.ndarray,
    imu_vec: np.ndarray,
    mocap_t: np.ndarray,
    mocap_vec: np.ndarray,
) -> dict:
    interval = lambda t: t >= start_sec
    best = {"score": -1.0}
    for scale in np.arange(initial["scale"] - 0.0011, initial["scale"] + 0.0011, 0.0001):
        for offset in np.arange(initial["offset_sec"] - 0.21, initial["offset_sec"] + 0.211, 0.005):
            result = evaluate_mapping(
                scale, offset, imu_t, imu_vec, mocap_t, mocap_vec,
                fit_mask_extra=interval, score_mask_extra=interval, stride=2,
            )
            if result["score"] > best["score"]:
                best = result
    for scale_step, offset_step in ((0.00002, 0.001), (0.000004, 0.0002), (0.000001, 0.00005)):
        candidates = []
        for ds in range(-5, 6):
            for do in range(-8, 9):
                candidates.append(evaluate_mapping(
                    best["scale"] + ds * scale_step,
                    best["offset_sec"] + do * offset_step,
                    imu_t, imu_vec, mocap_t, mocap_vec,
                    fit_mask_extra=interval, score_mask_extra=interval,
                ))
        best = max(candidates, key=lambda row: row["score"])
    best["fit_start_sec"] = float(start_sec)
    best["coarse_top"] = initial.get("coarse_top", [])
    best["scale_center_from_duration"] = initial.get("scale_center_from_duration")
    return best


def cross_validate_interval(
    best: dict,
    start_sec: float,
    imu_t: np.ndarray,
    imu_vec: np.ndarray,
    mocap_t: np.ndarray,
    mocap_vec: np.ndarray,
) -> dict:
    even = lambda t: (t >= start_sec) & ((np.floor((t - start_sec) / 10.0).astype(np.int64) % 2) == 0)
    odd = lambda t: (t >= start_sec) & ~even(t)
    fit_even = evaluate_mapping(best["scale"], best["offset_sec"], imu_t, imu_vec, mocap_t, mocap_vec, fit_mask_extra=even, score_mask_extra=even)
    fit_odd = evaluate_mapping(best["scale"], best["offset_sec"], imu_t, imu_vec, mocap_t, mocap_vec, fit_mask_extra=odd, score_mask_extra=odd)
    eval_odd = evaluate_mapping(best["scale"], best["offset_sec"], imu_t, imu_vec, mocap_t, mocap_vec, rotation=fit_even["rotation_mocap_to_imu"], score_mask_extra=odd)
    eval_even = evaluate_mapping(best["scale"], best["offset_sec"], imu_t, imu_vec, mocap_t, mocap_vec, rotation=fit_odd["rotation_mocap_to_imu"], score_mask_extra=even)
    return {
        "fit_even_10s_eval_odd_10s": {k: v for k, v in eval_odd.items() if k != "rotation_mocap_to_imu"},
        "fit_odd_10s_eval_even_10s": {k: v for k, v in eval_even.items() if k != "rotation_mocap_to_imu"},
        "mean_held_window_corr": float(np.mean([eval_odd["score"], eval_even["score"]])),
        "evaluation_start_sec": float(start_sec),
    }


def mapped_mocap_time(alignment: dict, camera_elapsed_sec: np.ndarray | float) -> np.ndarray | float:
    mapped = alignment["offset_sec"] + alignment["scale"] * camera_elapsed_sec
    warp = alignment.get("time_warp")
    if not warp:
        return mapped
    correction = np.interp(
        camera_elapsed_sec,
        np.asarray(warp["camera_time_knots_sec"], dtype=np.float64),
        np.asarray(warp["correction_knots_sec"], dtype=np.float64),
    )
    return mapped + correction


def evaluate_warped_mapping(
    alignment: dict,
    imu_t: np.ndarray,
    imu_vec: np.ndarray,
    mocap_t: np.ndarray,
    mocap_vec: np.ndarray,
    start_sec: float,
    rotation: np.ndarray | None = None,
    score_mask_extra=None,
) -> dict:
    mapped = np.asarray(mapped_mocap_time(alignment, imu_t), dtype=np.float64)
    valid = (mapped >= mocap_t[0]) & (mapped <= mocap_t[-1]) & (imu_t >= start_sec)
    t = imu_t[valid]
    imu = smooth_vectors(imu_vec[valid], 3)
    mocap = smooth_vectors(interpolate_vectors(mocap_t, mocap_vec, mapped[valid]), 3)
    activity = np.linalg.norm(imu - np.median(imu, axis=0), axis=1)
    active = activity >= np.percentile(activity, 40.0)
    if score_mask_extra is not None:
        active &= score_mask_extra(t)
    fitted = fit_rotation(mocap[active], imu[active]) if rotation is None else rotation
    prediction = mocap @ fitted
    axis_corr = [
        float(np.corrcoef(prediction[active, axis], imu[active, axis])[0, 1])
        for axis in range(3)
    ]
    return {
        "score": vector_corr(prediction[active], imu[active]),
        "scale": float(alignment["scale"]),
        "offset_sec": float(alignment["offset_sec"]),
        "rotation_mocap_to_imu": fitted,
        "axis_corr": axis_corr,
        "valid_samples": int(len(t)),
        "active_samples": int(np.sum(active)),
    }


def fit_mocap_time_warp(
    alignment: dict,
    start_sec: float,
    imu_t: np.ndarray,
    imu_vec: np.ndarray,
    mocap_t: np.ndarray,
    mocap_vec: np.ndarray,
) -> tuple[dict, dict]:
    """Fit slow/piecewise mocap timing deviations to head IMU motion."""
    knots = []
    rotation = alignment["rotation_mocap_to_imu"]
    for start in np.arange(start_sec, imu_t[-1] - 10.0, 10.0):
        mask = (imu_t >= start) & (imu_t < start + 10.0)
        if np.sum(mask) < 300:
            continue
        motion = float(np.percentile(np.linalg.norm(imu_vec[mask], axis=1), 90))
        local_best = None
        for delta in np.arange(-0.200, 0.2001, 0.002):
            result = evaluate_mapping(
                alignment["scale"], alignment["offset_sec"] + delta,
                imu_t[mask], imu_vec[mask], mocap_t, mocap_vec, rotation=rotation,
            )
            if local_best is None or result["score"] > local_best["score"]:
                local_best = {**result, "delta_sec": float(delta)}
        if local_best is not None and motion >= 0.15 and local_best["score"] >= 0.90:
            knots.append({
                "camera_time_sec": float(start + 5.0),
                "correction_sec": local_best["delta_sec"],
                "window_corr": local_best["score"],
                "motion_p90_rad_s": motion,
            })
    if len(knots) < 8:
        raise RuntimeError(f"Only {len(knots)} reliable mocap time-warp knots were found")
    knot_t = np.asarray([row["camera_time_sec"] for row in knots], dtype=np.float64)
    knot_d = np.asarray([row["correction_sec"] for row in knots], dtype=np.float64)
    if np.any(np.diff(alignment["offset_sec"] + alignment["scale"] * knot_t + knot_d) <= 0):
        raise RuntimeError("Mocap time-warp knots are not monotonic")
    warped = {
        **alignment,
        "time_warp": {
            "method": "piecewise-linear local lag fitted from 10 s high-motion windows",
            "camera_time_knots_sec": knot_t.tolist(),
            "correction_knots_sec": knot_d.tolist(),
            "window_details": knots,
        },
    }
    evaluated = evaluate_warped_mapping(warped, imu_t, imu_vec, mocap_t, mocap_vec, start_sec)
    warped.update(evaluated)
    leave_one_out_error = []
    for index in range(1, len(knots) - 1):
        predicted = np.interp(knot_t[index], np.delete(knot_t, index), np.delete(knot_d, index))
        leave_one_out_error.append(float((knot_d[index] - predicted) * 1000.0))
    stats = {
        "knots": len(knots),
        "correction_min_ms": float(np.min(knot_d) * 1000.0),
        "correction_max_ms": float(np.max(knot_d) * 1000.0),
        "leave_one_knot_out_p90_abs_ms": float(np.percentile(np.abs(leave_one_out_error), 90)),
    }
    return warped, stats


def cross_validate_warped(
    alignment: dict,
    start_sec: float,
    imu_t: np.ndarray,
    imu_vec: np.ndarray,
    mocap_t: np.ndarray,
    mocap_vec: np.ndarray,
) -> dict:
    even = lambda t: (np.floor((t - start_sec) / 10.0).astype(np.int64) % 2) == 0
    odd = lambda t: ~even(t)
    fit_even = evaluate_warped_mapping(alignment, imu_t, imu_vec, mocap_t, mocap_vec, start_sec, score_mask_extra=even)
    fit_odd = evaluate_warped_mapping(alignment, imu_t, imu_vec, mocap_t, mocap_vec, start_sec, score_mask_extra=odd)
    eval_odd = evaluate_warped_mapping(alignment, imu_t, imu_vec, mocap_t, mocap_vec, start_sec, rotation=fit_even["rotation_mocap_to_imu"], score_mask_extra=odd)
    eval_even = evaluate_warped_mapping(alignment, imu_t, imu_vec, mocap_t, mocap_vec, start_sec, rotation=fit_odd["rotation_mocap_to_imu"], score_mask_extra=even)
    return {
        "fit_even_10s_eval_odd_10s": {k: v for k, v in eval_odd.items() if k != "rotation_mocap_to_imu"},
        "fit_odd_10s_eval_even_10s": {k: v for k, v in eval_even.items() if k != "rotation_mocap_to_imu"},
        "mean_held_window_corr": float(np.mean([eval_odd["score"], eval_even["score"]])),
        "evaluation_start_sec": float(start_sec),
    }


def local_lag_validation_warped(
    alignment: dict,
    start_sec: float,
    imu_t: np.ndarray,
    imu_vec: np.ndarray,
    mocap_t: np.ndarray,
    mocap_vec: np.ndarray,
) -> dict:
    rows = []
    rotation = alignment["rotation_mocap_to_imu"]
    for start in np.arange(start_sec, imu_t[-1] - 10.0, 10.0):
        mask = (imu_t >= start) & (imu_t < start + 10.0)
        if np.sum(mask) < 300:
            continue
        local_t = imu_t[mask]
        local_v = imu_vec[mask]
        motion = float(np.percentile(np.linalg.norm(local_v, axis=1), 90))
        base_mapped = np.asarray(mapped_mocap_time(alignment, local_t), dtype=np.float64)
        best = None
        for delta in np.arange(-0.050, 0.0501, 0.001):
            valid = (base_mapped + delta >= mocap_t[0]) & (base_mapped + delta <= mocap_t[-1])
            mocap = smooth_vectors(interpolate_vectors(mocap_t, mocap_vec, base_mapped[valid] + delta), 3)
            imu = smooth_vectors(local_v[valid], 3)
            activity = np.linalg.norm(imu - np.median(imu, axis=0), axis=1)
            active = activity >= np.percentile(activity, 40.0)
            score = vector_corr((mocap @ rotation)[active], imu[active])
            if best is None or score > best["score"]:
                best = {"score": score, "delta_ms": float(delta * 1000.0)}
        rows.append({"window_start_sec": float(start), "window_end_sec": float(start + 10.0), "motion_p90_rad_s": motion, "best_delta_ms": best["delta_ms"], "vector_corr": best["score"]})
    reliable = [row for row in rows if row["vector_corr"] >= 0.90]
    abs_lag = np.abs([row["best_delta_ms"] for row in reliable])
    return {
        "window_sec": 10.0,
        "windows": rows,
        "reliable_windows": len(reliable),
        "median_abs_ms": float(np.median(abs_lag)),
        "p90_abs_ms": float(np.percentile(abs_lag, 90)),
        "max_abs_ms": float(np.max(abs_lag)),
    }


def cross_validate(best: dict, imu_t: np.ndarray, imu_vec: np.ndarray, mocap_t: np.ndarray, mocap_vec: np.ndarray) -> dict:
    even = lambda t: (np.floor(t / 10.0).astype(np.int64) % 2) == 0
    odd = lambda t: ~even(t)
    fit_even = evaluate_mapping(best["scale"], best["offset_sec"], imu_t, imu_vec, mocap_t, mocap_vec, fit_mask_extra=even)
    fit_odd = evaluate_mapping(best["scale"], best["offset_sec"], imu_t, imu_vec, mocap_t, mocap_vec, fit_mask_extra=odd)
    eval_odd = evaluate_mapping(
        best["scale"], best["offset_sec"], imu_t, imu_vec, mocap_t, mocap_vec,
        rotation=fit_even["rotation_mocap_to_imu"], score_mask_extra=odd,
    )
    eval_even = evaluate_mapping(
        best["scale"], best["offset_sec"], imu_t, imu_vec, mocap_t, mocap_vec,
        rotation=fit_odd["rotation_mocap_to_imu"], score_mask_extra=even,
    )
    return {
        "fit_even_10s_eval_odd_10s": {k: v for k, v in eval_odd.items() if k != "rotation_mocap_to_imu"},
        "fit_odd_10s_eval_even_10s": {k: v for k, v in eval_even.items() if k != "rotation_mocap_to_imu"},
        "mean_held_window_corr": float(np.mean([eval_odd["score"], eval_even["score"]])),
    }


def local_lag_validation(best: dict, imu_t: np.ndarray, imu_vec: np.ndarray, mocap_t: np.ndarray, mocap_vec: np.ndarray, start_sec: float = 0.0) -> dict:
    rotation = best["rotation_mocap_to_imu"]
    rows = []
    for start in np.arange(start_sec, max(start_sec, imu_t[-1] - 10.0), 10.0):
        mask = (imu_t >= start) & (imu_t < start + 10.0)
        if np.sum(mask) < 300:
            continue
        local_t = imu_t[mask]
        local_v = imu_vec[mask]
        motion = float(np.percentile(np.linalg.norm(local_v, axis=1), 90))
        if motion < 0.15:
            continue
        local_best = None
        for delta in np.arange(-0.050, 0.0501, 0.001):
            result = evaluate_mapping(
                best["scale"], best["offset_sec"] + delta,
                local_t, local_v, mocap_t, mocap_vec, rotation=rotation,
            )
            if local_best is None or result["score"] > local_best["score"]:
                local_best = result
                local_best["delta_sec"] = float(delta)
        if local_best is not None:
            rows.append({
                "window_start_sec": float(start),
                "window_end_sec": float(start + 10.0),
                "motion_p90_rad_s": motion,
                "best_delta_ms": local_best["delta_sec"] * 1000.0,
                "vector_corr": local_best["score"],
            })
    abs_lag = np.abs([row["best_delta_ms"] for row in rows])
    return {
        "window_sec": 10.0,
        "windows": rows,
        "median_abs_ms": float(np.median(abs_lag)) if len(abs_lag) else None,
        "p90_abs_ms": float(np.percentile(abs_lag, 90)) if len(abs_lag) else None,
        "max_abs_ms": float(np.max(abs_lag)) if len(abs_lag) else None,
    }


def refine_from_lag_trend(
    best: dict,
    lag: dict,
    imu_t: np.ndarray,
    imu_vec: np.ndarray,
    mocap_t: np.ndarray,
    mocap_vec: np.ndarray,
) -> tuple[dict, dict]:
    windows = lag["windows"]
    if len(windows) < 4:
        return best, {"applied": False, "reason": "fewer than four valid 10s windows"}
    centers = np.array([(row["window_start_sec"] + row["window_end_sec"]) * 0.5 for row in windows])
    residual_sec = np.array([row["best_delta_ms"] / 1000.0 for row in windows])
    slope, intercept = np.polyfit(centers, residual_sec, 1)
    seed_scale = float(best["scale"] + slope)
    seed_offset = float(best["offset_sec"] + intercept)
    original_meta = {
        "coarse_top": best.get("coarse_top", []),
        "scale_center_from_duration": best.get("scale_center_from_duration"),
    }
    refined = best
    stages = []
    for scale_step, offset_step, scale_radius, offset_radius in (
        (0.00005, 0.001, 6, 8),
        (0.00001, 0.00025, 5, 6),
        (0.000002, 0.00005, 4, 5),
    ):
        candidates = [refined]
        for ds in range(-scale_radius, scale_radius + 1):
            for do in range(-offset_radius, offset_radius + 1):
                candidates.append(
                    evaluate_mapping(
                        seed_scale + ds * scale_step,
                        seed_offset + do * offset_step,
                        imu_t,
                        imu_vec,
                        mocap_t,
                        mocap_vec,
                    )
                )
        refined = max(candidates, key=lambda row: row["score"])
        seed_scale = refined["scale"]
        seed_offset = refined["offset_sec"]
        stages.append({
            "scale_step": scale_step,
            "offset_step_sec": offset_step,
            "best_scale": seed_scale,
            "best_offset_sec": seed_offset,
            "best_score": refined["score"],
        })
    refined.update(original_meta)
    return refined, {
        "applied": True,
        "residual_lag_slope_sec_per_sec": float(slope),
        "residual_lag_intercept_sec": float(intercept),
        "trend_seed_scale": float(best["scale"] + slope),
        "trend_seed_offset_sec": float(best["offset_sec"] + intercept),
        "initial_scale": best["scale"],
        "initial_offset_sec": best["offset_sec"],
        "initial_score": best["score"],
        "final_scale": refined["scale"],
        "final_offset_sec": refined["offset_sec"],
        "final_score": refined["score"],
        "stages": stages,
    }


def serialize_alignment(best: dict) -> dict:
    return {
        **{k: v for k, v in best.items() if k != "rotation_mocap_to_imu"},
        "rotation_mocap_to_imu": best["rotation_mocap_to_imu"].tolist(),
        "fit_direction": "CH3_07 body-frame angular velocity @ rotation -> module01 IMU gyro",
    }


def build_aligned(
    dataset: Path,
    grouped: dict[tuple[int, str], list[dict]],
    module_clocks: dict[int, dict],
    imus: dict[int, dict],
    mocap: dict,
    alignment: dict,
    external_lookup: dict[tuple[str, int], dict] | None = None,
    valid_camera_elapsed_start_sec: float = 0.0,
) -> tuple[list[dict], dict]:
    lookup = {
        key: {int(row["seq"]): row for row in rows}
        for key, rows in grouped.items()
    }
    # Trim to the common camera capture span: start after every stream has
    # begun and stop before any stream has ended. Internal dropped frames stay
    # as blank camera cells so the nominal 50 Hz trigger timeline is preserved.
    seq_min = max(min(stream) for stream in lookup.values())
    seq_max = min(max(stream) for stream in lookup.values())
    output = []
    missing_imu = 0
    imu_dt_ms = {module: [] for module in MODULES}
    mocap_dt_ms = []
    missing_mocap_due_gap = 0
    missing_camera_values = {f"module{m:02d}_{c}": 0 for m in MODULES for c in CAMERAS}
    for source_seq in range(seq_min, seq_max + 1):
        module01_target_ms = (
            module_clocks[1]["intercept_ms_at_seq0"]
            + module_clocks[1]["slope_ms_per_seq"] * source_seq
        )
        camera_elapsed_sec = (module01_target_ms - module_clocks[1]["intercept_ms_at_seq0"]) / 1000.0
        if camera_elapsed_sec < valid_camera_elapsed_start_sec:
            continue
        row = {}
        for module in MODULES:
            for camera in CAMERAS:
                source = lookup[(module, camera)].get(source_seq)
                key = f"module{module:02d}_{camera}_device_ts_ms"
                row[key] = source[CAMERA_TS_FIELD] if source is not None else ""
                if source is None:
                    missing_camera_values[f"module{module:02d}_{camera}"] += 1
        if external_lookup is not None:
            for camera in CAMERAS:
                external_row = external_lookup.get((camera, source_seq))
                row[f"external_{camera}_exposure_end_device_timestamp_us"] = (
                    external_row["exposure_end_device_timestamp_us"] if external_row is not None else ""
                )

        imu_valid = True
        for module in MODULES:
            target_ms = (
                module_clocks[module]["intercept_ms_at_seq0"]
                + module_clocks[module]["slope_ms_per_seq"] * source_seq
            )
            imu = imus[module]
            idx = nearest_index(imu["gyro_t_ms"], target_ms)
            dt_ms = abs(float(imu["gyro_t_ms"][idx]) - target_ms)
            imu_dt_ms[module].append(dt_ms)
            if dt_ms > IMU_NEAREST_LIMIT_MS:
                imu_valid = False
                missing_imu += 1
                for field in imu["fields"]:
                    row[f"module{module:02d}_imu_{field}"] = ""
            else:
                for field, value in imu["rows"][idx].items():
                    row[f"module{module:02d}_imu_{field}"] = value

        mocap_target = float(mapped_mocap_time(alignment, camera_elapsed_sec))
        valid_mocap = mocap["time_sec"][0] <= mocap_target <= mocap["time_sec"][-1]
        if not valid_mocap or not imu_valid:
            continue
        mi = nearest_index(mocap["time_sec"], mocap_target)
        dt_ms = float((mocap["time_sec"][mi] - mocap_target) * 1000.0)
        if abs(dt_ms) > MOCAP_NEAREST_LIMIT_MS:
            missing_mocap_due_gap += 1
            continue
        source = mocap["rows"][mi]
        row["mocap_time_sec_target"] = f"{mocap_target:.9f}"
        row["mocap_nearest_time_sec"] = f"{mocap['time_sec'][mi]:.9f}"
        row["mocap_nearest_dt_ms"] = f"{dt_ms:.9f}"
        row["mocap_frame_index"] = mi
        row["mocap_raw_tick"] = source["raw_tick"]
        row["mocap_status"] = source["status"]
        for source_field, output_field in (
            ("x", "mocap_CH3_07_world_x"),
            ("y", "mocap_CH3_07_world_y"),
            ("z", "mocap_CH3_07_world_z"),
            ("qw", "mocap_CH3_07_world_qw"),
            ("qx", "mocap_CH3_07_world_qx"),
            ("qy", "mocap_CH3_07_world_qy"),
            ("qz", "mocap_CH3_07_world_qz"),
        ):
            row[output_field] = f"{float(source[source_field]):.9f}"
        mocap_dt_ms.append(dt_ms)
        output.append(row)

    for seq, row in enumerate(output):
        row = {"seq": seq, **row}
        output[seq] = row
    stats = {
        "source_seq_min_internal": seq_min,
        "source_seq_max_internal": seq_max,
        "rigid_coupling_trim_start_sec": float(valid_camera_elapsed_start_sec),
        "candidate_trigger_slots": seq_max - seq_min + 1,
        "missing_camera_values": missing_camera_values,
        "trigger_slots_with_no_camera_data": sum(
            1
            for source_seq in range(seq_min, seq_max + 1)
            if all(source_seq not in lookup[(module, camera)] for module in MODULES for camera in CAMERAS)
        ),
        "missing_imu_matches_before_trim": missing_imu,
        "rows_removed_for_mocap_gap_over_20ms": missing_mocap_due_gap,
        "imu_nearest_dt_ms": {
            f"module{module:02d}": {
                "median": float(np.median(imu_dt_ms[module])) if imu_dt_ms[module] else None,
                "max": float(np.max(imu_dt_ms[module])) if imu_dt_ms[module] else None,
            }
            for module in MODULES
        },
        "mocap_nearest_dt_ms": {
            "median": float(np.median(mocap_dt_ms)) if mocap_dt_ms else None,
            "max_abs": float(np.max(np.abs(mocap_dt_ms))) if mocap_dt_ms else None,
        },
    }
    return output, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Align 50 Hz cameras and module01 head IMU to CH07 rigid pose in ABX2.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--abx2", default=None, type=Path)
    parser.add_argument("--sensor-id", type=int, default=DEFAULT_SENSOR_ID)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument(
        "--external-camera-dir",
        default=None,
        type=Path,
        help="Optional second OAK recording driven by the same 50 Hz external trigger.",
    )
    parser.add_argument(
        "--abx2-exporter",
        default=str(Path(__file__).with_name("export_abx2_mocap_rigid_csv.py")),
        type=Path,
    )
    args = parser.parse_args()

    dataset = args.dataset
    abx2 = args.abx2 or dataset / "001.abx2"
    outdir = args.output_dir or dataset / "aligned_data"
    outdir.mkdir(parents=True, exist_ok=True)

    grouped = read_camera_rows(dataset)
    module_clocks, camera_fit_stats = build_camera_clock_model(grouped)
    external_lookup = None
    external_stats = None
    if args.external_camera_dir is not None:
        external_rows = read_external_camera_rows(args.external_camera_dir)
        external_lookup, external_stats = fit_external_exposure_end_mapping(external_rows, grouped)
    imus = {
        module: read_imu(next(dataset.glob(f"module{module:02d}_*_imu.csv")))
        for module in MODULES
    }
    mocap = extract_ch07(abx2, args.abx2_exporter, args.sensor_id)

    head_imu = imus[1]
    valid = head_imu["valid"]
    imu_t = (head_imu["gyro_t_ms"][valid] - module_clocks[1]["intercept_ms_at_seq0"]) / 1000.0
    imu_vec = head_imu["gyro"][valid]
    use = imu_t >= -1.0
    imu_t = imu_t[use]
    imu_vec = imu_vec[use]

    print(f"camera nominal period={module_clocks[1]['slope_ms_per_seq']:.6f}ms")
    print(f"module01 IMU rows={len(imu_t)} duration={imu_t[-1] - imu_t[0]:.3f}s")
    print(
        f"CH07 rows={len(mocap['rows'])} duration={mocap['time_sec'][-1]:.3f}s "
        f"nominal_fps={mocap['nominal_fps']:.6f} effective_rate={mocap['effective_rate_hz']:.6f}Hz"
    )
    print("searching time scale/offset and fitting mocap body axes to module01 IMU...")
    initial_alignment = search_alignment(imu_t, imu_vec, mocap["time_sec"], mocap["gyro"])
    coupling_start_sec, coupling_detection = detect_rigid_coupling_start(
        initial_alignment, imu_t, imu_vec, mocap["time_sec"], mocap["gyro"]
    )
    affine_alignment = refine_alignment_for_interval(
        initial_alignment, coupling_start_sec, imu_t, imu_vec, mocap["time_sec"], mocap["gyro"]
    )
    alignment, time_warp_stats = fit_mocap_time_warp(
        affine_alignment, coupling_start_sec, imu_t, imu_vec, mocap["time_sec"], mocap["gyro"]
    )
    validation = cross_validate_warped(
        alignment, coupling_start_sec, imu_t, imu_vec, mocap["time_sec"], mocap["gyro"]
    )
    lag = local_lag_validation_warped(
        alignment, coupling_start_sec, imu_t, imu_vec, mocap["time_sec"], mocap["gyro"]
    )
    drift_refinement = {
        "applied": True,
        "method": "interval-restricted joint scale/offset grid refinement",
        "initial_scale": initial_alignment["scale"],
        "initial_offset_sec": initial_alignment["offset_sec"],
        "initial_full_recording_score": initial_alignment["score"],
        "final_scale": affine_alignment["scale"],
        "final_offset_sec": affine_alignment["offset_sec"],
        "final_affine_rigid_interval_score": affine_alignment["score"],
        "final_time_warped_rigid_interval_score": alignment["score"],
        "time_warp_stats": time_warp_stats,
    }
    print(
        f"scale={alignment['scale']:.9f} offset={alignment['offset_sec']:.6f}s "
        f"corr={alignment['score']:.6f} cv={validation['mean_held_window_corr']:.6f}"
    )

    rows, build_stats = build_aligned(
        dataset, grouped, module_clocks, imus, mocap, alignment,
        external_lookup=external_lookup,
        valid_camera_elapsed_start_sec=coupling_start_sec,
    )
    if not rows:
        raise RuntimeError("No common valid rows after camera/IMU/mocap trim")
    out_csv = outdir / "aligned_50hz.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    source_camera_period_ms = float(np.median([model["slope_ms_per_seq"] for model in module_clocks.values()]))
    report = {
        "method": "single head rigid CH3_07 to module01 IMU body-frame angular-velocity alignment",
        "output_csv": str(out_csv),
        "rows": len(rows),
        "columns": len(rows[0]),
        "camera_nominal_rate_hz": 1000.0 / source_camera_period_ms,
        "camera_timestamp_field": CAMERA_TS_FIELD,
        "camera_streams": [f"module{m:02d}_{c}" for m in MODULES for c in CAMERAS],
        "camera_clock_models": {f"module{m:02d}": model for m, model in module_clocks.items()},
        "camera_clock_fit_stats": camera_fit_stats,
        "external_camera_alignment": external_stats,
        "mocap_sensor_id": args.sensor_id,
        "mocap_sensor_name": mocap["pwr"].get("name"),
        "mocap_nominal_fps": mocap["nominal_fps"],
        "mocap_effective_rate_hz_from_raw_tick": mocap["effective_rate_hz"],
        "mocap_raw_tick_rate_hz": mocap["tick_rate_hz"],
        "mocap_invalid_raw_tick_rows_interpolated": mocap["invalid_tick_rows"],
        "mocap_max_local_half_interval_ms": float(np.max(np.diff(mocap["time_sec"])) * 500.0),
        "rigid_coupling_detection": {
            "start_sec": coupling_start_sec,
            **coupling_detection,
        },
        "alignment": serialize_alignment(alignment),
        "time_drift_refinement": drift_refinement,
        "cross_validation": validation,
        "local_lag_validation": lag,
        "build": build_stats,
        "acceptance": {
            "global_vector_corr_gt_0_90": alignment["score"] > 0.90,
            "held_window_corr_gt_0_90": validation["mean_held_window_corr"] > 0.90,
            "local_lag_p90_le_20ms": lag["p90_abs_ms"] <= 20.0,
            "mocap_nearest_within_20ms": (
                build_stats["mocap_nearest_dt_ms"]["max_abs"] <= MOCAP_NEAREST_LIMIT_MS
            ),
            "missing_imu_matches_zero": build_stats["missing_imu_matches_before_trim"] == 0,
        },
        "source_files": {
            "dataset": str(dataset),
            "timestamps": str(dataset / "timestamps.csv"),
            "abx2": str(abx2),
            "module01_imu": str(imus[1]["path"]),
            "module02_imu": str(imus[2]["path"]),
            "module03_imu": str(imus[3]["path"]),
            "external_camera_timestamps": (
                str(args.external_camera_dir / "timestamps.csv") if args.external_camera_dir is not None else None
            ),
        },
        "notes": [
            "No BVH or skeleton data is used.",
            "CH3_07 raw_tick is used to model non-uniform mocap frame time; invalid 0xFFFFFFFF ticks are interpolated.",
            "The fixed proper rotation fits CH3_07 body-frame angular velocity to module01 IMU axes.",
            "Rows before the detected sustained CH3_07-module01 rigid-coupling interval are excluded.",
            "Residual non-uniform mocap timing is corrected by a piecewise-linear lag fit from reliable 10 s motion windows.",
            "Derived mocap angular velocity is used only for alignment and is not written to aligned_50hz.csv.",
            "Original camera seq is internal only; final seq is reindexed from zero.",
            "The optional external OAK system is joined only by shared-trigger exposure-end timestamps, not by its local sequence counter.",
        ],
    }
    report_path = outdir / "aligned_50hz_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
