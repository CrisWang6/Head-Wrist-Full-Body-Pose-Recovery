from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


CAMERAS = ("CAM_A", "CAM_D")
SYSTEMS = ("head", "external01", "external02")
CAMERA_MATCH_LIMIT_MS = 8.0
PAIR_CLUSTER_LIMIT_MS = 4.0
IMU_NEAREST_LIMIT_MS = 20.0
MOCAP_NEAREST_LIMIT_MS = 8.34
MIN_ALIGNMENT_CORR = 0.95
UINT32_INVALID = 0xFFFFFFFE


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def nearest_index(values: np.ndarray, target: float) -> int:
    index = int(np.searchsorted(values, target))
    if index <= 0:
        return 0
    if index >= len(values):
        return len(values) - 1
    return index - 1 if abs(values[index - 1] - target) <= abs(values[index] - target) else index


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def camera_rows(path: Path, kind: str) -> dict[str, list[dict]]:
    _fields, rows = read_csv(path)
    result = {camera: [] for camera in CAMERAS}
    for row in rows:
        camera = row.get("camera")
        if camera not in result:
            continue
        if kind == "head":
            if row.get("module") != "1":
                continue
            timestamp_ms = float(row["device_ts_ms"])
            source_frame = int(row["seq"])
        else:
            if row.get("jpeg_valid") != "1":
                continue
            if row.get("timestamp_reference") != "exposure_end":
                raise ValueError(f"{path}: camera timestamp is not exposure_end")
            timestamp_ms = float(row["exposure_end_device_timestamp_us"]) / 1000.0
            source_frame = int(row["frame_index"])
        result[camera].append({
            "camera": camera,
            "timestamp_ms": timestamp_ms,
            "source_frame": source_frame,
            "source_row": row,
        })
    for camera, stream in result.items():
        if not stream:
            raise ValueError(f"{path}: missing {camera}")
        stream.sort(key=lambda row: row["timestamp_ms"])
    return result


def cluster_stereo_events(streams: dict[str, list[dict]]) -> list[dict]:
    tagged = sorted(
        (row["timestamp_ms"], camera, row)
        for camera, rows in streams.items()
        for row in rows
    )
    events: list[list[tuple[float, str, dict]]] = []
    for item in tagged:
        if (
            not events
            or item[0] - float(np.median([old[0] for old in events[-1]])) > PAIR_CLUSTER_LIMIT_MS
            or any(old[1] == item[1] for old in events[-1])
        ):
            events.append([item])
        else:
            events[-1].append(item)
    output = []
    for index, event in enumerate(events):
        cameras = {camera: row for _time, camera, row in event}
        output.append({
            "event_index": index,
            "time_ms": float(np.median([time for time, _camera, _row in event])),
            "cameras": cameras,
        })
    return output


def fit_event_clock(reference: list[dict], other: list[dict]) -> tuple[dict, dict[int, dict]]:
    ref_t = np.asarray([event["time_ms"] for event in reference], dtype=np.float64)
    oth_t = np.asarray([event["time_ms"] for event in other], dtype=np.float64)
    ref_anchor = next(event["time_ms"] for event in reference if "CAM_A" in event["cameras"])
    oth_anchor = next(event["time_ms"] for event in other if "CAM_A" in event["cameras"])
    scale = 1.0
    intercept = ref_anchor
    for _ in range(5):
        pairs = []
        for time_ms in oth_t:
            mapped = intercept + scale * (time_ms - oth_anchor)
            index = nearest_index(ref_t, mapped)
            if abs(ref_t[index] - mapped) <= 12.0:
                pairs.append((time_ms - oth_anchor, ref_t[index]))
        if len(pairs) < 100:
            raise RuntimeError(f"Only {len(pairs)} common trigger events found while fitting clocks")
        x = np.asarray([row[0] for row in pairs])
        y = np.asarray([row[1] for row in pairs])
        scale, intercept = np.polyfit(x, y, 1)
        residual = y - (intercept + scale * x)
        center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - center)))
        keep = np.abs(residual - center) <= max(0.5, 6.0 * 1.4826 * mad)
        if np.sum(keep) >= 100:
            scale, intercept = np.polyfit(x[keep], y[keep], 1)

    matches: dict[int, dict] = {}
    residuals = []
    for event in other:
        mapped = intercept + scale * (event["time_ms"] - oth_anchor)
        index = nearest_index(ref_t, mapped)
        residual = float(ref_t[index] - mapped)
        if abs(residual) > CAMERA_MATCH_LIMIT_MS:
            continue
        existing = matches.get(index)
        if existing is None or abs(residual) < abs(existing["residual_ms"]):
            matches[index] = {"event": event, "residual_ms": residual, "mapped_time_ms": float(mapped)}
        residuals.append(residual)
    return {
        "method": "first CAM_A trigger anchor plus robust affine fit of actual trigger events",
        "reference_anchor_ms": float(ref_anchor),
        "source_anchor_ms": float(oth_anchor),
        "scale": float(scale),
        "reference_time_at_source_anchor_ms": float(intercept),
        "matched_events": len(matches),
        "unmatched_events": len(other) - len(matches),
        "residual_ms": {
            "median": float(np.median(residuals)),
            "p90_abs": float(np.percentile(np.abs(residuals), 90)),
            "p99_abs": float(np.percentile(np.abs(residuals), 99)),
            "max_abs": float(np.max(np.abs(residuals))),
        },
    }, matches


def discover_group(group: Path) -> dict:
    abx2 = next(group.glob("*.abx2"))
    head = next(path for path in group.iterdir() if path.is_dir() and (path / "module01_D45D2E00_imu.csv").exists())
    external = next(path for path in group.iterdir() if path.is_dir() and path.name.startswith("2026"))
    return {"group": group, "abx2": abx2, "head": head, "external": external}


def build_camera_intersection(paths: dict) -> tuple[list[dict], dict]:
    streams = {
        "head": camera_rows(paths["head"] / "timestamps.csv", "head"),
        "external01": camera_rows(paths["external"] / "external_01" / "timestamps.csv", "external"),
        "external02": camera_rows(paths["external"] / "external_02" / "timestamps.csv", "external"),
    }
    events = {name: cluster_stereo_events(value) for name, value in streams.items()}
    fit01, match01 = fit_event_clock(events["head"], events["external01"])
    fit02, match02 = fit_event_clock(events["head"], events["external02"])
    kept = []
    rejected = {"head_stereo_missing": 0, "external01_stereo_missing": 0, "external02_stereo_missing": 0, "external_clock_unmatched": 0}
    for index, head_event in enumerate(events["head"]):
        if set(head_event["cameras"]) != set(CAMERAS):
            rejected["head_stereo_missing"] += 1
            continue
        if index not in match01 or index not in match02:
            rejected["external_clock_unmatched"] += 1
            continue
        ext01 = match01[index]["event"]
        ext02 = match02[index]["event"]
        if set(ext01["cameras"]) != set(CAMERAS):
            rejected["external01_stereo_missing"] += 1
            continue
        if set(ext02["cameras"]) != set(CAMERAS):
            rejected["external02_stereo_missing"] += 1
            continue
        kept.append({
            "head_event_index": index,
            "head_time_ms": head_event["time_ms"],
            "head": head_event,
            "external01": ext01,
            "external02": ext02,
            "external01_clock_residual_ms": match01[index]["residual_ms"],
            "external02_clock_residual_ms": match02[index]["residual_ms"],
        })
    pair_stats = {}
    for name in SYSTEMS:
        deltas = [
            abs(event["cameras"]["CAM_A"]["timestamp_ms"] - event["cameras"]["CAM_D"]["timestamp_ms"])
            for event in events[name] if set(event["cameras"]) == set(CAMERAS)
        ]
        pair_stats[name] = {
            "events": len(events[name]),
            "complete_stereo_events": len(deltas),
            "pair_p90_abs_ms": float(np.percentile(deltas, 90)),
            "pair_max_abs_ms": float(np.max(deltas)),
        }
    return kept, {
        "clock_fit_external01_to_head": fit01,
        "clock_fit_external02_to_head": fit02,
        "stereo_event_stats": pair_stats,
        "rejected_reference_events": rejected,
        "six_camera_complete_events": len(kept),
    }


def extract_rigids(abx2: Path, exporter, sensor_ids: tuple[int, ...]) -> dict:
    info, config = exporter.read_abx2_header(abx2)
    fps = float(info.get("ABXInfo", {}).get("fps") or 60.0)
    pwrs = exporter.pwr_map_for_sensors(config, sensor_ids)
    by_sensor = exporter.extract_ch3_rigids(abx2, pwrs, fps)
    lengths = {sensor_id: len(by_sensor[sensor_id]) for sensor_id in sensor_ids}
    if len(set(lengths.values())) != 1:
        raise RuntimeError(f"Rigid row counts differ: {lengths}")
    count = next(iter(lengths.values()))
    # ABX2 frames are one common optical frame stream. raw_tick is a per-rigid
    # tracker field and can be absent or non-monotonic, so frame_index/fps is the
    # only valid common time base; the IMU fit below estimates drift explicitly.
    time_sec = np.arange(count, dtype=np.float64) / fps
    head_rows = by_sensor[308]
    quaternion = np.asarray([[row["qw"], row["qx"], row["qy"], row["qz"]] for row in head_rows], dtype=np.float64)
    validity = {}
    raw_tick_validity = {}
    for sensor_id in sensor_ids:
        rows = by_sensor[sensor_id]
        raw_tick = np.asarray([int(row["raw_tick"]) for row in rows], dtype=np.int64)
        pose = np.asarray([[row["x"], row["y"], row["z"], row["qw"], row["qx"], row["qy"], row["qz"]] for row in rows], dtype=np.float64)
        raw_tick_validity[sensor_id] = (raw_tick >= 0) & (raw_tick < UINT32_INVALID)
        quaternion_norm = np.linalg.norm(pose[:, 3:], axis=1)
        validity[sensor_id] = np.all(np.isfinite(pose), axis=1) & (quaternion_norm > 0.5) & (quaternion_norm < 1.5)
    return {
        "fps": fps,
        "time_sec": time_sec,
        "rows": by_sensor,
        "validity": validity,
        "raw_tick_validity": raw_tick_validity,
        "head_quaternion": quaternion,
        "count": count,
        "invalid_rows": {str(sensor_id): int(np.sum(~validity[sensor_id])) for sensor_id in sensor_ids},
        "missing_raw_tick_rows": {str(sensor_id): int(np.sum(~raw_tick_validity[sensor_id])) for sensor_id in sensor_ids},
    }


def load_imu(head: Path, anchor_ms: float, alignment_module) -> dict:
    path = head / "module01_D45D2E00_imu.csv"
    imu = alignment_module.read_imu(path)
    valid_indices = np.flatnonzero(imu["valid"])
    return {
        **imu,
        "valid_indices": valid_indices,
        "time_sec": imu["gyro_t_ms"][valid_indices] / 1000.0 - anchor_ms / 1000.0,
        "gyro_valid": imu["gyro"][valid_indices],
    }


def align_imu_mocap(imu: dict, mocap: dict, alignment_module) -> tuple[dict, dict]:
    mocap_gyro = alignment_module.body_angular_velocity(mocap["head_quaternion"], mocap["time_sec"])
    best = alignment_module.search_alignment(imu["time_sec"], imu["gyro_valid"], mocap["time_sec"], mocap_gyro)
    cross_validation = alignment_module.cross_validate(best, imu["time_sec"], imu["gyro_valid"], mocap["time_sec"], mocap_gyro)
    lag = alignment_module.local_lag_validation(best, imu["time_sec"], imu["gyro_valid"], mocap["time_sec"], mocap_gyro)
    if best["score"] < MIN_ALIGNMENT_CORR or cross_validation["mean_held_window_corr"] < MIN_ALIGNMENT_CORR:
        raise RuntimeError(
            f"CH3-8/IMU alignment rejected: score={best['score']:.6f}, "
            f"cross_validation={cross_validation['mean_held_window_corr']:.6f}"
        )
    serializable = {
        **{key: value for key, value in best.items() if key != "rotation_mocap_to_imu"},
        "rotation_mocap_body_gyro_to_head_imu": best["rotation_mocap_to_imu"].tolist(),
        "fit_direction": "CH3-8 body-frame angular velocity @ rotation -> module01 head IMU gyro",
        "cross_validation": cross_validation,
        "local_lag_validation": lag,
        "accepted_threshold": MIN_ALIGNMENT_CORR,
    }
    return best, serializable


def value(row: dict, key: str):
    item = row.get(key, "")
    try:
        return int(item)
    except (TypeError, ValueError):
        try:
            return float(item)
        except (TypeError, ValueError):
            return item


def assemble_rows(camera_events: list[dict], imu: dict, mocap: dict, alignment: dict, sensor_ids: tuple[int, ...], alignment_module) -> tuple[list[dict], dict]:
    output = []
    rejected = {"outside_imu": 0, "outside_mocap": 0, "invalid_rigid": 0}
    imu_t = imu["time_sec"]
    mocap_t = mocap["time_sec"]
    anchor_ms = camera_events[0]["head"]["cameras"]["CAM_A"]["timestamp_ms"]
    # Use the true first CAM_A trigger, not the first six-camera-complete event.
    anchor_ms = min(event["head_time_ms"] for event in camera_events) if not math.isfinite(anchor_ms) else anchor_ms
    source_anchor_ms = camera_events[0].get("global_head_cam_a_anchor_ms", anchor_ms)
    for event in camera_events:
        camera_time_sec = (event["head_time_ms"] - source_anchor_ms) / 1000.0
        mocap_target = alignment["offset_sec"] + alignment["scale"] * camera_time_sec
        mocap_index = nearest_index(mocap_t, mocap_target)
        mocap_delta_ms = float((mocap_t[mocap_index] - mocap_target) * 1000.0)
        imu_index = nearest_index(imu_t, camera_time_sec)
        imu_delta_ms = float((imu_t[imu_index] - camera_time_sec) * 1000.0)
        if abs(imu_delta_ms) > IMU_NEAREST_LIMIT_MS:
            rejected["outside_imu"] += 1
            continue
        if abs(mocap_delta_ms) > MOCAP_NEAREST_LIMIT_MS:
            rejected["outside_mocap"] += 1
            continue
        if any(not mocap["validity"][sensor_id][mocap_index] for sensor_id in sensor_ids):
            rejected["invalid_rigid"] += 1
            continue
        row = {
            "seq": len(output),
            "camera_elapsed_sec": camera_time_sec,
            "head_trigger_time_device_ts_ms": event["head_time_ms"],
            "external01_clock_residual_ms": event["external01_clock_residual_ms"],
            "external02_clock_residual_ms": event["external02_clock_residual_ms"],
        }
        for system in SYSTEMS:
            for camera in CAMERAS:
                cam = event[system]["cameras"][camera]
                prefix = f"{system}_{camera}"
                row[f"{prefix}_exposure_end_timestamp_ms"] = cam["timestamp_ms"]
        original_imu_index = int(imu["valid_indices"][imu_index])
        row["head_imu_nearest_dt_ms"] = imu_delta_ms
        for field in imu["fields"]:
            if field in {"module", "accel_seq", "gyro_seq"}:
                continue
            row[f"head_imu_{field}"] = value(imu["rows"][original_imu_index], field)
        row["mocap_target_time_sec"] = mocap_target
        row["mocap_frame_index"] = mocap_index
        row["mocap_time_sec"] = mocap_t[mocap_index]
        row["mocap_nearest_dt_ms"] = mocap_delta_ms
        for sensor_id in sensor_ids:
            rigid = mocap["rows"][sensor_id][mocap_index]
            prefix = f"mocap_CH3_{sensor_id - 300:02d}"
            for field in ("status", "raw_tick", "x", "y", "z", "qw", "qx", "qy", "qz"):
                row[f"{prefix}_{field}"] = rigid[field]
            row[f"{prefix}_raw_tick_valid"] = int(mocap["raw_tick_validity"][sensor_id][mocap_index])
        output.append(row)
    return output, rejected


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"No aligned rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def process_group(paths: dict, exporter, alignment_module) -> dict:
    group = paths["group"]
    sensor_ids = (301, 304, 308) if group.name == "无" else (301, 304, 306, 307, 308)
    camera_events, camera_report = build_camera_intersection(paths)
    head_events = cluster_stereo_events(camera_rows(paths["head"] / "timestamps.csv", "head"))
    head_cam_a_anchor_ms = next(event["time_ms"] for event in head_events if "CAM_A" in event["cameras"])
    for event in camera_events:
        event["global_head_cam_a_anchor_ms"] = head_cam_a_anchor_ms
    mocap = extract_rigids(paths["abx2"], exporter, sensor_ids)
    imu = load_imu(paths["head"], head_cam_a_anchor_ms, alignment_module)
    alignment, alignment_report = align_imu_mocap(imu, mocap, alignment_module)
    rows, sample_rejected = assemble_rows(camera_events, imu, mocap, alignment, sensor_ids, alignment_module)
    output_dir = group / "aligned_data"
    csv_path = output_dir / "aligned_30hz.csv"
    report_path = output_dir / "aligned_30hz_report.json"
    write_csv(csv_path, rows)
    report = {
        "group": group.name,
        "inputs": {key: str(value) for key, value in paths.items() if key != "group"},
        "policy": {
            "final_base": "actual common 30 fps external-trigger events",
            "camera_timestamp": "exposure-end device timestamp",
            "camera_drop_rule": "discard the complete row if any of six camera streams is absent",
            "mocap_time_base": "ABX2 common frame_index / declared 60 fps; drift fitted against actual IMU timestamps",
            "head_rigid": "CH3-8",
            "external01_rigid": "CH3-1",
            "external02_rigid": "CH3-4",
            "right_wrist_or_ankle": "CH3-7" if 307 in sensor_ids else None,
            "left_wrist_or_ankle": "CH3-6" if 306 in sensor_ids else None,
        },
        "camera_alignment": camera_report,
        "imu": {
            "rows": len(imu["rows"]),
            "valid_gyro_rows": len(imu["time_sec"]),
            "effective_rate_hz": float(1.0 / np.median(np.diff(imu["time_sec"]))),
            "first_head_cam_a_trigger_ms": head_cam_a_anchor_ms,
        },
        "mocap": {
            "declared_fps": mocap["fps"],
            "frames": mocap["count"],
            "included_sensor_ids": list(sensor_ids),
            "invalid_rows_by_sensor": mocap["invalid_rows"],
            "missing_raw_tick_rows_by_sensor": mocap["missing_raw_tick_rows"],
        },
        "head_imu_mocap_alignment": alignment_report,
        "sampling": {
            "six_camera_events_before_imu_mocap_limits": len(camera_events),
            "rejected": sample_rejected,
            "final_rows": len(rows),
            "first_camera_elapsed_sec": rows[0]["camera_elapsed_sec"],
            "last_camera_elapsed_sec": rows[-1]["camera_elapsed_sec"],
            "imu_nearest_p99_abs_ms": float(np.percentile(np.abs([row["head_imu_nearest_dt_ms"] for row in rows]), 99)),
            "mocap_nearest_p99_abs_ms": float(np.percentile(np.abs([row["mocap_nearest_dt_ms"] for row in rows]), 99)),
        },
        "outputs": {"aligned_csv": str(csv_path), "report": str(report_path)},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align the 0806 head/external stereo, head IMU and ABX2 rigid data")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--exporter", type=Path, default=Path(__file__).with_name("export_abx2_mocap_rigid_csv.py"))
    parser.add_argument("--alignment-module", type=Path, default=Path(__file__).with_name("align_head_rigid_imu.py"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exporter = load_module("abx2_exporter_0806", args.exporter)
    alignment_module = load_module("head_alignment_0806", args.alignment_module)
    reports = []
    for group in sorted(path for path in args.dataset.iterdir() if path.is_dir()):
        reports.append(process_group(discover_group(group), exporter, alignment_module))
        print(json.dumps({
            "group": group.name,
            "rows": reports[-1]["sampling"]["final_rows"],
            "correlation": reports[-1]["head_imu_mocap_alignment"]["score"],
            "cross_validation": reports[-1]["head_imu_mocap_alignment"]["cross_validation"]["mean_held_window_corr"],
        }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
