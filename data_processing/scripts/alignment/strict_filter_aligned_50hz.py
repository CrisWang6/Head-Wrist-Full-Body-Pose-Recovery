#!/usr/bin/env python3
"""Build an all-camera intersection from shared-trigger exposure-end times."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CAMERAS = ("CAM_A", "CAM_D")
CLUSTER_TOLERANCE_MS = 4.0
MATCH_TOLERANCE_MS = 8.0


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def nearest_index(values: np.ndarray, target: float) -> int:
    index = int(np.searchsorted(values, target))
    if index <= 0:
        return 0
    if index >= len(values):
        return len(values) - 1
    return index - 1 if abs(values[index - 1] - target) <= abs(values[index] - target) else index


def cluster_events(streams: dict[str, list[dict]], timestamp_ms) -> list[dict]:
    tagged = sorted(
        (timestamp_ms(row), camera, row)
        for camera, rows in streams.items()
        for row in rows
    )
    events: list[dict] = []
    for time_ms, camera, row in tagged:
        if not events or time_ms - events[-1]["time_ms"] > CLUSTER_TOLERANCE_MS:
            events.append({"time_ms": time_ms, "rows": {camera: row}, "times": [time_ms]})
            continue
        if camera in events[-1]["rows"]:
            raise RuntimeError(f"Two {camera} frames fell in one trigger cluster")
        events[-1]["rows"][camera] = row
        events[-1]["times"].append(time_ms)
        events[-1]["time_ms"] = float(np.median(events[-1]["times"]))
    return events


def event_with_camera_sequence(events: list[dict], field: str, value: int) -> dict:
    for event in events:
        row = event["rows"].get("CAM_A")
        if row is not None and int(row[field]) == value:
            return event
    raise RuntimeError(f"CAM_A anchor sequence {value} is absent from reference system")


def fit_clock_to_reference(events: list[dict], reference: list[dict], seed_offset_ms: float) -> dict:
    reference_time = np.asarray([event["time_ms"] for event in reference], dtype=np.float64)
    scale = 1.0
    offset = seed_offset_ms
    pairs = []
    for _ in range(4):
        pairs = []
        for event in events:
            mapped = offset + scale * event["time_ms"]
            index = nearest_index(reference_time, mapped)
            residual = float(reference_time[index] - mapped)
            if abs(residual) <= MATCH_TOLERANCE_MS:
                pairs.append((event["time_ms"], float(reference_time[index])))
        if len(pairs) < 100:
            raise RuntimeError(f"Only {len(pairs)} shared trigger events found")
        x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        scale, offset = np.polyfit(x, y, 1)
        residual = y - (offset + scale * x)
        center = float(np.median(residual))
        mad = float(np.median(np.abs(residual - center)))
        keep = np.abs(residual - center) <= max(0.5, 6.0 * 1.4826 * mad)
        if np.sum(keep) >= 100:
            scale, offset = np.polyfit(x[keep], y[keep], 1)
    residual = np.asarray([pair[1] - (offset + scale * pair[0]) for pair in pairs], dtype=np.float64)
    return {
        "scale": float(scale),
        "offset_ms": float(offset),
        "matched_events": len(pairs),
        "residual_p90_abs_ms": float(np.percentile(np.abs(residual), 90)),
        "residual_p99_abs_ms": float(np.percentile(np.abs(residual), 99)),
        "residual_max_abs_ms": float(np.max(np.abs(residual))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--external-timestamps", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    base_rows = read_rows(args.aligned)
    main_rows = read_rows(args.timestamps)
    external_rows = [row for row in read_rows(args.external_timestamps) if row.get("jpeg_valid", "1") == "1"]

    systems: dict[str, list[dict]] = {}
    for module in (1, 2, 3):
        streams = {}
        for camera in CAMERAS:
            rows = [row for row in main_rows if int(row["module"]) == module and row["camera"] == camera]
            for frame_index, row in enumerate(rows):
                row["_source_frame"] = frame_index
            streams[camera] = rows
        systems[f"module{module:02d}"] = cluster_events(streams, lambda row: float(row["device_ts_ms"]))

    external_streams = {}
    for camera in CAMERAS:
        rows = [row for row in external_rows if row["camera"] == camera]
        for row in rows:
            row["_source_frame"] = int(row["frame_index"])
        external_streams[camera] = rows
    systems["external"] = cluster_events(
        external_streams,
        lambda row: float(row["exposure_end_device_timestamp_us"]) / 1000.0,
    )

    reference = systems["module01"]
    clock_models = {"module01": {"scale": 1.0, "offset_ms": 0.0, "matched_events": len(reference)}}
    for system_name in ("module02", "module03", "external"):
        events = systems[system_name]
        anchor_row = events[0]["rows"].get("CAM_A")
        if anchor_row is None:
            raise RuntimeError(f"{system_name} first trigger cluster has no CAM_A")
        sequence_field = "sequence" if system_name == "external" else "seq"
        anchor_sequence = int(anchor_row[sequence_field])
        reference_anchor = event_with_camera_sequence(reference, "seq", anchor_sequence)
        seed_offset = reference_anchor["time_ms"] - events[0]["time_ms"]
        clock_models[system_name] = fit_clock_to_reference(events, reference, seed_offset)

    tagged_events = []
    for system_name, events in systems.items():
        model = clock_models[system_name]
        for event in events:
            mapped_time = model["offset_ms"] + model["scale"] * event["time_ms"]
            tagged_events.append((mapped_time, system_name, event))
    tagged_events.sort(key=lambda item: item[0])

    global_events: list[dict] = []
    for mapped_time, system_name, event in tagged_events:
        if not global_events or mapped_time - global_events[-1]["time_ms"] > CLUSTER_TOLERANCE_MS:
            global_events.append({"time_ms": mapped_time, "systems": {system_name: event}, "times": [mapped_time]})
            continue
        if system_name in global_events[-1]["systems"]:
            raise RuntimeError(f"Two {system_name} events fell in one global trigger cluster")
        global_events[-1]["systems"][system_name] = event
        global_events[-1]["times"].append(mapped_time)
        global_events[-1]["time_ms"] = float(np.median(global_events[-1]["times"]))

    base_by_module01_a = {
        round(float(row["module01_CAM_A_device_ts_ms"]), 6): row
        for row in base_rows
        if row.get("module01_CAM_A_device_ts_ms")
    }
    kept = []
    complete_camera_events = 0
    for global_event in global_events:
        if set(global_event["systems"]) != set(systems):
            continue
        if any(set(event["rows"]) != set(CAMERAS) for event in global_event["systems"].values()):
            continue
        complete_camera_events += 1
        module01_a = global_event["systems"]["module01"]["rows"]["CAM_A"]
        base = base_by_module01_a.get(round(float(module01_a["device_ts_ms"]), 6))
        if base is None:
            continue
        output_row = dict(base)
        for module in (1, 2, 3):
            system_name = f"module{module:02d}"
            for camera in CAMERAS:
                camera_row = global_event["systems"][system_name]["rows"][camera]
                output_row[f"{system_name}_{camera}_device_ts_ms"] = camera_row["device_ts_ms"]
                output_row[f"{system_name}_{camera}_source_frame"] = camera_row["_source_frame"]
        for camera in CAMERAS:
            camera_row = global_event["systems"]["external"]["rows"][camera]
            output_row[f"external_{camera}_exposure_end_device_timestamp_us"] = camera_row[
                "exposure_end_device_timestamp_us"
            ]
            output_row[f"external_{camera}_source_frame"] = camera_row["_source_frame"]
        kept.append(output_row)

    for aligned_index, row in enumerate(kept):
        row["seq"] = aligned_index
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(kept[0]))
        writer.writeheader()
        writer.writerows(kept)

    pair_sync = {}
    for system_name in systems:
        deltas = []
        for row in kept:
            if system_name == "external":
                a = float(row["external_CAM_A_exposure_end_device_timestamp_us"]) / 1000.0
                d = float(row["external_CAM_D_exposure_end_device_timestamp_us"]) / 1000.0
            else:
                a = float(row[f"{system_name}_CAM_A_device_ts_ms"])
                d = float(row[f"{system_name}_CAM_D_device_ts_ms"])
            deltas.append(a - d)
        pair_sync[system_name] = {
            "median_ms": float(np.median(deltas)),
            "p90_abs_ms": float(np.percentile(np.abs(deltas), 90)),
            "max_abs_ms": float(np.max(np.abs(deltas))),
            "rows_over_5ms": int(np.sum(np.abs(deltas) > 5.0)),
        }

    report = {
        "method": "first-trigger clock anchor, shared exposure-end event clustering, affine clock drift fit, then 8-camera intersection",
        "input_aligned_rows": len(base_rows),
        "global_trigger_events": len(global_events),
        "complete_8camera_events": complete_camera_events,
        "kept_rows_with_imu_and_mocap": len(kept),
        "removed_after_camera_intersection_or_sensor_trim": len(base_rows) - len(kept),
        "clock_models_to_module01": clock_models,
        "same_system_CAM_A_minus_CAM_D": pair_sync,
        "policy": "delete a trigger event if any of 8 cameras is missing; use per-camera source_frame to read original videos",
    }
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
