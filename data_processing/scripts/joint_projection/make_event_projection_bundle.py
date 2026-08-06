#!/usr/bin/env python3
"""Build compact timestamp/alignment tables for strict event-indexed videos."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


parser = argparse.ArgumentParser()
parser.add_argument("--aligned", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seconds", type=float, default=10.0)
args = parser.parse_args()

with args.aligned.open("r", encoding="utf-8-sig", newline="") as f:
    source_rows = list(csv.DictReader(f))
if not source_rows:
    raise RuntimeError("aligned table is empty")

t0 = float(source_rows[0]["module01_CAM_A_device_ts_ms"])
rows = [
    row for row in source_rows
    if float(row["module01_CAM_A_device_ts_ms"]) < t0 + args.seconds * 1000.0
]

# Event videos contain only retained strict rows, so their frame and sequence are
# both the compact event index.  The original exposure timestamp remains the key
# used to join back to the mocap/rigid-pose row.
external = []
head = []
aligned = []
for index, row in enumerate(rows):
    for camera in ("CAM_A", "CAM_D"):
        ts_us = int(float(row[f"external_{camera}_exposure_end_device_timestamp_us"]))
        external.append({
            "camera": camera,
            "frame_index": index,
            "sequence": index,
            "exposure_end_device_timestamp_us": ts_us,
        })
        head.append({
            "module": "1",
            "camera": camera,
            "seq": index,
            "device_ts_ms": row[f"module01_{camera}_device_ts_ms"],
        })
    out = dict(row)
    out["external_CAM_A_exposure_end_device_timestamp_us"] = external[-2]["exposure_end_device_timestamp_us"]
    out["external_CAM_D_exposure_end_device_timestamp_us"] = external[-1]["exposure_end_device_timestamp_us"]
    out["event_index"] = index
    aligned.append(out)

write_csv(
    args.output / "external_timestamps.csv",
    ["camera", "frame_index", "sequence", "exposure_end_device_timestamp_us"],
    external,
)
write_csv(
    args.output / "head_timestamps.csv",
    ["module", "camera", "seq", "device_ts_ms"],
    head,
)
aligned_fields = list(source_rows[0]) + (["event_index"] if "event_index" not in source_rows[0] else [])
write_csv(args.output / "aligned_50hz.csv", aligned_fields, aligned)
(args.output / "strict_sequences.json").write_text(
    json.dumps({"kept_sequences": list(range(len(rows)))}, indent=2),
    encoding="utf-8",
)
(args.output / "empty_hands.jsonl").write_text("", encoding="utf-8")
print({"rows": len(rows), "duration_source_seconds": (float(rows[-1]["module01_CAM_A_device_ts_ms"]) - t0) / 1000.0})
