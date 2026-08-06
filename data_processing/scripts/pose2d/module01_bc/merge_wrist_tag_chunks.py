#!/usr/bin/env python3
"""Merge chunked wrist-tag outputs into one successful-detections CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--report-json", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    reports = []
    candidates = {}
    fieldnames = None
    for report_path in sorted(args.chunks_dir.glob("chunk_*/wrist_pose_report_*.json")):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        reports.append((report_path, report))
        csv_path = report_path.parent / Path(report["outputs"]["csv"]).name
        with csv_path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            fieldnames = reader.fieldnames
            for row in reader:
                if not row.get("wrist_CAM_B_x_m", "").strip():
                    continue
                key = round(float(row["CAM_B_device_ts_ms"]), 6)
                previous = candidates.get(key)
                if previous is None:
                    candidates[key] = row
                    continue
                # Chunk boundaries can contain the same timestamp. Prefer the
                # estimate supported by more cameras/tags, then lower reprojection.
                score = (
                    row["observation_sources"].count("|") + 1,
                    -float(row["mean_reprojection_error_px"]),
                )
                old_score = (
                    previous["observation_sources"].count("|") + 1,
                    -float(previous["mean_reprojection_error_px"]),
                )
                if score > old_score:
                    candidates[key] = row

    if not candidates or fieldnames is None:
        raise RuntimeError("No successful wrist poses found in chunk outputs")
    rows = [candidates[key] for key in sorted(candidates)]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    by_id = {}
    by_source = {}
    for row in rows:
        by_id[row["detected_tag_ids"]] = by_id.get(row["detected_tag_ids"], 0) + 1
        source = row["observation_sources"]
        by_source[source] = by_source.get(source, 0) + 1
    summary = {
        "schema": "wrist_tag_full_merge.v1",
        "successful_unique_frames": len(rows),
        "first_CAM_B_device_ts_ms": float(rows[0]["CAM_B_device_ts_ms"]),
        "last_CAM_B_device_ts_ms": float(rows[-1]["CAM_B_device_ts_ms"]),
        "detections_by_id_combination": by_id,
        "frames_by_observation_source": by_source,
        "chunks": [
            {
                "report": str(path),
                "start_offset_s": report["start_offset_s"],
                "duration_s": report["duration_s"],
                "effective_tag_size_m": report["effective_tag_size_m"],
                "stereo_translation_scale": report["stereo_translation_scale"],
                "stats": report["stats"],
            }
            for path, report in reports
        ],
        "output_csv": str(args.output_csv),
    }
    args.report_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
