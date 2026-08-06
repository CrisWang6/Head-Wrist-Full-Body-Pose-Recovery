#!/usr/bin/env python3
"""Extract validation images directly from H.265 with piecewise decode-loss mapping."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
DATASET = Path(r"C:\Users\hand\Desktop\Dataset\0722\record_9cam_0711_021044")
ALIGNED = DATASET / "aligned_data" / "aligned_30hz.csv"
TIMESTAMPS = DATASET / "timestamps.csv"
FFMPEG = Path(r"C:\Users\hand\miniconda3\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe")
TIMELINE_REPORT = HERE / "validation_0722_h265_timeline" / "h265_timeline_report.json"
SELECTION_SUMMARY = HERE / "validation_0722_head_ch3_08_final" / "summary.json"
OUTPUT = HERE / "validation_0722_h265_random100"
WIDTH, HEIGHT = 1920, 1200


def camera_timestamp_rows(camera: str) -> list[dict[str, str]]:
    with TIMESTAMPS.open(newline="", encoding="utf-8-sig") as handle:
        return [
            row for row in csv.DictReader(handle)
            if row["module"] == "1" and row["camera"] == camera
        ]


def aligned_timestamps(selected: set[int]) -> dict[int, dict[str, float]]:
    result: dict[int, dict[str, float]] = {}
    with ALIGNED.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            seq = int(row["seq"])
            if seq not in selected:
                continue
            result[seq] = {
                camera: float(row[f"module01_{camera}_device_ts_ms"])
                for camera in ("CAM_B", "CAM_C")
                if row[f"module01_{camera}_device_ts_ms"]
            }
    return result


def decoded_to_capture(camera: str, decoded_count: int, events: list[dict[str, object]]) -> np.ndarray:
    # Only Duplicate POC events paired with an invalid NAL are actual dropped
    # image access units. "Could not find ref" may still yield an image.
    dropped_after = sorted(
        int(event["near_decoded_index"])
        for event in events
        if "Duplicate POC" in str(event["message"])
    )
    decoded = np.arange(decoded_count, dtype=np.int64)
    capture = decoded.copy()
    for after in dropped_after:
        capture += decoded > after
    return capture


def main() -> int:
    selected = [
        int(value)
        for value in json.loads(SELECTION_SUMMARY.read_text(encoding="utf-8"))["selected_sequences"]
    ]
    aligned = aligned_timestamps(set(selected))
    timeline = json.loads(TIMELINE_REPORT.read_text(encoding="utf-8"))
    mapping_report: dict[str, object] = {}

    for camera in ("CAM_B", "CAM_C"):
        raw_rows = camera_timestamp_rows(camera)
        timestamp_to_ordinal = {
            round(float(row["device_ts_ms"]), 6): ordinal
            for ordinal, row in enumerate(raw_rows)
        }
        decoded_count = int(timeline[camera]["decoded_frames"])
        capture_indices = decoded_to_capture(
            camera, decoded_count, timeline[camera]["decoder_events"]
        )
        requested: list[dict[str, object]] = []
        for seq in selected:
            if camera not in aligned.get(seq, {}):
                continue
            device_ts = aligned[seq][camera]
            raw_ordinal = timestamp_to_ordinal.get(round(device_ts, 6))
            if raw_ordinal is None:
                continue
            nearest = int(np.argmin(np.abs(capture_indices - raw_ordinal)))
            requested.append(
                {
                    "seq": seq,
                    "device_ts_ms": device_ts,
                    "timestamp_ordinal": raw_ordinal,
                    "decoded_index": nearest,
                    "capture_index_error": int(capture_indices[nearest] - raw_ordinal),
                }
            )
        requested.sort(key=lambda row: int(row["decoded_index"]))
        indices = [int(row["decoded_index"]) for row in requested]
        expression = "+".join(f"eq(n\\,{index})" for index in indices)
        source = DATASET / f"module01_D45D2E00_{camera}.h265"
        command = [
            str(FFMPEG), "-hide_banner", "-loglevel", "error", "-i", str(source),
            "-vf", f"select={expression}", "-vsync", "0", "-pix_fmt", "bgr24",
            "-f", "rawvideo", "pipe:1",
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert process.stdout is not None
        destination = OUTPUT / "source_frames" / "module01" / camera
        destination.mkdir(parents=True, exist_ok=True)
        frame_bytes = WIDTH * HEIGHT * 3
        written = 0
        for row in requested:
            buffer = process.stdout.read(frame_bytes)
            if len(buffer) != frame_bytes:
                break
            image = np.frombuffer(buffer, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)
            output_path = destination / f"seq_{int(row['seq']):06d}.jpg"
            cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, 96])
            written += 1
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0 or written != len(requested):
            raise RuntimeError(
                f"H.265 extraction failed for {camera}: return={return_code}, written={written}/{len(requested)}"
            )
        mapping_report[camera] = {
            "source": str(source),
            "timestamp_rows": len(raw_rows),
            "decoded_frames": decoded_count,
            "dropped_duplicate_poc_frames": len(
                [event for event in timeline[camera]["decoder_events"] if "Duplicate POC" in str(event["message"])]
            ),
            "written": written,
            "mapping": requested,
        }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "h265_frame_mapping.json").write_text(
        json.dumps(mapping_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({camera: {key: value for key, value in report.items() if key != "mapping"} for camera, report in mapping_report.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
