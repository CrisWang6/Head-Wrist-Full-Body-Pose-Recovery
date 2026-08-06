import argparse
import csv
import json
import re
import statistics
from bisect import bisect_left
from pathlib import Path


JOINT_COLUMNS = [
    "left_shoulder_x",
    "left_shoulder_y",
    "right_shoulder_x",
    "right_shoulder_y",
    "left_elbow_x",
    "left_elbow_y",
    "right_elbow_x",
    "right_elbow_y",
]


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def parse_decode_map(path, raw_count):
    pattern = re.compile(r"n:\s*(\d+).*?pts:\s*(-?\d+)")
    frames = []
    for line in Path(path).read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            frames.append((int(match.group(1)), int(match.group(2))))
    frames = sorted(set(frames))
    steps = [frames[index][1] - frames[index - 1][1] for index in range(1, len(frames))]
    normal = statistics.median(steps)
    missing_before = {}
    for index in range(1, len(frames)):
        step = frames[index][1] - frames[index - 1][1]
        missing = max(0, int(round(step / normal)) - 1)
        if missing:
            missing_before[frames[index][0]] = missing

    cumulative = 0
    mapping = {}
    for decoded, _ in frames:
        cumulative += missing_before.get(decoded, 0)
        mapping[decoded] = decoded + cumulative
    if len(mapping) + cumulative != raw_count:
        raise RuntimeError(
            f"{path}: decoded={len(mapping)}, inferred missing={cumulative}, raw={raw_count}"
        )
    return mapping, {
        "decoded_frames": len(mapping),
        "normal_pts_step": normal,
        "decode_gaps": [
            {"before_decoded_frame": frame, "missing_frames": count}
            for frame, count in missing_before.items()
        ],
    }


def exact_index(sorted_values, value, tolerance=1e-6):
    position = bisect_left(sorted_values, value)
    candidates = [
        index
        for index in (position - 1, position)
        if 0 <= index < len(sorted_values)
    ]
    if not candidates:
        return None
    result = min(candidates, key=lambda index: abs(sorted_values[index] - value))
    return result if abs(sorted_values[result] - value) <= tolerance else None


def process_camera(root, aligned, raw_all, camera):
    raw = [
        row
        for row in raw_all
        if row["module"] == "1" and row["camera"] == f"CAM_{camera}"
    ]
    raw_times = [float(row["device_ts_ms"]) for row in raw]
    decoded_to_recorded, decode_info = parse_decode_map(
        root / "output" / f"cam_{camera.lower()}_showinfo.log", len(raw)
    )
    recorded_to_decoded = {
        recorded: decoded for decoded, recorded in decoded_to_recorded.items()
    }

    predictions = {}
    summaries = []
    parts = root / "output" / "sapiens_parts"
    for path in sorted(parts.glob(f"cam_{camera.lower()}_part_*.jsonl")):
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                predictions[int(record["decoded_frame_index"])] = record
        summaries.append(json.loads(Path(f"{path}.summary.json").read_text()))

    seq_key = "\ufeffseq" if "\ufeffseq" in aligned[0] else "seq"
    timestamp_column = f"module01_CAM_{camera}_device_ts_ms"
    output_rows = []
    status_counts = {}
    valid_times = []
    for source in aligned:
        value = source[timestamp_column].strip()
        row = {
            "seq": source[seq_key],
            "module": "1",
            "camera": f"CAM_{camera}",
            "device_ts_ms": value,
            "raw_camera_seq": "",
            "decoded_frame_index": "",
            "status": "",
        }
        row.update({column: "" for column in JOINT_COLUMNS})
        if not value:
            row["status"] = "missing_aligned_timestamp"
        else:
            timestamp = float(value)
            valid_times.append(timestamp)
            raw_index = exact_index(raw_times, timestamp)
            if raw_index is None:
                row["status"] = "timestamp_not_found"
            else:
                row["raw_camera_seq"] = raw[raw_index]["seq"]
                decoded = recorded_to_decoded.get(raw_index)
                if decoded is None:
                    row["status"] = "video_decode_failed"
                else:
                    row["decoded_frame_index"] = decoded
                    prediction = predictions.get(decoded)
                    if prediction is None:
                        row["status"] = "inference_missing"
                    else:
                        row["status"] = prediction["status"]
                        for column in JOINT_COLUMNS:
                            row[column] = prediction.get(column, "")
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        output_rows.append(row)

    columns = [
        "seq",
        "module",
        "camera",
        "device_ts_ms",
        "raw_camera_seq",
        "decoded_frame_index",
        "status",
        *JOINT_COLUMNS,
    ]
    output_path = root / "output" / f"module01_cam_{camera.lower()}_shoulder_elbow_2d.csv"
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output_rows)

    duration_s = (max(valid_times) - min(valid_times)) / 1000.0
    max_wall = max(summary["wall_s"] for summary in summaries)
    return {
        "camera": f"CAM_{camera}",
        "output": output_path.name,
        "aligned_rows": len(aligned),
        "valid_timestamps": len(valid_times),
        "status_counts": status_counts,
        "decode_mapping": decode_info,
        "worker_summaries": summaries,
        "source_duration_s": duration_s,
        "effective_wall_s_excluding_model_load": max_wall,
        "processing_s_per_video_minute": max_wall / duration_s * 60.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    aligned = read_csv(root / "input" / "aligned_30hz.csv")
    raw = read_csv(root / "input" / "timestamps.csv")
    report = {
        "model": "Sapiens2-0.4B pose 1024x768 full-frame + flip test",
        "confidence_columns_saved": False,
        "cameras": [
            process_camera(root, aligned, raw, "B"),
            process_camera(root, aligned, raw, "C"),
        ],
    }
    report_path = root / "output" / "module01_cam_bc_shoulder_elbow_2d_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
