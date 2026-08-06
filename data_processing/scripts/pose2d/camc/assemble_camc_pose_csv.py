import argparse
import csv
import json
import math
import re
import statistics
from bisect import bisect_left
from pathlib import Path


JOINT_COLUMNS = [
    "left_shoulder_x",
    "left_shoulder_y",
    "left_shoulder_score",
    "right_shoulder_x",
    "right_shoulder_y",
    "right_shoulder_score",
    "left_elbow_x",
    "left_elbow_y",
    "left_elbow_score",
    "right_elbow_x",
    "right_elbow_y",
    "right_elbow_score",
]


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def parse_decoded_to_recorded_map(log_path, raw_count):
    pattern = re.compile(r"n:\s*(\d+).*?pts:\s*(-?\d+)")
    frames = []
    for line in Path(log_path).read_text(errors="replace").splitlines():
        match = pattern.search(line)
        if match:
            frames.append((int(match.group(1)), int(match.group(2))))
    frames = sorted(set(frames))
    pts_steps = [frames[i][1] - frames[i - 1][1] for i in range(1, len(frames))]
    normal_step = statistics.median(pts_steps)
    missing_before = {}
    cumulative = 0
    for index in range(1, len(frames)):
        step = frames[index][1] - frames[index - 1][1]
        missing = max(0, int(round(step / normal_step)) - 1)
        if missing:
            cumulative += missing
            missing_before[frames[index][0]] = missing

    mapping = {}
    cumulative = 0
    for decoded_index, _ in frames:
        cumulative += missing_before.get(decoded_index, 0)
        mapping[decoded_index] = decoded_index + cumulative
    if len(mapping) + cumulative != raw_count:
        raise RuntimeError(
            f"Decoded/raw mapping mismatch: decoded={len(mapping)}, "
            f"inserted={cumulative}, raw={raw_count}"
        )
    return mapping, {
        "decoded_frames": len(mapping),
        "normal_pts_step": normal_step,
        "decode_gaps": [
            {"before_decoded_frame": key, "missing_frames": value}
            for key, value in missing_before.items()
        ],
    }


def nearest_exact_index(sorted_values, value, tolerance=1e-6):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned", required=True)
    parser.add_argument("--raw-timestamps", required=True)
    parser.add_argument("--showinfo-log", required=True)
    parser.add_argument("--parts-dir", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()

    aligned = read_csv(args.aligned)
    raw = [
        row
        for row in read_csv(args.raw_timestamps)
        if row["module"] == "1" and row["camera"] == "CAM_C"
    ]
    raw_times = [float(row["device_ts_ms"]) for row in raw]
    decoded_to_recorded, decode_info = parse_decoded_to_recorded_map(
        args.showinfo_log, len(raw)
    )
    recorded_to_decoded = {
        recorded: decoded for decoded, recorded in decoded_to_recorded.items()
    }

    predictions = {}
    parts_dir = Path(args.parts_dir)
    for path in sorted(parts_dir.glob("part_*.jsonl")):
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                predictions[int(record["decoded_frame_index"])] = record

    summaries = [
        json.loads(path.read_text())
        for path in sorted(parts_dir.glob("part_*.summary.json"))
    ]
    seq_key = "\ufeffseq" if "\ufeffseq" in aligned[0] else "seq"
    rows = []
    status_counts = {}
    first_time = None
    for source in aligned:
        value = source["module01_CAM_C_device_ts_ms"].strip()
        row = {
            "seq": source[seq_key],
            "module": "1",
            "camera": "CAM_C",
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
            if first_time is None:
                first_time = timestamp
            row["_relative_s"] = (timestamp - first_time) / 1000.0
            raw_index = nearest_exact_index(raw_times, timestamp)
            if raw_index is None:
                row["status"] = "timestamp_not_found"
            else:
                row["raw_camera_seq"] = raw[raw_index]["seq"]
                decoded_index = recorded_to_decoded.get(raw_index)
                if decoded_index is None:
                    row["status"] = "video_decode_failed"
                else:
                    row["decoded_frame_index"] = decoded_index
                    prediction = predictions.get(decoded_index)
                    if prediction is None:
                        row["status"] = "inference_missing"
                    else:
                        row["status"] = prediction["status"]
                        for column in JOINT_COLUMNS:
                            row[column] = prediction[column]
                        row["_inference_s"] = prediction["inference_s"]
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        rows.append(row)

    output_columns = [
        "seq",
        "module",
        "camera",
        "device_ts_ms",
        "raw_camera_seq",
        "decoded_frame_index",
        "status",
        *JOINT_COLUMNS,
    ]
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    valid_times = [
        float(row["module01_CAM_C_device_ts_ms"])
        for row in aligned
        if row["module01_CAM_C_device_ts_ms"].strip()
    ]
    duration_s = (max(valid_times) - min(valid_times)) / 1000.0
    minute_metrics = []
    minute_count = int(math.ceil(duration_s / 60.0))
    for minute in range(minute_count):
        segment = [
            row
            for row in rows
            if "_relative_s" in row and minute * 60 <= row["_relative_s"] < (minute + 1) * 60
        ]
        completed = [row for row in segment if row["status"] == "ok"]
        aggregate = sum(row["_inference_s"] for row in completed)
        minute_metrics.append(
            {
                "video_minute": minute + 1,
                "source_start_s": minute * 60,
                "source_end_s": min((minute + 1) * 60, duration_s),
                "aligned_valid_frames": len(segment),
                "processed_frames": len(completed),
                "aggregate_worker_inference_s": aggregate,
                "estimated_four_worker_wall_s": aggregate / 4.0,
            }
        )

    max_worker_wall = max((item["wall_s"] for item in summaries), default=0)
    joint_quality = {}
    for joint in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow"):
        scores = sorted(
            float(row[f"{joint}_score"])
            for row in rows
            if row["status"] == "ok"
        )
        x_values = [
            float(row[f"{joint}_x"]) for row in rows if row["status"] == "ok"
        ]
        y_values = [
            float(row[f"{joint}_y"]) for row in rows if row["status"] == "ok"
        ]
        joint_quality[joint] = {
            "mean_score": statistics.mean(scores),
            "median_score": statistics.median(scores),
            "p05_score": scores[int(0.05 * (len(scores) - 1))],
            "below_0_3": sum(score < 0.3 for score in scores),
            "below_0_5": sum(score < 0.5 for score in scores),
            "out_of_image": sum(
                not (0 <= x < 1920 and 0 <= y < 1200)
                for x, y in zip(x_values, y_values)
            ),
        }
    report = {
        "formal_model": "Sapiens2-0.4B keypoints308, full-frame crop, flip test",
        "video": "module01_D45D2E00_CAM_C.h265",
        "image_size": [1920, 1200],
        "aligned_rows": len(aligned),
        "aligned_valid_cam_c_timestamps": len(valid_times),
        "source_duration_s": duration_s,
        "status_counts": status_counts,
        "joint_quality": joint_quality,
        "decode_mapping": decode_info,
        "workers": summaries,
        "predictions_loaded": len(predictions),
        "effective_full_run_wall_s_excluding_model_load": max_worker_wall,
        "effective_processing_s_per_video_minute": (
            max_worker_wall / duration_s * 60.0 if duration_s else None
        ),
        "per_video_minute": minute_metrics,
    }
    Path(args.output_report).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
