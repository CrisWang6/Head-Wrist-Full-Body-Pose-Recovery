#!/usr/bin/env python3
"""Build compact frame data and metrics for two optimized 3D pose sources."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


BODY = (
    "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)


def load_pose(path: Path) -> dict[int, dict[str, list[float]]]:
    frames: dict[int, dict[str, list[float]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["joint"] not in BODY:
                continue
            frames.setdefault(int(row["sequence"]), {})[row["joint"]] = [
                float(row["x_m"]), float(row["y_m"]), float(row["z_m"])]
    return frames


def temporal_steps(frames: dict[int, dict[str, list[float]]]) -> dict:
    values = []
    per_joint = {}
    sequences = sorted(frames)
    for joint in BODY:
        steps = []
        for first, second in zip(sequences, sequences[1:]):
            if second != first + 1 or joint not in frames[first] or joint not in frames[second]:
                continue
            steps.append(1000.0 * float(np.linalg.norm(
                np.asarray(frames[second][joint])-np.asarray(frames[first][joint]))))
        if steps:
            per_joint[joint] = {"median_mm_per_frame": float(np.median(steps)),
                                "p90_mm_per_frame": float(np.percentile(steps, 90))}
            values.extend(steps)
    return {"all_joints_median_mm_per_frame": float(np.median(values)),
            "all_joints_p90_mm_per_frame": float(np.percentile(values, 90)),
            "per_joint": per_joint}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--yolo-csv", type=Path, required=True)
    p.add_argument("--rtmpose-csv", type=Path, required=True)
    p.add_argument("--yolo-report", type=Path, required=True)
    p.add_argument("--rtmpose-report", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    yolo, rtmpose = load_pose(a.yolo_csv), load_pose(a.rtmpose_csv)
    sequences = sorted(set(yolo) & set(rtmpose))
    y_report = json.loads(a.yolo_report.read_text(encoding="utf-8"))
    r_report = json.loads(a.rtmpose_report.read_text(encoding="utf-8"))
    payload = {
        "joint_order": BODY,
        "frames": [{"sequence": seq,
                    "yolo": {k: [round(x, 5) for x in v] for k, v in yolo[seq].items()},
                    "rtmpose": {k: [round(x, 5) for x in v] for k, v in rtmpose[seq].items()}}
                   for seq in sequences],
        "metrics": {
            "yolo": {"temporal_step": temporal_steps(yolo),
                     "bone_targets_m": y_report["bone_targets_m"],
                     "nose_fit_median_mm": y_report["new_translation_fit_nose_error_median_mm"],
                     "reprojection_after_median_px": y_report["stereo_reprojection_after_median_px"],
                     "bone_error_after_median_mm": y_report["bone_error_after_median_mm"]},
            "rtmpose": {"temporal_step": temporal_steps(rtmpose),
                        "bone_targets_m": r_report["bone_targets_m"],
                        "nose_fit_median_mm": r_report["new_translation_fit_nose_error_median_mm"],
                        "reprojection_after_median_px": r_report["stereo_reprojection_after_median_px"],
                        "bone_error_after_median_mm": r_report["bone_error_after_median_mm"]},
        },
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                        encoding="utf-8")
    print(json.dumps({"frames": len(sequences), "output": str(a.output),
                      "metrics": payload["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
