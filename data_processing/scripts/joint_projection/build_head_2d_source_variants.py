#!/usr/bin/env python3
"""Build fair all-head-YOLO and all-external-projection 2-D variants."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from render_hybrid_first_third import (
    JOINTS, choose_candidate, load_candidates, load_projection, smooth_joint,
)


p = argparse.ArgumentParser()
p.add_argument("--candidate-a", type=Path, required=True)
p.add_argument("--candidate-d", type=Path, required=True)
p.add_argument("--projection-csv", type=Path, required=True)
p.add_argument("--output-dir", type=Path, required=True)
p.add_argument("--frames", type=int, required=True)
p.add_argument("--median-window", type=int, default=5)
a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)

projection = load_projection(a.projection_csv)
candidates = {
    "CAM_A": load_candidates(a.candidate_a),
    "CAM_D": load_candidates(a.candidate_d),
}
sequences = list(range(a.frames))


def filter_tracks(raw: np.ndarray, confidence: np.ndarray, source: list[list[str]]) -> np.ndarray:
    filtered = np.full_like(raw, np.nan)
    for joint_index, joint_name in enumerate(JOINTS):
        values = raw[:, joint_index].copy()
        valid = np.all(np.isfinite(values), axis=1)
        if not np.any(valid):
            continue
        valid_indices = np.flatnonzero(valid)
        for axis in range(2):
            values[:, axis] = np.interp(np.arange(len(values)), valid_indices, values[valid_indices, axis])
        interpolated = ~valid
        confidence[interpolated, joint_index] = 0.12
        for frame_index in np.flatnonzero(interpolated):
            source[frame_index][joint_index] += "_temporal_interpolation"
        filtered[:, joint_index] = smooth_joint(values, confidence[:, joint_index], joint_name, a.median_window)
    return filtered


def build_head_yolo(camera: str):
    raw = np.full((a.frames, len(JOINTS), 2), np.nan, np.float32)
    confidence = np.zeros((a.frames, len(JOINTS)), np.float32)
    source = [["head_yolo_missing"] * len(JOINTS) for _ in sequences]
    for sequence in sequences:
        projected = projection.get(sequence, {}).get(camera, {})
        record = candidates[camera][sequence] if sequence < len(candidates[camera]) else {"candidates": []}
        selected, _ = choose_candidate(record, projected)
        if selected is None:
            continue
        for joint_index, joint_name in enumerate(JOINTS):
            keypoint = selected.get("keypoints", {}).get(joint_name)
            if keypoint is None or float(keypoint[2]) < 0.10:
                continue
            x, y = map(float, keypoint[:2])
            if not (0.0 <= x < 1920.0 and 0.0 <= y < 1200.0):
                continue
            raw[sequence, joint_index] = (x, y)
            confidence[sequence, joint_index] = float(keypoint[2])
            source[sequence][joint_index] = "head_yolo_only"
    return raw, filter_tracks(raw, confidence, source), confidence, source


def build_external(camera: str):
    raw = np.full((a.frames, len(JOINTS), 2), np.nan, np.float32)
    confidence = np.zeros((a.frames, len(JOINTS)), np.float32)
    source = [["external_projection_missing"] * len(JOINTS) for _ in sequences]
    for sequence in sequences:
        frame = projection.get(sequence, {}).get(camera, {})
        for joint_index, joint_name in enumerate(JOINTS):
            if joint_name not in frame:
                continue
            uv = np.asarray(frame[joint_name], np.float32)
            if not (0.0 <= uv[0] < 1920.0 and 0.0 <= uv[1] < 1200.0):
                continue
            raw[sequence, joint_index] = uv
            confidence[sequence, joint_index] = 0.70
            source[sequence][joint_index] = "external_projection_only"
    return raw, filter_tracks(raw, confidence, source), confidence, source


def write_variant(path: Path, builder) -> None:
    fields = ["sequence", "camera", "source_frame_index", "joint", "source", "confidence", "raw_x", "raw_y", "filtered_x", "filtered_y"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for camera in ("CAM_A", "CAM_D"):
            raw, filtered, confidence, source = builder(camera)
            for sequence in sequences:
                for joint_index, joint_name in enumerate(JOINTS):
                    if not np.all(np.isfinite(filtered[sequence, joint_index])):
                        continue
                    writer.writerow({
                        "sequence": sequence, "camera": camera, "source_frame_index": sequence,
                        "joint": joint_name, "source": source[sequence][joint_index],
                        "confidence": float(confidence[sequence, joint_index]),
                        "raw_x": "" if not np.isfinite(raw[sequence, joint_index, 0]) else float(raw[sequence, joint_index, 0]),
                        "raw_y": "" if not np.isfinite(raw[sequence, joint_index, 1]) else float(raw[sequence, joint_index, 1]),
                        "filtered_x": float(filtered[sequence, joint_index, 0]),
                        "filtered_y": float(filtered[sequence, joint_index, 1]),
                    })


write_variant(a.output_dir / "head_yolo_only_2d.csv", build_head_yolo)
write_variant(a.output_dir / "external_projection_only_2d.csv", build_external)
print({"frames": a.frames, "outputs": ["head_yolo_only_2d.csv", "external_projection_only_2d.csv"]})
