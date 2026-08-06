#!/usr/bin/env python3
"""Reproduce the validated final_v3 refinement on a new recording.

The external-stereo nose trajectory is aligned to CH07 world translation with
multi-frame Kabsch.  The mechanically validated CH07 correction stays Z +90
degrees, and only then is the shared CH07 translation fitted from head-view
shoulder/elbow/wrist observations.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from process_external_stereo_to_head import Omni, load_json, qrot


p = argparse.ArgumentParser()
p.add_argument("--stereo-json", type=Path, required=True)
p.add_argument("--aligned", type=Path, required=True)
p.add_argument("--candidate-a", type=Path, required=True)
p.add_argument("--candidate-d", type=Path, required=True)
p.add_argument("--calib", type=Path, required=True)
p.add_argument("--output-refinement", type=Path, required=True)
p.add_argument("--output-report", type=Path, required=True)
p.add_argument("--offset-min", type=int, default=-120)
p.add_argument("--offset-max", type=int, default=120)
a = p.parse_args()

records = load_json(a.stereo_json)
with a.aligned.open("r", encoding="utf-8-sig", newline="") as f:
    aligned = list(csv.DictReader(f))


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0); target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_center).T @ (target - target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return rotation, target_center - rotation @ source_center


def ch07_position(row: dict) -> np.ndarray:
    return np.asarray([float(row[f"mocap_CH3_07_world_{axis}"]) for axis in "xyz"], np.float64)


def ch07_pose(row: dict) -> tuple[np.ndarray, np.ndarray]:
    rotation = qrot([float(row[f"mocap_CH3_07_world_q{axis}"]) for axis in "wxyz"])
    return rotation, ch07_position(row)


nose = {}
for record in records:
    joint = record["joints"].get("nose")
    if joint and float(joint.get("confidence", 0.0)) >= 0.35 and float(joint.get("ray_gap_m", 1.0)) <= 0.08:
        nose[int(record["sequence"])] = np.asarray(joint.get("xyz_smooth", joint["xyz"]), np.float64)

# Every candidate offset is evaluated on the same central sequence set.  This
# avoids falsely favoring a boundary offset merely because it uses fewer frames.
common_lo = max(0, -a.offset_min)
common_hi = min(len(aligned), len(aligned) - a.offset_max)
common_sequences = [seq for seq in sorted(nose) if common_lo <= seq < common_hi]
if len(common_sequences) < 80:
    # Short clips may not leave enough central samples for a wide scan.
    radius = min(60, max(5, len(aligned) // 4))
    common_lo = radius; common_hi = len(aligned) - radius
    common_sequences = [seq for seq in sorted(nose) if common_lo <= seq < common_hi]

scan = []
for offset in range(a.offset_min, a.offset_max + 1):
    valid = [seq for seq in common_sequences if 0 <= seq + offset < len(aligned)]
    if len(valid) < 40:
        continue
    source = np.asarray([nose[seq] for seq in valid])
    target = np.asarray([ch07_position(aligned[seq + offset]) for seq in valid])
    rotation, translation = kabsch(source, target)
    errors = np.linalg.norm(source @ rotation.T + translation - target, axis=1)
    scan.append({
        "offset_frames": offset,
        "median_anchor_error_m": float(np.median(errors)),
        "p90_anchor_error_m": float(np.percentile(errors, 90)),
        "samples": len(valid),
        "rotation": rotation,
        "translation": translation,
    })

best = min(scan, key=lambda item: (item["median_anchor_error_m"], item["p90_anchor_error_m"]))
offset = int(best["offset_frames"])

# Refit Kabsch with every valid high-quality nose sample at the chosen offset.
valid = [seq for seq in sorted(nose) if 0 <= seq + offset < len(aligned)]
source = np.asarray([nose[seq] for seq in valid])
target = np.asarray([ch07_position(aligned[seq + offset]) for seq in valid])
world_rotation, world_translation = kabsch(source, target)
anchor_errors = np.linalg.norm(source @ world_rotation.T + world_translation - target, axis=1)


def dominant_candidates(path: Path) -> dict[int, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line); people = row.get("candidates", [])
        if not people:
            continue
        def score(person):
            x1, y1, x2, y2 = person["box_xyxy"]
            return (x2 - x1) * (y2 - y1) * (0.7 + 0.3 * float(person["box_confidence"]))
        result[int(row["frame_index"])] = max(people, key=score)
    return result


intrinsics = load_json(a.calib / "head_intrinsics_kalibr_omni_1920x1200.json")
head_rigid = load_json(a.calib / "head_stereo_rigid_extrinsics.json")
cameras = {"A": Omni(intrinsics, "CAM_A"), "D": Omni(intrinsics, "CAM_C")}
head_transforms = {}
for camera, side in (("A", "left"), ("D", "right")):
    transform = np.asarray(head_rigid["cameras"][side]["T_camera_rigid"], np.float64)
    transform[:3, 3] /= 1000.0
    head_transforms[camera] = transform

axis_rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], np.float64)
record_by_sequence = {int(record["sequence"]): record for record in records}
observations = []
for camera, path in (("A", a.candidate_a), ("D", a.candidate_d)):
    for seq, person in dominant_candidates(path).items():
        shifted = seq + offset
        record = record_by_sequence.get(seq)
        if record is None or not (0 <= shifted < len(aligned)):
            continue
        ch07_rotation, ch07_translation = ch07_pose(aligned[shifted])
        for joint_name in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"):
            joint = record["joints"].get(joint_name)
            keypoint = person.get("keypoints", {}).get(joint_name)
            if joint is None or keypoint is None or float(keypoint[2]) < 0.22:
                continue
            p01 = np.asarray(joint.get("xyz_smooth", joint["xyz"]), np.float64)
            world = world_rotation @ p01 + world_translation
            p07_without_offset = axis_rotation @ (ch07_rotation.T @ (world - ch07_translation))
            observations.append((camera, p07_without_offset, np.asarray(keypoint[:2], np.float64), float(keypoint[2])))


def reprojection_residual(translation: np.ndarray) -> np.ndarray:
    residuals = []
    for camera, p07, observed, confidence in observations:
        point_camera = (head_transforms[camera] @ np.r_[p07 + translation, 1.0])[:3]
        projected = cameras[camera].project(point_camera)
        difference = np.asarray((500.0, 500.0) if projected is None else np.asarray(projected) - observed)
        residuals.extend(difference * np.sqrt(confidence))
    return np.asarray(residuals)


initial_translation = np.asarray([-0.00387265, -0.06366961, -0.14704196], np.float64)
fit = least_squares(
    reprojection_residual, initial_translation, bounds=(-0.8, 0.8),
    loss="soft_l1", f_scale=35.0, max_nfev=300,
)
pixel_errors = np.linalg.norm(reprojection_residual(fit.x).reshape(-1, 2), axis=1)

refinement = {
    "schema": "joint_projection.multiframe_axis_offset_refinement.v1",
    "dataset": "0711_214559",
    "R_world_ch01": world_rotation.tolist(),
    "t_world_ch01_m": world_translation.tolist(),
    "R_ch07_axis_correction": axis_rotation.tolist(),
    "t_ch07_axis_correction_m": fit.x.tolist(),
    "ch07_event_offset_frames": offset,
    "notes": [
        "Recomputed for this recording using the validated 0711_175408 final_v3 method.",
        "CH01-to-world is multi-frame Kabsch from the YOLO11x stereo nose trajectory to CH07 translation.",
        "The mechanically validated residual axis correction is CH07 Z +90 degrees.",
        "The shared translation is robustly fitted from head-view shoulder/elbow/wrist correspondences.",
    ],
}
report = {
    "chosen_ch07_event_offset_frames": offset,
    "chosen_ch07_event_offset_ms": offset * 20.0,
    "kabsch_samples": len(valid),
    "kabsch_median_anchor_error_mm": float(np.median(anchor_errors) * 1000.0),
    "kabsch_p90_anchor_error_mm": float(np.percentile(anchor_errors, 90) * 1000.0),
    "translation_observations": len(observations),
    "translation_ch07_m": fit.x.tolist(),
    "translation_fit_median_weighted_error_px": float(np.median(pixel_errors)),
    "translation_fit_p90_weighted_error_px": float(np.percentile(pixel_errors, 90)),
    "scan": [
        {key: value for key, value in item.items() if key not in {"rotation", "translation"}}
        for item in scan
    ],
}
a.output_refinement.write_text(json.dumps(refinement, indent=2), encoding="utf-8")
a.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report))
