#!/usr/bin/env python3
"""Triangulate the rendered head A/D 2-D skeleton and visualize it in 3-D."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from process_external_stereo_to_head import Omni, load_json, qrot


JOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
EDGES = [
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
]


p = argparse.ArgumentParser()
p.add_argument("--head-2d", type=Path, required=True)
p.add_argument("--aligned", type=Path, required=True)
p.add_argument("--calib", type=Path, required=True)
p.add_argument("--refinement", type=Path, required=True)
p.add_argument("--output-dir", type=Path, required=True)
p.add_argument("--ray-gap-max-m", type=float, default=0.25)
a = p.parse_args(); a.output_dir.mkdir(parents=True, exist_ok=True)


def closest_rays(origin_a, direction_a, origin_d, direction_d):
    delta = origin_a - origin_d
    aa = float(direction_a @ direction_a); ad = float(direction_a @ direction_d)
    dd = float(direction_d @ direction_d); ar = float(direction_a @ delta); dr = float(direction_d @ delta)
    denominator = aa * dd - ad * ad
    if abs(denominator) < 1e-10:
        return None
    depth_a = (ad * dr - dd * ar) / denominator
    depth_d = (aa * dr - ad * ar) / denominator
    point_a = origin_a + depth_a * direction_a
    point_d = origin_d + depth_d * direction_d
    return (point_a + point_d) * 0.5, float(np.linalg.norm(point_a - point_d)), depth_a, depth_d


intrinsics = load_json(a.calib / "head_intrinsics_kalibr_omni_1920x1200.json")
rigid = load_json(a.calib / "head_stereo_rigid_extrinsics.json")
cameras = {"CAM_A": Omni(intrinsics, "CAM_A"), "CAM_D": Omni(intrinsics, "CAM_C")}
origins = {}; rotations = {}
for camera, side in (("CAM_A", "left"), ("CAM_D", "right")):
    transform = np.asarray(rigid["cameras"][side]["T_rigid_camera"], np.float64)
    rotations[camera] = transform[:3, :3]
    origins[camera] = transform[:3, 3] / 1000.0

with a.head_2d.open("r", encoding="utf-8-sig", newline="") as f:
    rows_2d = list(csv.DictReader(f))
points_2d = {}
for row in rows_2d:
    try:
        uv = np.asarray([float(row["filtered_x"]), float(row["filtered_y"])], np.float64)
        confidence = float(row["confidence"])
    except (TypeError, ValueError):
        continue
    if not np.all(np.isfinite(uv)) or confidence < 0.10:
        continue
    # Do not extrapolate rays from annotations outside the physical image.
    if not (0.0 <= uv[0] < 1920.0 and 0.0 <= uv[1] < 1200.0):
        continue
    key = (int(row["sequence"]), row["joint"])
    points_2d.setdefault(key, {})[row["camera"]] = (uv, confidence, row["source"])

with a.aligned.open("r", encoding="utf-8-sig", newline="") as f:
    aligned = list(csv.DictReader(f))
refinement = load_json(a.refinement)
axis_rotation = np.asarray(refinement["R_ch07_axis_correction"], np.float64)
axis_translation = np.asarray(refinement["t_ch07_axis_correction_m"], np.float64)
event_offset = int(refinement.get("ch07_event_offset_frames", 0))

raw = {}
diagnostics = []
sequences = sorted({key[0] for key in points_2d})
for sequence in sequences:
    shifted = sequence + event_offset
    if not (0 <= shifted < len(aligned)):
        continue
    row = aligned[shifted]
    rotation_world_ch07 = qrot([float(row[f"mocap_CH3_07_world_q{axis}"]) for axis in "wxyz"])
    translation_world_ch07 = np.asarray([float(row[f"mocap_CH3_07_world_{axis}"]) for axis in "xyz"], np.float64)
    frame = {}
    for joint in JOINTS:
        pair = points_2d.get((sequence, joint), {})
        if "CAM_A" not in pair or "CAM_D" not in pair:
            continue
        uv_a, confidence_a, source_a = pair["CAM_A"]
        uv_d, confidence_d, source_d = pair["CAM_D"]
        direction_a = rotations["CAM_A"] @ cameras["CAM_A"].ray(uv_a)
        direction_d = rotations["CAM_D"] @ cameras["CAM_D"].ray(uv_d)
        answer = closest_rays(origins["CAM_A"], direction_a, origins["CAM_D"], direction_d)
        if answer is None:
            continue
        point_head, ray_gap, depth_a, depth_d = answer
        if depth_a <= 0.0 or depth_d <= 0.0 or ray_gap > a.ray_gap_max_m:
            continue
        # Inverse of the final_v3 world -> corrected CH07 chain used to draw
        # the head-view skeleton.
        point_world = rotation_world_ch07 @ (axis_rotation.T @ (point_head - axis_translation)) + translation_world_ch07
        frame[joint] = point_world
        diagnostics.append({
            "sequence": sequence, "joint": joint, "ray_gap_m": ray_gap,
            "confidence": min(confidence_a, confidence_d),
            "source_a": source_a, "source_d": source_d,
        })
    raw[sequence] = frame

# Mocap world Y is vertical for this setup.  The adjusted version changes only
# the per-frame global Y translation, preserving every bone vector exactly.
adjusted = {}; frame_shifts = {}
for sequence, frame in raw.items():
    if frame:
        minimum_y = min(point[1] for point in frame.values())
        shift_y = 0.10 - minimum_y
    else:
        shift_y = 0.0
    frame_shifts[sequence] = shift_y
    adjusted[sequence] = {
        joint: point + np.asarray([0.0, shift_y, 0.0])
        for joint, point in frame.items()
    }


def write_points(path: Path, data: dict, variant: str) -> None:
    fields = ["sequence", "joint", "x_m", "y_m", "z_m", "variant", "frame_vertical_shift_m"]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader()
        for sequence in sorted(data):
            for joint, point in data[sequence].items():
                writer.writerow({
                    "sequence": sequence, "joint": joint,
                    "x_m": point[0], "y_m": point[1], "z_m": point[2],
                    "variant": variant,
                    "frame_vertical_shift_m": 0.0 if variant == "raw" else frame_shifts[sequence],
                })


write_points(a.output_dir / "head_stereo_3d_raw_world.csv", raw, "raw")
write_points(a.output_dir / "head_stereo_3d_min10cm_world.csv", adjusted, "min10cm")
with (a.output_dir / "head_stereo_3d_diagnostics.csv").open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(diagnostics[0])); writer.writeheader(); writer.writerows(diagnostics)

all_raw = np.asarray([point for frame in raw.values() for point in frame.values()])
center_x = float(np.median(all_raw[:, 0])); center_z = float(np.median(all_raw[:, 2]))
vertical_values = all_raw[:, 1]
y_lo = min(-0.05, float(np.percentile(vertical_values, 1) - 0.15))
y_hi = max(2.15, float(np.percentile(vertical_values, 99) + 0.15))
scale = min(350.0, 700.0 / max(y_hi - y_lo, 0.5))


def project(point: np.ndarray, panel_left: int) -> tuple[int, int]:
    x, y, z = point
    screen_x = panel_left + 480 + scale * ((x - center_x) + 0.36 * (z - center_z))
    screen_y = 820 - scale * ((y - y_lo) - 0.20 * (z - center_z))
    return int(round(screen_x)), int(round(screen_y))


def draw_panel(canvas: np.ndarray, data: dict, sequence: int, panel_left: int, title: str, adjusted_panel: bool) -> None:
    cv2.rectangle(canvas, (panel_left, 0), (panel_left + 959, 959), (22, 22, 26), -1)
    floor_y = 0.10 if adjusted_panel else 0.0
    floor_points = [
        np.asarray([center_x + dx, floor_y, center_z + dz])
        for dx, dz in ((-0.9, -0.9), (0.9, -0.9), (0.9, 0.9), (-0.9, 0.9))
    ]
    floor_uv = np.asarray([project(point, panel_left) for point in floor_points], np.int32)
    cv2.polylines(canvas, [floor_uv], True, (70, 70, 78), 2, cv2.LINE_AA)
    for fraction in (-0.5, 0.0, 0.5):
        a0 = project(np.asarray([center_x - 0.9, floor_y, center_z + fraction]), panel_left)
        a1 = project(np.asarray([center_x + 0.9, floor_y, center_z + fraction]), panel_left)
        cv2.line(canvas, a0, a1, (45, 45, 52), 1, cv2.LINE_AA)
    frame = data.get(sequence, {})
    for first, second in EDGES:
        if first in frame and second in frame:
            cv2.line(canvas, project(frame[first], panel_left), project(frame[second], panel_left), (0, 225, 255), 5, cv2.LINE_AA)
    for joint, point in frame.items():
        cv2.circle(canvas, project(point, panel_left), 6, (20, 80, 255), -1, cv2.LINE_AA)
    cv2.putText(canvas, title, (panel_left + 28, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (240, 240, 245), 2, cv2.LINE_AA)
    cv2.putText(canvas, "mocap world Y = vertical", (panel_left + 28, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 180), 1, cv2.LINE_AA)
    if frame:
        minimum = min(point[1] for point in frame.values())
        cv2.putText(canvas, f"lowest valid joint: {minimum * 100:.1f} cm", (panel_left + 28, 916), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 225), 2, cv2.LINE_AA)


fourcc = cv2.VideoWriter_fourcc(*"mp4v")
comparison = cv2.VideoWriter(str(a.output_dir / "head_stereo_3d_raw_vs_min10cm.mp4"), fourcc, 50, (1920, 960))
raw_video = cv2.VideoWriter(str(a.output_dir / "head_stereo_3d_raw.mp4"), fourcc, 50, (960, 960))
adjusted_video = cv2.VideoWriter(str(a.output_dir / "head_stereo_3d_min10cm.mp4"), fourcc, 50, (960, 960))
for sequence in sequences:
    canvas = np.zeros((960, 1920, 3), np.uint8)
    draw_panel(canvas, raw, sequence, 0, "Raw head-stereo 3D", False)
    draw_panel(canvas, adjusted, sequence, 960, "Lowest joint fixed at 10 cm", True)
    cv2.putText(canvas, f"event {sequence}", (875, 940), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 150, 160), 1, cv2.LINE_AA)
    comparison.write(canvas); raw_video.write(canvas[:, :960]); adjusted_video.write(canvas[:, 960:])
comparison.release(); raw_video.release(); adjusted_video.release()

gaps = np.asarray([row["ray_gap_m"] for row in diagnostics])
valid_counts = np.asarray([len(frame) for frame in raw.values()])
report = {
    "frames": len(sequences),
    "valid_3d_points": len(diagnostics),
    "median_valid_joints_per_frame": float(np.median(valid_counts)),
    "median_ray_gap_mm": float(np.median(gaps) * 1000.0),
    "p90_ray_gap_mm": float(np.percentile(gaps, 90) * 1000.0),
    "coordinate_frame": "mocap world; Y is vertical",
    "adjustment": "per-frame Y translation only; minimum valid joint Y equals 0.10 m",
    "ch07_event_offset_frames": event_offset,
}
(a.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report))
