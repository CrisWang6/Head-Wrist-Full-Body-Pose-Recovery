#!/usr/bin/env python3
"""Project CH07-nose/bone-optimized world skeletons into both head cameras."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from process_external_stereo_to_head import NAMES, Omni, draw_pose, load_json, qrot, writer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--world-csv", type=Path, required=True)
    p.add_argument("--aligned", type=Path, required=True)
    p.add_argument("--refinement", type=Path, required=True)
    p.add_argument("--head-intrinsics", type=Path, required=True)
    p.add_argument("--head-rigid", type=Path, required=True)
    p.add_argument("--head-a", type=Path, required=True)
    p.add_argument("--head-d", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ch07-event-offset", type=int, default=71)
    p.add_argument("--label", default="nose GT + statistical bones")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with a.world_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    points = {}
    for row in rows:
        points.setdefault(int(row["sequence"]), {})[row["joint"]] = np.asarray(
            [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])], np.float64)
    with a.aligned.open("r", encoding="utf-8-sig", newline="") as f:
        aligned = list(csv.DictReader(f))

    refinement = load_json(a.refinement)
    axis_r = np.asarray(refinement["R_ch07_axis_correction"], np.float64)
    axis_t = np.asarray(refinement["t_ch07_axis_correction_m"], np.float64)
    head_intrinsics = load_json(a.head_intrinsics)
    head_rigid = load_json(a.head_rigid)
    right_name = "CAM_D" if "CAM_D" in head_intrinsics["cameras"] else "CAM_C"
    cameras = {"A": Omni(head_intrinsics, "CAM_A"), "D": Omni(head_intrinsics, right_name)}
    transforms = {}
    for key, side in (("A", "left"), ("D", "right")):
        matrix = np.asarray(head_rigid["cameras"][side]["T_camera_rigid"], np.float64)
        matrix[:3, 3] /= 1000.0
        transforms[key] = matrix

    projected = {"A": {}, "D": {}}
    csv_rows = []
    for sequence, joints in points.items():
        ch07_index = sequence + a.ch07_event_offset
        if not 0 <= ch07_index < len(aligned):
            continue
        row = aligned[ch07_index]
        rotation = qrot([float(row[f"mocap_CH3_07_world_q{x}"]) for x in "wxyz"])
        translation = np.asarray([float(row[f"mocap_CH3_07_world_{x}"]) for x in "xyz"])
        for name, world in joints.items():
            # Face policy: nose only.
            if name != "nose" and name not in NAMES[5:]:
                continue
            ch07 = axis_r @ (rotation.T @ (world-translation)) + axis_t
            out = {"sequence": sequence, "joint": name,
                   "world_x_m": world[0], "world_y_m": world[1], "world_z_m": world[2],
                   "ch07_x_m": ch07[0], "ch07_y_m": ch07[1], "ch07_z_m": ch07[2]}
            for key in ("A", "D"):
                camera_point = (transforms[key] @ np.r_[ch07, 1.0])[:3]
                uv = cameras[key].project(camera_point)
                if uv is not None and -200 <= uv[0] < 2120 and -200 <= uv[1] < 1400:
                    projected[key].setdefault(sequence, {})[name] = uv
                    out[f"head_{key}_u_px"], out[f"head_{key}_v_px"] = uv
                else:
                    out[f"head_{key}_u_px"], out[f"head_{key}_v_px"] = float("nan"), float("nan")
            csv_rows.append(out)

    if csv_rows:
        with (a.output_dir/"optimized_world_to_head_2d.csv").open("w", encoding="utf-8-sig", newline="") as f:
            output = csv.DictWriter(f, fieldnames=list(csv_rows[0])); output.writeheader(); output.writerows(csv_rows)

    for key, video_path in (("A", a.head_a), ("D", a.head_d)):
        capture = cv2.VideoCapture(str(video_path))
        output = writer(a.output_dir/f"head_CAM_{key}_nose_bone_optimized_event.mp4", (1920, 1200), 50)
        sequence = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            draw_pose(frame, projected[key].get(sequence, {}), (0, 255, 255),
                      f"{a.label} -> HEAD_{key} seq={sequence}", body_only=True)
            output.write(frame)
            sequence += 1
        capture.release(); output.release()

    left_path = a.output_dir/"head_CAM_A_nose_bone_optimized_event.mp4"
    right_path = a.output_dir/"head_CAM_D_nose_bone_optimized_event.mp4"
    left_capture, right_capture = cv2.VideoCapture(str(left_path)), cv2.VideoCapture(str(right_path))
    stereo_output = writer(a.output_dir/"head_stereo_nose_bone_optimized_event.mp4", (1920, 600), 50)
    while True:
        ok_left, left_frame = left_capture.read(); ok_right, right_frame = right_capture.read()
        if not ok_left or not ok_right:
            break
        stereo_output.write(np.hstack((cv2.resize(left_frame, (960, 600)), cv2.resize(right_frame, (960, 600)))))
    left_capture.release(); right_capture.release(); stereo_output.release()

    report = {
        "frames_with_3d": len(points),
        "projected_rows": len(csv_rows),
        "ch07_event_offset_frames": a.ch07_event_offset,
        "face_policy": "nose only",
        "source": str(a.world_csv),
        "head_extrinsic": str(a.head_rigid),
        "head_extrinsic_schema": head_rigid.get("schema"),
        "axis_mode": refinement.get("axis_mode", "unspecified"),
        "R_ch07_axis_correction": axis_r.tolist(),
        "t_ch07_axis_correction_m": axis_t.tolist(),
        "label": a.label,
    }
    (a.output_dir/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
