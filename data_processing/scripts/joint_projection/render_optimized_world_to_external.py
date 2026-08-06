#!/usr/bin/env python3
"""Render an optimized world-space skeleton back into the external stereo pair."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import yaml

from process_external_stereo_to_head import Omni, draw_pose, load_json, writer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--world-csv", type=Path, required=True)
    p.add_argument("--optimization-report", type=Path, required=True)
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--external-stereo-yaml", type=Path, required=True)
    p.add_argument("--external-a", type=Path, required=True)
    p.add_argument("--external-d", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)

    points = {}
    with a.world_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            points.setdefault(int(row["sequence"]), {})[row["joint"]] = np.asarray(
                [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])], np.float64)

    report = load_json(a.optimization_report)
    world_r = np.asarray(report["R_world_ch01_preserved"], np.float64)
    world_t = np.asarray(report["t_world_ch01_nose_gt_translation_fit_m"], np.float64)
    intrinsics = load_json(a.calib/"handle_ac_intrinsics_kalibr_omni_1920x1200_20260729.json")
    cameras = {"A": Omni(intrinsics, "CAM_A"), "D": Omni(intrinsics, "CAM_C")}
    rigid = load_json(a.calib/"external_stereo_rigid_k_extrinsics.json")
    r_ch01_left = np.asarray(rigid["cameras"]["left"]["R_rigid_camera"], np.float64)
    o_ch01_left = np.asarray(rigid["cameras"]["left"]["p_rigid_camera_mm"], np.float64)/1000.0
    stereo = yaml.safe_load(a.external_stereo_yaml.read_text(encoding="utf-8"))
    transform = np.asarray(stereo["cam1"]["T_cn_cnm1"], np.float64)
    r_rl, t_rl = transform[:3, :3], transform[:3, 3]

    projected = {"A": {}, "D": {}}
    rows = []
    for sequence, joints in points.items():
        for name, world in joints.items():
            ch01 = world_r.T @ (world-world_t)
            left = r_ch01_left.T @ (ch01-o_ch01_left)
            for camera_name, camera_point in (("A", left), ("D", r_rl @ left+t_rl)):
                uv = cameras[camera_name].project(camera_point)
                if uv is None:
                    continue
                upright = np.asarray([1920.0-uv[0], 1200.0-uv[1]])
                projected[camera_name].setdefault(sequence, {})[name] = upright
                rows.append({"sequence": sequence, "camera": camera_name, "joint": name,
                             "u_px": upright[0], "v_px": upright[1]})

    with (a.output_dir/"optimized_external_2d.csv").open("w", encoding="utf-8-sig", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(rows[0])); out.writeheader(); out.writerows(rows)

    for camera_name, video_path in (("A", a.external_a), ("D", a.external_d)):
        capture = cv2.VideoCapture(str(video_path))
        output = writer(a.output_dir/f"external_CAM_{camera_name}_optimized_pose_event.mp4", (1920, 1200), 50)
        sequence = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            draw_pose(frame, projected[camera_name].get(sequence, {}), (0, 255, 255),
                      f"optimized 3D reprojected onto fixed external 2D GT seq={sequence}",
                      body_only=True)
            output.write(frame)
            sequence += 1
        capture.release(); output.release()


if __name__ == "__main__":
    main()
