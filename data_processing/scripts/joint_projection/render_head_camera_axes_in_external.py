#!/usr/bin/env python3
"""Render both head-camera coordinate frames in the two external cameras."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter

from process_external_stereo_to_head import Omni, load_json, qrot, writer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--aligned", type=Path, required=True)
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--refinement", type=Path, required=True)
    p.add_argument("--nose-report", type=Path, required=True)
    p.add_argument("--nose-fit-csv", type=Path)
    p.add_argument("--external-stereo-yaml", type=Path, required=True)
    p.add_argument("--external-a", type=Path, required=True)
    p.add_argument("--external-d", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ch07-event-offset", type=int, default=71)
    p.add_argument("--axis-length-m", type=float, default=.15)
    p.add_argument("--only-frame", choices=("CH07_RAW", "CH07_CORR", "HEAD_A", "HEAD_D"))
    p.add_argument("--origin-dot-only", action="store_true")
    p.add_argument("--smooth-origin-window", type=int, default=11)
    p.add_argument("--show-nose", action="store_true",
                   help="Overlay filtered observed nose and CH07-derived GT nose.")
    return p.parse_args()


def draw_axes(image, projected, frame_name, thickness, origin_color=(0, 255, 255), label_offset=(10, -10)):
    origin = projected.get("origin")
    if origin is None:
        return
    origin_i = tuple(np.rint(origin).astype(int))
    cv2.circle(image, origin_i, 8, origin_color, -1, cv2.LINE_AA)
    cv2.putText(image, frame_name, (origin_i[0]+label_offset[0], origin_i[1]+label_offset[1]),
                cv2.FONT_HERSHEY_SIMPLEX, .65, origin_color, 2, cv2.LINE_AA)
    colors = {"x": (0, 0, 255), "y": (0, 255, 0), "z": (255, 0, 0)}
    for axis in "xyz":
        endpoint = projected.get(axis)
        if endpoint is None:
            continue
        endpoint_i = tuple(np.rint(endpoint).astype(int))
        cv2.arrowedLine(image, origin_i, endpoint_i, colors[axis], thickness, cv2.LINE_AA, tipLength=.18)


def main() -> None:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with a.aligned.open("r", encoding="utf-8-sig", newline="") as f:
        aligned = list(csv.DictReader(f))
    refinement = load_json(a.refinement)
    axis_r = np.asarray(refinement["R_ch07_axis_correction"], np.float64)
    axis_t = np.asarray(refinement["t_ch07_axis_correction_m"], np.float64)
    nose_report = load_json(a.nose_report)
    if "R_world_ch01_2d_gt_fit" in nose_report:
        world_r = np.asarray(nose_report["R_world_ch01_2d_gt_fit"], np.float64)
        world_t = np.asarray(nose_report["t_world_ch01_2d_gt_fit_m"], np.float64)
        mapping_label = "nose 2D-GT constrained fit"
    else:
        world_r = np.asarray(nose_report["R_world_ch01_preserved"], np.float64)
        world_t = np.asarray(nose_report["t_world_ch01_nose_gt_translation_fit_m"], np.float64)
        mapping_label = "nose-GT translation fit"

    nose_fit = {"A": {}, "D": {}}
    if a.nose_fit_csv:
        with a.nose_fit_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                sequence = int(row["sequence"])
                for camera_name in ("A", "D"):
                    observed = np.asarray([float(row[f"{camera_name}_observed_u_px"]),
                                           float(row[f"{camera_name}_observed_v_px"])] )
                    fitted = np.asarray([float(row[f"{camera_name}_fitted_u_px"]),
                                         float(row[f"{camera_name}_fitted_v_px"])] )
                    nose_fit[camera_name][sequence] = {
                        "observed": np.asarray([1920.0-observed[0], 1200.0-observed[1]]),
                        "fitted": np.asarray([1920.0-fitted[0], 1200.0-fitted[1]]),
                        "error": float(row[f"{camera_name}_fitted_error_px"]),
                    }

    head_rigid = load_json(a.calib/"head_stereo_rigid_extrinsics.json")
    inverse_head = {}
    for key, side in (("A", "left"), ("D", "right")):
        transform = np.asarray(head_rigid["cameras"][side]["T_camera_rigid"], np.float64)
        transform[:3, 3] /= 1000.0
        inverse_head[key] = np.linalg.inv(transform)

    external_intrinsics = load_json(a.calib/"handle_ac_intrinsics_kalibr_omni_1920x1200_20260729.json")
    external_rigid = load_json(a.calib/"external_stereo_rigid_k_extrinsics.json")
    stereo = yaml.safe_load(a.external_stereo_yaml.read_text(encoding="utf-8"))
    right_left = np.asarray(stereo["cam1"]["T_cn_cnm1"], np.float64)
    r_right_left, t_right_left = right_left[:3, :3], right_left[:3, 3]
    r_ch01_left = np.asarray(external_rigid["cameras"]["left"]["R_rigid_camera"], np.float64)
    o_ch01_left = np.asarray(external_rigid["cameras"]["left"]["p_rigid_camera_mm"], np.float64)/1000.0
    external_cameras = {"A": Omni(external_intrinsics, "CAM_A"), "D": Omni(external_intrinsics, "CAM_C")}

    def project_world_frame(world_points, external_name, sequence, frame_name):
        projected = {}
        for point_name, world in world_points.items():
            ch01 = world_r.T @ (world-world_t)
            left_camera = r_ch01_left.T @ (ch01-o_ch01_left)
            external_camera = left_camera if external_name == "A" else r_right_left @ left_camera + t_right_left
            uv = external_cameras[external_name].project(external_camera)
            if uv is not None:
                projected[point_name] = np.asarray([1920.0-uv[0], 1200.0-uv[1]])
                csv_rows.append({"sequence": sequence, "external_camera": external_name,
                                 "coordinate_frame": frame_name, "point": point_name,
                                 "u_px": projected[point_name][0], "v_px": projected[point_name][1],
                                 "world_x_m": world[0], "world_y_m": world[1], "world_z_m": world[2]})
        projections[external_name].setdefault(sequence, {})[frame_name] = projected

    camera_points = {
        "origin": np.array([0.0, 0.0, 0.0]),
        "x": np.array([a.axis_length_m, 0.0, 0.0]),
        "y": np.array([0.0, a.axis_length_m, 0.0]),
        "z": np.array([0.0, 0.0, a.axis_length_m]),
    }
    projections = {"A": {}, "D": {}}
    csv_rows = []
    for sequence in range(min(344, len(aligned)-a.ch07_event_offset)):
        ch07 = aligned[sequence+a.ch07_event_offset]
        r07 = qrot([float(ch07[f"mocap_CH3_07_world_q{x}"]) for x in "wxyz"])
        t07 = np.asarray([float(ch07[f"mocap_CH3_07_world_{x}"]) for x in "xyz"])
        raw_rigid_world = {point_name: r07 @ point + t07 for point_name, point in camera_points.items()}
        corrected_rigid_world = {
            point_name: r07 @ (axis_r.T @ (point-axis_t)) + t07
            for point_name, point in camera_points.items()
        }
        for external_name in ("A", "D"):
            project_world_frame(raw_rigid_world, external_name, sequence, "CH07_RAW")
            project_world_frame(corrected_rigid_world, external_name, sequence, "CH07_CORR")
        for head_name, inverse in inverse_head.items():
            world_points = {}
            for point_name, camera_point in camera_points.items():
                corrected_ch07 = (inverse @ np.r_[camera_point, 1.0])[:3]
                raw_ch07 = axis_r.T @ (corrected_ch07-axis_t)
                world_points[point_name] = r07 @ raw_ch07 + t07
            for external_name in ("A", "D"):
                project_world_frame(world_points, external_name, sequence, f"HEAD_{head_name}")

    if a.origin_dot_only and a.only_frame:
        for external_name in ("A", "D"):
            sequences = sorted(projections[external_name])
            points = np.asarray([
                projections[external_name][sequence][a.only_frame].get("origin", [np.nan, np.nan])
                for sequence in sequences
            ], np.float64)
            valid = np.isfinite(points).all(axis=1)
            if valid.sum() >= 5:
                indices = np.arange(len(points))
                window = min(a.smooth_origin_window, int(valid.sum()) if int(valid.sum()) % 2 else int(valid.sum())-1)
                window = max(5, window if window % 2 else window-1)
                for coordinate in range(2):
                    filled = np.interp(indices, indices[valid], points[valid, coordinate])
                    filled = median_filter(filled, size=3, mode="nearest")
                    points[:, coordinate] = savgol_filter(filled, window, 2, mode="interp")
                for sequence, point in zip(sequences, points):
                    projections[external_name][sequence][a.only_frame]["origin"] = point

    if csv_rows:
        with (a.output_dir/"head_camera_axes_in_external_2d.csv").open("w", encoding="utf-8-sig", newline="") as f:
            output = csv.DictWriter(f, fieldnames=list(csv_rows[0])); output.writeheader(); output.writerows(csv_rows)

    for external_name, video_path in (("A", a.external_a), ("D", a.external_d)):
        capture = cv2.VideoCapture(str(video_path))
        output = writer(a.output_dir/f"external_CAM_{external_name}_head_camera_axes_event.mp4", (1920, 1200), 50)
        sequence = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            frames = projections[external_name].get(sequence, {})
            frame_styles = {
                "CH07_RAW": (6, (255, 255, 255), (-115, -20)),
                "CH07_CORR": (4, (255, 0, 255), (-125, 25)),
                "HEAD_A": (3, (0, 255, 255), (10, -16)),
                "HEAD_D": (2, (0, 200, 255), (10, 24)),
            }
            names = (a.only_frame,) if a.only_frame else tuple(frame_styles)
            for name in names:
                thickness, color, offset = frame_styles[name]
                if a.origin_dot_only:
                    origin = frames.get(name, {}).get("origin")
                    if origin is not None:
                        cv2.circle(frame, tuple(np.rint(origin).astype(int)), 5, (0, 0, 255), -1, cv2.LINE_AA)
                else:
                    draw_axes(frame, frames.get(name, {}), name, thickness, color, offset)
            fit = nose_fit[external_name].get(sequence)
            if fit and (a.show_nose or not a.only_frame):
                observed = tuple(np.rint(fit["observed"]).astype(int))
                fitted = tuple(np.rint(fit["fitted"]).astype(int))
                cv2.circle(frame, observed, 6, (255, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, fitted, 6, (255, 0, 255), -1, cv2.LINE_AA)
                if a.origin_dot_only and a.only_frame:
                    origin = frames.get(a.only_frame, {}).get("origin")
                    if origin is not None:
                        cv2.line(frame, tuple(np.rint(origin).astype(int)), fitted,
                                 (0, 180, 255), 2, cv2.LINE_AA)
                else:
                    cv2.line(frame, observed, fitted, (255, 255, 255), 2, cv2.LINE_AA)
            if a.only_frame:
                legend = ("CH07 red / filtered nose cyan / CH07 GT nose magenta"
                          if a.origin_dot_only and a.show_nose else
                          (f"{a.only_frame} origin (smoothed)" if a.origin_dot_only else
                           f"{a.only_frame} only / X red / Y green / Z blue"))
                cv2.putText(frame, legend,
                            (24, 38), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.putText(frame, "nose observed=cyan cross / GT reprojection=magenta circle",
                            (24, 38), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "CH07 raw=white / corrected=magenta / X red / Y green / Z blue",
                            (24, 72), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2, cv2.LINE_AA)
            output.write(frame)
            sequence += 1
        capture.release(); output.release()

    report = {"frames": len(projections["A"]), "ch07_event_offset_frames": a.ch07_event_offset,
              "axis_length_m": a.axis_length_m, "external_world_mapping": mapping_label,
              "colors": {"X": "red", "Y": "green", "Z": "blue"}}
    (a.output_dir/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
