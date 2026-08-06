#!/usr/bin/env python3
"""Calibrate the external rig/world pose from filtered 2D nose and CH07 nose GT."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.ndimage import median_filter
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

from process_external_stereo_to_head import Omni, load_json, qrot


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--nose-2d-csv", type=Path, required=True)
    p.add_argument("--aligned", type=Path, required=True)
    p.add_argument("--calib", type=Path, required=True)
    p.add_argument("--external-stereo-yaml", type=Path, required=True)
    p.add_argument("--initial-world-report", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ch07-event-offset", type=int, default=71)
    p.add_argument("--nose-offset-mm", type=float, nargs=3, default=(0.0, -15.0, -125.0))
    p.add_argument("--min-confidence", type=float, default=.75)
    p.add_argument("--use-raw-2d", action="store_true",
                   help="Use same-frame unfiltered nose detections for geometric calibration.")
    p.add_argument("--zero-phase-filter", action="store_true",
                   help="Apply centered median + Savitzky-Golay smoothing without temporal lag.")
    p.add_argument("--filter-window", type=int, default=11)
    p.add_argument("--fit-rotation", action="store_true",
                   help="Also fit a small rotation. Default keeps the validated Z+90 orientation fixed.")
    p.add_argument("--max-rotation-deg", type=float, default=2.0)
    p.add_argument("--max-translation-m", type=float, default=.08)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    with a.nose_2d_csv.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [row for row in csv.DictReader(f)
                if row["joint"] == "nose" and float(row["stereo_confidence"]) >= a.min_confidence]
    with a.aligned.open("r", encoding="utf-8-sig", newline="") as f:
        aligned = list(csv.DictReader(f))

    intrinsics = load_json(a.calib/"handle_ac_intrinsics_kalibr_omni_1920x1200_20260729.json")
    cameras = {"A": Omni(intrinsics, "CAM_A"), "D": Omni(intrinsics, "CAM_C")}
    rigid = load_json(a.calib/"external_stereo_rigid_k_extrinsics.json")
    r_ch01_left = np.asarray(rigid["cameras"]["left"]["R_rigid_camera"], np.float64)
    o_ch01_left = np.asarray(rigid["cameras"]["left"]["p_rigid_camera_mm"], np.float64)/1000.0
    stereo = yaml.safe_load(a.external_stereo_yaml.read_text(encoding="utf-8"))
    right_left = np.asarray(stereo["cam1"]["T_cn_cnm1"], np.float64)
    r_right_left, t_right_left = right_left[:3, :3], right_left[:3, 3]

    initial = load_json(a.initial_world_report)
    world_r0 = np.asarray(initial["R_world_ch01_preserved"], np.float64)
    world_t0 = np.asarray(initial["t_world_ch01_nose_gt_translation_fit_m"], np.float64)
    nose_offset = np.asarray(a.nose_offset_mm, np.float64)/1000.0

    observations = []
    for row in rows:
        sequence = int(row["sequence"])
        ch07_index = sequence + a.ch07_event_offset
        if not 0 <= ch07_index < len(aligned):
            continue
        motion = aligned[ch07_index]
        try:
            r07 = qrot([float(motion[f"mocap_CH3_07_world_q{x}"]) for x in "wxyz"])
            t07 = np.asarray([float(motion[f"mocap_CH3_07_world_{x}"]) for x in "xyz"])
        except (KeyError, TypeError, ValueError):
            continue
        nose_world = r07 @ nose_offset + t07
        left_keys = ("left_u_raw_px", "left_v_raw_px") if a.use_raw_2d else ("left_u_px", "left_v_px")
        right_keys = ("right_u_raw_px", "right_v_raw_px") if a.use_raw_2d else ("right_u_px", "right_v_px")
        observations.append({
            "sequence": sequence,
            "world": nose_world,
            "A": np.asarray([float(row[key]) for key in left_keys]),
            "D": np.asarray([float(row[key]) for key in right_keys]),
            "confidence": float(row["stereo_confidence"]),
        })

    if a.zero_phase_filter and len(observations) >= 5:
        window = min(a.filter_window, len(observations) if len(observations) % 2 else len(observations)-1)
        window = max(5, window if window % 2 else window-1)
        for camera_name in ("A", "D"):
            points = np.asarray([item[camera_name] for item in observations], np.float64)
            for coordinate in range(2):
                clean = median_filter(points[:, coordinate], size=3, mode="nearest")
                points[:, coordinate] = savgol_filter(clean, window_length=window, polyorder=2,
                                                      mode="interp")
            for item, point in zip(observations, points):
                item[camera_name] = point

    def unpack(parameter):
        if a.fit_rotation:
            return parameter[:3], parameter[3:]
        return np.zeros(3, np.float64), parameter

    def project(parameter, world, camera_name):
        rotvec, translation = unpack(parameter)
        delta_r = Rotation.from_rotvec(rotvec).as_matrix()
        point_ch01_initial = world_r0.T @ (world-world_t0)
        point_ch01 = delta_r @ point_ch01_initial + translation
        point_left = r_ch01_left.T @ (point_ch01-o_ch01_left)
        point_camera = point_left if camera_name == "A" else r_right_left @ point_left + t_right_left
        return cameras[camera_name].project(point_camera), point_camera

    def reprojection_errors(parameter):
        errors = []
        for observation in observations:
            weight = np.sqrt(observation["confidence"])
            for camera_name in ("A", "D"):
                uv, _ = project(parameter, observation["world"], camera_name)
                if uv is not None:
                    errors.extend((np.asarray(uv)-observation[camera_name])*weight)
        # Scale priors with the observation count so hundreds of pixels cannot overwhelm physics.
        rotvec, translation = unpack(parameter)
        prior_weight = np.sqrt(max(1, len(observations)*2))
        if a.fit_rotation:
            errors.extend(rotvec/np.deg2rad(1.0)*prior_weight)
        errors.extend(translation/.03*prior_weight)
        return np.asarray(errors, np.float64)

    initial_parameter = np.zeros(6 if a.fit_rotation else 3, np.float64)
    if a.fit_rotation:
        lower = np.r_[-np.full(3, np.deg2rad(a.max_rotation_deg)),
                      -np.full(3, a.max_translation_m)]
        upper = -lower
    else:
        lower = -np.full(3, a.max_translation_m)
        upper = -lower
    solution = least_squares(reprojection_errors, initial_parameter, loss="soft_l1", f_scale=3.0,
                             bounds=(lower, upper), max_nfev=300,
                             ftol=1e-10, xtol=1e-10, gtol=1e-10)
    solved_rotvec, solved_translation = unpack(solution.x)
    delta_r = Rotation.from_rotvec(solved_rotvec).as_matrix()
    world_r = world_r0 @ delta_r.T
    world_t = world_t0 - world_r @ solved_translation

    output_rows = []
    before_errors, after_errors = {"A": [], "D": []}, {"A": [], "D": []}
    for observation in observations:
        out = {"sequence": observation["sequence"], "confidence": observation["confidence"]}
        for camera_name in ("A", "D"):
            before_uv, before_camera = project(initial_parameter, observation["world"], camera_name)
            after_uv, after_camera = project(solution.x, observation["world"], camera_name)
            observed = observation[camera_name]
            before_error = float(np.linalg.norm(np.asarray(before_uv)-observed))
            after_error = float(np.linalg.norm(np.asarray(after_uv)-observed))
            before_errors[camera_name].append(before_error); after_errors[camera_name].append(after_error)
            for axis, value in zip("uv", observed): out[f"{camera_name}_observed_{axis}_px"] = value
            for axis, value in zip("uv", before_uv): out[f"{camera_name}_before_{axis}_px"] = value
            for axis, value in zip("uv", after_uv): out[f"{camera_name}_fitted_{axis}_px"] = value
            out[f"{camera_name}_before_error_px"] = before_error
            out[f"{camera_name}_fitted_error_px"] = after_error
            out[f"{camera_name}_fitted_z_m"] = float(after_camera[2])
            out[f"{camera_name}_fitted_range_m"] = float(np.linalg.norm(after_camera))
        output_rows.append(out)

    with (a.output_dir/"nose_2d_gt_reprojection.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0])); writer.writeheader(); writer.writerows(output_rows)

    def stats(values):
        return {"median_px": float(np.median(values)), "p90_px": float(np.percentile(values, 90))}
    report = {
        "frames": len(observations),
        "nose_offset_ch07_mm": list(map(float, a.nose_offset_mm)),
        "ch07_event_offset_frames": a.ch07_event_offset,
        "optimization": "joint robust 2D-3D fit of a common external-rig SE3 correction; stereo relation fixed",
        "nose_2d_source": (("raw same-frame detections + centered zero-phase median/Savgol"
                            if a.use_raw_2d else "filtered detections + centered zero-phase median/Savgol")
                           if a.zero_phase_filter else
                           ("raw same-frame detections" if a.use_raw_2d else "temporally filtered detections")),
        "zero_phase_filter_window": a.filter_window if a.zero_phase_filter else None,
        "fit_policy": "bounded small SE3 around validated Z+90+offset" if a.fit_rotation
                      else "translation-only; validated Z+90 rotation held fixed",
        "delta_rotation_rotvec_rad": solved_rotvec.tolist(),
        "delta_rotation_deg": float(np.linalg.norm(solved_rotvec)*180.0/np.pi),
        "delta_translation_ch01_m": solved_translation.tolist(),
        "R_world_ch01_2d_gt_fit": world_r.tolist(),
        "t_world_ch01_2d_gt_fit_m": world_t.tolist(),
        "reprojection_before": {name: stats(before_errors[name]) for name in ("A", "D")},
        "reprojection_after": {name: stats(after_errors[name]) for name in ("A", "D")},
        "solver_cost": float(solution.cost), "solver_optimality": float(solution.optimality),
    }
    (a.output_dir/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
