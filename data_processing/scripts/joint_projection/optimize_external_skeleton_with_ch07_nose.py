#!/usr/bin/env python3
"""Anchor filtered external-stereo skeletons to a CH07-relative nose GT."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
from scipy.signal import savgol_filter

from process_external_stereo_to_head import Omni, closest_rays, load_json, qrot


BODY_NAMES = (
    "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)

# Bone groups share one robust target across the left/right sides when symmetric.
BONE_GROUPS = {
    "nose_shoulder": (("nose", "left_shoulder"), ("nose", "right_shoulder")),
    "shoulder_width": (("left_shoulder", "right_shoulder"),),
    "upper_arm": (("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow")),
    "forearm": (("left_elbow", "left_wrist"), ("right_elbow", "right_wrist")),
    "torso_side": (("left_shoulder", "left_hip"), ("right_shoulder", "right_hip")),
    "hip_width": (("left_hip", "right_hip"),),
    "thigh": (("left_hip", "left_knee"), ("right_hip", "right_knee")),
    "shin": (("left_knee", "left_ankle"), ("right_knee", "right_ankle")),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--stereo-json", type=Path, required=True)
    p.add_argument("--aligned", type=Path, required=True)
    p.add_argument("--old-refinement", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ch07-event-offset", type=int, default=71)
    p.add_argument("--nose-offset-mm", type=float, nargs=3, default=(0.0, -15.0, -125.0))
    p.add_argument("--nose-2d-csv", type=Path,
                   help="Filtered cyan nose 2D GT and its stereo observations.")
    p.add_argument("--calib", type=Path)
    p.add_argument("--external-stereo-yaml", type=Path)
    p.add_argument("--world-calibration-report", type=Path)
    p.add_argument("--anchor-filter-window", type=int, default=11)
    p.add_argument("--use-subject-gt-bones", action="store_true")
    p.add_argument("--disable-bone-constraints", action="store_true",
                   help="Ablation: do not use measured or estimated bone lengths in either solve pass.")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel per-frame nonlinear solves; results are deterministic and reassembled by sequence.")
    p.add_argument("--parallel-backend", choices=("thread", "process"), default="process")
    return p.parse_args()


def kabsch(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source-source_center).T @ (target-target_center))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def percentile(values, q):
    return float(np.percentile(np.asarray(values, np.float64), q)) if values else float("nan")


def main() -> None:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    records = json.loads(a.stereo_json.read_text(encoding="utf-8"))
    with a.aligned.open("r", encoding="utf-8-sig", newline="") as f:
        aligned = list(csv.DictReader(f))
    old = json.loads(a.old_refinement.read_text(encoding="utf-8"))
    old_r = np.asarray(old["R_world_ch01"], np.float64)
    old_t = np.asarray(old["t_world_ch01_m"], np.float64)
    if a.world_calibration_report:
        calibrated = load_json(a.world_calibration_report)
        old_r = np.asarray(calibrated["R_world_ch01_2d_gt_fit"], np.float64)
        old_t = np.asarray(calibrated["t_world_ch01_2d_gt_fit_m"], np.float64)
    nose_offset = np.asarray(a.nose_offset_mm, np.float64) / 1000.0

    nose_ch01_from_2d = {}
    stereo_projection_ready = False
    if a.nose_2d_csv:
        if not a.calib or not a.external_stereo_yaml:
            raise ValueError("--nose-2d-csv requires --calib and --external-stereo-yaml")
        intrinsics = load_json(a.calib/"handle_ac_intrinsics_kalibr_omni_1920x1200_20260729.json")
        right_name = "CAM_D" if "CAM_D" in intrinsics["cameras"] else "CAM_C"
        cams = {"A": Omni(intrinsics, "CAM_A"), "D": Omni(intrinsics, right_name)}
        rigid = load_json(a.calib/"external_stereo_rigid_k_extrinsics.json")
        r_ch01_left = np.asarray(rigid["cameras"]["left"]["R_rigid_camera"], np.float64)
        o_ch01_left = np.asarray(rigid["cameras"]["left"]["p_rigid_camera_mm"], np.float64)/1000.0
        stereo = yaml.safe_load(a.external_stereo_yaml.read_text(encoding="utf-8"))
        transform = np.asarray(stereo["cam1"]["T_cn_cnm1"], np.float64)
        r_rl, t_rl = transform[:3, :3], transform[:3, 3]
        r_left, r_right = r_ch01_left, r_ch01_left @ r_rl.T
        o_left = o_ch01_left
        o_right = o_ch01_left + r_ch01_left @ (-r_rl.T @ t_rl)
        stereo_projection_ready = True
        with a.nose_2d_csv.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                # Accept either the dedicated nose-GT export or the generic
                # filtered stereo-keypoint table emitted by the pose pipeline.
                if row.get("joint") not in (None, "", "nose"):
                    continue
                if "A_observed_u_px" in row:
                    left_uv = [float(row["A_observed_u_px"]), float(row["A_observed_v_px"])]
                    right_uv = [float(row["D_observed_u_px"]), float(row["D_observed_v_px"])]
                else:
                    left_uv = [float(row["left_u_px"]), float(row["left_v_px"])]
                    right_uv = [float(row["right_u_px"]), float(row["right_v_px"])]
                answer = closest_rays(o_left, r_left @ cams["A"].ray(left_uv),
                                      o_right, r_right @ cams["D"].ray(right_uv))
                if answer is not None and answer[2] > 0 and answer[3] > 0:
                    nose_ch01_from_2d[int(row["sequence"])] = answer[0]

    usable = []
    for record in records:
        seq = int(record["sequence"])
        ch07_index = seq + a.ch07_event_offset
        if "nose" not in record["joints"] or not 0 <= ch07_index < len(aligned):
            continue
        row = aligned[ch07_index]
        try:
            rotation = qrot([float(row[f"mocap_CH3_07_world_q{x}"]) for x in "wxyz"])
            translation = np.asarray([float(row[f"mocap_CH3_07_world_{x}"]) for x in "xyz"])
        except (KeyError, TypeError, ValueError):
            continue
        nose_gt = rotation @ nose_offset + translation
        nose_ch01 = nose_ch01_from_2d.get(seq, np.asarray(record["joints"]["nose"]["xyz"], np.float64))
        usable.append((record, rotation, translation, nose_ch01, nose_gt))

    source_nose = np.asarray([item[3] for item in usable])
    target_nose = np.asarray([item[4] for item in usable])
    # A single tracked point must not re-estimate a full 3-axis rotation: that
    # solution is weakly observable and can tilt the complete body.  Preserve
    # the validated rigid rotation and use the CH07 nose GT to solve translation.
    world_r = old_r.copy()
    world_t = np.median(target_nose-(world_r @ source_nose.T).T, axis=0)
    old_nose_error = np.linalg.norm((old_r @ source_nose.T).T + old_t - target_nose, axis=1)
    fitted_nose_error = np.linalg.norm((world_r @ source_nose.T).T + world_t - target_nose, axis=1)

    raw_frames = {}
    uv_frames = {}
    anchored_frames = {}
    exact_shift_by_seq = {}
    gt_by_seq = {}
    for record, _, _, _, nose_gt in usable:
        seq = int(record["sequence"])
        raw = {name: world_r @ np.asarray(joint["xyz"], np.float64) + world_t
               for name, joint in record["joints"].items() if name in BODY_NAMES}
        uv_frames[seq] = {
            name: {"A": np.asarray(joint["left_uv"], np.float64),
                   "D": np.asarray(joint["right_uv"], np.float64)}
            for name, joint in record["joints"].items()
            if name in BODY_NAMES and "left_uv" in joint and "right_uv" in joint
        }
        if "nose" not in raw:
            continue
        shift = nose_gt - raw["nose"]
        exact_shift_by_seq[seq] = shift
        gt_by_seq[seq] = nose_gt
        raw_frames[seq] = raw

    sequences = sorted(exact_shift_by_seq)
    shifts = np.asarray([exact_shift_by_seq[seq] for seq in sequences], np.float64)
    window = min(a.anchor_filter_window, len(sequences) if len(sequences) % 2 else len(sequences)-1)
    window = max(5, window if window % 2 else window-1)
    smoothed_shifts = shifts.copy()
    if len(sequences) >= window:
        for coordinate in range(3):
            clean = median_filter(shifts[:, coordinate], size=3, mode="nearest")
            smoothed_shifts[:, coordinate] = savgol_filter(clean, window, 2, mode="interp")
    anchor_shifts = np.linalg.norm(shifts, axis=1)
    smoothed_anchor_residuals = []
    offset_rows = []
    for index, seq in enumerate(sequences):
        shift = smoothed_shifts[index]
        raw = raw_frames[seq]
        anchored_frames[seq] = {name: point + shift for name, point in raw.items()}
        anchored_frames[seq]["nose"] = gt_by_seq[seq].copy()
        smoothed_anchor_residuals.append(float(np.linalg.norm(shifts[index]-shift)))
        offset_rows.append({"sequence": seq,
                            **{f"exact_d{axis}_m": shifts[index, i] for i, axis in enumerate("xyz")},
                            **{f"smoothed_d{axis}_m": shift[i] for i, axis in enumerate("xyz")},
                            "smoothed_reference_residual_m": smoothed_anchor_residuals[-1]})

    with (a.output_dir/"nose_3d_anchor_offset.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(offset_rows[0])); writer.writeheader(); writer.writerows(offset_rows)

    targets = {}
    for group, edges in BONE_GROUPS.items():
        lengths = []
        for frame in anchored_frames.values():
            for first, second in edges:
                if first in frame and second in frame:
                    lengths.append(float(np.linalg.norm(frame[first]-frame[second])))
        targets[group] = float(np.median(lengths))

    estimated_targets = targets.copy()
    if a.use_subject_gt_bones:
        shoulder_width = .390
        nose_to_shoulder_center = .110 + .160/2.0
        targets.update({
            "nose_shoulder": float(np.hypot(nose_to_shoulder_center, shoulder_width/2.0)),
            "shoulder_width": shoulder_width,
            "upper_arm": .300,
            "forearm": .270,
            "torso_side": .550,
            "hip_width": .240,
            "thigh": .430,
            "shin": .480,
        })

    edge_to_group = {edge: group for group, edges in BONE_GROUPS.items() for edge in edges}
    variable_names = BODY_NAMES[1:]
    if a.use_subject_gt_bones:
        # Depth from the external baseline is the weak measurement. Allow points to
        # move substantially along their image rays while 2D and metric bones stay strong.
        data_sigma = {name: .120 if ("shoulder" in name or "hip" in name) else .150
                      for name in variable_names}
    else:
        data_sigma = {
            "left_shoulder": .025, "right_shoulder": .025, "left_hip": .025, "right_hip": .025,
            "left_elbow": .035, "right_elbow": .035, "left_knee": .035, "right_knee": .035,
            "left_wrist": .045, "right_wrist": .045, "left_ankle": .045, "right_ankle": .045,
        }
    optimized_frames = {}
    raw_bone_errors, optimized_bone_errors = [], []
    reprojection_before, reprojection_after = [], []

    def project_world(point, camera_name):
        point_ch01 = world_r.T @ (point-world_t)
        point_left = r_ch01_left.T @ (point_ch01-o_ch01_left)
        point_camera = point_left if camera_name == "A" else r_rl @ point_left + t_rl
        return cams[camera_name].project(point_camera)

    def solve_first(item):
        seq, anchored = item
        present = [name for name in variable_names if name in anchored]
        nose = anchored["nose"]
        reference = {name: anchored[name] for name in present}
        x0 = np.concatenate([reference[name] for name in present])

        def unpack(vector):
            points = {"nose": nose}
            for index, name in enumerate(present):
                points[name] = vector[index*3:index*3+3]
            return points

        def residual(vector):
            points = unpack(vector)
            values = []
            for name in present:
                values.extend((points[name]-reference[name]) / data_sigma[name])
                if stereo_projection_ready and name in uv_frames.get(seq, {}):
                    pixel_sigma = 4.0 if ("shoulder" in name or "hip" in name) else 6.0
                    for camera_name in ("A", "D"):
                        uv = project_world(points[name], camera_name)
                        if uv is not None:
                            values.extend((np.asarray(uv)-uv_frames[seq][name][camera_name]) / pixel_sigma)
            if not a.disable_bone_constraints:
                for edge, group in edge_to_group.items():
                    first, second = edge
                    if first in points and second in points:
                        sigma = (.008 if group == "nose_shoulder" else .004) if a.use_subject_gt_bones else (.020 if group == "nose_shoulder" else .010)
                        values.append((np.linalg.norm(points[first]-points[second])-targets[group]) / sigma)
            return np.asarray(values)

        result = least_squares(residual, x0, method="trf", max_nfev=80, ftol=1e-8, xtol=1e-8)
        optimized = unpack(result.x)
        before_values=[];after_values=[];raw_bones=[];optimized_bones=[]
        if stereo_projection_ready:
            for name in present:
                if name not in uv_frames.get(seq, {}):
                    continue
                for camera_name in ("A", "D"):
                    target_uv = uv_frames[seq][name][camera_name]
                    before_uv = project_world(reference[name], camera_name)
                    after_uv = project_world(optimized[name], camera_name)
                    if before_uv is not None:
                        before_values.append(float(np.linalg.norm(np.asarray(before_uv)-target_uv)))
                    if after_uv is not None:
                        after_values.append(float(np.linalg.norm(np.asarray(after_uv)-target_uv)))
        for edge, group in edge_to_group.items():
            first, second = edge
            if first in anchored and second in anchored:
                raw_bones.append(abs(np.linalg.norm(anchored[first]-anchored[second])-targets[group]))
            if first in optimized and second in optimized:
                optimized_bones.append(abs(np.linalg.norm(optimized[first]-optimized[second])-targets[group]))
        return seq, optimized, before_values, after_values, raw_bones, optimized_bones

    items=list(anchored_frames.items())
    if a.workers > 1 and a.parallel_backend == "process":
        from joblib import Parallel, delayed
        first_results=Parallel(n_jobs=a.workers,backend="loky",batch_size=128)(delayed(solve_first)(item) for item in items)
    elif a.workers > 1:
        with ThreadPoolExecutor(max_workers=a.workers) as pool: first_results=list(pool.map(solve_first,items))
    else:
        first_results=list(map(solve_first,items))
    for seq,optimized,before_values,after_values,raw_bones,optimized_bones in first_results:
        optimized_frames[seq]=optimized
        reprojection_before.extend(before_values);reprojection_after.extend(after_values)
        raw_bone_errors.extend(raw_bones);optimized_bone_errors.extend(optimized_bones)

    # Multi-frame refinement: centered trajectory estimates are used only as soft
    # temporal references. Every final point is re-solved against stereo 2D rays,
    # metric bones and the hard CH07 nose anchor, so this is not a post-3D filter.
    temporal_reference = {seq: {} for seq in optimized_frames}
    for name in variable_names:
        seqs = [seq for seq in sorted(optimized_frames) if name in optimized_frames[seq]]
        if len(seqs) < 5:
            continue
        values = np.asarray([optimized_frames[seq][name] for seq in seqs], np.float64)
        temporal_window = min(11, len(seqs) if len(seqs) % 2 else len(seqs)-1)
        for coordinate in range(3):
            clean = median_filter(values[:, coordinate], size=3, mode="nearest")
            values[:, coordinate] = savgol_filter(clean, temporal_window, 2, mode="interp")
        for seq, point in zip(seqs, values):
            temporal_reference[seq][name] = point

    multiframe_frames = {}
    def solve_multiframe(item):
        seq, anchored = item
        present = [name for name in variable_names if name in anchored and name in optimized_frames[seq]]
        nose = anchored["nose"]
        x0 = np.concatenate([optimized_frames[seq][name] for name in present])

        def unpack_multiframe(vector):
            points = {"nose": nose}
            for index, name in enumerate(present):
                points[name] = vector[index*3:index*3+3]
            return points

        def residual_multiframe(vector):
            points = unpack_multiframe(vector)
            values = []
            for name in present:
                values.extend((points[name]-anchored[name]) / data_sigma[name])
                if name in temporal_reference.get(seq, {}):
                    temporal_sigma = (.012 if ("shoulder" in name or "hip" in name) else
                                      (.020 if ("elbow" in name or "wrist" in name) else .025))
                    values.extend((points[name]-temporal_reference[seq][name]) / temporal_sigma)
                if stereo_projection_ready and name in uv_frames.get(seq, {}):
                    pixel_sigma = 4.0 if ("shoulder" in name or "hip" in name) else 6.0
                    for camera_name in ("A", "D"):
                        uv = project_world(points[name], camera_name)
                        if uv is not None:
                            values.extend((np.asarray(uv)-uv_frames[seq][name][camera_name]) / pixel_sigma)
            if not a.disable_bone_constraints:
                for edge, group in edge_to_group.items():
                    first, second = edge
                    if first in points and second in points:
                        sigma = (.008 if group == "nose_shoulder" else .004) if a.use_subject_gt_bones else (.020 if group == "nose_shoulder" else .010)
                        values.append((np.linalg.norm(points[first]-points[second])-targets[group]) / sigma)
            return np.asarray(values)

        result = least_squares(residual_multiframe, x0, method="trf", max_nfev=80,
                               ftol=1e-8, xtol=1e-8)
        return seq, unpack_multiframe(result.x)
    if a.workers > 1 and a.parallel_backend == "process":
        multi_results=Parallel(n_jobs=a.workers,backend="loky",batch_size=128)(delayed(solve_multiframe)(item) for item in items)
    elif a.workers > 1:
        with ThreadPoolExecutor(max_workers=a.workers) as pool: multi_results=list(pool.map(solve_multiframe,items))
    else:
        multi_results=list(map(solve_multiframe,items))
    multiframe_frames=dict(multi_results)
    optimized_frames = multiframe_frames

    optimized_bone_errors = []
    reprojection_after = []
    for seq, points in optimized_frames.items():
        for edge, group in edge_to_group.items():
            first, second = edge
            if first in points and second in points:
                optimized_bone_errors.append(abs(np.linalg.norm(points[first]-points[second])-targets[group]))
        if stereo_projection_ready:
            for name, point in points.items():
                if name == "nose" or name not in uv_frames.get(seq, {}):
                    continue
                for camera_name in ("A", "D"):
                    uv = project_world(point, camera_name)
                    if uv is not None:
                        reprojection_after.append(float(np.linalg.norm(
                            np.asarray(uv)-uv_frames[seq][name][camera_name])))

    raw_rows, optimized_rows = [], []
    for seq in sorted(optimized_frames):
        for variant, frames, rows in (("nose_anchor_only", anchored_frames, raw_rows),
                                      ("nose_anchor_bone_optimized", optimized_frames, optimized_rows)):
            for name, point in frames[seq].items():
                rows.append({"sequence": seq, "joint": name, "x_m": point[0], "y_m": point[1], "z_m": point[2], "variant": variant})
    for path, rows in ((a.output_dir/"nose_anchor_only_world.csv", raw_rows),
                       (a.output_dir/"nose_anchor_bone_optimized_world.csv", optimized_rows)):
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    report = {
        "frames": len(optimized_frames),
        "ch07_event_offset_frames": a.ch07_event_offset,
        "nose_offset_ch07_mm": list(map(float, a.nose_offset_mm)),
        "world_rotation_policy": "preserve validated R_world_ch01; a single nose trajectory does not re-fit rotation",
        "R_world_ch01_preserved": world_r.tolist(),
        "t_world_ch01_nose_gt_translation_fit_m": world_t.tolist(),
        "old_mapping_nose_error_median_mm": float(np.median(old_nose_error)*1000),
        "old_mapping_nose_error_p90_mm": float(np.percentile(old_nose_error, 90)*1000),
        "new_translation_fit_nose_error_median_mm": float(np.median(fitted_nose_error)*1000),
        "new_translation_fit_nose_error_p90_mm": float(np.percentile(fitted_nose_error, 90)*1000),
        "per_frame_nose_anchor_shift_median_mm": float(np.median(anchor_shifts)*1000),
        "per_frame_nose_anchor_shift_p90_mm": float(np.percentile(anchor_shifts, 90)*1000),
        "anchor_definition": "CH07 nose 3D GT minus stereo triangulated cyan nose 2D GT, applied to the full frame skeleton",
        "anchor_filter": f"centered median(3) + Savitzky-Golay({window}), no temporal phase shift",
        "smoothed_anchor_reference_residual_median_mm": float(np.median(smoothed_anchor_residuals)*1000),
        "smoothed_anchor_reference_residual_p90_mm": float(np.percentile(smoothed_anchor_residuals, 90)*1000),
        "bone_targets_m": targets,
        "triangulated_bone_targets_before_gt_override_m": estimated_targets,
        "subject_gt_bone_profile_cm": ({"head": 16.0, "neck": 11.0, "shoulder_width": 39.0,
                                         "upper_arm_left_right": [30.0, 30.0],
                                         "forearm_left_right": [27.0, 27.0], "palm_left_right": [20.0, 20.0],
                                         "hip_width": 24.0, "torso": 55.0,
                                         "thigh_left_right": [43.0, 43.0],
                                         "shin_left_right": [48.0, 48.0]}
                                        if a.use_subject_gt_bones else None),
        "unavailable_direct_constraints": (["palm: no hand endpoint in COCO pose",
                                               "head/neck: no head-top or neck joint; nose-to-shoulder uses neck + half head"]
                                              if a.use_subject_gt_bones else []),
        "bone_error_before_median_mm": float(np.median(raw_bone_errors)*1000),
        "bone_error_before_p90_mm": float(np.percentile(raw_bone_errors, 90)*1000),
        "bone_error_after_median_mm": float(np.median(optimized_bone_errors)*1000),
        "bone_error_after_p90_mm": float(np.percentile(optimized_bone_errors, 90)*1000),
        "stereo_reprojection_before_median_px": float(np.median(reprojection_before)) if reprojection_before else None,
        "stereo_reprojection_before_p90_px": float(np.percentile(reprojection_before, 90)) if reprojection_before else None,
        "stereo_reprojection_after_median_px": float(np.median(reprojection_after)) if reprojection_after else None,
        "stereo_reprojection_after_p90_px": float(np.percentile(reprojection_after, 90)) if reprojection_after else None,
        "optimization_constraints": ("CH07 nose 3D hard anchor + immutable stereo 2D GT reprojection + robust anchor-shift prior + centered multi-frame temporal reference"
                                     if a.disable_bone_constraints else
                                     "CH07 nose 3D hard anchor + immutable stereo 2D GT reprojection + robust anchor-shift prior + bone lengths + centered multi-frame temporal reference"),
        "bone_constraints_enabled": not a.disable_bone_constraints,
        "temporal_policy": "centered median(3)+Savitzky-Golay(11) reference inside the constrained solve; no direct post-3D filtering",
        "optimization_variables": "3D joint coordinates only (world-space x,y,z)",
        "external_2d_policy": "immutable GT observations; never modified by the 3D optimizer",
        "face_output_policy": "nose only; eyes and ears excluded",
    }
    (a.output_dir/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
