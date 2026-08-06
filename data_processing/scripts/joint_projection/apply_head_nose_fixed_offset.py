#!/usr/bin/env python3
"""Fit one CH07 translation from fixed RTMW head-stereo nose 2D and reproject 3D pose."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from process_external_stereo_to_head import NAMES, Omni, closest_rays, draw_pose, load_json, qrot, writer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--head-a-nose-csv", type=Path, required=True)
    p.add_argument("--head-d-nose-csv", type=Path, required=True)
    p.add_argument("--head-intrinsics", type=Path, required=True)
    p.add_argument("--head-rigid", type=Path, required=True)
    p.add_argument("--aligned", type=Path, required=True)
    p.add_argument("--world-csv", type=Path, required=True)
    p.add_argument("--head-a", type=Path, required=True)
    p.add_argument("--head-d", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ch07-event-offset", type=int, default=71)
    p.add_argument("--nose-offset-mm", type=float, nargs=3, default=(0.0, -15.0, -125.0))
    return p.parse_args()


def load_nose(path: Path) -> tuple[dict[int, np.ndarray], int]:
    result = {}
    total = 0
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            total += 1
            try:
                point = np.asarray([float(row["face_nose_u_px"]), float(row["face_nose_v_px"])], np.float64)
                score = float(row["face_nose_score"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(point).all() and np.isfinite(score) and 0 <= point[0] < 1920 and 0 <= point[1] < 1200:
                result[int(row["frame_index"])] = point
    return result, total


def robust_fixed_point(points: dict[int, np.ndarray]) -> tuple[np.ndarray, dict[int, bool], dict]:
    indices = sorted(points)
    values = np.asarray([points[index] for index in indices], np.float64)
    center = np.median(values, axis=0)
    radial = np.linalg.norm(values-center, axis=1)
    radial_center = float(np.median(radial))
    mad = float(np.median(np.abs(radial-radial_center)))
    threshold = radial_center + max(4.0, 4.5 * 1.4826 * mad)
    keep_array = radial <= threshold
    fixed = np.median(values[keep_array], axis=0)
    keep = {index: bool(flag) for index, flag in zip(indices, keep_array)}
    residual = np.linalg.norm(values[keep_array]-fixed, axis=1)
    stats = {"raw_samples": len(values), "inlier_samples": int(keep_array.sum()),
             "outlier_samples": int((~keep_array).sum()), "outlier_threshold_px": threshold,
             "raw_to_fixed_median_px": float(np.median(residual)),
             "raw_to_fixed_p90_px": float(np.percentile(residual, 90)),
             "fixed_uv_px": fixed.tolist()}
    return fixed, keep, stats


def main() -> None:
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    raw_a, total_a = load_nose(a.head_a_nose_csv)
    raw_d, total_d = load_nose(a.head_d_nose_csv)
    fixed_a, keep_a, stats_a = robust_fixed_point(raw_a)
    fixed_d, keep_d, stats_d = robust_fixed_point(raw_d)

    intrinsics = load_json(a.head_intrinsics)
    right_name = "CAM_D" if "CAM_D" in intrinsics["cameras"] else "CAM_C"
    cameras = {"A": Omni(intrinsics, "CAM_A"), "D": Omni(intrinsics, right_name)}
    stereo = np.asarray(intrinsics["stereo_extrinsics"]["T_CAM_D_CAM_A"], np.float64)
    r_da, t_da = stereo[:3, :3], stereo[:3, 3]
    origin_a = np.zeros(3, np.float64)
    origin_d_in_a = -r_da.T @ t_da
    ray_a = cameras["A"].ray(fixed_a)
    ray_d_in_a = r_da.T @ cameras["D"].ray(fixed_d)
    answer = closest_rays(origin_a, ray_a, origin_d_in_a, ray_d_in_a)
    if answer is None or answer[2] <= 0 or answer[3] <= 0:
        raise RuntimeError("Fixed RTMW nose rays do not form a valid forward stereo solution")
    nose_a, ray_gap, depth_a, depth_d = answer

    rigid = load_json(a.head_rigid)
    transforms = {}
    for camera, side in (("A", "left"), ("D", "right")):
        camera_to_rigid = np.asarray(rigid["cameras"][side]["T_rigid_camera"], np.float64)
        camera_to_rigid[:3, 3] /= 1000.0
        transforms[camera] = {"camera_to_rigid": camera_to_rigid,
                              "rigid_to_camera": np.linalg.inv(camera_to_rigid)}
    nose_rigid = (transforms["A"]["camera_to_rigid"] @ np.r_[nose_a, 1.0])[:3]
    nose_gt_rigid = np.asarray(a.nose_offset_mm, np.float64)/1000.0
    fixed_offset_rigid = nose_rigid - nose_gt_rigid

    with a.world_csv.open("r", encoding="utf-8-sig", newline="") as f:
        points = {}
        for row in csv.DictReader(f):
            points.setdefault(int(row["sequence"]), {})[row["joint"]] = np.asarray(
                [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])], np.float64)
    with a.aligned.open("r", encoding="utf-8-sig", newline="") as f:
        aligned = list(csv.DictReader(f))

    projected = {"A": {}, "D": {}}
    output_rows = []
    for sequence, joints in points.items():
        ch07_index = sequence + a.ch07_event_offset
        if not 0 <= ch07_index < len(aligned):
            continue
        row = aligned[ch07_index]
        rotation = qrot([float(row[f"mocap_CH3_07_world_q{x}"]) for x in "wxyz"])
        translation = np.asarray([float(row[f"mocap_CH3_07_world_{x}"]) for x in "xyz"])
        for name, world in joints.items():
            if name != "nose" and name not in NAMES[5:]:
                continue
            ch07_before = rotation.T @ (world-translation)
            ch07_after = ch07_before + fixed_offset_rigid
            output = {"sequence": sequence, "joint": name,
                      **{f"ch07_before_{axis}_m": ch07_before[i] for i, axis in enumerate("xyz")},
                      **{f"ch07_after_{axis}_m": ch07_after[i] for i, axis in enumerate("xyz")}}
            for camera in ("A", "D"):
                camera_point = (transforms[camera]["rigid_to_camera"] @ np.r_[ch07_after, 1.0])[:3]
                uv = cameras[camera].project(camera_point)
                if uv is not None and -200 <= uv[0] < 2120 and -200 <= uv[1] < 1400:
                    projected[camera].setdefault(sequence, {})[name] = np.asarray(uv)
                    output[f"head_{camera}_u_px"], output[f"head_{camera}_v_px"] = uv
                else:
                    output[f"head_{camera}_u_px"], output[f"head_{camera}_v_px"] = float("nan"), float("nan")
            output_rows.append(output)
    with (a.output_dir/"rtmw_nose_offset_projection_2d.csv").open("w", encoding="utf-8-sig", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(output_rows[0])); out.writeheader(); out.writerows(output_rows)

    detection_rows = []
    for frame_index in range(max(total_a, total_d)):
        row = {"frame_index": frame_index}
        for camera, raw, fixed, keep in (("A", raw_a, fixed_a, keep_a), ("D", raw_d, fixed_d, keep_d)):
            point = raw.get(frame_index)
            row[f"head_{camera}_raw_u_px"] = point[0] if point is not None else ""
            row[f"head_{camera}_raw_v_px"] = point[1] if point is not None else ""
            row[f"head_{camera}_inlier"] = int(keep.get(frame_index, False))
            row[f"head_{camera}_fixed_u_px"] = fixed[0]
            row[f"head_{camera}_fixed_v_px"] = fixed[1]
        detection_rows.append(row)
    with (a.output_dir/"rtmw_head_nose_raw_and_fixed_2d.csv").open("w", encoding="utf-8-sig", newline="") as f:
        out = csv.DictWriter(f, fieldnames=list(detection_rows[0])); out.writeheader(); out.writerows(detection_rows)

    for camera, video in (("A", a.head_a), ("D", a.head_d)):
        capture = cv2.VideoCapture(str(video))
        output = writer(a.output_dir/f"head_CAM_{camera}_rtmw_nose_offset_projection.mp4", (1920, 1200), 50)
        frame_index = 0
        fixed = fixed_a if camera == "A" else fixed_d
        raw = raw_a if camera == "A" else raw_d
        while True:
            ok, image = capture.read()
            if not ok:
                break
            draw_pose(image, projected[camera].get(frame_index, {}), (0, 255, 255),
                      f"RTMW fixed nose 3D offset -> HEAD_{camera} seq={frame_index}", body_only=True)
            raw_point = raw.get(frame_index)
            if raw_point is not None:
                cv2.circle(image, tuple(np.rint(raw_point).astype(int)), 7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(image, tuple(np.rint(fixed).astype(int)), 8, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(image, "green=fixed RTMW nose  cyan ring=raw RTMW", (24, 76),
                        cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2, cv2.LINE_AA)
            output.write(image)
            frame_index += 1
        capture.release(); output.release()

    cap_a = cv2.VideoCapture(str(a.output_dir/"head_CAM_A_rtmw_nose_offset_projection.mp4"))
    cap_d = cv2.VideoCapture(str(a.output_dir/"head_CAM_D_rtmw_nose_offset_projection.mp4"))
    stereo_out = writer(a.output_dir/"head_stereo_rtmw_nose_offset_projection.mp4", (1920, 600), 50)
    while True:
        ok_a, image_a = cap_a.read(); ok_d, image_d = cap_d.read()
        if not ok_a or not ok_d:
            break
        stereo_out.write(np.hstack((cv2.resize(image_a, (960, 600)), cv2.resize(image_d, (960, 600)))))
    cap_a.release(); cap_d.release(); stereo_out.release()

    reprojection = {}
    for camera, fixed in (("A", fixed_a), ("D", fixed_d)):
        camera_point = (transforms[camera]["rigid_to_camera"] @ np.r_[nose_rigid, 1.0])[:3]
        uv = np.asarray(cameras[camera].project(camera_point))
        reprojection[camera] = {"uv_px": uv.tolist(), "error_to_fixed_px": float(np.linalg.norm(uv-fixed))}
    report = {"model": "RTMW WholeBody performance (68-point face subset)",
              "face_nose_landmark": {"face_index_0_based": 30, "wholebody_global_index_0_based": 53},
              "head_A_detection": stats_a, "head_D_detection": stats_d,
              "head_stereo_source": str(a.head_intrinsics),
              "fixed_nose_xyz_head_A_m": nose_a.tolist(),
              "fixed_nose_stereo_ray_gap_mm": float(ray_gap*1000.0),
              "fixed_nose_ray_depth_m": {"A": float(depth_a), "D": float(depth_d)},
              "fixed_nose_xyz_ch07_m": nose_rigid.tolist(),
              "external_nose_gt_xyz_ch07_m": nose_gt_rigid.tolist(),
              "fixed_offset_added_to_external_skeleton_ch07_m": fixed_offset_rigid.tolist(),
              "fixed_offset_norm_mm": float(np.linalg.norm(fixed_offset_rigid)*1000.0),
              "fixed_nose_reprojection": reprojection,
              "ch07_event_offset_frames": a.ch07_event_offset,
              "offset_policy": "single fixed CH07 translation; rotation unchanged; external 2D and optimized 3D unchanged"}
    (a.output_dir/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
