#!/usr/bin/env python3
"""Fit external-stereo 3D poses to sparse manual head-stereo 2D/3D GT.

The head annotations are triangulated with the same omni model and the same
xy-swap + z-flip head-camera basis used by the approved head projections.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from process_external_stereo_to_head import Omni, closest_rays, load_json, qrot, writer


BASE_JOINTS = (
    "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
NEW_JOINTS = BASE_JOINTS + ("left_toe", "right_toe")
HIGH = {
    "nose", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_toe", "right_toe",
}
EDGES = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("left_ankle", "left_toe"), ("right_ankle", "right_toe"),
)
BONE_EDGES = {
    "shoulder_width": ("left_shoulder", "right_shoulder"),
    "left_upper_arm": ("left_shoulder", "left_elbow"),
    "right_upper_arm": ("right_shoulder", "right_elbow"),
    "left_forearm": ("left_elbow", "left_wrist"),
    "right_forearm": ("right_elbow", "right_wrist"),
}


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manual-csv", type=Path, required=True)
    p.add_argument("--head-intrinsics", type=Path, required=True)
    p.add_argument("--head-rigid", type=Path, required=True)
    p.add_argument("--external-world", type=Path)
    p.add_argument("--external-ch07-csv", type=Path,
                   help="Direct raw external-stereo triangulation CSV with ch07_x/y/z columns")
    p.add_argument("--aligned", type=Path, required=True)
    p.add_argument("--head-a", type=Path, required=True)
    p.add_argument("--head-d", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--start-seq", type=int, default=250)
    p.add_argument("--end-seq-exclusive", type=int, default=750)
    p.add_argument("--ch07-event-offset", type=int, default=0)
    return p.parse_args()


def read_manual(path: Path):
    observations = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if str(row.get("visible", "true")).lower() not in ("1", "true", "yes"):
                continue
            seq, cam, joint = int(row["aligned_sequence"]), row["camera"], row["joint"]
            observations.setdefault(seq, {}).setdefault(joint, {})[cam] = np.asarray(
                [float(row["x_px"]), float(row["y_px"])], np.float64
            )
    return observations


def head_geometry(intrinsics: dict, rigid: dict):
    right_name = "CAM_D" if "CAM_D" in intrinsics["cameras"] else "CAM_C"
    cams = {"CAM_A": Omni(intrinsics, "CAM_A"), "CAM_D": Omni(intrinsics, right_name)}
    stereo = np.asarray(intrinsics["stereo_extrinsics"]["T_CAM_D_CAM_A"], np.float64)
    r_da = stereo[:3, :3]
    # Validated z+90-equivalent basis used by the existing head projection.
    r_ra = np.asarray([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]], np.float64)
    c_a = np.asarray(rigid["cameras"]["left"]["p_rigid_camera_mm"], np.float64) / 1000.0
    c_d = np.asarray(rigid["cameras"]["right"]["p_rigid_camera_mm"], np.float64) / 1000.0
    rotations = {"CAM_A": r_ra, "CAM_D": r_ra @ r_da.T}
    centers = {"CAM_A": c_a, "CAM_D": c_d}
    return cams, rotations, centers


def triangulate_manual(observations, cams, rotations, centers):
    gt, diagnostics = {}, []
    for seq, joints in sorted(observations.items()):
        for joint, views in joints.items():
            if "CAM_A" not in views or "CAM_D" not in views:
                continue
            da = rotations["CAM_A"] @ cams["CAM_A"].ray(views["CAM_A"])
            dd = rotations["CAM_D"] @ cams["CAM_D"].ray(views["CAM_D"])
            ans = closest_rays(centers["CAM_A"], da, centers["CAM_D"], dd)
            if ans is None:
                continue
            point, miss, depth_a, depth_d = ans
            gt.setdefault(seq, {})[joint] = point
            reproj = {}
            for cam in ("CAM_A", "CAM_D"):
                pc = rotations[cam].T @ (point - centers[cam])
                uv = cams[cam].project(pc)
                reproj[cam] = float(np.linalg.norm(np.asarray(uv) - views[cam])) if uv else float("nan")
            diagnostics.append({
                "sequence": seq, "joint": joint,
                "x_m": point[0], "y_m": point[1], "z_m": point[2],
                "ray_miss_mm": miss * 1000.0,
                "depth_A_m": depth_a, "depth_D_m": depth_d,
                "reprojection_A_px": reproj["CAM_A"],
                "reprojection_D_px": reproj["CAM_D"],
            })
    return gt, diagnostics


def read_external_world(path: Path, wanted: set[int]):
    frames = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            seq = int(row["sequence"])
            if seq not in wanted:
                continue
            frames.setdefault(seq, {})[row["joint"]] = np.asarray(
                [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])], np.float64
            )
    return frames


def read_external_ch07(path: Path, wanted: set[int]):
    frames = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            seq, joint = int(row["sequence"]), row["joint"]
            if seq not in wanted or joint not in BASE_JOINTS:
                continue
            values = [float(row[f"ch07_{axis}_m"]) for axis in "xyz"]
            if not np.all(np.isfinite(values)):
                continue
            frames.setdefault(seq, {})[joint] = np.asarray(values, np.float64)
    return frames


def external_to_ch07(world, aligned, offset):
    result = {}
    for seq, joints in world.items():
        row = aligned[seq + offset]
        r = qrot([float(row[f"mocap_CH3_07_world_q{x}"]) for x in "wxyz"])
        t = np.asarray([float(row[f"mocap_CH3_07_world_{x}"]) for x in "xyz"], np.float64)
        result[seq] = {name: r.T @ (point - t) for name, point in joints.items()}
    return result


def weighted_kabsch(source, target, weights):
    x, y, w = map(np.asarray, (source, target, weights))
    w = w / np.sum(w)
    mx, my = np.sum(x * w[:, None], axis=0), np.sum(y * w[:, None], axis=0)
    h = (x - mx).T @ ((y - my) * w[:, None])
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    return r, my - r @ mx


def fit_global(external, head_gt):
    source, target, weights, labels = [], [], [], []
    for seq, gt in sorted(head_gt.items()):
        if seq not in external:
            continue
        for joint in BASE_JOINTS:
            if joint not in gt or joint not in external[seq]:
                continue
            source.append(external[seq][joint]); target.append(gt[joint])
            weights.append(6.0 if joint in HIGH else 1.0)
            labels.append((seq, joint))
    r, t = weighted_kabsch(source, target, weights)
    source, target, weights = map(np.asarray, (source, target, weights))
    before = np.linalg.norm(source - target, axis=1)
    after = np.linalg.norm((r @ source.T).T + t - target, axis=1)
    return r, t, labels, before, after, weights


def median_bones(head_gt):
    raw_samples = {}
    for label, (u, v) in BONE_EDGES.items():
        values = [np.linalg.norm(j[u] - j[v]) for j in head_gt.values() if u in j and v in j]
        if not values:
            raise ValueError(f"No head GT samples for {label}")
        raw_samples[label] = values
    upper = raw_samples["left_upper_arm"] + raw_samples["right_upper_arm"]
    forearm = raw_samples["left_forearm"] + raw_samples["right_forearm"]
    shared_upper = float(np.median(upper))
    shared_forearm = float(np.median(forearm))
    targets = {
        "shoulder_width": float(np.median(raw_samples["shoulder_width"])),
        "left_upper_arm": shared_upper, "right_upper_arm": shared_upper,
        "left_forearm": shared_forearm, "right_forearm": shared_forearm,
    }
    samples = dict(raw_samples)
    samples["upper_arm_bilateral_pool"] = upper
    samples["forearm_bilateral_pool"] = forearm
    return targets, samples


def foot_models(head_gt):
    values = {"left": [], "right": []}
    for joints in head_gt.values():
        if not all(n in joints for n in ("left_hip", "right_hip")):
            continue
        lateral = joints["right_hip"] - joints["left_hip"]
        lateral /= max(np.linalg.norm(lateral), 1e-9)
        for side in ("left", "right"):
            need = (f"{side}_knee", f"{side}_ankle", f"{side}_toe")
            if not all(n in joints for n in need):
                continue
            down = joints[need[1]] - joints[need[0]]
            down /= max(np.linalg.norm(down), 1e-9)
            forward = np.cross(lateral, down)
            forward /= max(np.linalg.norm(forward), 1e-9)
            vec = joints[need[2]] - joints[need[1]]
            values[side].append([vec @ down, vec @ lateral, vec @ forward])
    return {side: np.median(np.asarray(coeffs), axis=0) for side, coeffs in values.items()}


def predicted_toe(points, side, coefficients):
    lateral = points["right_hip"] - points["left_hip"]
    lateral /= max(np.linalg.norm(lateral), 1e-9)
    down = points[f"{side}_ankle"] - points[f"{side}_knee"]
    down /= max(np.linalg.norm(down), 1e-9)
    forward = np.cross(lateral, down)
    forward /= max(np.linalg.norm(forward), 1e-9)
    c = coefficients[side]
    return points[f"{side}_ankle"] + c[0] * down + c[1] * lateral + c[2] * forward


def optimize_frames(external, head_gt, r_fit, t_fit, bone_targets, foot_coeffs, seqs):
    optimized, transformed = {}, {}
    for seq in seqs:
        raw = external[seq]
        base = {name: r_fit @ raw[name] + t_fit for name in BASE_JOINTS if name in raw}
        if not all(name in base for name in BASE_JOINTS):
            continue
        base["left_toe"] = predicted_toe(base, "left", foot_coeffs)
        base["right_toe"] = predicted_toe(base, "right", foot_coeffs)
        transformed[seq] = {name: point.copy() for name, point in base.items()}
        x0 = np.concatenate([base[name] for name in NEW_JOINTS])

        def unpack(x):
            return {name: x[i * 3:i * 3 + 3] for i, name in enumerate(NEW_JOINTS)}

        def residual(x):
            p = unpack(x); values = []
            for name in BASE_JOINTS:
                sigma = .012 if name in HIGH else .025
                values.extend((p[name] - base[name]) / sigma)
            if seq in head_gt:
                for name, target in head_gt[seq].items():
                    if name not in p:
                        continue
                    sigma = .004 if name in HIGH else .012
                    values.extend((p[name] - target) / sigma)
            for label, (u, v) in BONE_EDGES.items():
                values.append((np.linalg.norm(p[u] - p[v]) - bone_targets[label]) / .003)
            for side in ("left", "right"):
                toe_reference = predicted_toe(p, side, foot_coeffs)
                values.extend((p[f"{side}_toe"] - toe_reference) / .018)
            return np.asarray(values)

        result = least_squares(residual, x0, loss="soft_l1", f_scale=2.0, max_nfev=100)
        optimized[seq] = unpack(result.x)
    return transformed, optimized


def project(point, cam, cams, rotations, centers):
    return cams[cam].project(rotations[cam].T @ (point - centers[cam]))


def draw_skeleton(image, uv, color, thickness=3):
    for u, v in EDGES:
        if u in uv and v in uv:
            cv2.line(image, tuple(np.round(uv[u]).astype(int)), tuple(np.round(uv[v]).astype(int)), color, thickness, cv2.LINE_AA)
    for name, point in uv.items():
        radius = 5 if name in HIGH else 4
        cv2.circle(image, tuple(np.round(point).astype(int)), radius, color, -1, cv2.LINE_AA)


def write_pose_csv(path, frames, variant):
    rows = []
    for seq in sorted(frames):
        for name, point in frames[seq].items():
            rows.append({"sequence": seq, "joint": name, "x_m": point[0], "y_m": point[1], "z_m": point[2], "variant": variant})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def render_head(a, external, optimized, observations, cams, rotations, centers):
    paths = {}
    for cam, video in (("CAM_A", a.head_a), ("CAM_D", a.head_d)):
        cap = cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_POS_FRAMES, a.start_seq)
        suffix = "A" if cam == "CAM_A" else "D"
        path = a.output_dir / f"head_CAM_{suffix}_external_vs_headGT_fit_5s15s.mp4"
        out = writer(path, (1920, 1200), 50)
        for seq in range(a.start_seq, a.end_seq_exclusive):
            ok, image = cap.read()
            if not ok: break
            raw_uv, opt_uv = {}, {}
            for name, point in external.get(seq, {}).items():
                uv = project(point, cam, cams, rotations, centers)
                if uv is not None: raw_uv[name] = np.asarray(uv)
            for name, point in optimized.get(seq, {}).items():
                uv = project(point, cam, cams, rotations, centers)
                if uv is not None: opt_uv[name] = np.asarray(uv)
            draw_skeleton(image, raw_uv, (255, 255, 0), 2)
            draw_skeleton(image, opt_uv, (0, 255, 255), 4)
            for name, views in observations.get(seq, {}).items():
                if cam in views:
                    cv2.circle(image, tuple(np.round(views[cam]).astype(int)), 7, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"seq={seq}  5-15s validation", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,255,255), 2, cv2.LINE_AA)
            cv2.putText(image, "cyan=external 3D GT   yellow=weighted head-GT fit   green=manual head 2D GT", (24, 74), cv2.FONT_HERSHEY_SIMPLEX, .65, (0,255,255), 2, cv2.LINE_AA)
            out.write(image)
        cap.release(); out.release(); paths[cam] = path
    ca, cd = cv2.VideoCapture(str(paths["CAM_A"])), cv2.VideoCapture(str(paths["CAM_D"]))
    stereo = a.output_dir / "head_stereo_external_vs_headGT_weighted_fit_5s15s.mp4"
    out = writer(stereo, (1920, 600), 50); frames = 0
    while True:
        oka, ia = ca.read(); okd, id_ = cd.read()
        if not oka or not okd: break
        out.write(np.hstack([cv2.resize(ia, (960,600)), cv2.resize(id_, (960,600))])); frames += 1
    ca.release(); cd.release(); out.release()
    return stereo, frames


def main():
    a = arguments(); a.output_dir.mkdir(parents=True, exist_ok=True)
    intr, rigid = load_json(a.head_intrinsics), load_json(a.head_rigid)
    cams, rotations, centers = head_geometry(intr, rigid)
    observations = read_manual(a.manual_csv)
    head_gt, tri_diag = triangulate_manual(observations, cams, rotations, centers)
    with a.aligned.open("r", encoding="utf-8-sig", newline="") as f:
        aligned = list(csv.DictReader(f))
    validation_seqs = list(range(a.start_seq, a.end_seq_exclusive))
    wanted = set(validation_seqs) | set(head_gt)
    if a.external_ch07_csv:
        external = read_external_ch07(a.external_ch07_csv, wanted)
        external_source_policy = "direct raw external-stereo triangulation ch07_x/y/z"
    else:
        if not a.external_world:
            raise ValueError("Provide --external-ch07-csv or --external-world")
        world = read_external_world(a.external_world, wanted)
        external = external_to_ch07(world, aligned, a.ch07_event_offset)
        external_source_policy = "world skeleton transformed into CH07"

    r_fit, t_fit, labels, before, after, fit_weights = fit_global(external, head_gt)
    bone_targets, bone_samples = median_bones(head_gt)
    foot_coeffs = foot_models(head_gt)
    transformed, optimized = optimize_frames(external, head_gt, r_fit, t_fit, bone_targets, foot_coeffs, validation_seqs)

    write_pose_csv(a.output_dir / "external_triangulated_ch07_5s15s.csv", {s: external[s] for s in validation_seqs if s in external}, "external_triangulated_gt")
    write_pose_csv(a.output_dir / "external_global_fit_ch07_5s15s.csv", transformed, "external_global_weighted_fit")
    write_pose_csv(a.output_dir / "headGT_weighted_bone_optimized_ch07_5s15s.csv", optimized, "headGT_weighted_bone_optimized")
    with (a.output_dir / "head_stereo_manual_triangulated_ch07.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(tri_diag[0])); w.writeheader(); w.writerows(tri_diag)

    comparison = {
        "start_sequence": a.start_seq,
        "end_sequence_exclusive": a.end_seq_exclusive,
        "fps": 50,
        "joint_names_external": list(BASE_JOINTS),
        "joint_names_optimized": list(NEW_JOINTS),
        "frames": [{
            "sequence": seq,
            "external": [external[seq][n].round(6).tolist() for n in BASE_JOINTS],
            "optimized": [optimized[seq][n].round(6).tolist() for n in NEW_JOINTS],
            "head_gt": {n: p.round(6).tolist() for n, p in head_gt.get(seq, {}).items()},
        } for seq in validation_seqs if seq in external and seq in optimized],
    }
    (a.output_dir / "comparison_3d_5s15s.json").write_text(json.dumps(comparison, separators=(",", ":")), encoding="utf-8")
    stereo, rendered_frames = render_head(a, {s: external[s] for s in validation_seqs if s in external}, optimized, observations, cams, rotations, centers)

    high_mask = np.asarray([joint in HIGH for _, joint in labels])
    report = {
        "validation_sequences": [a.start_seq, a.end_seq_exclusive - 1],
        "validation_frames": len(validation_seqs),
        "external_source_policy": external_source_policy,
        "manual_annotated_sequences_all": sorted(head_gt),
        "manual_annotated_sequences_inside_validation": sorted(set(head_gt) & set(validation_seqs)),
        "head_triangulated_points": len(tri_diag),
        "head_triangulation_ray_miss_median_mm": float(np.median([r["ray_miss_mm"] for r in tri_diag])),
        "head_triangulation_ray_miss_p90_mm": float(np.percentile([r["ray_miss_mm"] for r in tri_diag], 90)),
        "global_fit_rotation": r_fit.tolist(),
        "global_fit_translation_m": t_fit.tolist(),
        "weighted_common_correspondences": len(labels),
        "fit_error_before_median_mm": float(np.median(before) * 1000),
        "fit_error_after_median_mm": float(np.median(after) * 1000),
        "fit_error_after_high_weight_median_mm": float(np.median(after[high_mask]) * 1000),
        "fit_error_after_low_weight_median_mm": float(np.median(after[~high_mask]) * 1000),
        "joint_weights": {"high": 6.0, "low": 1.0, "high_joints": sorted(HIGH)},
        "toe_policy": "External RTMPose has ankles but no toe keypoints; manual head toes are high-weight head-only joints. Unlabelled toe trajectories use the median head-GT foot vector in a leg-local basis.",
        "bone_targets_from_head_gt_m": bone_targets,
        "bone_samples_from_head_gt_m": bone_samples,
        "foot_local_coefficients_m": {k: v.tolist() for k, v in foot_coeffs.items()},
        "external_position_prior_sigma_m": {"high": .012, "low": .025},
        "manual_head_gt_sigma_m": {"high": .004, "low": .012},
        "bone_constraint_sigma_m": .003,
        "temporal_filter": "none",
        "rendered_frames": rendered_frames,
        "rendered_video": str(stereo),
        "head_axis_basis": "xy_swap + z_flip (same as approved head projection)",
    }
    (a.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
