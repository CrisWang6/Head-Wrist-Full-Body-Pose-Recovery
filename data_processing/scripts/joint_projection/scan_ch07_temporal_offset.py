#!/usr/bin/env python3
"""Scan CH07 mocap timing while keeping the shared-trigger camera frames fixed."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from process_external_stereo_to_head import Omni, load_json, qrot


p = argparse.ArgumentParser()
p.add_argument("--projection-csv", type=Path, required=True)
p.add_argument("--aligned", type=Path, required=True)
p.add_argument("--candidate-a", type=Path, required=True)
p.add_argument("--candidate-d", type=Path, required=True)
p.add_argument("--calib", type=Path, required=True)
p.add_argument("--base-refinement", type=Path, required=True)
p.add_argument("--min-offset", type=int, default=-25)
p.add_argument("--max-offset", type=int, default=25)
p.add_argument("--step", type=int, default=1)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--output-refinement", type=Path)
a = p.parse_args()


def dominant_candidates(path: Path) -> dict[int, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line); people = row.get("candidates", [])
        if not people:
            continue
        def score(person):
            x1, y1, x2, y2 = person["box_xyxy"]
            return (x2 - x1) * (y2 - y1) * (0.7 + 0.3 * person["box_confidence"])
        result[int(row["frame_index"])] = max(people, key=score)
    return result


with a.aligned.open("r", encoding="utf-8-sig", newline="") as f:
    aligned = list(csv.DictReader(f))

points = {}
with a.projection_csv.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        if row["joint"] in {"left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"}:
            points[(int(row["sequence"]), row["joint"])] = np.asarray(
                [float(row["ch01_x_m"]), float(row["ch01_y_m"]), float(row["ch01_z_m"])], np.float64
            )

intr = load_json(a.calib / "head_intrinsics_kalibr_omni_1920x1200.json")
rigid = load_json(a.calib / "head_stereo_rigid_extrinsics.json")
refine = load_json(a.base_refinement)
cams = {"A": Omni(intr, "CAM_A"), "D": Omni(intr, "CAM_C")}
head_T = {}
for key, side in (("A", "left"), ("D", "right")):
    mat = np.asarray(rigid["cameras"][side]["T_camera_rigid"], np.float64)
    mat[:3, 3] /= 1000.0; head_T[key] = mat
world_R = np.asarray(refine["R_world_ch01"], np.float64)
world_t = np.asarray(refine["t_world_ch01_m"], np.float64)
axis_R = np.asarray(refine["R_ch07_axis_correction"], np.float64)
base_t = np.asarray(refine["t_ch07_axis_correction_m"], np.float64)

observations = []
for camera, path in (("A", a.candidate_a), ("D", a.candidate_d)):
    for seq, person in dominant_candidates(path).items():
        for joint in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"):
            kp = person["keypoints"].get(joint); p01 = points.get((seq, joint))
            if p01 is not None and kp is not None and float(kp[2]) >= 0.24:
                observations.append((seq, camera, p01, np.asarray(kp[:2], np.float64), float(kp[2])))


def pose(row):
    R = qrot([float(row[f"mocap_CH3_07_world_q{x}"]) for x in "wxyz"])
    t = np.asarray([float(row[f"mocap_CH3_07_world_{x}"]) for x in "xyz"], np.float64)
    return R, t


results = []
for offset in range(a.min_offset, a.max_offset + 1, a.step):
    prepared = []
    for seq, camera, p01, uv, confidence in observations:
        shifted = seq + offset
        if shifted < 0 or shifted >= len(aligned):
            continue
        R07, t07 = pose(aligned[shifted])
        pw = world_R @ p01 + world_t
        p07_without_t = axis_R @ (R07.T @ (pw - t07))
        prepared.append((camera, p07_without_t, uv, confidence))

    def residual(t):
        out = []
        for camera, p07, uv, confidence in prepared:
            pc = (head_T[camera] @ np.r_[p07 + t, 1.0])[:3]
            pred = cams[camera].project(pc)
            diff = np.asarray((500.0, 500.0) if pred is None else np.asarray(pred) - uv)
            out.extend(diff * np.sqrt(confidence))
        return np.asarray(out)

    fit = least_squares(residual, base_t, bounds=(-0.8, 0.8), loss="soft_l1", f_scale=35.0, max_nfev=100)
    errors = np.linalg.norm(residual(fit.x).reshape(-1, 2), axis=1)
    results.append({
        "offset_frames": offset,
        "offset_ms": offset * 20.0,
        "translation_ch07_m": fit.x.tolist(),
        "median_weighted_error_px": float(np.median(errors)),
        "p90_weighted_error_px": float(np.percentile(errors, 90)),
        "observations": len(prepared),
    })

best = min(results, key=lambda x: x["median_weighted_error_px"])
report = {"best": best, "definition": "head frame s uses CH07 pose from aligned row s + offset_frames", "scan": results}
a.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
if a.output_refinement:
    refine["dataset"] = "0711_214559"
    refine["t_ch07_axis_correction_m"] = best["translation_ch07_m"]
    refine["ch07_event_offset_frames"] = best["offset_frames"]
    refine.setdefault("notes", []).append(
        "This recording uses the scanned CH07 event offset; camera event s remains fixed while CH07 uses s + offset."
    )
    a.output_refinement.write_text(json.dumps(refine, indent=2), encoding="utf-8")
print(json.dumps(best))
