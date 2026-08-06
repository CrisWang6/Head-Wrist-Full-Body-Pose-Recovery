#!/usr/bin/env python3
"""Fit the residual CH07 translation offset from head-view shoulder/elbow cues."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from process_external_stereo_to_head import Omni, load_json


p = argparse.ArgumentParser()
p.add_argument("--projection-csv", type=Path, required=True)
p.add_argument("--candidate-a", type=Path, required=True)
p.add_argument("--candidate-d", type=Path, required=True)
p.add_argument("--calib", type=Path, required=True)
p.add_argument("--base-refinement", type=Path, required=True)
p.add_argument("--output-refinement", type=Path, required=True)
p.add_argument("--output-report", type=Path, required=True)
a = p.parse_args()


def candidates(path: Path) -> dict[int, dict]:
    result = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            # The wearer is the dominant, top-clipped person in the head view.
            # Prefer area, with confidence only as a mild tie breaker.
            people = row.get("candidates", [])
            if people:
                def score(person):
                    x1, y1, x2, y2 = person["box_xyxy"]
                    return (x2 - x1) * (y2 - y1) * (0.7 + 0.3 * person["box_confidence"])
                result[int(row["frame_index"])] = max(people, key=score)
    return result


head_intr = load_json(a.calib / "head_intrinsics_kalibr_omni_1920x1200.json")
head_rigid = load_json(a.calib / "head_stereo_rigid_extrinsics.json")
base = load_json(a.base_refinement)
cams = {"A": Omni(head_intr, "CAM_A"), "D": Omni(head_intr, "CAM_C")}
transforms = {}
for key, side in (("A", "left"), ("D", "right")):
    mat = np.asarray(head_rigid["cameras"][side]["T_camera_rigid"], np.float64)
    mat[:3, 3] /= 1000.0
    transforms[key] = mat

points: dict[tuple[int, str], np.ndarray] = {}
with a.projection_csv.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        if row["joint"] in {"left_shoulder", "right_shoulder", "left_elbow", "right_elbow"}:
            points[(int(row["sequence"]), row["joint"])] = np.asarray(
                [float(row["ch07_x_m"]), float(row["ch07_y_m"]), float(row["ch07_z_m"])],
                np.float64,
            )

observations = []
for key, path in (("A", a.candidate_a), ("D", a.candidate_d)):
    for seq, person in candidates(path).items():
        for joint in ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow"):
            kp = person["keypoints"].get(joint)
            p3 = points.get((seq, joint))
            if p3 is not None and kp is not None and float(kp[2]) >= 0.22:
                observations.append((key, p3, np.asarray(kp[:2], np.float64), float(kp[2])))


def residual(delta: np.ndarray) -> np.ndarray:
    values = []
    for key, p3, uv, confidence in observations:
        pc = (transforms[key] @ np.r_[p3 + delta, 1.0])[:3]
        predicted = cams[key].project(pc)
        if predicted is None:
            values.extend((500.0, 500.0))
        else:
            weight = np.sqrt(max(confidence, 0.22))
            values.extend((np.asarray(predicted) - uv) * weight)
    return np.asarray(values)


before = residual(np.zeros(3)).reshape(-1, 2)
fit = least_squares(residual, np.zeros(3), bounds=(-0.8, 0.8), loss="soft_l1", f_scale=35.0, max_nfev=300)
after = residual(fit.x).reshape(-1, 2)
old_t = np.asarray(base["t_ch07_axis_correction_m"], np.float64)
base["t_ch07_axis_correction_m"] = (old_t + fit.x).tolist()
base["dataset"] = "0711_214559"
base.setdefault("notes", []).append(
    "Residual CH07 translation was re-fitted on this recording from robust multi-frame head-view shoulder/elbow correspondences."
)
a.output_refinement.write_text(json.dumps(base, indent=2), encoding="utf-8")

norm = lambda x: np.linalg.norm(x, axis=1)
report = {
    "observations": len(observations),
    "delta_ch07_m": fit.x.tolist(),
    "old_t_ch07_m": old_t.tolist(),
    "new_t_ch07_m": (old_t + fit.x).tolist(),
    "median_error_before_px": float(np.median(norm(before))),
    "median_error_after_px": float(np.median(norm(after))),
    "p90_error_before_px": float(np.percentile(norm(before), 90)),
    "p90_error_after_px": float(np.percentile(norm(after), 90)),
    "optimizer_success": bool(fit.success),
}
a.output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report))
