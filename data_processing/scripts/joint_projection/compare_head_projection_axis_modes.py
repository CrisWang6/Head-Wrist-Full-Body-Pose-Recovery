#!/usr/bin/env python3
"""Compare discrete CH07 axis corrections against head-view pose detections."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from process_external_stereo_to_head import Omni, load_json, qrot


AXIS_MODES = {
    "none": np.eye(3, dtype=np.float64),
    "z+90": np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    "z-90": np.asarray([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
    "z180": np.asarray([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]]),
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-csv", type=Path, required=True)
    parser.add_argument("--aligned", type=Path, required=True)
    parser.add_argument("--candidate-a", type=Path, required=True)
    parser.add_argument("--candidate-d", type=Path, required=True)
    parser.add_argument("--head-intrinsics", type=Path, required=True)
    parser.add_argument("--head-rigid", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-refinement", type=Path, required=True)
    parser.add_argument("--ch07-event-offset", type=int, default=71)
    return parser.parse_args()


def dominant_candidates(path: Path) -> dict[int, dict]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        people = row.get("candidates", [])
        if not people:
            continue
        def score(person: dict) -> float:
            x1, y1, x2, y2 = person["box_xyxy"]
            return (x2-x1) * (y2-y1) * (0.7 + 0.3*float(person["box_confidence"]))
        result[int(row["frame_index"])] = max(people, key=score)
    return result


def main() -> None:
    a = arguments()
    a.output_report.parent.mkdir(parents=True, exist_ok=True)
    points: dict[int, dict[str, np.ndarray]] = {}
    with a.world_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            points.setdefault(int(row["sequence"]), {})[row["joint"]] = np.asarray(
                [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])])
    with a.aligned.open("r", encoding="utf-8-sig", newline="") as stream:
        aligned = list(csv.DictReader(stream))

    intrinsics = load_json(a.head_intrinsics)
    rigid = load_json(a.head_rigid)
    cameras = {"A": Omni(intrinsics, "CAM_A"), "D": Omni(intrinsics, "CAM_C")}
    transforms = {}
    for camera, side in (("A", "left"), ("D", "right")):
        transform = np.asarray(rigid["cameras"][side]["T_camera_rigid"], np.float64)
        transform[:3, 3] /= 1000.0
        transforms[camera] = transform

    detections = {"A": dominant_candidates(a.candidate_a), "D": dominant_candidates(a.candidate_d)}
    joint_names = ("left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist")
    base_observations = []
    for sequence, joints in points.items():
        shifted = sequence + a.ch07_event_offset
        if not 0 <= shifted < len(aligned):
            continue
        row = aligned[shifted]
        rotation = qrot([float(row[f"mocap_CH3_07_world_q{x}"]) for x in "wxyz"])
        translation = np.asarray([float(row[f"mocap_CH3_07_world_{x}"]) for x in "xyz"])
        rigid_unrotated = {name: rotation.T @ (world-translation) for name, world in joints.items()}
        for camera in ("A", "D"):
            person = detections[camera].get(sequence)
            if person is None:
                continue
            for name in joint_names:
                observed = person.get("keypoints", {}).get(name)
                point = rigid_unrotated.get(name)
                if point is None or observed is None or float(observed[2]) < 0.22:
                    continue
                base_observations.append((camera, point, np.asarray(observed[:2], np.float64), float(observed[2])))

    results = []
    for mode, axis_rotation in AXIS_MODES.items():
        observations = [(camera, axis_rotation @ point, observed, confidence)
                        for camera, point, observed, confidence in base_observations]

        def residual(translation: np.ndarray, weighted: bool = True) -> np.ndarray:
            values = []
            for camera, point, observed, confidence in observations:
                camera_point = (transforms[camera] @ np.r_[point+translation, 1.0])[:3]
                projected = cameras[camera].project(camera_point)
                difference = np.asarray([500.0, 500.0]) if projected is None else np.asarray(projected)-observed
                values.extend(difference * (np.sqrt(confidence) if weighted else 1.0))
            return np.asarray(values)

        fit = least_squares(lambda value: residual(value, True), np.asarray([0.0, -0.09, -0.125]),
                            bounds=(-0.8, 0.8), loss="soft_l1", f_scale=35.0, max_nfev=500)
        weighted_errors = np.linalg.norm(residual(fit.x, True).reshape(-1, 2), axis=1)
        raw_errors = np.linalg.norm(residual(fit.x, False).reshape(-1, 2), axis=1)
        results.append({
            "axis_mode": mode,
            "R_ch07_axis_correction": axis_rotation.tolist(),
            "t_ch07_axis_correction_m": fit.x.tolist(),
            "observations": len(observations),
            "median_weighted_error_px": float(np.median(weighted_errors)),
            "p90_weighted_error_px": float(np.percentile(weighted_errors, 90)),
            "median_raw_error_px": float(np.median(raw_errors)),
            "p90_raw_error_px": float(np.percentile(raw_errors, 90)),
            "cost": float(fit.cost),
        })

    results.sort(key=lambda item: (item["median_weighted_error_px"], item["p90_weighted_error_px"]))
    chosen = results[0]
    report = {
        "schema": "joint_projection.head_axis_mode_comparison.v1",
        "world_source": str(a.world_csv),
        "head_extrinsic_schema": rigid.get("schema"),
        "ch07_event_offset_frames": a.ch07_event_offset,
        "selection_reference": "largest head-view YOLO person; shoulder/elbow/wrist confidence >= 0.22",
        "chosen": chosen,
        "all_modes": results,
    }
    refinement = {
        "schema": "joint_projection.head_projection_axis_translation_refinement.v1",
        "dataset": "0711_214559",
        "source_3d": str(a.world_csv),
        "head_extrinsic_schema": rigid.get("schema"),
        "R_ch07_axis_correction": chosen["R_ch07_axis_correction"],
        "t_ch07_axis_correction_m": chosen["t_ch07_axis_correction_m"],
        "ch07_event_offset_frames": a.ch07_event_offset,
        "axis_mode": chosen["axis_mode"],
        "notes": [
            "Discrete CH07 axis modes were compared after the head extrinsic XY correction.",
            "Only a shared translation was fitted for each mode; no free rotation was optimized.",
            "The source skeleton is nose 2D/3D GT plus triangulated statistical bone lengths.",
        ],
    }
    a.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    a.output_refinement.write_text(json.dumps(refinement, ensure_ascii=False, indent=2), encoding="utf-8")
    for item in results:
        variant = dict(refinement)
        variant["R_ch07_axis_correction"] = item["R_ch07_axis_correction"]
        variant["t_ch07_axis_correction_m"] = item["t_ch07_axis_correction_m"]
        variant["axis_mode"] = item["axis_mode"]
        variant["selection_metrics"] = {
            key: item[key] for key in (
                "median_weighted_error_px", "p90_weighted_error_px",
                "median_raw_error_px", "p90_raw_error_px", "observations")
        }
        safe_mode = item["axis_mode"].replace("+", "plus").replace("-", "minus")
        (a.output_refinement.parent/f"axis_mode_refinement_{safe_mode}.json").write_text(
            json.dumps(variant, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
