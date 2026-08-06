#!/usr/bin/env python3
"""Compare raw BVH with rigid-wrist/C7/bone-length/upper-arm constrained IK."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

import project_0722_abx2_subject_scaled as kin
import project_joints as base
from compare_0722_rigid_wrist_replacement import R_WRIST_RIGID, SENSOR, WRIST_TO_RIGID_MM


HERE = Path(__file__).resolve().parent
BASELINE = HERE / "validation_0722_2_ch308_raw_bvh"
BASELINE_SUMMARY = BASELINE / "summary.json"
ALIGNED = Path(r"C:\Users\hand\Desktop\Dataset\0722_2\0711_035935\aligned_data\aligned_30hz.csv")
CONFIG = HERE / "projection_config_0722_head_ch3_08.json"
CALIBRATION = HERE / "validation_0722_h265_fixed_time_calibration" / "calibration_fixed_time.json"
OUTPUT = HERE / "validation_0722_2_rigid_wrist_c7_upperarm_ik_comparison"
CAMERAS = {"CAM_B": "module01_CAM_B", "CAM_C": "module01_CAM_C"}


def unit(vector: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm > 1e-9:
        return vector / norm
    if fallback is None:
        raise ValueError("Cannot normalize zero vector")
    return unit(fallback)


def sphere_intersection_nearest(
    center_a: np.ndarray,
    radius_a: float,
    center_b: np.ndarray,
    radius_b: float,
    prior: np.ndarray,
) -> np.ndarray:
    """Point on the intersection circle of two spheres nearest to prior."""
    delta = center_b - center_a
    distance = float(np.linalg.norm(delta))
    axis = unit(delta, np.asarray([1.0, 0.0, 0.0]))
    x = (radius_a * radius_a - radius_b * radius_b + distance * distance) / max(2.0 * distance, 1e-9)
    height2 = max(radius_a * radius_a - x * x, 0.0)
    circle_center = center_a + x * axis
    perpendicular = prior - circle_center
    perpendicular -= axis * float(np.dot(perpendicular, axis))
    if float(np.linalg.norm(perpendicular)) < 1e-8:
        fallback = np.asarray([0.0, 0.0, 1.0])
        if abs(float(np.dot(fallback, axis))) > 0.9:
            fallback = np.asarray([0.0, 1.0, 0.0])
        perpendicular = fallback - axis * float(np.dot(fallback, axis))
    return circle_center + unit(perpendicular) * np.sqrt(height2)


def solve_shoulder_chain(
    c7: np.ndarray,
    clavicle: np.ndarray,
    shoulder: np.ndarray,
    elbow: np.ndarray,
    wrist: np.ndarray,
    wrist_target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Hard-constrained solution preserving upper-arm direction and all lengths."""
    l_c7_clavicle = float(np.linalg.norm(clavicle - c7))
    l_clavicle_shoulder = float(np.linalg.norm(shoulder - clavicle))
    l_upper = float(np.linalg.norm(elbow - shoulder))
    l_fore = float(np.linalg.norm(wrist - elbow))
    upper_direction = unit(elbow - shoulder)

    # E = S + L_upper*u and |W-E|=L_fore, hence the shoulder S lies on a
    # sphere centered at (W-L_upper*u) with radius L_fore.
    wrist_sphere_center = wrist_target - l_upper * upper_direction
    center_distance = float(np.linalg.norm(wrist_sphere_center - c7))
    original_c7_shoulder = float(np.linalg.norm(shoulder - c7))

    shoulder_chain_min = abs(l_c7_clavicle - l_clavicle_shoulder)
    shoulder_chain_max = l_c7_clavicle + l_clavicle_shoulder
    sphere_min = abs(center_distance - l_fore)
    sphere_max = center_distance + l_fore
    feasible_min = max(shoulder_chain_min, sphere_min) + 1e-6
    feasible_max = min(shoulder_chain_max, sphere_max) - 1e-6
    reachable = feasible_min <= feasible_max
    if not reachable:
        # Nearest bounded approximation; report it explicitly.
        c7_shoulder_distance = float(np.clip(original_c7_shoulder, shoulder_chain_min, shoulder_chain_max))
    else:
        c7_shoulder_distance = float(np.clip(original_c7_shoulder, feasible_min, feasible_max))

    solved_shoulder = sphere_intersection_nearest(
        c7, c7_shoulder_distance, wrist_sphere_center, l_fore, shoulder
    )
    solved_elbow = solved_shoulder + l_upper * upper_direction

    # Solve C7->clavicle->shoulder exactly, retaining the BVH clavicle bend side.
    c7_to_shoulder = solved_shoulder - c7
    distance = float(np.linalg.norm(c7_to_shoulder))
    axis = unit(c7_to_shoulder)
    along = (l_c7_clavicle**2 - l_clavicle_shoulder**2 + distance**2) / max(2.0 * distance, 1e-9)
    height = np.sqrt(max(l_c7_clavicle**2 - along**2, 0.0))
    circle_center = c7 + along * axis
    bend = clavicle - circle_center
    bend -= axis * float(np.dot(bend, axis))
    if float(np.linalg.norm(bend)) < 1e-8:
        bend = np.cross(axis, np.asarray([0.0, 0.0, 1.0]))
        if float(np.linalg.norm(bend)) < 1e-8:
            bend = np.cross(axis, np.asarray([0.0, 1.0, 0.0]))
    solved_clavicle = circle_center + unit(bend) * height

    diagnostics = {
        "reachable": reachable,
        "lengths_mm": {
            "c7_clavicle": l_c7_clavicle,
            "clavicle_shoulder": l_clavicle_shoulder,
            "upper_arm": l_upper,
            "forearm": l_fore,
        },
        "wrist_residual_mm": float(np.linalg.norm(wrist_target - (solved_elbow + unit(wrist_target - solved_elbow) * l_fore))),
        "forearm_length_residual_mm": abs(float(np.linalg.norm(wrist_target - solved_elbow)) - l_fore),
        "upper_direction_error_deg": float(np.degrees(np.arccos(np.clip(np.dot(unit(solved_elbow - solved_shoulder), upper_direction), -1.0, 1.0)))),
        "c7_shoulder_distance_mm": c7_shoulder_distance,
    }
    return solved_clavicle, solved_shoulder, solved_elbow, diagnostics


def load_selected_rows(sequences: set[int]) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    with ALIGNED.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            seq = int(row["seq"])
            if seq in sequences:
                result[seq] = row
    missing = sequences - set(result)
    if missing:
        raise RuntimeError(f"Missing aligned rows: {sorted(missing)}")
    return result


def project_point(point: np.ndarray, camera_position: np.ndarray, camera_rotation: np.ndarray, model: dict[str, object]) -> np.ndarray:
    camera_point = camera_rotation.T @ (point - camera_position)
    uv, valid = base.omni_project(camera_point[None, :], model)
    return uv[0] if valid[0] else np.asarray([np.nan, np.nan])


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    baseline_summary = json.loads(BASELINE_SUMMARY.read_text(encoding="utf-8"))
    sequences = [int(value) for value in baseline_summary["selected_sequences"]]
    aligned_rows = load_selected_rows(set(sequences))
    models = base.load_camera_models(base.load_json(CONFIG))
    calibration = base.load_json(CALIBRATION)
    joint_index = {name: i for i, name in enumerate(kin.JOINT_NAMES)}

    head_axes = np.column_stack(([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]))
    head_to_rigid_mm = np.asarray([-2.0, 53.8, 135.5], dtype=np.float64)
    head_joint_in_rigid_mm = -head_axes.T @ head_to_rigid_mm
    reports: dict[str, object] = {}

    for camera, camera_key in CAMERAS.items():
        model = models[camera_key]
        r_rigid_camera = np.asarray(calibration[f"R_rigid_cam_{camera[-1]}"], dtype=np.float64)
        p_rigid_camera = np.asarray(calibration[f"p_rigid_cam_{camera[-1]}_mm"], dtype=np.float64)
        head_to_camera_rigid = p_rigid_camera - head_joint_in_rigid_mm
        destination = OUTPUT / camera_key
        overlay_destination = OUTPUT / "overlay" / camera_key
        destination.mkdir(parents=True, exist_ok=True)
        overlay_destination.mkdir(parents=True, exist_ok=True)
        frame_reports: list[dict[str, object]] = []

        for seq in sequences:
            row = aligned_rows[seq]
            raw = np.asarray([
                [float(row[f"mocap_{name}_world_{axis}"]) * 10.0 for axis in "xyz"]
                for name in kin.JOINT_NAMES
            ], dtype=np.float64)
            solved = raw.copy()

            head_position = np.asarray([float(row[f"mocap_CH3_08_Rigid_K_world_{axis}"]) * 1000.0 for axis in "xyz"])
            head_q = np.asarray([float(row[f"mocap_CH3_08_Rigid_K_world_q{axis}"]) for axis in "wxyz"])
            head_q /= np.linalg.norm(head_q)
            r_world_head = kin.q_to_matrix(head_q)
            bvh_head = raw[joint_index["Head"]]
            pwr_head_joint = head_position + r_world_head @ head_joint_in_rigid_mm
            pwr_to_bvh_translation = bvh_head - pwr_head_joint
            arms: dict[str, object] = {}

            for side in ("Left", "Right"):
                rigid_name = f"CH3_{'01' if side == 'Left' else '07'}_Rigid_K"
                rigid_position = np.asarray([float(row[f"mocap_{rigid_name}_world_{axis}"]) * 1000.0 for axis in "xyz"])
                rigid_q = np.asarray([float(row[f"mocap_{rigid_name}_world_q{axis}"]) for axis in "wxyz"])
                rigid_q /= np.linalg.norm(rigid_q)
                r_world_rigid = kin.q_to_matrix(rigid_q)
                r_world_wrist = r_world_rigid @ R_WRIST_RIGID[side].T
                wrist_target = rigid_position - r_world_wrist @ WRIST_TO_RIGID_MM[side] + pwr_to_bvh_translation

                names = ["Spine2", f"{side}Shoulder", f"{side}Arm", f"{side}ForeArm", f"{side}Hand"]
                indices = [joint_index[name] for name in names]
                c7, clavicle, shoulder, elbow, wrist = raw[indices]
                new_clavicle, new_shoulder, new_elbow, diagnostics = solve_shoulder_chain(
                    c7, clavicle, shoulder, elbow, wrist, wrist_target
                )
                solved[indices[1]] = new_clavicle
                solved[indices[2]] = new_shoulder
                solved[indices[3]] = new_elbow
                solved[indices[4]] = wrist_target
                diagnostics.update({
                    "wrist_target_mm": wrist_target.tolist(),
                    "wrist_delta_mm": float(np.linalg.norm(wrist_target - wrist)),
                    "raw_bvh_wrist_in_rigid_frame_mm": (
                        r_world_rigid.T @ ((wrist - pwr_to_bvh_translation) - rigid_position)
                    ).tolist(),
                    "configured_wrist_in_rigid_frame_mm": (
                        -R_WRIST_RIGID[side].T @ WRIST_TO_RIGID_MM[side]
                    ).tolist(),
                    "shoulder_delta_mm": float(np.linalg.norm(new_shoulder - shoulder)),
                    "elbow_delta_mm": float(np.linalg.norm(new_elbow - elbow)),
                    "clavicle_delta_mm": float(np.linalg.norm(new_clavicle - clavicle)),
                })
                arms[side] = diagnostics

            camera_position = bvh_head + r_world_head @ head_to_camera_rigid
            camera_rotation = r_world_head @ r_rigid_camera
            source_path = BASELINE / "selected_clean" / camera_key / f"seq_{seq:06d}_projection.jpg"
            # Use the undecorated source image so neither result is visually privileged.
            undecorated_path = BASELINE / "source_frames" / camera / f"seq_{seq:06d}.jpg"
            source = cv2.imread(str(undecorated_path), cv2.IMREAD_COLOR)
            if source is None:
                raise RuntimeError(f"Missing source image {undecorated_path}")

            raw_panel, solved_panel = source.copy(), source.copy()
            kin.draw(raw_panel, raw, camera_position, camera_rotation, model, (0, 255, 255), 4)
            kin.draw(solved_panel, solved, camera_position, camera_rotation, model, (255, 0, 255), 4)
            for image, title, color in (
                (raw_panel, f"Raw BVH  seq={seq:06d}", (0, 255, 255)),
                (solved_panel, "Rigid wrist + C7 + lengths + upper-arm pose", (255, 0, 255)),
            ):
                cv2.rectangle(image, (0, 0), (image.shape[1], 64), (0, 0, 0), -1)
                cv2.putText(image, title, (22, 43), cv2.FONT_HERSHEY_SIMPLEX, .72, color, 2, cv2.LINE_AA)
            comparison = np.hstack((raw_panel, solved_panel))
            cv2.imwrite(str(destination / f"seq_{seq:06d}_comparison.jpg"), comparison, [cv2.IMWRITE_JPEG_QUALITY, 94])

            overlay = source.copy()
            kin.draw(overlay, raw, camera_position, camera_rotation, model, (0, 255, 255), 2)
            kin.draw(overlay, solved, camera_position, camera_rotation, model, (255, 0, 255), 4)
            c7_uv = project_point(raw[joint_index["Spine2"]], camera_position, camera_rotation, model)
            if np.all(np.isfinite(c7_uv)):
                point = tuple(np.rint(c7_uv).astype(int))
                cv2.circle(overlay, point, 9, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(overlay, "C7", (point[0] + 10, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(overlay, "yellow=raw BVH  magenta=constrained IK  green=C7", (22, 43), cv2.FONT_HERSHEY_SIMPLEX, .72, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imwrite(str(overlay_destination / f"seq_{seq:06d}_overlay.jpg"), overlay, [cv2.IMWRITE_JPEG_QUALITY, 94])

            for side in ("Left", "Right"):
                for label, name in (("shoulder", f"{side}Arm"), ("elbow", f"{side}ForeArm"), ("wrist", f"{side}Hand")):
                    ji = joint_index[name]
                    old_uv = project_point(raw[ji], camera_position, camera_rotation, model)
                    new_uv = project_point(solved[ji], camera_position, camera_rotation, model)
                    arms[side][f"{label}_pixel_delta"] = (
                        float(np.linalg.norm(new_uv - old_uv))
                        if np.all(np.isfinite(old_uv)) and np.all(np.isfinite(new_uv)) else None
                    )
            frame_reports.append({
                "seq": seq,
                "pwr_to_bvh_translation_mm": pwr_to_bvh_translation.tolist(),
                "arms": arms,
            })
        reports[camera_key] = {"sample_count": len(frame_reports), "frames": frame_reports}

        thumbnails = []
        for seq in sequences:
            image = cv2.imread(str(destination / f"seq_{seq:06d}_comparison.jpg"))
            if image is not None:
                thumbnails.append(cv2.resize(image, (640, 200), interpolation=cv2.INTER_AREA))
        if len(thumbnails) == len(sequences):
            overview = np.vstack([np.hstack(thumbnails[i:i + 4]) for i in range(0, len(thumbnails), 4)])
            cv2.imwrite(str(OUTPUT / f"overview_{camera_key}.jpg"), overview, [cv2.IMWRITE_JPEG_QUALITY, 92])

    all_frames = [frame for report in reports.values() for frame in report["frames"]]
    statistics: dict[str, object] = {}
    for side in ("Left", "Right"):
        rows = [frame["arms"][side] for frame in all_frames]
        statistics[side] = {
            "reachable": f"{sum(bool(row['reachable']) for row in rows)}/{len(rows)}",
            "wrist_delta_mm_median": float(np.median([row["wrist_delta_mm"] for row in rows])),
            "shoulder_delta_mm_median": float(np.median([row["shoulder_delta_mm"] for row in rows])),
            "elbow_delta_mm_median": float(np.median([row["elbow_delta_mm"] for row in rows])),
            "forearm_length_residual_mm_max": float(np.max([row["forearm_length_residual_mm"] for row in rows])),
            "upper_direction_error_deg_max": float(np.max([row["upper_direction_error_deg"] for row in rows])),
            "raw_bvh_wrist_in_rigid_frame_mm_median": np.median(
                np.asarray([row["raw_bvh_wrist_in_rigid_frame_mm"] for row in rows]), axis=0
            ).tolist(),
            "raw_bvh_wrist_in_rigid_frame_mm_std": np.std(
                np.asarray([row["raw_bvh_wrist_in_rigid_frame_mm"] for row in rows]), axis=0
            ).tolist(),
            "configured_wrist_in_rigid_frame_mm": rows[0]["configured_wrist_in_rigid_frame_mm"],
        }
    summary = {
        "schema": "0722_2_rigid_wrist_c7_upperarm_ik.v1",
        "c7_mapping": "Spine2",
        "constraints": ["CH3_01/07 rigid-derived wrist position", "C7/Spine2 position", "all four chain segment lengths", "raw BVH upper-arm direction"],
        "right_axis_mapping_determinant": float(np.linalg.det(R_WRIST_RIGID["Right"])),
        "head_alignment_translation_mm_std": np.std(
            np.asarray([
                frame["pwr_to_bvh_translation_mm"]
                for frame in reports["module01_CAM_C"]["frames"]
            ]), axis=0
        ).tolist(),
        "statistics": statistics,
        "reports": reports,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(statistics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
