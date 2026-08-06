#!/usr/bin/env python3
"""Calibrate the head rig from H.265 images while keeping clock sync fixed."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pupil_apriltags import Detector
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

import project_joints as base


HERE = Path(__file__).resolve().parent
IMAGE_ROOT = HERE / "validation_0722_h265_random100" / "source_frames" / "module01"
MAPPING = HERE / "validation_0722_h265_random100" / "h265_frame_mapping.json"
OUTPUT = HERE / "validation_0722_h265_fixed_time_calibration"
CONFIG = HERE / "projection_config_0722_head_ch3_08.json"
OLD_CALIBRATION = HERE / "validation_0722_head_ch3_08_random100_final" / "joint_calibration_final.json"
SYNC_REPORT = HERE / "validation_0722_head_imu_sync" / "head_imu_mocap_sync_report.json"
MOCAP = Path(r"C:\Users\hand\Desktop\Dataset\0722\abx2_mocap_rigid_csv\mocap_rigid_20260722.csv")
RIGIDS = {9: "CH3_01_Rigid_K", 498: "CH3_07_Rigid_K"}
HEAD = "CH3_08_Rigid_K"
EXPOSURE_CENTER_FROM_DEVICE_END_SEC = -0.006


def load_rigid_motion() -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Slerp]]:
    names = [HEAD, *RIGIDS.values()]
    times: list[float] = []
    positions = {name: [] for name in names}
    quaternions = {name: [] for name in names}
    with MOCAP.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            times.append(float(row["time_sec"]))
            for name in names:
                positions[name].append([float(row[f"{name}_world_{axis}"]) * 1000.0 for axis in "xyz"])
                qw, qx, qy, qz = [float(row[f"{name}_world_q{axis}"]) for axis in "wxyz"]
                quaternions[name].append([qx, qy, qz, qw])
    time_array = np.asarray(times)
    return (
        time_array,
        {name: np.asarray(value) for name, value in positions.items()},
        {name: Slerp(time_array, Rotation.from_quat(np.asarray(value))) for name, value in quaternions.items()},
    )


def pose(name: str, t: float, times: np.ndarray, positions: dict[str, np.ndarray], rotations: dict[str, Slerp]) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray([np.interp(t, times, positions[name][:, axis]) for axis in range(3)])
    return p, rotations[name]([t]).as_matrix()[0]


def detect_tags() -> list[dict[str, object]]:
    cache_path = OUTPUT / "apriltag_detections.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    detector = Detector(families="tag36h11", nthreads=8, quad_decimate=1.0, quad_sigma=0.0, refine_edges=1, decode_sharpening=0.25)
    found: list[dict[str, object]] = []
    for camera in ("CAM_B", "CAM_C"):
        for path in sorted((IMAGE_ROOT / camera).glob("seq_*.jpg")):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            for detection in detector.detect(image):
                if detection.tag_id not in RIGIDS or detection.decision_margin < 20.0:
                    continue
                found.append({
                    "camera": camera,
                    "seq": int(path.stem.split("_")[1]),
                    "tag_id": int(detection.tag_id),
                    "center": np.asarray(detection.center).tolist(),
                    "corners": np.asarray(detection.corners).tolist(),
                    "decision_margin": float(detection.decision_margin),
                })
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(found, ensure_ascii=False, indent=2), encoding="utf-8")
    return found


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = base.load_json(CONFIG)
    models = base.load_camera_models(config)
    old = base.load_json(OLD_CALIBRATION)
    sync = base.load_json(SYNC_REPORT)
    mapping = base.load_json(MAPPING)
    observations = detect_tags()
    device_times = {
        (camera, int(row["seq"])): float(row["device_ts_ms"])
        for camera in ("CAM_B", "CAM_C") for row in mapping[camera]["mapping"]
    }
    scale = float(sync["global_scale"])
    offset = float(sync["global_offset_sec"])
    origin_ms = float(sync["module01_camera_origin_ms"])
    head_delta = float(sync["results"][HEAD]["additional_delta_sec"])
    for item in observations:
        device_ms = device_times[(str(item["camera"]), int(item["seq"]))]
        item["mocap_time_sec"] = offset + scale * ((device_ms - origin_ms) / 1000.0) + head_delta + EXPOSURE_CENTER_FROM_DEVICE_END_SEC

    times, positions, rotations = load_rigid_motion()
    stereo = models["module01_CAM_B"]["relative_transform_calibrated"]
    r_cb, t_cb_mm = stereo[:3, :3], stereo[:3, 3] * 1000.0
    old_rb = np.asarray(old["R_rigid_cam_B"])
    old_pb = np.asarray(old["p_rigid_cam_B_mm"])
    old_tags = {tag: np.asarray(old["tag_offsets_rigid_mm"][str(tag)]) for tag in RIGIDS}

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
        rb = Rotation.from_rotvec(x[:3]).as_matrix() @ old_rb
        pb = x[3:6]
        tags = {9: x[6:9], 498: x[9:12]}
        return rb, pb, tags

    def camera_pose(camera: str, rb: np.ndarray, pb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if camera == "CAM_B":
            return rb, pb
        return rb @ r_cb.T, pb + rb @ (-r_cb.T @ t_cb_mm)

    def residual(x: np.ndarray, selected: list[dict[str, object]] = observations) -> np.ndarray:
        rb, pb, tags = unpack(x)
        values: list[float] = []
        for item in selected:
            camera = str(item["camera"])
            t = float(item["mocap_time_sec"])
            ph, rh = pose(HEAD, t, times, positions, rotations)
            wrist_name = RIGIDS[int(item["tag_id"])]
            pw, rw = pose(wrist_name, t, times, positions, rotations)
            world_tag = pw + rw @ tags[int(item["tag_id"])]
            rc, pc = camera_pose(camera, rb, pb)
            world_camera = ph + rh @ pc
            world_rotation = rh @ rc
            point_camera = world_rotation.T @ (world_tag - world_camera)
            uv, valid = base.omni_project(point_camera[None, :], models[f"module01_{camera}"])
            if not valid[0] or not np.all(np.isfinite(uv[0])):
                values.extend((500.0, 500.0))
            else:
                values.extend((uv[0] - np.asarray(item["center"])).tolist())
        return np.asarray(values)

    x0 = np.r_[np.zeros(3), old_pb, old_tags[9], old_tags[498]]
    first = least_squares(residual, x0, loss="huber", f_scale=18.0, max_nfev=600, verbose=1)
    errors = np.linalg.norm(residual(first.x).reshape(-1, 2), axis=1)
    inliers = errors < max(45.0, float(np.percentile(errors, 85)))
    inlier_obs = [item for item, keep in zip(observations, inliers) if keep]
    final = least_squares(lambda x: residual(x, inlier_obs), first.x, loss="soft_l1", f_scale=8.0, max_nfev=1000, verbose=1)
    rb, pb, tags = unpack(final.x)
    rc, pc = camera_pose("CAM_C", rb, pb)

    # Evaluate all detections and expose radial/polar-angle dependence.
    all_error = np.linalg.norm(residual(final.x).reshape(-1, 2), axis=1)
    detail: list[dict[str, object]] = []
    for item, error in zip(observations, all_error):
        model = models[f"module01_{item['camera']}"]
        center = np.asarray(item["center"])
        normalized_radius = float(np.linalg.norm((center - [model["cx"], model["cy"]]) / [model["width"] / 2.0, model["height"] / 2.0]))
        # The inverse gives the actual calibrated polar angle of this observation.
        from project_0722_head_final import omni_unproject_unit
        ray = omni_unproject_unit(center[None, :], model)[0]
        polar = float(np.degrees(np.arccos(np.clip(ray[2], -1.0, 1.0))))
        detail.append({**item, "reprojection_error_px": float(error), "normalized_image_radius": normalized_radius, "polar_angle_deg": polar})

    report = {
        "schema": "head_h265_fixed_time_calibration.v1",
        "timing": {
            "formula": "mocap = global_offset + scale*(camera_device_ms-origin_ms)/1000 + head_imu_delta + exposure_center_shift",
            "global_scale": scale, "global_offset_sec": offset, "origin_ms": origin_ms,
            "head_imu_delta_sec": head_delta, "exposure_center_shift_sec": EXPOSURE_CENTER_FROM_DEVICE_END_SEC,
            "net_additional_to_device_end_sec": head_delta + EXPOSURE_CENTER_FROM_DEVICE_END_SEC,
        },
        "detections": len(observations), "inliers": len(inlier_obs),
        "median_error_px": float(np.median(all_error)), "p90_error_px": float(np.percentile(all_error, 90)),
        "R_rigid_cam_B": rb.tolist(), "p_rigid_cam_B_mm": pb.tolist(),
        "R_rigid_cam_C": rc.tolist(), "p_rigid_cam_C_mm": pc.tolist(),
        "tag_offsets_rigid_mm": {str(tag): value.tolist() for tag, value in tags.items()},
        "stereo_transform_source": str(config["head_relative_bc_yaml"]),
        "observations": detail,
    }
    (OUTPUT / "calibration_fixed_time.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=160)
    for camera, marker in (("CAM_B", "o"), ("CAM_C", "s")):
        rows = [row for row in detail if row["camera"] == camera]
        ax.scatter([row["polar_angle_deg"] for row in rows], [row["reprojection_error_px"] for row in rows], s=22, alpha=.7, marker=marker, label=camera)
    ax.axvspan(65, 90, color="#ffb000", alpha=.12, label="intrinsics weakly covered (approx.)")
    ax.set(xlabel="Calibrated polar angle (deg)", ylabel="AprilTag center reprojection error (px)", title="H.265 + fixed-time head-camera validation")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(OUTPUT / "reprojection_error_vs_angle.png")
    print(json.dumps({key: report[key] for key in ("detections", "inliers", "median_error_px", "p90_error_px")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
