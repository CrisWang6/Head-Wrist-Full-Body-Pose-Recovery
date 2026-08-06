#!/usr/bin/env python3
"""Fuse CAM_B/C shoulder-elbow detections into 3-D and render CAM_C."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter

import project_joints as base


WIDTH = 1920
HEIGHT = 1200
FRAME_BYTES = WIDTH * HEIGHT * 3
JOINTS = {
    "LeftArm": "left_shoulder",
    "RightArm": "right_shoulder",
    "LeftForeArm": "left_elbow",
    "RightForeArm": "right_elbow",
}
SIDES = (
    ("LeftArm", "LeftForeArm"),
    ("RightArm", "RightForeArm"),
)


def omni_unproject_unit(
    uv: np.ndarray, model: dict[str, object]
) -> np.ndarray:
    """Inverse Kalibr omni+radtan pixels to unit camera rays."""
    pixels = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    camera_matrix = np.asarray(
        [
            [model["fx"], 0.0, model["cx"]],
            [0.0, model["fy"], model["cy"]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    normalized = cv2.undistortPoints(
        pixels, camera_matrix, np.asarray(model["distortion"], dtype=np.float64)
    ).reshape(-1, 2)
    x, y = normalized[:, 0], normalized[:, 1]
    radius2 = x * x + y * y
    xi = float(model["xi"])
    lam = (
        xi
        + np.sqrt(np.maximum(1.0 + (1.0 - xi * xi) * radius2, 1e-12))
    ) / (1.0 + radius2)
    rays = np.column_stack((lam * x, lam * y, lam - xi))
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    return rays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Rebuild CSV outputs without rendering the stereo-only preview.",
    )
    return parser.parse_args()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def triangulate_rays(
    ray_b: np.ndarray,
    ray_c: np.ndarray,
    r_c_b: np.ndarray,
    t_c_b_mm: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    center_c_b = -r_c_b.T @ t_c_b_mm
    ray_c_b = r_c_b.T @ ray_c
    matrix = np.column_stack((ray_b, -ray_c_b))
    distances, _, _, _ = np.linalg.lstsq(matrix, center_c_b, rcond=None)
    point_on_b = distances[0] * ray_b
    point_on_c = center_c_b + distances[1] * ray_c_b
    point_b = 0.5 * (point_on_b + point_on_c)
    angle_deg = math.degrees(
        math.acos(float(np.clip(abs(np.dot(ray_b, ray_c_b)), -1.0, 1.0)))
    )
    return point_b, {
        "distance_b_mm": float(distances[0]),
        "distance_c_mm": float(distances[1]),
        "ray_gap_mm": float(np.linalg.norm(point_on_b - point_on_c)),
        "ray_angle_deg": angle_deg,
    }


def camera_pose_in_skeleton(
    row: dict[str, str],
    calibration: dict[str, object],
    camera: str,
) -> tuple[np.ndarray, np.ndarray]:
    rigid_position = np.asarray(
        [float(row[f"mocap_CH3_08_Rigid_K_world_{axis}"]) * 1000.0 for axis in "xyz"]
    )
    quaternion = [
        float(row[f"mocap_CH3_08_Rigid_K_world_q{axis}"]) for axis in "wxyz"
    ]
    r_world_rigid = base.quaternion_to_rotation(*quaternion)

    bvh_head = np.asarray(
        [float(row[f"mocap_Head_world_{axis}"]) * 10.0 for axis in "xyz"]
    )
    head_axes = np.column_stack(
        ([0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
    )
    head_to_rigid_mm = np.asarray([-2.0, 53.8, 135.5], dtype=np.float64)
    head_joint_in_rigid_mm = -head_axes.T @ head_to_rigid_mm
    rigid_head_joint = rigid_position + r_world_rigid @ head_joint_in_rigid_mm
    rigid_to_skeleton = bvh_head - rigid_head_joint

    r_rigid_camera = np.asarray(
        calibration[f"R_rigid_cam_{camera}"], dtype=np.float64
    )
    p_rigid_camera = np.asarray(
        calibration[f"p_rigid_cam_{camera}_mm"], dtype=np.float64
    )
    r_world_camera = r_world_rigid @ r_rigid_camera
    p_world_camera = (
        rigid_position + r_world_rigid @ p_rigid_camera + rigid_to_skeleton
    )
    return p_world_camera, r_world_camera


def reprojection_error(
    point_b: np.ndarray,
    uv_b: np.ndarray,
    uv_c: np.ndarray,
    model_b: dict[str, object],
    model_c: dict[str, object],
    r_c_b: np.ndarray,
    t_c_b_mm: np.ndarray,
) -> tuple[float, float]:
    projected_b, valid_b = base.omni_project(point_b[None, :], model_b)
    point_c = r_c_b @ point_b + t_c_b_mm
    projected_c, valid_c = base.omni_project(point_c[None, :], model_c)
    if not valid_b[0] or not valid_c[0]:
        return float("inf"), float("inf")
    return (
        float(np.linalg.norm(projected_b[0] - uv_b)),
        float(np.linalg.norm(projected_c[0] - uv_c)),
    )


def read_exact(stream, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def smooth_contiguous(
    points_by_seq: dict[int, np.ndarray],
) -> dict[int, np.ndarray]:
    if not points_by_seq:
        return {}
    sequences = sorted(points_by_seq)
    groups: list[list[int]] = []
    current = [sequences[0]]
    for seq in sequences[1:]:
        if seq == current[-1] + 1:
            current.append(seq)
        else:
            groups.append(current)
            current = [seq]
    groups.append(current)

    result = dict(points_by_seq)
    for group in groups:
        if len(group) < 5:
            continue
        values = np.vstack([points_by_seq[seq] for seq in group])
        window = min(7, len(group) if len(group) % 2 else len(group) - 1)
        if window < 5:
            continue
        filtered = savgol_filter(values, window_length=window, polyorder=2, axis=0)
        for seq, point in zip(group, filtered):
            result[seq] = point
    return result


def in_image(point: np.ndarray) -> bool:
    return bool(
        np.all(np.isfinite(point))
        and 0.0 <= point[0] < WIDTH
        and 0.0 <= point[1] < HEIGHT
    )


def render_video(
    recording: Path,
    ffmpeg: Path,
    fused_rows: list[dict[str, object]],
    cam_c_rows: list[dict[str, str]],
    output_path: Path,
) -> dict[str, object]:
    fused_by_seq = {int(row["seq"]): row for row in fused_rows}
    decoded_to_seq = {
        int(row["decoded_frame_index"]): int(row["seq"])
        for row in cam_c_rows
        if row["decoded_frame_index"].strip()
    }
    decoder = subprocess.Popen(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-err_detect",
            "ignore_err",
            "-i",
            str(recording / "module01_D45D2E00_CAM_C.mp4"),
            "-vsync",
            "0",
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "-",
        ],
        stdout=subprocess.PIPE,
    )
    encoder = subprocess.Popen(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{WIDTH}x{HEIGHT}",
            "-r",
            "30",
            "-i",
            "-",
            "-an",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-rc",
            "vbr",
            "-cq",
            "19",
            "-b:v",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output_path),
        ],
        stdin=subprocess.PIPE,
    )

    decoded_index = 0
    annotated = 0
    while True:
        raw = read_exact(decoder.stdout, FRAME_BYTES)
        if not raw:
            break
        if len(raw) != FRAME_BYTES:
            raise RuntimeError(f"Partial decoded frame: {len(raw)} bytes")
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3).copy()
        seq = decoded_to_seq.get(decoded_index)
        row = fused_by_seq.get(seq) if seq is not None else None
        points: dict[str, tuple[int, int]] = {}
        if row:
            for joint, field in JOINTS.items():
                x_value = row.get(f"{field}_x", "")
                y_value = row.get(f"{field}_y", "")
                if x_value == "" or y_value == "":
                    continue
                point = np.asarray([float(x_value), float(y_value)])
                if in_image(point):
                    points[joint] = tuple(np.rint(point).astype(int))
            colors = {"Left": (0, 220, 255), "Right": (255, 150, 0)}
            for shoulder, elbow in SIDES:
                side = shoulder.removesuffix("Arm")
                if shoulder in points and elbow in points:
                    cv2.line(
                        frame,
                        points[shoulder],
                        points[elbow],
                        colors[side],
                        5,
                        cv2.LINE_AA,
                    )
            if "LeftArm" in points and "RightArm" in points:
                cv2.line(
                    frame,
                    points["LeftArm"],
                    points["RightArm"],
                    (80, 255, 80),
                    4,
                    cv2.LINE_AA,
                )
            for joint, point in points.items():
                side = "Left" if joint.startswith("Left") else "Right"
                cv2.circle(frame, point, 10, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, point, 7, colors[side], -1, cv2.LINE_AA)
            if points:
                annotated += 1
        cv2.rectangle(frame, (0, 0), (630, 58), (0, 0, 0), -1)
        label = (
            f"CAM_C stereo-fused shoulder/elbow  seq={seq}"
            if seq is not None
            else "CAM_C stereo-fused shoulder/elbow"
        )
        cv2.putText(
            frame,
            label,
            (20, 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        encoder.stdin.write(frame.tobytes())
        decoded_index += 1
        if decoded_index % 1000 == 0:
            print(f"rendered {decoded_index} frames", flush=True)

    decoder_code = decoder.wait()
    encoder.stdin.close()
    encoder_code = encoder.wait()
    if decoder_code or encoder_code:
        raise RuntimeError(
            f"ffmpeg failed: decoder={decoder_code}, encoder={encoder_code}"
        )
    return {
        "decoded_frames": decoded_index,
        "annotated_frames": annotated,
        "output": str(output_path),
    }


def main() -> int:
    args = parse_args()
    recording = args.recording.resolve()
    project_dir = args.project_dir.resolve()
    aligned_path = recording / "aligned_data" / "aligned_30hz.csv"
    pose_b_path = recording / "aligned_data" / "module01_cam_b_shoulder_elbow_2d.csv"
    pose_c_path = recording / "aligned_data" / "module01_cam_c_shoulder_elbow_2d.csv"
    output_aligned = recording / "aligned_data" / "aligned_30hz_stereo_upper_body.csv"
    output_fused_2d = (
        recording / "aligned_data" / "module01_cam_c_stereo_fused_shoulder_elbow_2d.csv"
    )
    output_video = recording / "module01_CAM_C_stereo_fused_shoulder_elbow.mp4"
    output_report = recording / "aligned_data" / "stereo_upper_body_report.json"

    sys.path.insert(0, str(project_dir))
    config = base.load_json(project_dir / "projection_config_0722_head_ch3_08.json")
    calibration = base.load_json(
        project_dir
        / "validation_0722_h265_fixed_time_calibration"
        / "calibration_fixed_time.json"
    )
    models = base.load_camera_models(config)
    model_b = models["module01_CAM_B"]
    model_c = models["module01_CAM_C"]
    transform = model_b["relative_transform_calibrated"]
    r_c_b = transform[:3, :3]
    t_c_b_mm = transform[:3, 3] * 1000.0

    aligned_rows = load_csv(aligned_path)
    pose_b = {int(row["seq"]): row for row in load_csv(pose_b_path)}
    pose_c_list = load_csv(pose_c_path)
    pose_c = {int(row["seq"]): row for row in pose_c_list}

    candidates: dict[int, dict[str, dict[str, object]]] = {}
    same_label_costs: list[float] = []
    swapped_label_costs: list[float] = []
    swapped_frames = 0
    for row in aligned_rows:
        seq = int(row["seq"])
        row_b = pose_b.get(seq)
        row_c = pose_c.get(seq)
        if not row_b or not row_c or row_b["status"] != "ok" or row_c["status"] != "ok":
            continue
        frame: dict[str, dict[str, object]] = {}
        for joint, field in JOINTS.items():
            uv_b = np.asarray(
                [float(row_b[f"{field}_x"]), float(row_b[f"{field}_y"])],
                dtype=np.float64,
            )
            uv_c = np.asarray(
                [float(row_c[f"{field}_x"]), float(row_c[f"{field}_y"])],
                dtype=np.float64,
            )
            if not in_image(uv_b) or not in_image(uv_c):
                continue
            ray_b = omni_unproject_unit(uv_b[None, :], model_b)[0]
            ray_c = omni_unproject_unit(uv_c[None, :], model_c)[0]
            point_b, metrics = triangulate_rays(ray_b, ray_c, r_c_b, t_c_b_mm)
            error_b, error_c = reprojection_error(
                point_b, uv_b, uv_c, model_b, model_c, r_c_b, t_c_b_mm
            )
            metrics.update(reprojection_b_px=error_b, reprojection_c_px=error_c)
            valid = (
                150.0 <= metrics["distance_b_mm"] <= 4000.0
                and 150.0 <= metrics["distance_c_mm"] <= 4000.0
                and metrics["ray_gap_mm"] <= 150.0
                and metrics["ray_angle_deg"] >= 1.0
                and error_b <= 50.0
                and error_c <= 50.0
            )
            if not valid:
                continue
            camera_position, camera_rotation = camera_pose_in_skeleton(
                row, calibration, "B"
            )
            world = camera_position + camera_rotation @ point_b
            original = np.asarray(
                [float(row[f"mocap_{joint}_world_{axis}"]) * 10.0 for axis in "xyz"]
            )
            metrics["original_delta_mm"] = float(np.linalg.norm(world - original))
            if metrics["original_delta_mm"] > 750.0:
                continue
            frame[joint] = {
                "world_mm": world,
                "point_b_mm": point_b,
                "metrics": metrics,
            }

        if all(joint in frame for joint in JOINTS):
            originals = {
                joint: np.asarray(
                    [
                        float(row[f"mocap_{joint}_world_{axis}"]) * 10.0
                        for axis in "xyz"
                    ]
                )
                for joint in JOINTS
            }
            same = sum(
                np.linalg.norm(frame[joint]["world_mm"] - originals[joint])
                for joint in JOINTS
            )
            swapped_pairs = {
                "LeftArm": "RightArm",
                "RightArm": "LeftArm",
                "LeftForeArm": "RightForeArm",
                "RightForeArm": "LeftForeArm",
            }
            swapped = sum(
                np.linalg.norm(
                    frame[swapped_pairs[joint]]["world_mm"] - originals[joint]
                )
                for joint in JOINTS
            )
            same_label_costs.append(float(same))
            swapped_label_costs.append(float(swapped))
            if swapped + 100.0 < same:
                frame = {
                    joint: frame[source] for joint, source in swapped_pairs.items()
                }
                swapped_frames += 1

        for shoulder, elbow in SIDES:
            if shoulder in frame and elbow in frame:
                upper_arm = float(
                    np.linalg.norm(
                        frame[shoulder]["world_mm"] - frame[elbow]["world_mm"]
                    )
                )
                if not 120.0 <= upper_arm <= 550.0:
                    frame.pop(shoulder, None)
                    frame.pop(elbow, None)
        candidates[seq] = frame

    smoothed: dict[str, dict[int, np.ndarray]] = {}
    for joint in JOINTS:
        points = {
            seq: frame[joint]["world_mm"]
            for seq, frame in candidates.items()
            if joint in frame
        }
        smoothed[joint] = smooth_contiguous(points)

    audit_columns: list[str] = []
    for joint in JOINTS:
        audit_columns.extend(
            [
                f"stereo_{joint}_valid",
                f"stereo_{joint}_reprojection_mean_px",
                f"stereo_{joint}_ray_gap_mm",
                f"stereo_{joint}_ray_angle_deg",
                f"stereo_{joint}_original_delta_mm",
            ]
        )
    audit_columns.append("stereo_upper_body_valid_joint_count")
    aligned_header = list(aligned_rows[0])
    output_header = aligned_header + [
        column for column in audit_columns if column not in aligned_header
    ]
    fused_rows: list[dict[str, object]] = []
    valid_counts = {joint: 0 for joint in JOINTS}
    with output_aligned.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_header)
        writer.writeheader()
        for row in aligned_rows:
            seq = int(row["seq"])
            frame = candidates.get(seq, {})
            valid_count = 0
            fused: dict[str, object] = {"seq": seq, "valid_joint_count": 0}
            p_world_c, r_world_c = camera_pose_in_skeleton(row, calibration, "C")
            for joint, field in JOINTS.items():
                item = frame.get(joint)
                point_world = smoothed[joint].get(seq)
                valid = item is not None and point_world is not None
                row[f"stereo_{joint}_valid"] = int(valid)
                if not valid:
                    for suffix in (
                        "reprojection_mean_px",
                        "ray_gap_mm",
                        "ray_angle_deg",
                        "original_delta_mm",
                    ):
                        row[f"stereo_{joint}_{suffix}"] = ""
                    fused[f"{field}_x"] = ""
                    fused[f"{field}_y"] = ""
                    continue
                valid_count += 1
                valid_counts[joint] += 1
                for axis, value in zip("xyz", point_world / 10.0):
                    row[f"mocap_{joint}_world_{axis}"] = f"{value:.9f}"
                metrics = item["metrics"]
                row[f"stereo_{joint}_reprojection_mean_px"] = (
                    f"{0.5 * (metrics['reprojection_b_px'] + metrics['reprojection_c_px']):.6f}"
                )
                row[f"stereo_{joint}_ray_gap_mm"] = f"{metrics['ray_gap_mm']:.6f}"
                row[f"stereo_{joint}_ray_angle_deg"] = f"{metrics['ray_angle_deg']:.6f}"
                row[f"stereo_{joint}_original_delta_mm"] = (
                    f"{metrics['original_delta_mm']:.6f}"
                )
                point_c = r_world_c.T @ (point_world - p_world_c)
                uv_c, projection_valid = base.omni_project(point_c[None, :], model_c)
                if projection_valid[0] and in_image(uv_c[0]):
                    fused[f"{field}_x"] = float(uv_c[0, 0])
                    fused[f"{field}_y"] = float(uv_c[0, 1])
                else:
                    fused[f"{field}_x"] = ""
                    fused[f"{field}_y"] = ""
            row["stereo_upper_body_valid_joint_count"] = valid_count
            fused["valid_joint_count"] = valid_count
            fused_rows.append(fused)
            writer.writerow(row)

    fused_header = ["seq", "valid_joint_count"]
    for field in JOINTS.values():
        fused_header.extend([f"{field}_x", f"{field}_y"])
    with output_fused_2d.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fused_header)
        writer.writeheader()
        writer.writerows(fused_rows)

    metric_values: dict[str, dict[str, list[float]]] = {
        joint: {
            "reprojection_mean_px": [],
            "ray_gap_mm": [],
            "ray_angle_deg": [],
            "original_delta_mm": [],
        }
        for joint in JOINTS
    }
    for frame in candidates.values():
        for joint, item in frame.items():
            metrics = item["metrics"]
            metric_values[joint]["reprojection_mean_px"].append(
                0.5
                * (metrics["reprojection_b_px"] + metrics["reprojection_c_px"])
            )
            for key in ("ray_gap_mm", "ray_angle_deg", "original_delta_mm"):
                metric_values[joint][key].append(float(metrics[key]))

    quality = {}
    for joint, metrics in metric_values.items():
        quality[joint] = {
            name: {
                "median": float(np.median(values)) if values else None,
                "p95": float(np.percentile(values, 95)) if values else None,
            }
            for name, values in metrics.items()
        }
    report = {
        "schema": "stereo_upper_body_fusion.v1",
        "recording": str(recording),
        "stereo_baseline_mm": float(np.linalg.norm(t_c_b_mm)),
        "aligned_rows": len(aligned_rows),
        "valid_joint_counts": valid_counts,
        "all_four_valid_frames": sum(
            all(seq in smoothed[joint] for joint in JOINTS)
            for seq in range(len(aligned_rows))
        ),
        "left_right_swapped_frames": swapped_frames,
        "same_label_cost_median_mm": (
            float(np.median(same_label_costs)) if same_label_costs else None
        ),
        "swapped_label_cost_median_mm": (
            float(np.median(swapped_label_costs)) if swapped_label_costs else None
        ),
        "quality": quality,
        "outputs": {
            "aligned": str(output_aligned),
            "cam_c_fused_2d": str(output_fused_2d),
            "video": None if args.skip_render else str(output_video),
        },
    }
    output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if not args.skip_render:
        render_report = render_video(
            recording, args.ffmpeg, fused_rows, pose_c_list, output_video
        )
        report["render"] = render_report
        output_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(render_report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
