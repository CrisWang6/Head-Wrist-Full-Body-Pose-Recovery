#!/usr/bin/env python3
"""Build two-head-camera heatmap labels from the 0717 real capture."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re

import cv2
import numpy as np


JOINTS = (
    "Head", "Neck", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm",
    "LeftHand", "RightHand", "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg",
    "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase",
)
SOURCE_JOINT = {name: name for name in JOINTS}
# The real BVH export has no toe-base joints. Their legacy output channels are
# retained for checkpoint compatibility but masked out of the real-data loss.
SOURCE_JOINT.update({"LeftToeBase": "LeftFoot", "RightToeBase": "RightFoot"})
SKELETON = (
    (0, 1), (1, 2), (1, 3), (2, 4), (4, 6), (3, 5), (5, 7),
    (2, 8), (3, 9), (8, 9), (8, 10), (10, 12), (12, 14),
    (9, 11), (11, 13), (13, 15),
)
CAMERAS = ("CAM_B", "CAM_C")


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="/home/gaoweijian/Desktop/0717_training")
    parser.add_argument("--csv", default="")
    parser.add_argument("--images-root", default="")
    parser.add_argument(
        "--intrinsics",
        default=str(root / "configs/calibration/head/head_intrinsics_kalibr_omni_1920x1200.json"),
    )
    parser.add_argument(
        "--head-bc-extrinsics",
        default=str(root / "configs/calibration/head/head_BC-camchain.yaml"),
    )
    parser.add_argument(
        "--output",
        default=str(root / "data/labels/real_0717_head2cam/heatmap_labels_114x64.npz"),
    )
    parser.add_argument(
        "--preview-dir",
        default=str(root / "logs/real_0717_head2cam_label_previews"),
    )
    parser.add_argument("--heatmap-width", type=int, default=114)
    parser.add_argument("--heatmap-height", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--clock-fit-rows", type=int, default=800)
    parser.add_argument("--preview-count", type=int, default=12)
    parser.add_argument("--limit", type=int, default=0, help="Optional frame cap for debugging.")
    return parser.parse_args()


def f64(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def image_sequences(path: Path) -> set[int]:
    output: set[int] = set()
    for item in path.glob("seq_*.jpg"):
        try:
            output.add(int(item.stem.split("_")[-1]))
        except ValueError:
            pass
    return output


def load_capture(csv_path: Path, images_root: Path, clock_fit_rows: int) -> dict[str, object]:
    image_dirs = {cam: images_root / "module01" / cam for cam in CAMERAS}
    common_sequences = set.intersection(*(image_sequences(path) for path in image_dirs.values()))
    if not common_sequences:
        raise RuntimeError(f"No common CAM_B/C images found below {images_root}")

    query_rows: list[tuple[int, float, float]] = []
    fit = {cam: [] for cam in CAMERAS}
    mocap_time: list[float] = []
    joints_world_cm: list[list[list[float]]] = []
    head_quat: list[list[float]] = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        col = {name: idx for idx, name in enumerate(header)}
        required = [
            "seq", "mocap_valid", "mocap_time_sec", "mocap_time_sec_target",
            *(f"module01_{cam}_exposure_middle_ts_ms" for cam in CAMERAS),
            *(f"module01_{cam}_gap_filled" for cam in CAMERAS),
            *(f"mocap_Head_world_q{axis}" for axis in "wxyz"),
            *(f"mocap_{joint}_world_{axis}" for joint in sorted(set(SOURCE_JOINT.values())) for axis in "xyz"),
        ]
        missing = [name for name in required if name not in col]
        if missing:
            raise KeyError(f"CSV is missing columns: {missing}")

        for row in reader:
            seq = int(row[col["seq"]])
            target = f64(row[col["mocap_time_sec_target"]])
            exposures = {
                cam: f64(row[col[f"module01_{cam}_exposure_middle_ts_ms"]])
                for cam in CAMERAS
            }
            if seq in common_sequences and all(np.isfinite(exposures[cam]) for cam in CAMERAS):
                query_rows.append((seq, exposures["CAM_B"], exposures["CAM_C"]))

            if np.isfinite(target):
                for cam in CAMERAS:
                    if (
                        len(fit[cam]) < clock_fit_rows
                        and np.isfinite(exposures[cam])
                        and not truthy(row[col[f"module01_{cam}_gap_filled"]])
                    ):
                        fit[cam].append((exposures[cam] / 1000.0, target))

            if not truthy(row[col["mocap_valid"]]):
                continue
            mt = f64(row[col["mocap_time_sec"]])
            points = [
                [f64(row[col[f"mocap_{SOURCE_JOINT[joint]}_world_{axis}"]]) for axis in "xyz"]
                for joint in JOINTS
            ]
            quat = [f64(row[col[f"mocap_Head_world_q{axis}"]]) for axis in "wxyz"]
            if np.isfinite(mt) and np.isfinite(points).all() and np.isfinite(quat).all():
                mocap_time.append(mt)
                joints_world_cm.append(points)
                head_quat.append(quat)

    clock = {}
    for cam in CAMERAS:
        values = np.asarray(fit[cam], dtype=np.float64)
        if len(values) < 10:
            raise RuntimeError(f"Not enough rows to fit the {cam} exposure clock")
        scale, offset = np.polyfit(values[:, 0], values[:, 1], 1)
        residual_ms = (values[:, 0] * scale + offset - values[:, 1]) * 1000.0
        clock[cam] = {
            "scale": float(scale),
            "offset_sec": float(offset),
            "fit_rows": int(len(values)),
            "fit_rmse_ms": float(np.sqrt(np.mean(residual_ms**2))),
            "fit_max_abs_ms": float(np.max(np.abs(residual_ms))),
        }

    order = np.argsort(np.asarray(mocap_time))
    mt = np.asarray(mocap_time, dtype=np.float64)[order]
    points = np.asarray(joints_world_cm, dtype=np.float64)[order] * 10.0  # mocap cm -> mm
    quat = np.asarray(head_quat, dtype=np.float64)[order]
    unique = np.r_[True, np.diff(mt) > 1e-9]
    return {
        "queries": query_rows,
        "mocap_time": mt[unique],
        "joints_world_mm": points[unique],
        "head_quat": normalize_quaternions(quat[unique]),
        "clock": clock,
        "image_dirs": image_dirs,
        "common_image_count": len(common_sequences),
    }


def normalize_quaternions(q: np.ndarray) -> np.ndarray:
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)


def interpolate_motion(times: np.ndarray, values: np.ndarray, quats: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    right = np.searchsorted(times, query, side="right")
    right = np.clip(right, 1, len(times) - 1)
    left = right - 1
    alpha = ((query - times[left]) / np.maximum(times[right] - times[left], 1e-12)).clip(0.0, 1.0)
    positions = values[left] + alpha[:, None, None] * (values[right] - values[left])

    q0 = quats[left].copy()
    q1 = quats[right].copy()
    dot = np.sum(q0 * q1, axis=1)
    negative = dot < 0.0
    q1[negative] *= -1.0
    dot = np.abs(dot).clip(-1.0, 1.0)
    result = np.empty_like(q0)
    close = dot > 0.9995
    result[close] = normalize_quaternions(q0[close] + alpha[close, None] * (q1[close] - q0[close]))
    far = ~close
    if np.any(far):
        theta = np.arccos(dot[far])
        sin_theta = np.sin(theta)
        a = alpha[far]
        result[far] = (
            np.sin((1.0 - a) * theta)[:, None] / sin_theta[:, None] * q0[far]
            + np.sin(a * theta)[:, None] / sin_theta[:, None] * q1[far]
        )
    return positions, normalize_quaternions(result)


def quaternion_matrices(q: np.ndarray) -> np.ndarray:
    q = normalize_quaternions(q)
    w, x, y, z = q.T
    return np.stack(
        [
            1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
            2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
            2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)


def load_head_extrinsics(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    text = path.read_text(encoding="utf-8")
    section = text.split("T_cn_cnm1:", 1)[1]
    rows = []
    for line in section.splitlines():
        match = re.search(r"\[([^\]]+)\]", line)
        if match:
            rows.append([float(v.strip()) for v in match.group(1).split(",")])
            if len(rows) == 4:
                break
    transform_c_b = np.asarray(rows, dtype=np.float64)
    if transform_c_b.shape != (4, 4):
        raise ValueError(f"Could not parse T_C_B from {path}")

    # Columns describe camera B axes in the mocap Head coordinate system.
    r_h_b = np.asarray([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    p_h_b = np.asarray([87.0, -26.0, 161.0], dtype=np.float64)
    r_c_b = transform_c_b[:3, :3]
    t_c_b_mm = transform_c_b[:3, 3] * 1000.0
    r_h_c = r_h_b @ r_c_b.T
    p_h_c = p_h_b - r_h_c @ t_c_b_mm
    return {"CAM_B": (r_h_b, p_h_b), "CAM_C": (r_h_c, p_h_c)}


def project_omni(points_cam: np.ndarray, calibration: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    xi, fx, fy, cx, cy = [float(v) for v in calibration["intrinsics"]]
    k1, k2, p1, p2 = [float(v) for v in calibration["distortion_coeffs"]]
    norm = np.linalg.norm(points_cam, axis=-1)
    denom = points_cam[..., 2] + xi * norm
    x = points_cam[..., 0] / np.maximum(denom, 1e-12)
    y = points_cam[..., 1] / np.maximum(denom, 1e-12)
    r2 = x*x + y*y
    radial = 1.0 + k1*r2 + k2*r2*r2
    xd = x*radial + 2.0*p1*x*y + p2*(r2 + 2.0*x*x)
    yd = y*radial + p1*(r2 + 2.0*y*y) + 2.0*p2*x*y
    pixels = np.stack((fx*xd + cx, fy*yd + cy), axis=-1)
    width, height = [int(v) for v in calibration["resolution"]]
    visible = (
        (norm > 1e-6) & (points_cam[..., 2] > 0.0) & (denom > 1e-9)
        & np.isfinite(pixels).all(axis=-1)
        & (pixels[..., 0] >= 0.0) & (pixels[..., 0] < width)
        & (pixels[..., 1] >= 0.0) & (pixels[..., 1] < height)
    )
    return pixels.astype(np.float32), visible


def build_labels(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve() if args.csv else dataset_root / "aligned_30hz.csv"
    images_root = Path(args.images_root).expanduser().resolve() if args.images_root else dataset_root / "images_h265_strict"
    capture = load_capture(csv_path, images_root, args.clock_fit_rows)
    intrinsics_data = json.loads(Path(args.intrinsics).read_text(encoding="utf-8"))
    extrinsics = load_head_extrinsics(Path(args.head_bc_extrinsics))

    rows = capture["queries"]
    if args.limit > 0:
        rows = rows[: args.limit]
    seq = np.asarray([r[0] for r in rows], dtype=np.int32)
    exposure = np.asarray([[r[1], r[2]] for r in rows], dtype=np.float64) / 1000.0
    query_times = np.empty_like(exposure)
    for camera_idx, cam in enumerate(CAMERAS):
        fit = capture["clock"][cam]
        query_times[:, camera_idx] = exposure[:, camera_idx] * fit["scale"] + fit["offset_sec"]

    mocap_time = capture["mocap_time"]
    in_range = (query_times >= mocap_time[0]).all(axis=1) & (query_times <= mocap_time[-1]).all(axis=1)
    seq, query_times = seq[in_range], query_times[in_range]
    if len(seq) == 0:
        raise RuntimeError("No camera exposure times overlap the valid mocap time range")

    keypoints = np.full((len(seq), 2, len(JOINTS), 2), np.nan, dtype=np.float32)
    visible = np.zeros((len(seq), 2, len(JOINTS)), dtype=bool)
    camera_metadata = {}
    for camera_idx, cam in enumerate(CAMERAS):
        world_joints, head_quat = interpolate_motion(
            mocap_time, capture["joints_world_mm"], capture["head_quat"], query_times[:, camera_idx]
        )
        r_w_h = quaternion_matrices(head_quat)
        head_origin = world_joints[:, JOINTS.index("Head"), :]
        r_h_c, p_h_c = extrinsics[cam]
        r_w_c = np.einsum("fij,jk->fik", r_w_h, r_h_c)
        p_w_c = head_origin + np.einsum("fij,j->fi", r_w_h, p_h_c)
        points_cam = np.einsum("fji,fkj->fki", r_w_c, world_joints - p_w_c[:, None, :])
        pixels, is_visible = project_omni(points_cam, intrinsics_data["cameras"][cam])
        keypoints[:, camera_idx] = pixels
        visible[:, camera_idx] = is_visible
        camera_metadata[cam] = {
            "R_head_camera": r_h_c.tolist(),
            "p_head_camera_mm": p_h_c.tolist(),
            "visible_points": int(is_visible.sum()),
            "visible_fraction": float(is_visible.mean()),
            "query_time_min": float(query_times[:, camera_idx].min()),
            "query_time_max": float(query_times[:, camera_idx].max()),
        }

    image_paths = np.asarray(
        [[str(capture["image_dirs"][cam] / f"seq_{int(s):06d}.jpg") for cam in CAMERAS] for s in seq]
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    head_joint_mask = np.ones((2, len(JOINTS)), dtype=bool)
    head_joint_mask[:, JOINTS.index("LeftToeBase")] = False
    head_joint_mask[:, JOINTS.index("RightToeBase")] = False
    visible &= head_joint_mask[None, :, :]
    wrist_keypoints = np.full((len(seq), 2, 7, 2), np.nan, dtype=np.float32)
    wrist_visible = np.zeros((len(seq), 2, 7), dtype=bool)
    np.savez_compressed(
        output,
        schema_version=np.asarray(["egorear_real_head2cam_kalibr_v1"]),
        keypoints=keypoints,
        visible=visible,
        joint_mask=head_joint_mask,
        head_keypoints=keypoints,
        head_visible=visible,
        head_joint_mask=head_joint_mask,
        wrist_keypoints=wrist_keypoints,
        wrist_visible=wrist_visible,
        wrist_joint_mask=np.zeros((2, 7), dtype=bool),
        camera_is_head=np.ones(2, dtype=bool),
        camera_is_wrist=np.zeros(2, dtype=bool),
        camera_names=np.asarray(["module01_CAM_B", "module01_CAM_C"]),
        joints=np.asarray(JOINTS),
        head_camera_joints=np.asarray(JOINTS),
        head_source_mocap_joints=np.asarray(JOINTS),
        wrist_camera_joints=np.asarray(("L_Ankle", "R_Ankle", "L_Knee", "R_Knee", "L_Hip", "R_Hip", "Spine1")),
        video_paths=np.asarray(["", ""]),
        image_paths=image_paths,
        frame_indices=seq,
        camera_query_time_sec=query_times,
        video_size=np.asarray([1920, 1200], dtype=np.int32),
        heatmap_size=np.asarray([args.heatmap_width, args.heatmap_height], dtype=np.int32),
        sigma=np.asarray([args.sigma], dtype=np.float32),
        projection_model=np.asarray(["kalibr_omni_radtan"]),
        fisheye_fov_deg=np.asarray([180.0], dtype=np.float32),
        source_csv=np.asarray([str(csv_path)]),
        source_render_dir=np.asarray([""]),
    )

    manifest = {
        "schema": "egorear.real_head2cam_manifest.v1",
        "output": str(output),
        "source_csv": str(csv_path),
        "images_root": str(images_root),
        "frames": int(len(seq)),
        "sequence_min": int(seq.min()),
        "sequence_max": int(seq.max()),
        "common_image_count_before_time_filter": int(capture["common_image_count"]),
        "joints": list(JOINTS),
        "projection": "Kalibr omni+radtan, front 180 degrees",
        "clock_alignment": capture["clock"],
        "cameras": camera_metadata,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_previews(Path(args.preview_dir), image_paths, keypoints, visible, seq, args.preview_count)
    return output, manifest


def write_previews(preview_dir: Path, paths: np.ndarray, points: np.ndarray, visible: np.ndarray, seq: np.ndarray, count: int) -> None:
    preview_dir.mkdir(parents=True, exist_ok=True)
    if count <= 0:
        return
    selected = np.linspace(0, len(seq) - 1, min(count, len(seq)), dtype=int)
    for frame_index in selected:
        for camera_index, cam in enumerate(CAMERAS):
            image = cv2.imread(str(paths[frame_index, camera_index]), cv2.IMREAD_COLOR)
            if image is None:
                continue
            p = points[frame_index, camera_index]
            v = visible[frame_index, camera_index]
            for a, b in SKELETON:
                if v[a] and v[b]:
                    cv2.line(image, tuple(np.round(p[a]).astype(int)), tuple(np.round(p[b]).astype(int)), (0, 0, 255), 3, cv2.LINE_AA)
            for joint_index in np.flatnonzero(v):
                cv2.circle(image, tuple(np.round(p[joint_index]).astype(int)), 7, (0, 0, 255), -1, cv2.LINE_AA)
            image = cv2.resize(image, (960, 600), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(preview_dir / f"seq_{int(seq[frame_index]):06d}_{cam}.jpg"), image, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> int:
    output, manifest = build_labels(parse_args())
    print(json.dumps({"label_file": str(output), **manifest}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
