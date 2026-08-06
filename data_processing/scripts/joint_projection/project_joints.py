#!/usr/bin/env python3
"""Project aligned mocap joints into the eight head/wrist fisheye cameras."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "projection_config.json"
SEQ_RE = re.compile(r"^seq_(\d{6})\.jpg$", re.IGNORECASE)
WORLD_X_RE = re.compile(r"^mocap_(.+)_world_x$")


POSITION_UNIT_TO_MM = {
    "m": 1000.0,
    "meter": 1000.0,
    "meters": 1000.0,
    "cm": 10.0,
    "centimeter": 10.0,
    "centimeters": 10.0,
    "mm": 1.0,
    "millimeter": 1.0,
    "millimeters": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--samples", type=int, default=None, help="Override samples per camera.")
    parser.add_argument("--seed", type=int, default=None, help="Override deterministic random seed.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def position_scale_to_mm(unit: str) -> float:
    try:
        return POSITION_UNIT_TO_MM[unit.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported mocap position unit: {unit!r}") from exc


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a zero vector")
    return vector / norm


def camera_rotation_from_forward_up(forward: list[float], image_up: list[float]) -> np.ndarray:
    """Return R_anchor_camera; columns are camera axes expressed in anchor coordinates."""
    z_axis = normalize(np.asarray(forward, dtype=np.float64))
    up = normalize(np.asarray(image_up, dtype=np.float64))
    up = normalize(up - z_axis * float(np.dot(up, z_axis)))
    y_axis = -up
    x_axis = normalize(np.cross(y_axis, z_axis))
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    if np.linalg.det(rotation) < 0.999999:
        raise ValueError(f"Invalid camera basis, determinant={np.linalg.det(rotation)}")
    return rotation


def camera_rotation_from_axes(axes: dict[str, list[float]]) -> np.ndarray:
    """Return the exact user-specified camera axes expressed in anchor coordinates."""
    rotation = np.column_stack(
        [normalize(np.asarray(axes[axis], dtype=np.float64)) for axis in ("x", "y", "z")]
    )
    gram = rotation.T @ rotation
    if not np.allclose(gram, np.eye(3), atol=1e-7):
        raise ValueError(f"Camera axes are not orthonormal:\n{rotation}")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(abs(determinant), 1.0, abs_tol=1e-7):
        raise ValueError(f"Camera axis mapping determinant must be +/-1, got {determinant}")
    return rotation


def parse_kalibr_relative_transform(path: Path) -> np.ndarray:
    """Read the first T_cn_cnm1 4x4 matrix without requiring PyYAML."""
    text = path.read_text(encoding="utf-8")
    block = re.search(r"T_cn_cnm1:\s*((?:\s*-\s*\[[^\n]+\]\s*){4})", text)
    if not block:
        raise RuntimeError(f"Could not find T_cn_cnm1 in {path}")
    rows = []
    for row_text in re.findall(r"\[([^\]]+)\]", block.group(1)):
        rows.append([float(item.strip()) for item in row_text.split(",")])
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise RuntimeError(f"Expected a 4x4 transform in {path}, got {matrix.shape}")
    return matrix


def quaternion_to_rotation(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    q = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )




def load_camera_models(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    for key, spec in config["cameras"].items():
        intrinsics_doc = load_json(Path(spec["intrinsics"]))
        calibrated = intrinsics_doc["cameras"][spec["camera"]]
        width, height = map(int, calibrated.get("resolution", intrinsics_doc["resolution"]))
        if [width, height] != list(config["image_size"]):
            raise RuntimeError(f"{key}: intrinsics resolution {width}x{height} does not match config")
        values = calibrated["intrinsics"]
        if "camera_axes_anchor" in spec:
            rotation = camera_rotation_from_axes(spec["camera_axes_anchor"])
        else:
            rotation = camera_rotation_from_forward_up(
                spec["optical_axis_anchor"], spec["image_up_anchor"]
            )
        models[key] = {
            **spec,
            "width": width,
            "height": height,
            "xi": float(calibrated.get("xi", values[0])),
            "fx": float(values[1]),
            "fy": float(values[2]),
            "cx": float(values[3]),
            "cy": float(values[4]),
            "distortion": np.asarray(calibrated["distortion_coeffs"], dtype=np.float64),
            "calibration_std_px": [float(v) for v in calibrated.get("reprojection_error_std_px", [])],
            "R_anchor_camera": rotation,
            "axis_mapping_determinant": float(np.linalg.det(rotation)),
        }

    # Kalibr T_cn_cnm1 maps CAM_B coordinates into CAM_C coordinates.
    relative_bc = parse_kalibr_relative_transform(Path(config["head_relative_bc_yaml"]))
    if models["module01_CAM_C"].get("orientation_source") == "calibrated_relative_to_module01_CAM_B":
        r_c_b = relative_bc[:3, :3]
        r_head_b = models["module01_CAM_B"]["R_anchor_camera"]
        models["module01_CAM_C"]["R_anchor_camera"] = r_head_b @ r_c_b.T
        models["module01_CAM_C"]["axis_mapping_determinant"] = float(
            np.linalg.det(models["module01_CAM_C"]["R_anchor_camera"])
        )
    models["module01_CAM_C"]["relative_transform_calibrated"] = relative_bc
    models["module01_CAM_B"]["relative_transform_calibrated"] = relative_bc
    return models


def discover_image_sequences(images_root: Path, model: dict[str, Any]) -> dict[int, Path]:
    folder = images_root / model["module"] / model["camera"]
    found: dict[int, Path] = {}
    for path in folder.glob("seq_*.jpg"):
        match = SEQ_RE.match(path.name)
        if match:
            found[int(match.group(1))] = path
    if not found:
        raise RuntimeError(f"No seq_XXXXXX.jpg images found in {folder}")
    return found


def csv_layout(
    header: list[str], anchor_joints: set[str], requested_joints: list[str]
) -> dict[str, Any]:
    index = {name: idx for idx, name in enumerate(header)}
    joints = list(requested_joints)
    required = ["seq", "mocap_valid", "mocap_nearest_dt_ms"]
    for anchor in anchor_joints:
        required.extend(
            [f"mocap_{anchor}_world_{axis}" for axis in "xyz"]
            + [f"mocap_{anchor}_world_q{axis}" for axis in "wxyz"]
        )
    for joint in joints:
        required.extend(f"mocap_{joint}_world_{axis}" for axis in "xyz")
    missing = [name for name in required if name not in index]
    if missing:
        raise RuntimeError(f"Missing required aligned CSV columns: {missing}")
    return {"index": index, "joints": joints}


def load_candidate_rows(
    csv_path: Path,
    candidate_sequences: set[int],
    anchor_scales_to_mm: dict[str, float],
    requested_joints: list[str],
    joint_scale_to_mm: float,
) -> tuple[dict[int, dict[str, Any]], list[str]]:
    """Read a 543 MB CSV once, fully splitting only candidate rows."""
    rows: dict[int, dict[str, Any]] = {}
    # The exporter writes a UTF-8 BOM, so use utf-8-sig to expose the first field as exactly "seq".
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        header_line = handle.readline()
        header = next(csv.reader([header_line]))
        layout = csv_layout(header, set(anchor_scales_to_mm), requested_joints)
        idx = layout["index"]
        max_idx = len(header)
        for line in handle:
            first_comma = line.find(",")
            if first_comma <= 0:
                continue
            try:
                seq = int(line[:first_comma])
            except ValueError:
                continue
            if seq not in candidate_sequences:
                continue
            values = next(csv.reader([line]))
            if len(values) != max_idx or values[idx["mocap_valid"]] not in {"1", "1.0", "True", "true"}:
                continue
            parsed: dict[str, Any] = {
                "seq": seq,
                "mocap_nearest_dt_ms": float(values[idx["mocap_nearest_dt_ms"]]),
                "joints": {},
                "anchors": {},
            }
            for joint in layout["joints"]:
                try:
                    position_cm = np.asarray(
                        [float(values[idx[f"mocap_{joint}_world_{axis}"]]) for axis in "xyz"],
                        dtype=np.float64,
                    )
                except ValueError:
                    continue
                if np.all(np.isfinite(position_cm)):
                    parsed["joints"][joint] = position_cm * joint_scale_to_mm
            for anchor, anchor_scale_to_mm in anchor_scales_to_mm.items():
                position_mm = np.asarray(
                    [float(values[idx[f"mocap_{anchor}_world_{axis}"]]) for axis in "xyz"],
                    dtype=np.float64,
                ) * anchor_scale_to_mm
                quaternion = [float(values[idx[f"mocap_{anchor}_world_q{axis}"]]) for axis in "wxyz"]
                parsed["anchors"][anchor] = {
                    "position_world_mm": position_mm,
                    "R_world_anchor": quaternion_to_rotation(*quaternion),
                }
            rows[seq] = parsed
    return rows, layout["joints"]


def omni_project(points_camera: np.ndarray, model: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Kalibr omni + radtan projection, with the requested 180-degree forward hemisphere."""
    points = np.asarray(points_camera, dtype=np.float64)
    norm = np.linalg.norm(points, axis=1)
    denom = points[:, 2] + model["xi"] * norm
    model_valid = (norm > 1e-9) & (denom > 1e-9) & (points[:, 2] > 0.0)
    uv = np.full((len(points), 2), np.nan, dtype=np.float64)
    if not np.any(model_valid):
        return uv, model_valid
    x = points[model_valid, 0] / denom[model_valid]
    y = points[model_valid, 1] / denom[model_valid]
    k1, k2, p1, p2 = model["distortion"][:4]
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    uv[model_valid, 0] = model["fx"] * xd + model["cx"]
    uv[model_valid, 1] = model["fy"] * yd + model["cy"]
    return uv, model_valid


def transform_joints(
    row: dict[str, Any], model: dict[str, Any], position_delta_anchor_mm: np.ndarray | None = None
) -> tuple[list[str], np.ndarray]:
    anchor = row["anchors"][model["anchor_joint"]]
    r_world_anchor = anchor["R_world_anchor"]
    r_anchor_camera = model["R_anchor_camera"]
    r_world_camera = r_world_anchor @ r_anchor_camera
    offset = np.asarray(model["position_anchor_mm"], dtype=np.float64)
    if position_delta_anchor_mm is not None:
        offset = offset + position_delta_anchor_mm
    camera_world = anchor["position_world_mm"] + r_world_anchor @ offset
    names = list(row["joints"])
    points_world = np.vstack([row["joints"][name] for name in names])
    points_camera = (r_world_camera.T @ (points_world - camera_world).T).T
    return names, points_camera


def in_image(uv: np.ndarray, valid: np.ndarray, width: int, height: int) -> np.ndarray:
    return (
        valid
        & np.isfinite(uv[:, 0])
        & np.isfinite(uv[:, 1])
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] < width)
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] < height)
    )


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def round_or_none(value: float | None, digits: int = 4) -> float | None:
    return None if value is None else round(float(value), digits)


def head_consistency_metrics(models: dict[str, dict[str, Any]]) -> dict[str, float]:
    transform = models["module01_CAM_B"]["relative_transform_calibrated"]
    measured_baseline = float(np.linalg.norm(transform[:3, 3]) * 1000.0)
    rotation = transform[:3, :3]
    angle = math.degrees(math.acos(float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))
    return {
        "theoretical_baseline_mm": 174.0,
        "calibrated_baseline_mm": measured_baseline,
        "baseline_difference_mm": measured_baseline - 174.0,
        "relative_rotation_angle_deg": angle,
    }




def base_assumptions() -> list[str]:
    return [
        "aligned_30hz.csv 的 seq 与 images/.../seq_XXXXXX.jpg 直接对应。",
        "动捕 world 位置单位是 cm，转换为 mm 后与相机外参统一。",
        "相机使用 Kalibr unified omni + radtan 内参；180°定义按光轴前半球 z_camera > 0 过滤。",
        "仅做视场和图像边界过滤；没有人体网格/深度，因此不判断身体自遮挡。",
        "相机姿态使用 projection_config.json 的 camera_axes_anchor 完整三轴映射。",
        "仅保留踝、膝、髋、脊柱、颈、肩、肘和腕关节。",
        "骨骼边在 3D 中采样后按鱼眼模型投影，因此在图像中可以呈曲线。",
    ]


def write_camera_report(path: Path, report: dict[str, Any]) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    uncertainty = report["uncertainty_metrics"]
    lines = [
        f"# {report['camera_key']} 误差与不确定性报告",
        "",
        "> 没有人工标注的 2D joint 真值，因此不能给出绝对像素误差。以下报告内参重投影误差、同步误差和外参位置不确定性。",
        "",
        f"- 输出图像：{report['sample_count']} 张",
        f"- 可见 joint：总计 {report['visible_joint_observations']}，每帧均值 {report['visible_joints_per_frame_mean']:.2f}",
        f"- 空投影帧：{report['frames_with_zero_visible_joints']}",
        f"- Kalibr 内参重投影标准差 (x, y)：{report['intrinsic_reprojection_std_xy_px']} px",
        f"- Kalibr 内参径向 RMS：{report['intrinsic_reprojection_rms_px']:.4f} px",
        f"- 动捕最近时间差 |dt|：均值 {report['timing_abs_dt_ms']['mean']:.4f} ms，P95 {report['timing_abs_dt_ms']['p95']:.4f} ms，最大 {report['timing_abs_dt_ms']['max']:.4f} ms",
        f"- 相机轴映射行列式：{report['axis_mapping_determinant']:.1f}",
    ]
    if uncertainty["joint_observations_evaluated"]:
        lines.append(
            f"- 外参位置不确定性传播：均值 {uncertainty['pixel_displacement_mean_px']} px，P95 {uncertainty['pixel_displacement_p95_px']} px，最大 {uncertainty['pixel_displacement_max_px']} px"
        )
    else:
        lines.append("- 本相机没有 ± 位置不确定性输入，因此不估计外参位置传播误差。")
    lines.extend(["", "## 假设", ""])
    lines.extend(f"- {item}" for item in report["assumptions"])
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_camera(
    key: str,
    model: dict[str, Any],
    image_paths: dict[int, Path],
    selected_sequences: list[int],
    rows: dict[int, dict[str, Any]],
    output_root: Path,
    dot_radius: int,
    bone_thickness: int,
    skeleton_joint_names: list[str],
    skeleton_edges: list[list[str]],
    head_metrics: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    camera_dir = output_root / key
    camera_dir.mkdir(parents=True, exist_ok=True)
    projections: list[dict[str, Any]] = []
    visible_counts: list[int] = []
    timing: list[float] = []
    uncertainty_displacements: list[float] = []

    for seq in selected_sequences:
        row = rows[seq]
        image = cv2.imread(str(image_paths[seq]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read {image_paths[seq]}")
        if image.shape[1] != model["width"] or image.shape[0] != model["height"]:
            raise RuntimeError(f"Unexpected image size for {image_paths[seq]}: {image.shape[1]}x{image.shape[0]}")

        names, points_camera = transform_joints(row, model)
        uv, projection_valid = omni_project(points_camera, model)
        visible = in_image(uv, projection_valid, model["width"], model["height"])
        visible_counts.append(int(np.count_nonzero(visible)))
        timing.append(abs(float(row["mocap_nearest_dt_ms"])))

        uncertainty = np.asarray(model.get("position_uncertainty_anchor_mm", [0.0, 0.0, 0.0]), dtype=np.float64)
        if np.any(uncertainty):
            _, points_plus = transform_joints(row, model, uncertainty)
            _, points_minus = transform_joints(row, model, -uncertainty)
            uv_plus, valid_plus = omni_project(points_plus, model)
            uv_minus, valid_minus = omni_project(points_minus, model)
            comparable = visible & valid_plus & valid_minus
            if np.any(comparable):
                plus_delta = np.linalg.norm(uv_plus[comparable] - uv[comparable], axis=1)
                minus_delta = np.linalg.norm(uv_minus[comparable] - uv[comparable], axis=1)
                uncertainty_displacements.extend(np.maximum(plus_delta, minus_delta).tolist())

        name_to_index = {name: idx for idx, name in enumerate(names)}
        for first, second in skeleton_edges:
            first_idx = name_to_index[first]
            second_idx = name_to_index[second]
            if visible[first_idx] and visible[second_idx]:
                # A straight 3D bone generally becomes a curve in an omni/fisheye image.
                interpolation = np.linspace(0.0, 1.0, 33, dtype=np.float64)[:, None]
                bone_points_camera = (
                    points_camera[first_idx][None, :] * (1.0 - interpolation)
                    + points_camera[second_idx][None, :] * interpolation
                )
                bone_uv, bone_valid = omni_project(bone_points_camera, model)
                bone_visible = in_image(
                    bone_uv, bone_valid, model["width"], model["height"]
                )
                start = None
                for sample_idx, is_visible in enumerate(bone_visible):
                    if is_visible and start is None:
                        start = sample_idx
                    at_end = sample_idx == len(bone_visible) - 1
                    if start is not None and ((not is_visible) or at_end):
                        stop = sample_idx + 1 if is_visible and at_end else sample_idx
                        if stop - start >= 2:
                            curve = np.rint(bone_uv[start:stop]).astype(np.int32).reshape(-1, 1, 2)
                            cv2.polylines(
                                image,
                                [curve],
                                isClosed=False,
                                color=(0, 0, 255),
                                thickness=bone_thickness,
                                lineType=cv2.LINE_AA,
                            )
                        start = None

        for idx in np.flatnonzero(visible):
            u, v = uv[idx]
            cv2.circle(image, (int(round(u)), int(round(v))), dot_radius, (0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)
            projections.append(
                {
                    "camera_key": key,
                    "seq": seq,
                    "joint": names[idx],
                    "u_px": round(float(u), 4),
                    "v_px": round(float(v), 4),
                    "forward_depth_mm": round(float(points_camera[idx, 2]), 4),
                }
            )
        destination = camera_dir / f"seq_{seq:06d}_joints.jpg"
        if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"Failed to write {destination}")

    std_xy = model["calibration_std_px"]
    intrinsic_rms = math.sqrt(sum(value * value for value in std_xy)) if std_xy else float("nan")
    uncertainty_vector = model.get("position_uncertainty_anchor_mm", [0.0, 0.0, 0.0])
    report = {
        "schema": "joint_projection.error_report.v1",
        "camera_key": key,
        "sample_count": len(selected_sequences),
        "sampled_sequences": selected_sequences,
        "visible_joint_observations": int(sum(visible_counts)),
        "visible_joints_per_frame_mean": float(np.mean(visible_counts)),
        "visible_joints_per_frame_min": int(min(visible_counts)),
        "visible_joints_per_frame_max": int(max(visible_counts)),
        "frames_with_zero_visible_joints": int(sum(count == 0 for count in visible_counts)),
        "skeleton_joint_names": skeleton_joint_names,
        "skeleton_edges": skeleton_edges,
        "axis_mapping_determinant": model["axis_mapping_determinant"],
        "intrinsic_reprojection_std_xy_px": std_xy,
        "intrinsic_reprojection_rms_px": intrinsic_rms,
        "timing_abs_dt_ms": {
            "mean": float(np.mean(timing)),
            "p95": float(np.percentile(timing, 95)),
            "max": float(np.max(timing)),
        },
        "uncertainty_metrics": {
            "position_uncertainty_anchor_mm": uncertainty_vector,
            "joint_observations_evaluated": len(uncertainty_displacements),
            "pixel_displacement_mean_px": round_or_none(float(np.mean(uncertainty_displacements)) if uncertainty_displacements else None),
            "pixel_displacement_p95_px": round_or_none(percentile(uncertainty_displacements, 95)),
            "pixel_displacement_max_px": round_or_none(max(uncertainty_displacements) if uncertainty_displacements else None),
        },
        "absolute_2d_error_available": False,
        "assumptions": [
            "aligned_30hz.csv 的 seq 与 images/.../seq_XXXXXX.jpg 直接对应。",
            "动捕 world 位置单位是 cm，转换为 mm 后与相机外参统一。",
            "相机使用 Kalibr unified omni + radtan 内参；180°定义按光轴前半球 z_camera > 0 过滤。",
            "仅做视场和图像边界过滤；没有人体网格/深度，因此不判断身体自遮挡。",
            "相机姿态使用 projection_config.json 的 camera_axes_anchor 完整三轴映射，不再从光轴推断滚转。",
            "仅保留踝、膝、髋、脊柱、颈、肩、肘和腕关节；两端都在画面内时才绘制骨架边。",
            "红点和骨架端点采用浮点投影四舍五入，绘制半径/线宽不计入误差。",
        ],
    }
    report["assumptions"] = base_assumptions()
    if key.startswith("module01_"):
        report["head_bc_consistency"] = head_metrics
        report["head_orientation"] = model["head_orientation_meta"]
        report["assumptions"] = base_assumptions() + [
            "CAM_B 与 CAM_C 均使用用户提供的 Head 完整三轴映射；Kalibr B→C 外参仅用于一致性报告。",
            "头部相机位置逐帧来自 mocap Head world 位置。",
            "头部相机姿态逐帧来自 mocap Head world 四元数，不使用 module01 IMU。",
        ]
    if model["axis_mapping_determinant"] < 0.0:
        report["assumptions"].append(
            "用户给出的三轴映射行列式为 -1，包含一次镜像；本次严格按原始轴符号执行。"
        )
    write_camera_report(output_root / f"{key}_error_report.json", report)
    return report, projections


def main() -> int:
    args = parse_args()
    config = load_json(args.config.resolve())
    samples = int(args.samples if args.samples is not None else config["samples_per_camera"])
    seed = int(args.seed if args.seed is not None else config["random_seed"])
    if samples <= 0:
        raise ValueError("samples must be positive")

    aligned_csv = Path(config["aligned_csv"])
    images_root = Path(config["images_root"])
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    models = load_camera_models(config)
    image_maps = {key: discover_image_sequences(images_root, model) for key, model in models.items()}

    rng = random.Random(seed)
    candidate_by_camera: dict[str, list[int]] = {}
    candidate_union: set[int] = set()
    candidate_count = max(samples * 4, samples + 20)
    for key, paths in image_maps.items():
        candidates = list(paths)
        rng.shuffle(candidates)
        candidates = candidates[: min(candidate_count, len(candidates))]
        candidate_by_camera[key] = candidates
        candidate_union.update(candidates)
    default_position_unit = config.get("mocap_position_unit", "cm")
    joint_scale_to_mm = position_scale_to_mm(default_position_unit)
    anchor_scales_to_mm: dict[str, float] = {}
    for model in models.values():
        anchor = model["anchor_joint"]
        scale = position_scale_to_mm(model.get("anchor_position_unit", default_position_unit))
        previous = anchor_scales_to_mm.setdefault(anchor, scale)
        if not math.isclose(previous, scale):
            raise ValueError(f"Conflicting position units configured for anchor {anchor}")
    print(f"Reading selected rows from {aligned_csv} ...")
    skeleton_joint_names = list(config["skeleton_joint_names"])
    skeleton_edges = [list(edge) for edge in config["skeleton_edges"]]
    unknown_edge_joints = sorted(
        {joint for edge in skeleton_edges for joint in edge} - set(skeleton_joint_names)
    )
    if unknown_edge_joints:
        raise RuntimeError(f"Skeleton edges contain unconfigured joints: {unknown_edge_joints}")
    rows, joint_names = load_candidate_rows(
        aligned_csv,
        candidate_union,
        anchor_scales_to_mm,
        skeleton_joint_names,
        joint_scale_to_mm,
    )
    for key in ("module01_CAM_B", "module01_CAM_C"):
        models[key]["head_orientation_meta"] = dict(config["head_orientation"])
    selected_by_camera: dict[str, list[int]] = {}
    for key, candidates in candidate_by_camera.items():
        selected = [seq for seq in candidates if seq in rows][:samples]
        if len(selected) < samples:
            raise RuntimeError(
                f"{key}: only {len(selected)} valid mocap rows among {len(candidates)} candidates; increase candidate_count"
            )
        selected_by_camera[key] = sorted(selected)

    head_metrics = head_consistency_metrics(models)
    reports: dict[str, Any] = {}
    all_projections: list[dict[str, Any]] = []
    for key, model in models.items():
        print(f"Projecting {key}: {samples} images ...")
        report, projections = process_camera(
            key,
            model,
            image_maps[key],
            selected_by_camera[key],
            rows,
            output_root,
            int(config["dot_radius_px"]),
            int(config["bone_thickness_px"]),
            skeleton_joint_names,
            skeleton_edges,
            head_metrics,
        )
        reports[key] = report
        all_projections.extend(projections)

    projection_csv = output_root / "visible_joint_projections.csv"
    with projection_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["camera_key", "seq", "joint", "u_px", "v_px", "forward_depth_mm"]
        )
        writer.writeheader()
        writer.writerows(all_projections)

    manifest = {
        "schema": "joint_projection.run.v1",
        "config": str(args.config.resolve()),
        "aligned_csv": str(aligned_csv.resolve()),
        "images_root": str(images_root.resolve()),
        "output_root": str(output_root.resolve()),
        "random_seed": seed,
        "samples_per_camera": samples,
        "camera_count": len(models),
        "joint_count_in_mocap": len(joint_names),
        "visible_joint_projection_count": len(all_projections),
        "reports": {key: f"{key}_error_report.json" for key in models},
        "selected_sequences": selected_by_camera,
    }
    (output_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_root / "summary.json").write_text(
        json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Done. Results: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
