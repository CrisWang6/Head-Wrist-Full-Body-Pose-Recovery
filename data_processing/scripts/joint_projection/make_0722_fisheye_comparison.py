#!/usr/bin/env python3
"""Rectify the final omni overlays and create raw/rectilinear comparisons."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

import project_joints as base


HERE = Path(__file__).resolve().parent
CONFIG = HERE / "projection_config_0722_head_ch3_08.json"
CALIBRATION = HERE / "validation_0722_h265_fixed_time_calibration" / "calibration_fixed_time.json"
RAW = HERE / "validation_0722_h265_fixed_time_final"
OUTPUT = HERE / "validation_0722_h265_fixed_time_rectified_140deg"
HFOV_DEG = 140.0
WIDTH, HEIGHT = 1920, 1200


def rectification_map(model: dict[str, object]) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    focal = (WIDTH / 2.0) / np.tan(np.radians(HFOV_DEG / 2.0))
    u, v = np.meshgrid(np.arange(WIDTH, dtype=np.float64), np.arange(HEIGHT, dtype=np.float64))
    rays = np.column_stack(((u.ravel() - (WIDTH - 1) / 2.0) / focal, (v.ravel() - (HEIGHT - 1) / 2.0) / focal, np.ones(WIDTH * HEIGHT)))
    source, valid = base.omni_project(rays, model)
    valid &= (source[:, 0] >= 0) & (source[:, 0] < WIDTH - 1) & (source[:, 1] >= 0) & (source[:, 1] < HEIGHT - 1)
    map_x = np.where(valid, source[:, 0], -1).reshape(HEIGHT, WIDTH).astype(np.float32)
    map_y = np.where(valid, source[:, 1], -1).reshape(HEIGHT, WIDTH).astype(np.float32)
    vfov = np.degrees(2.0 * np.arctan((HEIGHT / 2.0) / focal))
    return map_x, map_y, {"horizontal_fov_deg": HFOV_DEG, "vertical_fov_deg": float(vfov), "focal_px": float(focal), "valid_output_fraction": float(np.mean(valid))}


def main() -> int:
    models = base.load_camera_models(base.load_json(CONFIG))
    calibration = base.load_json(CALIBRATION)
    report: dict[str, object] = {
        "schema": "fisheye_rectification_comparison.v1",
        "interpretation": "The colored overlay is remapped with the raw image, which is geometrically equivalent to projecting the same 3-D rays with the rectified pinhole model.",
        "warning": "A single rectilinear view cannot retain a full 180-degree hemisphere; 140 degrees is used to avoid infinite edge magnification.",
        "cameras": {},
    }
    comparison_tiles: list[np.ndarray] = []
    for camera in ("CAM_B", "CAM_C"):
        key = f"module01_{camera}"
        map_x, map_y, info = rectification_map(models[key])
        destination = OUTPUT / key
        destination.mkdir(parents=True, exist_ok=True)
        paths = sorted((RAW / key).glob("seq_*_joints.jpg"))
        for index, path in enumerate(paths):
            raw = cv2.imread(str(path))
            rectified = cv2.remap(raw, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            cv2.putText(rectified, f"rectilinear {HFOV_DEG:.0f} deg HFOV", (28, 82), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2, cv2.LINE_AA)
            cv2.imwrite(str(destination / path.name), rectified, [cv2.IMWRITE_JPEG_QUALITY, 94])
            if index in np.linspace(0, len(paths) - 1, 6, dtype=int):
                left = cv2.resize(raw, (768, 480), interpolation=cv2.INTER_AREA)
                right = cv2.resize(rectified, (768, 480), interpolation=cv2.INTER_AREA)
                cv2.putText(left, f"{camera} raw omni", (18, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,255,255), 2, cv2.LINE_AA)
                cv2.putText(right, f"{camera} rectified 140 deg", (18, 35), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,255,255), 2, cv2.LINE_AA)
                comparison_tiles.append(np.hstack((left, right)))
        report["cameras"][camera] = {**info, "images_written": len(paths)}

    pages = []
    for start in range(0, len(comparison_tiles), 4):
        page = np.vstack(comparison_tiles[start:start + 4])
        path = OUTPUT / f"raw_vs_rectified_contact_{start // 4 + 1:02d}.jpg"
        cv2.imwrite(str(path), page, [cv2.IMWRITE_JPEG_QUALITY, 94])
        pages.append(str(path))
    report["contact_sheets"] = pages

    observations = calibration["observations"]
    bins = [(0, 35), (35, 50), (50, 65), (65, 90)]
    radial = []
    for low, high in bins:
        values = [float(row["reprojection_error_px"]) for row in observations if low <= float(row["polar_angle_deg"]) < high]
        radial.append({"polar_angle_bin_deg": [low, high], "count": len(values), "median_error_px": float(np.median(values)) if values else None, "p90_error_px": float(np.percentile(values, 90)) if values else None})
    report["apriltag_error_by_polar_angle"] = radial
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "fisheye_assessment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
