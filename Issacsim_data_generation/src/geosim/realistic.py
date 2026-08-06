from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CameraPhysicalConfig:
    exposure_time_s: float = 1.0 / 120.0
    readout_time_s: float = 0.0
    iso: float = 400.0
    aperture_f_number: float = 2.0
    sensor_width_mm: float = 5.6
    sensor_height_mm: float = 3.2
    lens_transmission: float = 0.85


@dataclass(frozen=True)
class ImageDegradationConfig:
    motion_blur_px: float = 0.0
    motion_blur_angle_deg: float = 0.0
    gaussian_blur_sigma_px: float = 0.0
    shot_noise_std: float = 0.0
    read_noise_std: float = 0.0
    brightness_scale: float = 1.0
    contrast_scale: float = 1.0
    glare_probability: float = 0.0
    glare_radius_px: int = 24
    dropout_probability: float = 0.0
    occluder_probability: float = 0.0
    occluder_count: int = 0


@dataclass(frozen=True)
class RealisticInputConfig:
    camera: CameraPhysicalConfig = CameraPhysicalConfig()
    degradation: ImageDegradationConfig = ImageDegradationConfig()
    seed: int = 0


def load_realistic_config(path: str | Path) -> RealisticInputConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    camera_data = data.get("camera_physical", {})
    degradation_data = data.get("image_degradation", {})
    return RealisticInputConfig(
        camera=CameraPhysicalConfig(
            exposure_time_s=float(camera_data.get("exposure_time_s", 1.0 / 120.0)),
            readout_time_s=float(camera_data.get("readout_time_s", 0.0)),
            iso=float(camera_data.get("iso", 400.0)),
            aperture_f_number=float(camera_data.get("aperture_f_number", 2.0)),
            sensor_width_mm=float(camera_data.get("sensor_width_mm", 5.6)),
            sensor_height_mm=float(camera_data.get("sensor_height_mm", 3.2)),
            lens_transmission=float(camera_data.get("lens_transmission", 0.85)),
        ),
        degradation=ImageDegradationConfig(
            motion_blur_px=float(degradation_data.get("motion_blur_px", 0.0)),
            motion_blur_angle_deg=float(degradation_data.get("motion_blur_angle_deg", 0.0)),
            gaussian_blur_sigma_px=float(degradation_data.get("gaussian_blur_sigma_px", 0.0)),
            shot_noise_std=float(degradation_data.get("shot_noise_std", 0.0)),
            read_noise_std=float(degradation_data.get("read_noise_std", 0.0)),
            brightness_scale=float(degradation_data.get("brightness_scale", 1.0)),
            contrast_scale=float(degradation_data.get("contrast_scale", 1.0)),
            glare_probability=float(degradation_data.get("glare_probability", 0.0)),
            glare_radius_px=int(degradation_data.get("glare_radius_px", 24)),
            dropout_probability=float(degradation_data.get("dropout_probability", 0.0)),
            occluder_probability=float(degradation_data.get("occluder_probability", 0.0)),
            occluder_count=int(degradation_data.get("occluder_count", 0)),
        ),
        seed=int(data.get("seed", 0)),
    )


def apply_realistic_input_effects(
    image: np.ndarray,
    config: RealisticInputConfig,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Apply a lightweight sensor/input degradation pass to a rendered frame.

    This stage is intentionally image-space and independent of the geometric
    simulator. Later photorealistic stages can replace the renderer but still use
    this function as a deterministic sensor-noise baseline.
    """
    cv2 = _require_cv2()
    rng = np.random.default_rng(config.seed) if rng is None else rng
    degradation = config.degradation
    if rng.random() < degradation.dropout_probability:
        return np.zeros_like(image)

    result = np.asarray(image, dtype=np.float32)
    result = 127.5 + (result - 127.5) * degradation.contrast_scale
    result *= degradation.brightness_scale

    if degradation.motion_blur_px > 1.0:
        result = cv2.filter2D(result, -1, _motion_blur_kernel(degradation.motion_blur_px, degradation.motion_blur_angle_deg))
    if degradation.gaussian_blur_sigma_px > 0.0:
        result = cv2.GaussianBlur(result, (0, 0), degradation.gaussian_blur_sigma_px)
    if degradation.shot_noise_std > 0.0:
        signal_scale = np.sqrt(np.maximum(result, 0.0) / 255.0)
        result += rng.normal(0.0, degradation.shot_noise_std * 255.0, size=result.shape) * signal_scale
    if degradation.read_noise_std > 0.0:
        result += rng.normal(0.0, degradation.read_noise_std * 255.0, size=result.shape)
    if degradation.occluder_probability > 0.0 and rng.random() < degradation.occluder_probability:
        result = _add_occluders(result, rng, degradation.occluder_count)
    if degradation.glare_probability > 0.0 and rng.random() < degradation.glare_probability:
        result = _add_glare(result, rng, degradation.glare_radius_px)
    return np.clip(result, 0.0, 255.0).astype(np.uint8)


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for realistic image effects.") from exc
    return cv2


def _motion_blur_kernel(length_px: float, angle_deg: float) -> np.ndarray:
    cv2 = _require_cv2()
    length = max(1, int(round(length_px)))
    if length % 2 == 0:
        length += 1
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    kernel[center, :] = 1.0 / length
    matrix = cv2.getRotationMatrix2D((center, center), angle_deg, 1.0)
    rotated = cv2.warpAffine(kernel, matrix, (length, length))
    total = float(rotated.sum())
    return rotated / total if total > 1e-12 else kernel


def _add_glare(image: np.ndarray, rng: np.random.Generator, radius_px: int) -> np.ndarray:
    cv2 = _require_cv2()
    h, w = image.shape[:2]
    cx = int(rng.integers(0, max(w, 1)))
    cy = int(rng.integers(0, max(h, 1)))
    radius = max(2, int(radius_px))
    overlay = np.zeros_like(image, dtype=np.float32)
    cv2.circle(overlay, (cx, cy), radius, (255.0, 255.0, 255.0), -1, cv2.LINE_AA)
    overlay = cv2.GaussianBlur(overlay, (0, 0), radius * 0.45)
    return image + overlay * float(rng.uniform(0.25, 0.75))


def _add_occluders(image: np.ndarray, rng: np.random.Generator, count: int) -> np.ndarray:
    cv2 = _require_cv2()
    result = image.copy()
    h, w = result.shape[:2]
    for _ in range(max(1, count)):
        cx = int(rng.integers(0, max(w, 1)))
        cy = int(rng.integers(0, max(h, 1)))
        rx = int(rng.integers(max(8, w // 80), max(12, w // 12)))
        ry = int(rng.integers(max(8, h // 80), max(12, h // 10)))
        angle = float(rng.uniform(0.0, 180.0))
        color = tuple(float(v) for v in rng.uniform(20.0, 90.0, size=3))
        cv2.ellipse(result, (cx, cy), (rx, ry), angle, 0.0, 360.0, color, -1, cv2.LINE_AA)
    return result
