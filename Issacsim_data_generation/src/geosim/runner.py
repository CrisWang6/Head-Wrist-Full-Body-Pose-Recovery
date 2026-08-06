from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from geosim.camera import make_default_camera_rig
from geosim.config import SimulationConfig
from geosim.geometry import Ray, triangulate_rays
from geosim.linalg import rigid_align, rotation_error_deg
from geosim.motion import MotionSequence
from geosim.tag_rig import make_wrist_tag_rig


@dataclass(frozen=True)
class FrameResult:
    frame_idx: int
    success: bool
    position_error_m: float | None
    rotation_error_deg: float | None
    reconstructed_points: int
    visible_corners: int
    visible_observations: int


def run_motion_sequence(
    motion: MotionSequence,
    config: SimulationConfig,
    rng: np.random.Generator | None = None,
) -> list[FrameResult]:
    cameras = make_default_camera_rig(config.camera_rig)
    tag_rig = make_wrist_tag_rig(config.tag_rig)
    model_points = tag_rig.model_points
    rng = np.random.default_rng(0) if rng is None else rng
    frame_count = motion.frames if config.max_frames <= 0 else min(motion.frames, config.max_frames)

    results = []
    for frame_idx in range(frame_count):
        true_points = tag_rig.world_points(motion.left_wrist_pos[frame_idx], motion.left_wrist_rot[frame_idx])
        rays_by_corner: dict[str, list[Ray]] = {key: [] for key in model_points}
        visible_observations = 0

        for camera in cameras:
            for key, point_world in true_points.items():
                pixel, visible = camera.project_world(point_world, motion.head_pos[frame_idx], motion.head_rot[frame_idx])
                if not visible:
                    continue
                if config.noise.pixel_noise_std > 0.0:
                    pixel = pixel + rng.normal(0.0, config.noise.pixel_noise_std, size=2)
                    if not camera.pixel_in_image(pixel):
                        continue
                origin, direction = camera.ray_world(pixel, motion.head_pos[frame_idx], motion.head_rot[frame_idx])
                rays_by_corner[key].append(Ray(camera.name, origin, direction))
                visible_observations += 1

        reconstructed = {}
        for key, rays in rays_by_corner.items():
            if len(rays) >= config.min_views_per_corner:
                reconstructed[key] = triangulate_rays(rays)

        if len(reconstructed) >= 4:
            keys = sorted(reconstructed)
            source = np.stack([model_points[key] for key in keys], axis=0)
            target = np.stack([reconstructed[key] for key in keys], axis=0)
            if np.linalg.matrix_rank(source - source.mean(axis=0), tol=1e-8) < 2:
                pos_error = None
                rot_error = None
                success = False
            else:
                try:
                    est_rot, est_pos = rigid_align(source, target)
                    pos_error = float(np.linalg.norm(est_pos - motion.left_wrist_pos[frame_idx]))
                    rot_error = float(rotation_error_deg(motion.left_wrist_rot[frame_idx], est_rot))
                    success = math.isfinite(pos_error) and math.isfinite(rot_error)
                except np.linalg.LinAlgError:
                    pos_error = None
                    rot_error = None
                    success = False
        else:
            pos_error = None
            rot_error = None
            success = False

        visible_corners = sum(1 for rays in rays_by_corner.values() if rays)
        results.append(
            FrameResult(
                frame_idx=frame_idx,
                success=success,
                position_error_m=pos_error,
                rotation_error_deg=rot_error,
                reconstructed_points=len(reconstructed),
                visible_corners=visible_corners,
                visible_observations=visible_observations,
            )
        )
    return results


def summarize_results(results: list[FrameResult]) -> dict[str, float | int]:
    total = len(results)
    successes = [result for result in results if result.success]
    summary: dict[str, float | int] = {
        "frames": total,
        "success_frames": len(successes),
        "success_rate": len(successes) / total if total else 0.0,
    }
    if not successes:
        return summary
    pos = np.array([result.position_error_m for result in successes], dtype=float)
    rot = np.array([result.rotation_error_deg for result in successes], dtype=float)
    obs = np.array([result.visible_observations for result in results], dtype=float)
    summary.update(
        {
            "position_mean_m": float(np.mean(pos)),
            "position_p95_m": float(np.percentile(pos, 95)),
            "position_max_m": float(np.max(pos)),
            "rotation_mean_deg": float(np.mean(rot)),
            "rotation_p95_deg": float(np.percentile(rot, 95)),
            "rotation_max_deg": float(np.max(rot)),
            "visible_observations_mean": float(np.mean(obs)),
        }
    )
    return summary


def summarize_result_sets(result_sets: list[list[FrameResult]]) -> dict[str, float | int]:
    merged = [result for result_set in result_sets for result in result_set]
    summary = summarize_results(merged)
    summary["motions"] = len(result_sets)
    return summary
