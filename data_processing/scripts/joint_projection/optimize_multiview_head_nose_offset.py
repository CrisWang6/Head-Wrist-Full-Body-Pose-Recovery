#!/usr/bin/env python3
"""Stage E (part 2): head-nose refine for 0806 multiview → head projection.

Upstream `replace_limb_mocap_gt.py` already does per-frame world nose align +
limb tip GT. This stage fits one head-rigid translation so reprojected nose
matches a **fixed** RTMW 2D tip (robust median of a small sample of detections;
see `detect_head_nose_rtmw.py --sample-count`).

Production path (大道至简):
  - Layer1: residual rigid translation vs CH3-08 nominal tip (robust median).
  - Layer2: small Gauss-Newton refine against the **same** fixed 2D nose UV on
    CAM_A/D (not per-frame jittery detections).

Constraint ban (explicit):
  Do NOT pull toe/knee toward a foot direction derived from the ankle mocap
  rigid frame (踝刚体坐标系和骨架关节坐标系没对齐).

Face draw policy: nose tip only. Feet: one big toe (delivery edges).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from multiview_geometry import (
    CameraPose,
    OmniCamera,
    load_json,
    rigid_world_transform,
)
from render_multiview_to_head import (
    CAMERA_ORDER,
    DEFAULT_CONFIG,
    DEFAULT_HEAD_INTRINSICS,
    DEFAULT_HEAD_RIGID,
    H265CaptureReader,
    HeadTimestampIndex,
    discover_head_dir,
    draw_laterality_labels,
    find_head_video,
    infer_head_module_from_video,
    head_mocap_correction,
    load_skeleton_playback,
    open_writer,
    project_joints,
    resolve_repo_path,
    rigid_camera_mounts,
)


from delivery_keypoints import DELIVERY_EDGES, FACE_HIDE

SCRIPT_DIR = Path(__file__).resolve().parent
BODY_EDGES = tuple(DELIVERY_EDGES)
DEFAULT_NOSE_OFFSET_MM = (0.0, -15.0, -125.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--skeleton-playback", type=Path)
    parser.add_argument("--head-a-nose-csv", type=Path, required=True)
    parser.add_argument("--head-d-nose-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--nose-offset-mm", type=float, nargs=3, default=DEFAULT_NOSE_OFFSET_MM
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--min-depth-m", type=float, default=0.01)
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="Only fit offset / write report (no videos).",
    )
    parser.add_argument(
        "--mode",
        choices=("per_frame", "fixed"),
        default="per_frame",
        help="Layer1 nose fit mode (default per_frame).",
    )
    parser.add_argument(
        "--soft-shoulder-weight",
        type=float,
        default=0.0,
        help="Optional soft pull of shoulders toward nose-relative mid (0=off).",
    )
    parser.add_argument(
        "--skip-head-2d-refine",
        action="store_true",
        help="Ablation: skip Layer2 fixed-RTMW 2D refine (Layer1 only).",
    )
    parser.add_argument(
        "--snapshot-seq", type=int, action="append", default=[0, 830, 1995]
    )
    return parser.parse_args()


def filter_face(points: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: uv for name, uv in points.items() if name not in FACE_HIDE}


def draw_body_skeleton(
    image: np.ndarray,
    points: dict[str, np.ndarray],
    color: tuple[int, int, int],
    *,
    radius: int = 4,
) -> None:
    visible = filter_face(points)
    for first, second in BODY_EDGES:
        if first in visible and second in visible:
            cv2.line(
                image,
                tuple(np.rint(visible[first]).astype(int)),
                tuple(np.rint(visible[second]).astype(int)),
                color,
                3,
                cv2.LINE_AA,
            )
    for point in visible.values():
        cv2.circle(
            image,
            tuple(np.rint(point).astype(int)),
            radius,
            color,
            -1,
            cv2.LINE_AA,
        )


def load_nose_csv(path: Path) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                point = np.asarray(
                    [float(row["face_nose_u_px"]), float(row["face_nose_v_px"])],
                    dtype=np.float64,
                )
                score = float(row["face_nose_score"])
            except (KeyError, TypeError, ValueError):
                continue
            if (
                np.isfinite(point).all()
                and np.isfinite(score)
                and score >= 0.05
                and 0.0 <= point[0] < 1920.0
                and 0.0 <= point[1] < 1200.0
            ):
                result[int(row["frame_index"])] = point
    return result


def fixed_nose_uv_from_csv(path: Path) -> tuple[np.ndarray, dict]:
    meta_path = path.with_suffix(".fixed.json")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fixed = np.asarray(meta["fixed_uv_px"], dtype=np.float64)
        return fixed, meta
    raw = load_nose_csv(path)
    if not raw:
        raise RuntimeError(f"No valid nose rows in {path}")
    fixed, _, stats = robust_fixed_point(raw)
    return fixed, {"source": "csv_robust_median", "sample_stats": stats}


def apply_rigid_offset(
    xyz_world: dict[str, list[float]],
    world_rigid: np.ndarray,
    offset_rigid: np.ndarray,
) -> dict[str, list[float]]:
    rotation = world_rigid[:3, :3]
    delta_world = rotation @ offset_rigid
    return {
        name: (np.asarray(xyz, dtype=np.float64) + delta_world).tolist()
        for name, xyz in xyz_world.items()
    }


def fit_layer1_external_vs_rigid(
    frames: list[dict],
    aligned_by_seq: dict[int, dict],
    rigid_prefix: str,
    mocap_to_head: np.ndarray,
    nose_nominal_rigid: np.ndarray,
    *,
    mode: str = "per_frame",
) -> tuple[np.ndarray, dict]:
    deltas = []
    for frame in frames:
        sequence = int(frame["seq"])
        aligned = aligned_by_seq.get(sequence)
        if aligned is None:
            continue
        if int(float(aligned.get(f"{rigid_prefix}_status", "0"))) != 1:
            continue
        if int(float(aligned.get(f"{rigid_prefix}_raw_tick_valid", "1"))) != 1:
            continue
        if "nose" not in frame["xyz_world_m"]:
            continue
        world_rigid = rigid_world_transform(aligned, rigid_prefix) @ mocap_to_head
        nose_world = np.asarray(frame["xyz_world_m"]["nose"], dtype=np.float64)
        nose_h = np.r_[nose_world, 1.0]
        nose_rigid = (np.linalg.inv(world_rigid) @ nose_h)[:3]
        # Move external skeleton in rigid so its nose matches the nominal
        # head-rigid nose used by the head-camera chain.
        deltas.append(nose_nominal_rigid - nose_rigid)
    if not deltas:
        raise RuntimeError("Layer1: no external/rigid nose pairs")
    values = np.asarray(deltas, dtype=np.float64)
    center = np.median(values, axis=0)
    radii = np.linalg.norm(values - center, axis=1)
    keep = radii <= (
        float(np.median(radii))
        + 3.0 * 1.4826 * float(np.median(np.abs(radii - np.median(radii))))
        + 1e-9
    )
    # Both modes currently emit one rigid offset used by the head render path.
    # per_frame: median of per-frame nose deltas (after upstream per-frame world
    # nose align). fixed: same robust median (legacy ablation label).
    offset = np.median(values[keep], axis=0)
    residual = np.linalg.norm(values[keep] - offset, axis=1)
    stats = {
        "mode": mode,
        "samples": int(len(values)),
        "inliers": int(keep.sum()),
        "offset_rigid_m": offset.tolist(),
        "offset_norm_mm": float(np.linalg.norm(offset) * 1000.0),
        "residual_median_mm": float(np.median(residual) * 1000.0),
        "residual_p90_mm": float(np.percentile(residual, 90) * 1000.0),
        "ankle_rigid_to_toe_constraint": False,
    }
    return offset, stats


def robust_fixed_point(
    points: dict[int, np.ndarray],
) -> tuple[np.ndarray, dict[int, bool], dict]:
    indices = sorted(points)
    values = np.asarray([points[index] for index in indices], dtype=np.float64)
    center = np.median(values, axis=0)
    radial = np.linalg.norm(values - center, axis=1)
    radial_center = float(np.median(radial))
    mad = float(np.median(np.abs(radial - radial_center)))
    threshold = radial_center + max(4.0, 4.5 * 1.4826 * mad)
    keep_array = radial <= threshold
    fixed = np.median(values[keep_array], axis=0)
    keep = {index: bool(flag) for index, flag in zip(indices, keep_array)}
    residual = np.linalg.norm(values[keep_array] - fixed, axis=1)
    stats = {
        "raw_samples": int(len(values)),
        "inlier_samples": int(keep_array.sum()),
        "outlier_samples": int((~keep_array).sum()),
        "outlier_threshold_px": float(threshold),
        "raw_to_fixed_median_px": float(np.median(residual)),
        "raw_to_fixed_p90_px": float(np.percentile(residual, 90)),
        "fixed_uv_px": fixed.tolist(),
    }
    return fixed, keep, stats


def rtmw_inlier_masks(
    raw_a: dict[int, np.ndarray], raw_d: dict[int, np.ndarray]
) -> tuple[dict[str, dict[int, bool]], dict]:
    """Robust inliers for nearly-fixed wearer tip UV in each head camera."""
    _, keep_a, stats_a = robust_fixed_point(raw_a)
    _, keep_d, stats_d = robust_fixed_point(raw_d)
    return (
        {"CAM_A": keep_a, "CAM_D": keep_d},
        {"CAM_A": stats_a, "CAM_D": stats_d},
    )


def refine_fixed_2d_offset(
    frames: list[dict],
    aligned_by_seq: dict[int, dict],
    rigid_prefix: str,
    mocap_to_head: np.ndarray,
    mounts: dict[str, np.ndarray],
    models: dict[str, OmniCamera],
    indexes: dict[str, HeadTimestampIndex],
    fixed_uv: dict[str, np.ndarray],
    offset0: np.ndarray,
    *,
    min_depth_m: float,
    max_samples: int = 400,
) -> tuple[np.ndarray, dict]:
    """Refine one rigid translation against constant RTMW nose UV per camera."""
    packed = []
    for frame in frames:
        sequence = int(frame["seq"])
        aligned = aligned_by_seq.get(sequence)
        if aligned is None or "nose" not in frame["xyz_world_m"]:
            continue
        if int(float(aligned.get(f"{rigid_prefix}_status", "0"))) != 1:
            continue
        world_rigid = rigid_world_transform(aligned, rigid_prefix) @ mocap_to_head
        nose_world = np.asarray(frame["xyz_world_m"]["nose"], dtype=np.float64)
        for socket in CAMERA_ORDER:
            column = f"head_{socket}_exposure_end_timestamp_ms"
            try:
                capture_index, _, _ = indexes[socket].nearest(float(aligned[column]))
            except KeyError:
                continue
            packed.append((world_rigid, nose_world, socket, fixed_uv[socket], capture_index))
            if len(packed) >= max_samples:
                break
        if len(packed) >= max_samples:
            break
    if len(packed) < 40:
        return offset0, {
            "applied": False,
            "reason": f"insufficient pose pairs ({len(packed)})",
            "samples": len(packed),
        }

    def reproj_errors(candidate: np.ndarray) -> np.ndarray:
        errs = []
        for world_rigid, nose_world, socket, target, _capture_index in packed:
            nose = nose_world + world_rigid[:3, :3] @ candidate
            pose = CameraPose(models[socket], world_rigid @ mounts[socket])
            projected = project_joints(
                pose,
                {"nose": nose.tolist()},
                min_depth_m=min_depth_m,
                width=models[socket].width,
                height=models[socket].height,
            )
            if "nose" not in projected:
                continue
            errs.append(projected["nose"] - target)
        if not errs:
            return np.zeros(0, dtype=np.float64)
        return np.asarray(errs, dtype=np.float64).reshape(-1)

    err0 = reproj_errors(offset0)
    if err0.size < 80:
        return offset0, {
            "applied": False,
            "reason": "too few finite reprojections",
            "samples": int(err0.size // 2),
        }
    before = float(np.median(np.linalg.norm(err0.reshape(-1, 2), axis=1)))
    offset = offset0.copy()
    history = []
    for _ in range(6):
        residual = reproj_errors(offset)
        if residual.size < 80:
            break
        jac = []
        eps = 1e-3
        for axis in range(3):
            probe = offset.copy()
            probe[axis] += eps
            jac.append((reproj_errors(probe) - residual) / eps)
        jacobian = np.asarray(jac, dtype=np.float64).T
        if jacobian.shape[0] != residual.shape[0]:
            break
        delta, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        delta = np.clip(delta, -0.02, 0.02)
        offset = offset + delta
        history.append(float(np.linalg.norm(delta) * 1000.0))
        if float(np.linalg.norm(delta)) < 2e-4:
            break
    err1 = reproj_errors(offset)
    after = float(np.median(np.linalg.norm(err1.reshape(-1, 2), axis=1)))
    if after > before * 1.05:
        return offset0, {
            "applied": False,
            "reason": "refine increased reprojection error",
            "median_px_before": before,
            "median_px_after": after,
            "samples": int(err0.size // 2),
        }
    return offset, {
        "applied": True,
        "policy": "fixed_rtmw_uv_per_camera",
        "fixed_uv_px": {socket: fixed_uv[socket].tolist() for socket in CAMERA_ORDER},
        "samples": int(err0.size // 2),
        "median_px_before": before,
        "median_px_after": after,
        "delta_from_previous_mm": ((offset - offset0) * 1000.0).tolist(),
        "delta_norm_mm": float(np.linalg.norm(offset - offset0) * 1000.0),
        "step_norms_mm": history,
    }


def refine_layer2_2d(
    frames: list[dict],
    aligned_by_seq: dict[int, dict],
    rigid_prefix: str,
    mocap_to_head: np.ndarray,
    mounts: dict[str, np.ndarray],
    models: dict[str, OmniCamera],
    indexes: dict[str, HeadTimestampIndex],
    raw_nose: dict[str, dict[int, np.ndarray]],
    keep: dict[str, dict[int, bool]],
    offset0: np.ndarray,
    *,
    min_depth_m: float,
    max_samples: int = 400,
) -> tuple[np.ndarray, dict]:
    packed = []
    for frame in frames:
        sequence = int(frame["seq"])
        aligned = aligned_by_seq.get(sequence)
        if aligned is None or "nose" not in frame["xyz_world_m"]:
            continue
        if int(float(aligned.get(f"{rigid_prefix}_status", "0"))) != 1:
            continue
        world_rigid = rigid_world_transform(aligned, rigid_prefix) @ mocap_to_head
        nose_world = np.asarray(frame["xyz_world_m"]["nose"], dtype=np.float64)
        for socket in CAMERA_ORDER:
            column = f"head_{socket}_exposure_end_timestamp_ms"
            try:
                capture_index, _, _ = indexes[socket].nearest(float(aligned[column]))
            except KeyError:
                continue
            detected = raw_nose[socket].get(capture_index)
            if detected is None or not keep[socket].get(capture_index, False):
                continue
            packed.append((world_rigid, nose_world, socket, detected))
            if len(packed) >= max_samples:
                break
        if len(packed) >= max_samples:
            break
    if len(packed) < 40:
        return offset0, {
            "applied": False,
            "reason": f"insufficient 2D pairs ({len(packed)})",
            "samples": len(packed),
        }

    def reproj_errors(candidate: np.ndarray) -> np.ndarray:
        errs = []
        for world_rigid, nose_world, socket, detected in packed:
            nose = nose_world + world_rigid[:3, :3] @ candidate
            pose = CameraPose(models[socket], world_rigid @ mounts[socket])
            projected = project_joints(
                pose,
                {"nose": nose.tolist()},
                min_depth_m=min_depth_m,
                width=models[socket].width,
                height=models[socket].height,
            )
            if "nose" not in projected:
                continue
            errs.append(projected["nose"] - detected)
        if not errs:
            return np.zeros(0, dtype=np.float64)
        return np.asarray(errs, dtype=np.float64).reshape(-1)

    err0 = reproj_errors(offset0)
    if err0.size < 80:
        return offset0, {
            "applied": False,
            "reason": "too few finite reprojections",
            "samples": int(err0.size // 2),
        }
    before = float(np.median(np.linalg.norm(err0.reshape(-1, 2), axis=1)))
    offset = offset0.copy()
    history = []
    for _ in range(6):
        residual = reproj_errors(offset)
        if residual.size < 80:
            break
        jac = []
        eps = 1e-3
        for axis in range(3):
            probe = offset.copy()
            probe[axis] += eps
            jac.append((reproj_errors(probe) - residual) / eps)
        jacobian = np.asarray(jac, dtype=np.float64).T
        if jacobian.shape[0] != residual.shape[0]:
            break
        delta, *_ = np.linalg.lstsq(jacobian, -residual, rcond=None)
        delta = np.clip(delta, -0.02, 0.02)
        offset = offset + delta
        history.append(float(np.linalg.norm(delta) * 1000.0))
        if float(np.linalg.norm(delta)) < 2e-4:
            break
    err1 = reproj_errors(offset)
    after = float(np.median(np.linalg.norm(err1.reshape(-1, 2), axis=1)))
    if after > before * 1.05:
        return offset0, {
            "applied": False,
            "reason": "refine increased reprojection error",
            "median_px_before": before,
            "median_px_after": after,
            "samples": int(err0.size // 2),
        }
    return offset, {
        "applied": True,
        "samples": int(err0.size // 2),
        "median_px_before": before,
        "median_px_after": after,
        "delta_from_previous_mm": ((offset - offset0) * 1000.0).tolist(),
        "delta_norm_mm": float(np.linalg.norm(offset - offset0) * 1000.0),
        "step_norms_mm": history,
    }


def render_pair(
    *,
    records_before: list[dict],
    records_after: list[dict],
    videos: dict[str, Path],
    indexes: dict[str, HeadTimestampIndex],
    models: dict[str, OmniCamera],
    output_dir: Path,
    fps: float,
    snapshot_seq: set[int],
) -> dict[str, Path]:
    paths = {
        "before_a": output_dir / "head_CAM_A_direct_noseonly.mp4",
        "before_d": output_dir / "head_CAM_D_direct_noseonly.mp4",
        "after_a": output_dir / "head_CAM_A_nose_offset_opt.mp4",
        "after_d": output_dir / "head_CAM_D_nose_offset_opt.mp4",
        "grid": output_dir / "head_2x2_direct_vs_nose_offset_opt.mp4",
    }
    readers = {
        socket: H265CaptureReader(videos[socket], indexes[socket].rows)
        for socket in CAMERA_ORDER
    }
    writers = {
        "before_a": open_writer(
            paths["before_a"], (models["CAM_A"].width, models["CAM_A"].height), fps
        ),
        "before_d": open_writer(
            paths["before_d"], (models["CAM_D"].width, models["CAM_D"].height), fps
        ),
        "after_a": open_writer(
            paths["after_a"], (models["CAM_A"].width, models["CAM_A"].height), fps
        ),
        "after_d": open_writer(
            paths["after_d"], (models["CAM_D"].width, models["CAM_D"].height), fps
        ),
        "grid": open_writer(paths["grid"], (1920, 1200), fps),
    }
    try:
        for record_index, (before, after) in enumerate(
            zip(records_before, records_after)
        ):
            images = {}
            for socket in CAMERA_ORDER:
                base = readers[socket].read(int(before["frames"][socket]))
                image_b = base.copy()
                image_a = base.copy()
                draw_body_skeleton(
                    image_b, before["projected"][socket], (255, 0, 255), radius=4
                )
                draw_body_skeleton(
                    image_a, after["projected"][socket], (0, 255, 255), radius=4
                )
                draw_laterality_labels(
                    image_b, filter_face(before["projected"][socket])
                )
                draw_laterality_labels(
                    image_a, filter_face(after["projected"][socket])
                )
                det = after.get("rtmw", {}).get(socket)
                if det is not None:
                    cv2.circle(
                        image_a,
                        tuple(np.rint(det).astype(int)),
                        8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    image_b,
                    f"BEFORE direct {socket} seq={before['seq']} magenta nose-only",
                    (24, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image_a,
                    f"AFTER nose-opt {socket} seq={after['seq']} cyan + green fixed RTMW",
                    (24, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                images[socket] = (image_b, image_a)
            writers["before_a"].write(images["CAM_A"][0])
            writers["after_a"].write(images["CAM_A"][1])
            writers["before_d"].write(images["CAM_D"][0])
            writers["after_d"].write(images["CAM_D"][1])
            grid = np.vstack(
                (
                    np.hstack(
                        (
                            cv2.resize(images["CAM_A"][0], (960, 600)),
                            cv2.resize(images["CAM_A"][1], (960, 600)),
                        )
                    ),
                    np.hstack(
                        (
                            cv2.resize(images["CAM_D"][0], (960, 600)),
                            cv2.resize(images["CAM_D"][1], (960, 600)),
                        )
                    ),
                )
            )
            writers["grid"].write(grid)
            if int(before["seq"]) in snapshot_seq or record_index == 0:
                snap = (
                    output_dir
                    / f"head_2x2_direct_vs_nose_offset_opt_seq{int(before['seq']):04d}.jpg"
                )
                ok, encoded = cv2.imencode(".jpg", grid)
                if ok:
                    encoded.tofile(str(snap))
    finally:
        for writer in writers.values():
            writer.release()
        for reader in readers.values():
            reader.close()
    return paths


def build_records(
    frames: list[dict],
    aligned_by_seq: dict[int, dict],
    rigid_prefix: str,
    mocap_to_head: np.ndarray,
    mounts: dict[str, np.ndarray],
    models: dict[str, OmniCamera],
    indexes: dict[str, HeadTimestampIndex],
    *,
    offset_rigid: np.ndarray,
    raw_nose: dict[str, dict[int, np.ndarray]] | None,
    fixed_nose_uv: dict[str, np.ndarray] | None = None,
    min_depth_m: float,
    tolerance_ms: float,
) -> list[dict]:
    records = []
    for frame in frames:
        sequence = int(frame["seq"])
        aligned = aligned_by_seq.get(sequence)
        if aligned is None:
            continue
        if int(float(aligned.get(f"{rigid_prefix}_status", "0"))) != 1:
            continue
        world_rigid = rigid_world_transform(aligned, rigid_prefix) @ mocap_to_head
        xyz = frame["xyz_world_m"]
        if float(np.linalg.norm(offset_rigid)) > 0:
            xyz = apply_rigid_offset(xyz, world_rigid, offset_rigid)
        item = {"seq": sequence, "frames": {}, "projected": {}, "rtmw": {}}
        for socket in CAMERA_ORDER:
            column = f"head_{socket}_exposure_end_timestamp_ms"
            capture_index, _, _ = indexes[socket].nearest(
                float(aligned[column]), tolerance_ms
            )
            pose = CameraPose(models[socket], world_rigid @ mounts[socket])
            projected = project_joints(
                pose,
                xyz,
                min_depth_m=min_depth_m,
                width=models[socket].width,
                height=models[socket].height,
            )
            item["frames"][socket] = capture_index
            item["projected"][socket] = {
                name: uv for name, uv in projected.items() if name not in FACE_HIDE
            }
            if fixed_nose_uv is not None:
                item["rtmw"][socket] = fixed_nose_uv[socket]
            elif raw_nose is not None:
                det = raw_nose[socket].get(capture_index)
                if det is not None:
                    item["rtmw"][socket] = det
        records.append(item)
    return records


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    config = load_json(args.config)
    head_cfg = config.get("head", {})
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else data_root
        / "multiview_3d_results"
        / "full"
        / "head_reprojection"
        / "nose_offset_opt"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    skeleton_path = (
        args.skeleton_playback.resolve()
        if args.skeleton_playback is not None
        else data_root / "multiview_3d_results" / "full" / "skeleton_playback.json"
    )
    _, _, frames = load_skeleton_playback(skeleton_path)
    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    aligned_path = data_root / "aligned_data" / "aligned_30hz.csv"
    with aligned_path.open("r", encoding="utf-8-sig", newline="") as stream:
        aligned_by_seq = {int(row["seq"]): row for row in csv.DictReader(stream)}

    head_dir = discover_head_dir(data_root, config)
    rigid_prefix = head_cfg.get("rigid_prefix", "mocap_CH3_08")
    head_intrinsics = load_json(
        resolve_repo_path(head_cfg.get("intrinsics", DEFAULT_HEAD_INTRINSICS))
    )
    head_rigid = load_json(
        resolve_repo_path(head_cfg.get("rigid_extrinsics", DEFAULT_HEAD_RIGID))
    )
    camera_basis = str(head_cfg.get("head_camera_rotation_basis", "xy_swap"))
    mounts = rigid_camera_mounts(head_rigid, head_intrinsics, basis=camera_basis)
    mocap_to_head = head_mocap_correction(head_cfg, config)
    models = {
        socket: OmniCamera.from_calibration(head_intrinsics, socket, name=socket)
        for socket in CAMERA_ORDER
    }
    videos = {
        socket: find_head_video(head_dir, socket, config) for socket in CAMERA_ORDER
    }
    indexes = {
        socket: HeadTimestampIndex(
            head_dir / "timestamps.csv",
            socket,
            module=infer_head_module_from_video(videos[socket]),
        )
        for socket in CAMERA_ORDER
    }
    tolerance_ms = float(head_cfg.get("timestamp_match_tolerance_ms", 1.0))
    nose_cfg = head_cfg.get("nose_offset_mm", args.nose_offset_mm)
    nose_nominal = np.asarray(nose_cfg, dtype=np.float64) / 1000.0

    # Layer1: 3D GT from head rigid fixed tip vs external multiview nose.
    layer1_offset, layer1_stats = fit_layer1_external_vs_rigid(
        frames,
        aligned_by_seq,
        rigid_prefix,
        mocap_to_head,
        nose_nominal,
        mode=args.mode,
    )

    # Layer2: refine the same rigid translation against fixed RTMW 2D nose UV.
    fixed_a, det_stats_a = fixed_nose_uv_from_csv(args.head_a_nose_csv)
    fixed_d, det_stats_d = fixed_nose_uv_from_csv(args.head_d_nose_csv)
    fixed_uv = {"CAM_A": fixed_a, "CAM_D": fixed_d}
    if args.skip_head_2d_refine:
        layer2_stats = {"applied": False, "reason": "ablation --skip-head-2d-refine"}
        final_offset = layer1_offset
        final_source = "layer1_3d_gt_rigid_tip"
    else:
        layer2_offset, layer2_stats = refine_fixed_2d_offset(
            frames,
            aligned_by_seq,
            rigid_prefix,
            mocap_to_head,
            mounts,
            models,
            indexes,
            fixed_uv,
            layer1_offset,
            min_depth_m=args.min_depth_m,
        )
        if layer2_stats.get("applied"):
            final_offset = layer2_offset
            final_source = "layer1_3d_gt_rigid_tip+fixed_rtmw_2d_refine"
        else:
            final_offset = layer1_offset
            final_source = "layer1_3d_gt_rigid_tip"
    det_stats = {"CAM_A": det_stats_a, "CAM_D": det_stats_d}

    report = {
        "schema": "joint_projection.multiview_head_nose_offset.v2",
        "mode": args.mode,
        "method_note": (
            "Upstream replace does per-frame world nose align + limb tip GT. "
            "Here: 3D nose GT = T_world_CH3_08 @ [0,-15,-125] mm; "
            "2D nose obs = fixed RTMW face tip on head CAM_A/D (sampled + robust median); "
            f"Layer1 mode={args.mode}; Layer2 small rigid refine vs fixed 2D UV. "
            "BAN: never pull toe/knee from ankle mocap rigid axes."
        ),
        "data_root": str(data_root),
        "skeleton_playback": str(skeleton_path),
        "nose_gt_rigid_mm": list(map(float, nose_cfg)),
        "head_camera_rotation_basis": camera_basis,
        "soft_shoulder_weight": float(args.soft_shoulder_weight),
        "skip_head_2d_refine": bool(args.skip_head_2d_refine),
        "ankle_rigid_to_toe_constraint": False,
        "layer1_3d_external_vs_rigid_gt": layer1_stats,
        "layer2_rtmw_2d_refine": layer2_stats,
        "fixed_rtmw_nose_uv": {
            socket: fixed_uv[socket].tolist() for socket in CAMERA_ORDER
        },
        "rtmw_detection_stats": det_stats,
        "chosen_offset_source": final_source,
        "final_offset_rigid_m": final_offset.tolist(),
        "final_offset_norm_mm": float(np.linalg.norm(final_offset) * 1000.0),
        "face_policy": "draw nose only; hide eyes/ears",
        "foot_policy": "one_big_toe_per_foot",
        "videos": {socket: str(path) for socket, path in videos.items()},
    }

    if not args.skip_render:
        before = build_records(
            frames,
            aligned_by_seq,
            rigid_prefix,
            mocap_to_head,
            mounts,
            models,
            indexes,
            offset_rigid=np.zeros(3),
            raw_nose=None,
            min_depth_m=args.min_depth_m,
            tolerance_ms=tolerance_ms,
        )
        after = build_records(
            frames,
            aligned_by_seq,
            rigid_prefix,
            mocap_to_head,
            mounts,
            models,
            indexes,
            offset_rigid=final_offset,
            raw_nose=None,
            fixed_nose_uv=fixed_uv,
            min_depth_m=args.min_depth_m,
            tolerance_ms=tolerance_ms,
        )
        paths = render_pair(
            records_before=before,
            records_after=after,
            videos=videos,
            indexes=indexes,
            models=models,
            output_dir=output_dir,
            fps=args.fps,
            snapshot_seq=set(args.snapshot_seq),
        )
        report["outputs"] = {key: str(path) for key, path in paths.items()}
        report["rendered_frames"] = len(before)

    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
