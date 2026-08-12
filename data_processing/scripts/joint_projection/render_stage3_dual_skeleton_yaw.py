#!/usr/bin/env python3
"""Strong 3D filter on Stage3 pred + side-by-side GT/pred yaw skeleton MP4."""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import cv2
import numpy as np

from delivery_keypoints import DELIVERY_JOINTS
from skeleton_3d_filter import filter_skeleton_playback_records

MISS = -32768
PRED_EDGE = (255, 110, 0)
PRED_JOINT = (255, 180, 80)
GT_EDGE = (0, 210, 70)
GT_JOINT = (120, 255, 160)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-playback", type=Path, required=True)
    p.add_argument("--gt-playback", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--filtered-playback", type=Path, default=None)
    p.add_argument("--yaw-deg", type=float, default=100.0)
    p.add_argument("--pitch-deg", type=float, default=18.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--panel-width", type=int, default=960)
    p.add_argument("--panel-height", type=int, default=720)
    p.add_argument("--max-frames", type=int, default=0)
    return p.parse_args()


def load_playback_records(path: Path) -> tuple[list[int], list[str], list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    joints = [str(x) for x in payload["joints"]]
    seqs = [int(x) for x in payload["seqs"]]
    raw = base64.b64decode(payload["xyz_i16_b64"])
    array = np.frombuffer(raw, dtype="<i2").reshape(len(seqs), len(joints), 3)
    records: list[dict] = []
    for fi, seq in enumerate(seqs):
        multiview = {}
        for ji, name in enumerate(joints):
            sample = array[fi, ji]
            if (sample == MISS).any():
                continue
            multiview[name] = {"xyz_world_m": (sample.astype(np.float32) / 1000.0).tolist()}
        records.append({"seq": seq, "methods": {"filtered": {"multiview": multiview}}})
    return seqs, joints, records


def world_to_display(xyz: np.ndarray) -> np.ndarray:
    out = np.empty_like(xyz)
    out[..., 0] = xyz[..., 0]
    out[..., 1] = xyz[..., 2]
    out[..., 2] = xyz[..., 1]
    return out


def records_to_traj(records: list[dict], joint_names: list[str]) -> np.ndarray:
    traj = np.full((len(records), len(joint_names), 3), np.nan, dtype=np.float64)
    for fi, record in enumerate(records):
        joints = record["methods"]["filtered"]["multiview"]
        for ji, name in enumerate(joint_names):
            payload = joints.get(name)
            if payload is None:
                continue
            traj[fi, ji] = np.asarray(payload["xyz_world_m"], dtype=np.float64)
    return traj


def compute_bounds(display_list: list[np.ndarray]) -> tuple[np.ndarray, float]:
    valid = np.concatenate([d.reshape(-1, 3) for d in display_list], axis=0)
    valid = valid[np.isfinite(valid).all(axis=1)]
    if valid.size == 0:
        raise RuntimeError("No valid skeleton points")
    low = np.nanpercentile(valid, 2, axis=0)
    high = np.nanpercentile(valid, 98, axis=0)
    center = 0.5 * (low + high)
    span = float(np.max(high - low))
    return center.astype(np.float64), max(span, 0.5)


def project_points(
    points: np.ndarray,
    center: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    width: int,
    height: int,
    span: float,
) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    d = points.astype(np.float64) - center
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    x1 = d[:, 0] * cos_y - d[:, 1] * sin_y
    y1 = d[:, 0] * sin_y + d[:, 1] * cos_y
    z1 = d[:, 2]
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    y2 = y1 * cos_p - z1 * sin_p
    z2 = y1 * sin_p + z1 * cos_p
    x2 = x1
    scale = (min(width, height) * 0.78) / span
    sx = width * 0.5 + x2 * scale
    sy = height * 0.5 - z2 * scale
    return np.stack([sx, sy, y2], axis=1)


def default_edges(joints: list[str]) -> list[tuple[int, int]]:
    pairs = [
        ("nose", "left_shoulder"),
        ("nose", "right_shoulder"),
        ("left_shoulder", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "right_hip"),
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
        ("left_ankle", "left_big_toe"),
        ("right_ankle", "right_big_toe"),
    ]
    index = {name: i for i, name in enumerate(joints)}
    return [(index[a], index[b]) for a, b in pairs if a in index and b in index]


def draw_panel(
    traj_frame: np.ndarray,
    *,
    center: np.ndarray,
    span: float,
    yaw_deg: float,
    pitch_deg: float,
    width: int,
    height: int,
    edges: list[tuple[int, int]],
    edge_color: tuple[int, int, int],
    joint_color: tuple[int, int, int],
    title: str,
    seq: int,
) -> np.ndarray:
    img = np.full((height, width, 3), 18, dtype=np.uint8)
    display = world_to_display(traj_frame)
    projected = project_points(display, center, yaw_deg, pitch_deg, width, height, span)
    for a, b in edges:
        if not (np.isfinite(display[a]).all() and np.isfinite(display[b]).all()):
            continue
        pa, pb = projected[a], projected[b]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        cv2.line(
            img,
            (int(round(pa[0])), int(round(pa[1]))),
            (int(round(pb[0])), int(round(pb[1]))),
            edge_color,
            3,
            cv2.LINE_AA,
        )
    for j, p in enumerate(projected):
        if not (np.isfinite(display[j]).all() and np.isfinite(p).all()):
            continue
        cv2.circle(
            img,
            (int(round(p[0])), int(round(p[1]))),
            4,
            joint_color,
            -1,
            cv2.LINE_AA,
        )
    cv2.putText(
        img,
        title,
        (20, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (230, 230, 230),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"seq {seq}",
        (20, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (170, 170, 170),
        1,
        cv2.LINE_AA,
    )
    return img


def mpjpe_mm(pred: np.ndarray, gt: np.ndarray, joint_names: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    vals = []
    for ji, name in enumerate(joint_names):
        diff = pred[:, ji] - gt[:, ji]
        valid = np.isfinite(diff).all(axis=1)
        if valid.sum() == 0:
            out[name] = float("nan")
            continue
        err = float(np.linalg.norm(diff[valid], axis=1).mean() * 1000.0)
        out[name] = err
        vals.append(err)
    out["overall"] = float(np.mean(vals)) if vals else float("nan")
    return out


def main() -> int:
    args = parse_args()
    seqs, joint_names, pred_records = load_playback_records(args.pred_playback)
    gt_seqs, gt_joints, gt_records = load_playback_records(args.gt_playback)
    if seqs != gt_seqs or joint_names != gt_joints:
        raise ValueError("pred/gt playback seq or joints mismatch")

    filtered_records, filter_report = filter_skeleton_playback_records(
        pred_records,
        joint_names,
        min_volume_score=0.0,
        speed_mad_factor=2.0,
        min_speed_m=0.04,
        bone_length_deviation=0.25,
        gap_interp_max=10,
        median_window=11,
        temporal_sigma=3.0,
    )

    if args.filtered_playback is not None:
        from delivery_keypoints import export_skeleton_playback

        export_skeleton_playback(
            filtered_records,
            args.filtered_playback,
            source="Stage3 test pred + strong 3D temporal filter",
            joint_names=joint_names,
        )

    pred_traj = records_to_traj(filtered_records, joint_names)
    gt_traj = records_to_traj(gt_records, joint_names)
    n_frames = pred_traj.shape[0]
    if args.max_frames > 0:
        n_frames = min(n_frames, int(args.max_frames))
        pred_traj = pred_traj[:n_frames]
        gt_traj = gt_traj[:n_frames]
        seqs = seqs[:n_frames]

    pred_display = world_to_display(pred_traj)
    gt_display = world_to_display(gt_traj)
    center, span = compute_bounds([pred_display, gt_display])
    edges = default_edges(joint_names)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_w = args.panel_width * 2
    out_h = args.panel_height
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(args.fps),
        (out_w, out_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open {args.output}")

    for i in range(n_frames):
        left = draw_panel(
            pred_traj[i],
            center=center,
            span=span,
            yaw_deg=args.yaw_deg,
            pitch_deg=args.pitch_deg,
            width=args.panel_width,
            height=args.panel_height,
            edges=edges,
            edge_color=PRED_EDGE,
            joint_color=PRED_JOINT,
            title="Stage3 + strong filter",
            seq=seqs[i],
        )
        right = draw_panel(
            gt_traj[i],
            center=center,
            span=span,
            yaw_deg=args.yaw_deg,
            pitch_deg=args.pitch_deg,
            width=args.panel_width,
            height=args.panel_height,
            edges=edges,
            edge_color=GT_EDGE,
            joint_color=GT_JOINT,
            title="GT pre_limb",
            seq=seqs[i],
        )
        cv2.line(left, (args.panel_width - 1, 0), (args.panel_width - 1, out_h), (80, 80, 80), 2)
        writer.write(np.hstack([left, right]))
        if (i + 1) % 500 == 0 or i + 1 == n_frames:
            print(f"rendered {i + 1}/{n_frames}", flush=True)
    writer.release()

    raw_traj = records_to_traj(pred_records, joint_names)
    report = {
        "frames": n_frames,
        "filter_report": filter_report,
        "mpjpe_mm_raw": mpjpe_mm(raw_traj[:n_frames], gt_traj, joint_names),
        "mpjpe_mm_filtered": mpjpe_mm(pred_traj, gt_traj, joint_names),
        "output": str(args.output.resolve()),
        "filtered_playback": str(args.filtered_playback.resolve()) if args.filtered_playback else None,
        "layout": "left=filtered pred, right=GT",
    }
    report_path = args.report or args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
