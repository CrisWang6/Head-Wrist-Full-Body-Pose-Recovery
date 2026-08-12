#!/usr/bin/env python3
"""Render a skeleton-only MP4 from a fixed yaw/pitch display viewpoint."""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import cv2
import numpy as np

MISS = -32768


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            r"C:\Users\hand\Desktop\双外部双目\0806\无\multiview_3d_results\full"
            r"\skeleton_playback.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"C:\Users\hand\Desktop\双外部双目\0806\无\multiview_3d_results\full"
            r"\visualization_fixed\skeleton_yaw100.mp4"
        ),
    )
    parser.add_argument("--yaw-deg", type=float, default=100.0)
    parser.add_argument("--pitch-deg", type=float, default=18.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def load_playback(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def decode_xyz(payload: dict) -> np.ndarray:
    """Return float32 array [F, J, 3] in world meters; NaN for missing."""
    raw = base64.b64decode(payload["xyz_i16_b64"])
    i16 = np.frombuffer(raw, dtype="<i2")
    n_joints = len(payload["joints"])
    n_frames = int(payload["frame_count"])
    expected = n_frames * n_joints * 3
    if i16.size != expected:
        raise RuntimeError(f"xyz size mismatch: got {i16.size}, expected {expected}")
    xyz = i16.astype(np.float32).reshape(n_frames, n_joints, 3)
    missing = xyz == MISS
    xyz = xyz / 1000.0
    xyz[missing] = np.nan
    return xyz


def world_to_display(xyz: np.ndarray) -> np.ndarray:
    """Mocap Y-up -> viewer Z-up: (x,y,z) -> (x,z,y)."""
    out = np.empty_like(xyz)
    out[..., 0] = xyz[..., 0]
    out[..., 1] = xyz[..., 2]
    out[..., 2] = xyz[..., 1]
    return out


def compute_bounds(display: np.ndarray) -> tuple[np.ndarray, float]:
    valid = display.reshape(-1, 3)
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
    """Orthographic-ish projection; yaw about display +Z (up), then pitch."""
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
    out = np.stack([sx, sy, y2], axis=1)
    return out


def default_edges(joints: list[str]) -> list[list[int]]:
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
        ("left_ankle", "left_toe"),
        ("right_ankle", "right_toe"),
    ]
    index = {name: i for i, name in enumerate(joints)}
    return [[index[a], index[b]] for a, b in pairs if a in index and b in index]


def main() -> None:
    args = parse_args()
    payload = load_playback(args.data)
    if "frame_count" not in payload:
        payload["frame_count"] = len(payload["seqs"])
    world = decode_xyz(payload)
    display = world_to_display(world)
    if args.max_frames is not None:
        display = display[: args.max_frames]
        seqs = payload["seqs"][: args.max_frames]
    else:
        seqs = payload["seqs"]
    joints = payload["joints"]
    edges = payload.get("edges") or default_edges(joints)
    ankle_idx = {name: joints.index(name) for name in ("left_ankle", "right_ankle") if name in joints}

    center, span = compute_bounds(display)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(args.output),
        fourcc,
        float(args.fps),
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {args.output}")

    n = display.shape[0]
    for i in range(n):
        img = np.full((args.height, args.width, 3), 18, dtype=np.uint8)
        projected = project_points(
            display[i],
            center,
            args.yaw_deg,
            args.pitch_deg,
            args.width,
            args.height,
            span,
        )
        for a, b in edges:
            if not (np.isfinite(display[i, a]).all() and np.isfinite(display[i, b]).all()):
                continue
            pa, pb = projected[a], projected[b]
            if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                continue
            cv2.line(
                img,
                (int(round(pa[0])), int(round(pa[1]))),
                (int(round(pb[0])), int(round(pb[1]))),
                (0, 200, 220),
                3,
                cv2.LINE_AA,
            )
        for j, p in enumerate(projected):
            if not (np.isfinite(display[i, j]).all() and np.isfinite(p).all()):
                continue
            radius = 5 if j in ankle_idx.values() else 3
            color = (255, 255, 255) if j in ankle_idx.values() else (240, 230, 80)
            cv2.circle(
                img,
                (int(round(p[0])), int(round(p[1]))),
                radius,
                color,
                -1,
                cv2.LINE_AA,
            )
        cv2.putText(
            img,
            f"seq {seqs[i]}  yaw {args.yaw_deg:.0f}  pitch {args.pitch_deg:.0f}",
            (24, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
        writer.write(img)
        if (i + 1) % 500 == 0 or i + 1 == n:
            print(f"rendered {i + 1}/{n}", flush=True)

    writer.release()
    print(
        json.dumps(
            {
                "frames": n,
                "yaw_deg": args.yaw_deg,
                "pitch_deg": args.pitch_deg,
                "fps": args.fps,
                "output": str(args.output),
                "size_bytes": args.output.stat().st_size,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
