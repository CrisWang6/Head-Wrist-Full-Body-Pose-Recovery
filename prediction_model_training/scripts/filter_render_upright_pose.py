#!/usr/bin/env python3
"""Rotate recordings upright, smooth tag-guided joints, and render results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


CAMERAS = ("CAM_A", "CAM_C")
SIDES = {"CAM_A": "left", "CAM_C": "right"}
KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
SKELETON_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)
TORSO_AND_HEAD = set(range(7)) | {11, 12}
MID_LIMBS = {7, 8, 13, 14}
END_LIMBS = {9, 10, 15, 16}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-video", type=Path, required=True)
    parser.add_argument("--right-video", type=Path, required=True)
    parser.add_argument("--pose-csv", type=Path, required=True)
    parser.add_argument("--tag-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--median-window", type=int, default=5)
    parser.add_argument("--no-annotated-video", action="store_true")
    return parser.parse_args()


def read_pose(
    path: Path,
) -> tuple[
    dict[str, dict[int, dict[str, dict[str, str]]]],
    dict[str, dict[int, dict[str, str]]],
]:
    pose: dict[str, dict[int, dict[str, dict[str, str]]]] = {
        camera: {} for camera in CAMERAS
    }
    meta: dict[str, dict[int, dict[str, str]]] = {
        camera: {} for camera in CAMERAS
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            camera = row["camera"]
            if camera not in pose:
                continue
            frame_index = int(row["frame_index"])
            meta[camera].setdefault(frame_index, row)
            if row["status"] == "ok" and row["joint"]:
                pose[camera].setdefault(frame_index, {})[row["joint"]] = row
    return pose, meta


def read_tags(path: Path) -> dict[str, dict[int, list[dict[str, object]]]]:
    tags: dict[str, dict[int, list[dict[str, object]]]] = {
        camera: {} for camera in CAMERAS
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["camera"] not in tags or row["detected"] != "1":
                continue
            frame_index = int(row["frame_index"])
            tags[row["camera"]].setdefault(frame_index, []).append(
                {
                    "tag_id": int(row["tag_id"]),
                    "center": np.asarray(
                        [float(row["center_x"]), float(row["center_y"])],
                        dtype=np.float32,
                    ),
                    "corners": np.asarray(
                        [
                            [
                                float(row[f"corner{corner}_x"]),
                                float(row[f"corner{corner}_y"]),
                            ]
                            for corner in range(4)
                        ],
                        dtype=np.float32,
                    ),
                }
            )
    return tags


def median_filter(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    radius = window // 2
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    return np.asarray(
        [
            np.median(padded[index : index + window], axis=0)
            for index in range(len(values))
        ],
        dtype=np.float32,
    )


def step_limit(joint_index: int, confidence: float) -> float:
    if joint_index in TORSO_AND_HEAD:
        base = 30.0
    elif joint_index in MID_LIMBS:
        base = 40.0
    else:
        base = 52.0
    if confidence < 0.20:
        base *= 0.55
    return base


def smooth_joint(
    raw: np.ndarray,
    confidence: np.ndarray,
    *,
    joint_index: int,
    median_window: int,
) -> np.ndarray:
    measurements = median_filter(raw, median_window)
    output = np.empty_like(measurements)
    output[0] = measurements[0]
    for frame_index in range(1, len(measurements)):
        previous = output[frame_index - 1]
        measurement = measurements[frame_index]
        residual = measurement - previous
        distance = float(np.linalg.norm(residual))
        score = float(confidence[frame_index])
        if score < 0.06:
            alpha = 0.03
        else:
            confidence_scale = float(np.clip((score - 0.05) / 0.80, 0.2, 1.0))
            alpha = float(
                np.clip((0.18 + 0.0042 * distance) * confidence_scale, 0.12, 0.68)
            )
        candidate = previous + alpha * residual
        delta = candidate - previous
        delta_norm = float(np.linalg.norm(delta))
        limit = step_limit(joint_index, score)
        if delta_norm > limit:
            candidate = previous + delta * (limit / delta_norm)
        output[frame_index] = candidate
    return output


def build_filtered_pose(
    pose: dict[str, dict[int, dict[str, dict[str, str]]]],
    *,
    width: int,
    height: int,
    median_window: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    raw_upright: dict[str, np.ndarray] = {}
    confidence: dict[str, np.ndarray] = {}
    filtered: dict[str, np.ndarray] = {}
    for camera in CAMERAS:
        frame_count = max(pose[camera]) + 1
        raw = np.full(
            (frame_count, len(KEYPOINT_NAMES), 2),
            np.nan,
            dtype=np.float32,
        )
        scores = np.zeros(
            (frame_count, len(KEYPOINT_NAMES)),
            dtype=np.float32,
        )
        for frame_index, joints in pose[camera].items():
            for joint_index, joint_name in enumerate(KEYPOINT_NAMES):
                row = joints.get(joint_name)
                if row is None:
                    continue
                raw[frame_index, joint_index] = [
                    float(width - 1) - float(row["x"]),
                    float(height - 1) - float(row["y"]),
                ]
                scores[frame_index, joint_index] = float(row["confidence"])

        smoothed = np.empty_like(raw)
        for joint_index in range(len(KEYPOINT_NAMES)):
            values = raw[:, joint_index]
            valid = np.all(np.isfinite(values), axis=1)
            if not np.all(valid):
                valid_indices = np.flatnonzero(valid)
                if len(valid_indices) == 0:
                    smoothed[:, joint_index] = values
                    continue
                for coordinate in range(2):
                    values[:, coordinate] = np.interp(
                        np.arange(frame_count),
                        valid_indices,
                        values[valid_indices, coordinate],
                    )
            smoothed[:, joint_index] = smooth_joint(
                values,
                scores[:, joint_index],
                joint_index=joint_index,
                median_window=median_window,
            )
        raw_upright[camera] = raw
        confidence[camera] = scores
        filtered[camera] = smoothed
    return raw_upright, confidence, filtered


def displacement_statistics(points: np.ndarray) -> dict[str, float]:
    displacement = np.linalg.norm(np.diff(points, axis=0), axis=-1).reshape(-1)
    finite = displacement[np.isfinite(displacement)]
    return {
        "count": int(len(finite)),
        "median_px": float(np.median(finite)),
        "p90_px": float(np.percentile(finite, 90)),
        "p95_px": float(np.percentile(finite, 95)),
        "p99_px": float(np.percentile(finite, 99)),
        "max_px": float(np.max(finite)),
    }


def write_filtered_csv(
    path: Path,
    *,
    raw: dict[str, np.ndarray],
    confidence: dict[str, np.ndarray],
    filtered: dict[str, np.ndarray],
    meta: dict[str, dict[int, dict[str, str]]],
) -> None:
    fields = [
        "camera",
        "side",
        "frame_index",
        "sequence",
        "device_timestamp_us",
        "association",
        "joint",
        "confidence",
        "raw_upright_x",
        "raw_upright_y",
        "filtered_upright_x",
        "filtered_upright_y",
        "raw_interframe_delta_px",
        "filtered_interframe_delta_px",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for camera in CAMERAS:
            for frame_index in range(len(filtered[camera])):
                row_meta = meta[camera][frame_index]
                for joint_index, joint_name in enumerate(KEYPOINT_NAMES):
                    raw_delta = (
                        0.0
                        if frame_index == 0
                        else float(
                            np.linalg.norm(
                                raw[camera][frame_index, joint_index]
                                - raw[camera][frame_index - 1, joint_index]
                            )
                        )
                    )
                    filtered_delta = (
                        0.0
                        if frame_index == 0
                        else float(
                            np.linalg.norm(
                                filtered[camera][frame_index, joint_index]
                                - filtered[camera][frame_index - 1, joint_index]
                            )
                        )
                    )
                    writer.writerow(
                        {
                            "camera": camera,
                            "side": SIDES[camera],
                            "frame_index": frame_index,
                            "sequence": row_meta["sequence"],
                            "device_timestamp_us": row_meta["device_timestamp_us"],
                            "association": row_meta["association"],
                            "joint": joint_name,
                            "confidence": f"{confidence[camera][frame_index, joint_index]:.8f}",
                            "raw_upright_x": f"{raw[camera][frame_index, joint_index, 0]:.6f}",
                            "raw_upright_y": f"{raw[camera][frame_index, joint_index, 1]:.6f}",
                            "filtered_upright_x": (
                                f"{filtered[camera][frame_index, joint_index, 0]:.6f}"
                            ),
                            "filtered_upright_y": (
                                f"{filtered[camera][frame_index, joint_index, 1]:.6f}"
                            ),
                            "raw_interframe_delta_px": f"{raw_delta:.6f}",
                            "filtered_interframe_delta_px": f"{filtered_delta:.6f}",
                        }
                    )


def rotate_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    output = np.asarray(points, dtype=np.float32).copy()
    output[..., 0] = float(width - 1) - output[..., 0]
    output[..., 1] = float(height - 1) - output[..., 1]
    return output


def draw_tags(
    frame: np.ndarray,
    detections: list[dict[str, object]],
    width: int,
    height: int,
) -> None:
    for detection in detections:
        corners = np.rint(
            rotate_points(detection["corners"], width, height)
        ).astype(np.int32)
        center = tuple(
            np.rint(
                rotate_points(detection["center"], width, height)
            ).astype(int)
        )
        cv2.polylines(frame, [corners], True, (0, 255, 80), 4, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"TAG {detection['tag_id']}",
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 255, 80),
            2,
            cv2.LINE_AA,
        )


def draw_pose(
    frame: np.ndarray,
    points: np.ndarray,
    confidence: np.ndarray,
    confidence_threshold: float = 0.15,
) -> None:
    for first, second in SKELETON_EDGES:
        if (
            confidence[first] >= confidence_threshold
            and confidence[second] >= confidence_threshold
        ):
            cv2.line(
                frame,
                tuple(np.rint(points[first]).astype(int)),
                tuple(np.rint(points[second]).astype(int)),
                (255, 160, 20),
                5,
                cv2.LINE_AA,
            )
    for joint_index, point in enumerate(points):
        if confidence[joint_index] >= confidence_threshold:
            cv2.circle(
                frame,
                tuple(np.rint(point).astype(int)),
                7,
                (255, 160, 20),
                -1,
                cv2.LINE_AA,
            )


def render_video(
    *,
    camera: str,
    source: Path,
    destination: Path,
    filtered: np.ndarray,
    confidence: np.ndarray,
    meta: dict[int, dict[str, str]],
    tags: dict[int, list[dict[str, object]]],
    width: int,
    height: int,
    fps: float,
) -> None:
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        stdin=subprocess.PIPE,
    )
    cap = cv2.VideoCapture(str(source))
    frame_index = 0
    try:
        while frame_index < len(filtered):
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            draw_pose(frame, filtered[frame_index], confidence[frame_index])
            draw_tags(
                frame,
                tags.get(frame_index, []),
                width,
                height,
            )
            cv2.putText(
                frame,
                f"{camera} {SIDES[camera]} | upright | filtered | frame {frame_index:06d}",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"tag-guided target | {meta[frame_index]['association']}",
                (24, 78),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 160, 20),
                2,
                cv2.LINE_AA,
            )
            assert encoder.stdin is not None
            encoder.stdin.write(frame.tobytes())
            frame_index += 1
    finally:
        cap.release()
        if encoder.stdin is not None:
            encoder.stdin.close()
        return_code = encoder.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed for {destination}: {return_code}")


def probe_frame_size(source: Path) -> tuple[int, int]:
    """Get dimensions from a decoded frame, also for raw H.265 without metadata."""
    cap = cv2.VideoCapture(str(source))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not decode the first video frame: {source}")
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid decoded frame size {width}x{height}: {source}")
    return width, height


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pose, meta = read_pose(args.pose_csv)
    tags = read_tags(args.tag_csv)
    sources = {
        "CAM_A": args.left_video,
        "CAM_C": args.right_video,
    }
    width, height = probe_frame_size(args.left_video)

    raw, confidence, filtered = build_filtered_pose(
        pose,
        width=width,
        height=height,
        median_window=args.median_window,
    )
    output_csv = args.output_dir / "tag_guided_pose_2d_filtered_upright.csv"
    write_filtered_csv(
        output_csv,
        raw=raw,
        confidence=confidence,
        filtered=filtered,
        meta=meta,
    )

    annotated = {}
    if not args.no_annotated_video:
        for camera in CAMERAS:
            destination = args.output_dir / (
                f"{SIDES[camera]}_{camera}_tag_guided_filtered_upright.mp4"
            )
            render_video(
                camera=camera,
                source=sources[camera],
                destination=destination,
                filtered=filtered[camera],
                confidence=confidence[camera],
                meta=meta[camera],
                tags=tags[camera],
                width=width,
                height=height,
                fps=args.fps,
            )
            annotated[camera] = destination.name

    summary = {
        "status": "complete",
        "schema": "hearwristcam.filtered_upright_pose.v1",
        "rotation_degrees": 180,
        "filter": {
            "median_window_frames": args.median_window,
            "adaptive_ema_alpha_range": [0.12, 0.68],
            "max_step_px_per_frame": {
                "head_and_torso": 30.0,
                "elbows_and_knees": 40.0,
                "wrists_and_ankles": 52.0,
                "low_confidence_scale": 0.55,
            },
        },
        "motion_statistics": {
            camera: {
                "raw": displacement_statistics(raw[camera]),
                "filtered": displacement_statistics(filtered[camera]),
            }
            for camera in CAMERAS
        },
        "outputs": {
            "filtered_pose_csv": output_csv.name,
            "annotated_videos": annotated,
        },
    }
    (args.output_dir / "filtered_upright_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
