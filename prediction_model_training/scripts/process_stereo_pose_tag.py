#!/usr/bin/env python3
"""Run EgoRear-style stereo 2-D pose and AprilTag detection on two videos.

This script is intended to run inside the gaoweijian EgoRear_w_hand checkout.
It preserves a stage-1 prediction for every decoded frame and uses the trained
stage-2 stereo refiner whenever CAM_A/C device timestamps form a valid pair.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from pupil_apriltags import Detector
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egorear_sim2d.refinement import (  # noqa: E402
    HeadBCHeatmapRefinementNet,
    load_refiner_state,
    load_stage1_model,
)


CAMERAS = ("CAM_A", "CAM_C")
SIDES = {"CAM_A": "left", "CAM_C": "right"}
JOINT_EDGES = (
    ("LeftFoot", "LeftUpLeg"),
    ("RightFoot", "RightUpLeg"),
    ("LeftUpLeg", "Spine"),
    ("RightUpLeg", "Spine"),
    ("Spine", "Spine2"),
    ("Spine2", "LeftArm"),
    ("Spine2", "RightArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
)
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-video", type=Path, required=True)
    parser.add_argument("--right-video", type=Path, required=True)
    parser.add_argument("--timestamps", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--fps",
        type=float,
        default=50.0,
        help="source/output frame rate; raw H.265 containers do not carry reliable FPS metadata",
    )
    parser.add_argument("--pair-threshold-ms", type=float, default=10.0)
    parser.add_argument("--model-width", type=int, default=456)
    parser.add_argument("--model-height", type=int, default=256)
    parser.add_argument("--tag-decimate", type=float, default=1.5)
    parser.add_argument("--tag-threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--no-annotated-video", action="store_true")
    return parser.parse_args()


def load_timestamps(path: Path) -> dict[str, list[dict[str, int]]]:
    by_camera: dict[str, list[dict[str, int]]] = {camera: [] for camera in CAMERAS}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            camera = row["camera"]
            if camera not in by_camera:
                continue
            by_camera[camera].append(
                {
                    "sequence": int(row["sequence"]),
                    "device_timestamp_us": int(row["device_timestamp_us"]),
                }
            )
    return by_camera


def pair_timestamps(
    timestamps: dict[str, list[dict[str, int]]],
    threshold_us: int,
) -> list[tuple[int, int, int]]:
    left = timestamps["CAM_A"]
    right = timestamps["CAM_C"]
    pairs: list[tuple[int, int, int]] = []
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        delta_us = (
            left[left_index]["device_timestamp_us"]
            - right[right_index]["device_timestamp_us"]
        )
        if abs(delta_us) <= threshold_us:
            pairs.append((left_index, right_index, delta_us))
            left_index += 1
            right_index += 1
        elif delta_us < 0:
            left_index += 1
        else:
            right_index += 1
    return pairs


def video_properties(path: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    properties = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "container_reported_fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "container_reported_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return properties


def preprocess(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


def decode_heatmaps(
    heatmaps: np.ndarray,
    source_width: int,
    source_height: int,
) -> np.ndarray:
    count, joints, heatmap_height, heatmap_width = heatmaps.shape
    flat = heatmaps.reshape(count, joints, -1)
    indices = flat.argmax(axis=-1)
    confidence = flat.max(axis=-1)
    x = (
        (indices % heatmap_width + 0.5)
        * float(source_width)
        / float(heatmap_width)
    )
    y = (
        (indices // heatmap_width + 0.5)
        * float(source_height)
        / float(heatmap_height)
    )
    return np.stack((x, y, confidence), axis=-1).astype(np.float32)


def make_tag_detector(args: argparse.Namespace) -> Detector:
    return Detector(
        families="tag36h11",
        nthreads=max(1, int(args.tag_threads)),
        quad_decimate=max(1.0, float(args.tag_decimate)),
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )


def detect_tags(detector: Detector, frame: np.ndarray) -> list[dict[str, object]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = []
    for detection in detector.detect(gray, estimate_tag_pose=False):
        tag_id = int(detection.tag_id)
        if not 0 <= tag_id <= 5:
            continue
        detections.append(
            {
                "tag_id": tag_id,
                "center": np.asarray(detection.center, dtype=np.float32).tolist(),
                "corners": np.asarray(detection.corners, dtype=np.float32).tolist(),
                "decision_margin": float(detection.decision_margin),
                "hamming": int(detection.hamming),
            }
        )
    return detections


def infer_stage1_and_tags(
    *,
    camera: str,
    video_path: Path,
    model,
    device: torch.device,
    detector: Detector,
    args: argparse.Namespace,
    source_width: int,
    source_height: int,
) -> tuple[np.ndarray, list[list[dict[str, object]]]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    all_coordinates: list[np.ndarray] = []
    all_tags: list[list[dict[str, object]]] = []
    batch: list[np.ndarray] = []
    frame_index = 0
    started = time.perf_counter()

    def flush_batch() -> None:
        if not batch:
            return
        tensor = torch.from_numpy(np.asarray(batch)).to(device).float().unsqueeze(1)
        with torch.inference_mode():
            heatmaps = model(tensor, "head")["head"][:, 0].detach().cpu().numpy()
        all_coordinates.extend(
            decode_heatmaps(heatmaps, source_width, source_height)
        )
        batch.clear()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        batch.append(preprocess(frame, args.model_width, args.model_height))
        all_tags.append(detect_tags(detector, frame))
        frame_index += 1
        if len(batch) >= args.batch_size:
            flush_batch()
            print(
                json.dumps(
                    {
                        "stage": "stage1_and_tags",
                        "camera": camera,
                        "processed": frame_index,
                        "seconds": round(time.perf_counter() - started, 2),
                    }
                ),
                flush=True,
            )
    flush_batch()
    cap.release()
    return np.asarray(all_coordinates, dtype=np.float32), all_tags


def advance_to(cap: cv2.VideoCapture, current: int, target: int):
    frame = None
    while current < target:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Video ended before paired frame {target}")
        current += 1
    return current, frame


def infer_stage2_pairs(
    *,
    pairs: list[tuple[int, int, int]],
    left_video: Path,
    right_video: Path,
    refiner,
    device: torch.device,
    args: argparse.Namespace,
    source_width: int,
    source_height: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    caps = {
        "CAM_A": cv2.VideoCapture(str(left_video)),
        "CAM_C": cv2.VideoCapture(str(right_video)),
    }
    if not all(cap.isOpened() for cap in caps.values()):
        raise RuntimeError("Could not reopen videos for stage-2 refinement")
    current = {"CAM_A": -1, "CAM_C": -1}
    batch_images: list[list[np.ndarray]] = []
    batch_pairs: list[tuple[int, int, int]] = []
    refined_by_camera: dict[str, dict[int, np.ndarray]] = {
        camera: {} for camera in CAMERAS
    }

    def flush_batch() -> None:
        if not batch_images:
            return
        tensor = torch.from_numpy(np.asarray(batch_images)).to(device).float()
        with torch.inference_mode():
            heatmaps = refiner(tensor)["refined"].detach().cpu().numpy()
        for view_index, camera in enumerate(CAMERAS):
            coordinates = decode_heatmaps(
                heatmaps[:, view_index], source_width, source_height
            )
            pair_position = 0 if camera == "CAM_A" else 1
            for row_index, pair in enumerate(batch_pairs):
                refined_by_camera[camera][pair[pair_position]] = coordinates[row_index]
        batch_images.clear()
        batch_pairs.clear()

    for pair_number, pair in enumerate(pairs):
        left_index, right_index, _ = pair
        current["CAM_A"], left_frame = advance_to(
            caps["CAM_A"], current["CAM_A"], left_index
        )
        current["CAM_C"], right_frame = advance_to(
            caps["CAM_C"], current["CAM_C"], right_index
        )
        batch_images.append(
            [
                preprocess(left_frame, args.model_width, args.model_height),
                preprocess(right_frame, args.model_width, args.model_height),
            ]
        )
        batch_pairs.append(pair)
        if len(batch_images) >= args.batch_size:
            flush_batch()
            print(
                json.dumps(
                    {
                        "stage": "stage2_refinement",
                        "processed_pairs": pair_number + 1,
                        "total_pairs": len(pairs),
                    }
                ),
                flush=True,
            )
    flush_batch()
    for cap in caps.values():
        cap.release()
    return refined_by_camera["CAM_A"], refined_by_camera["CAM_C"]


def write_pose_csv(
    path: Path,
    *,
    joint_names: list[str],
    stage1: dict[str, np.ndarray],
    stage2: dict[str, dict[int, np.ndarray]],
    timestamps: dict[str, list[dict[str, int]]],
    pairs: list[tuple[int, int, int]],
) -> None:
    pair_maps: dict[str, dict[int, tuple[int, int]]] = {
        "CAM_A": {
            left: (right, delta) for left, right, delta in pairs
        },
        "CAM_C": {
            right: (left, delta) for left, right, delta in pairs
        },
    }
    fields = [
        "camera",
        "side",
        "frame_index",
        "sequence",
        "device_timestamp_us",
        "paired_frame_index",
        "stereo_delta_us_CAM_A_minus_CAM_C",
        "source_stage",
        "joint",
        "x",
        "y",
        "confidence",
        "stage1_x",
        "stage1_y",
        "stage1_confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for camera in CAMERAS:
            for frame_index, initial in enumerate(stage1[camera]):
                paired = pair_maps[camera].get(frame_index)
                refined = stage2[camera].get(frame_index)
                selected = refined if refined is not None else initial
                source_stage = "stage2_stereo_refined" if refined is not None else "stage1"
                timestamp = (
                    timestamps[camera][frame_index]
                    if frame_index < len(timestamps[camera])
                    else {"sequence": "", "device_timestamp_us": ""}
                )
                for joint_index, joint_name in enumerate(joint_names):
                    writer.writerow(
                        {
                            "camera": camera,
                            "side": SIDES[camera],
                            "frame_index": frame_index,
                            "sequence": timestamp["sequence"],
                            "device_timestamp_us": timestamp["device_timestamp_us"],
                            "paired_frame_index": paired[0] if paired else "",
                            "stereo_delta_us_CAM_A_minus_CAM_C": paired[1] if paired else "",
                            "source_stage": source_stage,
                            "joint": joint_name,
                            "x": f"{selected[joint_index, 0]:.6f}",
                            "y": f"{selected[joint_index, 1]:.6f}",
                            "confidence": f"{selected[joint_index, 2]:.8f}",
                            "stage1_x": f"{initial[joint_index, 0]:.6f}",
                            "stage1_y": f"{initial[joint_index, 1]:.6f}",
                            "stage1_confidence": f"{initial[joint_index, 2]:.8f}",
                        }
                    )


def write_tag_csv(
    path: Path,
    *,
    tags: dict[str, list[list[dict[str, object]]]],
    timestamps: dict[str, list[dict[str, int]]],
) -> None:
    fields = [
        "camera",
        "side",
        "frame_index",
        "sequence",
        "device_timestamp_us",
        "detected",
        "tag_id",
        "center_x",
        "center_y",
        "corner0_x",
        "corner0_y",
        "corner1_x",
        "corner1_y",
        "corner2_x",
        "corner2_y",
        "corner3_x",
        "corner3_y",
        "decision_margin",
        "hamming",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for camera in CAMERAS:
            for frame_index, frame_detections in enumerate(tags[camera]):
                timestamp = (
                    timestamps[camera][frame_index]
                    if frame_index < len(timestamps[camera])
                    else {"sequence": "", "device_timestamp_us": ""}
                )
                detections = frame_detections or [None]
                for detection in detections:
                    row = {
                        "camera": camera,
                        "side": SIDES[camera],
                        "frame_index": frame_index,
                        "sequence": timestamp["sequence"],
                        "device_timestamp_us": timestamp["device_timestamp_us"],
                        "detected": int(detection is not None),
                    }
                    if detection is not None:
                        row.update(
                            {
                                "tag_id": detection["tag_id"],
                                "center_x": f"{detection['center'][0]:.6f}",
                                "center_y": f"{detection['center'][1]:.6f}",
                                "decision_margin": f"{detection['decision_margin']:.6f}",
                                "hamming": detection["hamming"],
                            }
                        )
                        for corner_index, corner in enumerate(detection["corners"]):
                            row[f"corner{corner_index}_x"] = f"{corner[0]:.6f}"
                            row[f"corner{corner_index}_y"] = f"{corner[1]:.6f}"
                    writer.writerow(row)


def draw_pose(
    frame: np.ndarray,
    coordinates: np.ndarray,
    joint_names: list[str],
    refined: bool,
) -> None:
    index = {name: number for number, name in enumerate(joint_names)}
    color = (255, 140, 20) if refined else (255, 210, 80)
    for first, second in JOINT_EDGES:
        if first in index and second in index:
            p1 = tuple(np.rint(coordinates[index[first], :2]).astype(int))
            p2 = tuple(np.rint(coordinates[index[second], :2]).astype(int))
            cv2.line(frame, p1, p2, color, 4, cv2.LINE_AA)
    for joint_index, joint_name in enumerate(joint_names):
        point = tuple(np.rint(coordinates[joint_index, :2]).astype(int))
        cv2.circle(frame, point, 7, color, -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            joint_name,
            (point[0] + 8, point[1] - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def draw_tags(frame: np.ndarray, detections: list[dict[str, object]]) -> None:
    for detection in detections:
        corners = np.rint(np.asarray(detection["corners"])).astype(np.int32)
        cv2.polylines(frame, [corners], True, (0, 255, 80), 4, cv2.LINE_AA)
        center = tuple(np.rint(detection["center"]).astype(int))
        cv2.circle(frame, center, 6, (0, 255, 80), -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"36h11 ID {detection['tag_id']}",
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 80),
            2,
            cv2.LINE_AA,
        )


def render_annotated_video(
    *,
    camera: str,
    source: Path,
    destination: Path,
    fps: float,
    width: int,
    height: int,
    stage1: np.ndarray,
    stage2: dict[int, np.ndarray],
    tags: list[list[dict[str, object]]],
    joint_names: list[str],
) -> None:
    command = [
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
    ]
    encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
    cap = cv2.VideoCapture(str(source))
    frame_index = 0
    try:
        while frame_index < len(stage1):
            ok, frame = cap.read()
            if not ok:
                break
            refined = stage2.get(frame_index)
            coordinates = refined if refined is not None else stage1[frame_index]
            draw_pose(frame, coordinates, joint_names, refined is not None)
            draw_tags(frame, tags[frame_index])
            cv2.putText(
                frame,
                f"{camera} {SIDES[camera]} | frame {frame_index:06d} | "
                f"{'EgoRear stage2' if refined is not None else 'EgoRear stage1'}",
                (24, 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
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
        raise RuntimeError(f"ffmpeg failed for {destination} with code {return_code}")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        args.device
        if args.device == "cpu" or torch.cuda.is_available()
        else "cpu"
    )
    properties = {
        "CAM_A": video_properties(args.left_video),
        "CAM_C": video_properties(args.right_video),
    }
    width = int(properties["CAM_A"]["width"])
    height = int(properties["CAM_A"]["height"])
    if any(
        int(properties[camera]["width"]) != width
        or int(properties[camera]["height"]) != height
        for camera in CAMERAS
    ):
        raise ValueError(f"Stereo resolution mismatch: {properties}")

    stage1_checkpoint = torch.load(
        args.stage1_checkpoint, map_location="cpu", weights_only=False
    )
    stage1_config = stage1_checkpoint.get("config", {})
    joint_names = [
        str(name) for name in stage1_config.get("head_joint_names", [])
    ]
    if not joint_names:
        raise ValueError("Stage-1 checkpoint does not contain head_joint_names")
    stage1_model = load_stage1_model(
        args.stage1_checkpoint,
        base_channels=int(stage1_config.get("base_channels", 64)),
        num_head_heatmaps=len(joint_names),
    ).to(device).eval()

    stage2_checkpoint = torch.load(
        args.stage2_checkpoint, map_location="cpu", weights_only=False
    )
    stage2_config = stage2_checkpoint.get("config", {})
    refiner = HeadBCHeatmapRefinementNet(
        stage1_model,
        num_joints=len(joint_names),
        heatmap_size=(114, 64),
        base_channels=int(stage2_config.get("base_channels", 64)),
        query_dim=int(stage2_config.get("query_dim", 256)),
        sampling_points=int(stage2_config.get("sampling_points", 8)),
        freeze_stage1=True,
    )
    load_refiner_state(refiner, stage2_checkpoint["refiner"])
    refiner = refiner.to(device).eval()

    timestamps = load_timestamps(args.timestamps)
    pairs = pair_timestamps(
        timestamps,
        round(float(args.pair_threshold_ms) * 1000.0),
    )
    detector = make_tag_detector(args)
    stage1: dict[str, np.ndarray] = {}
    tags: dict[str, list[list[dict[str, object]]]] = {}
    sources = {"CAM_A": args.left_video, "CAM_C": args.right_video}
    for camera in CAMERAS:
        stage1[camera], tags[camera] = infer_stage1_and_tags(
            camera=camera,
            video_path=sources[camera],
            model=stage1_model,
            device=device,
            detector=detector,
            args=args,
            source_width=width,
            source_height=height,
        )

    refined_left, refined_right = infer_stage2_pairs(
        pairs=pairs,
        left_video=args.left_video,
        right_video=args.right_video,
        refiner=refiner,
        device=device,
        args=args,
        source_width=width,
        source_height=height,
    )
    stage2 = {"CAM_A": refined_left, "CAM_C": refined_right}

    pose_csv = args.output_dir / "pose_2d.csv"
    tag_csv = args.output_dir / "apriltag_36h11_id0-5.csv"
    write_pose_csv(
        pose_csv,
        joint_names=joint_names,
        stage1=stage1,
        stage2=stage2,
        timestamps=timestamps,
        pairs=pairs,
    )
    write_tag_csv(tag_csv, tags=tags, timestamps=timestamps)

    annotated_videos = {}
    if not args.no_annotated_video:
        for camera in CAMERAS:
            destination = args.output_dir / (
                f"{SIDES[camera]}_{camera}_pose_tag_annotated.mp4"
            )
            render_annotated_video(
                camera=camera,
                source=sources[camera],
                destination=destination,
                fps=float(args.fps),
                width=width,
                height=height,
                stage1=stage1[camera],
                stage2=stage2[camera],
                tags=tags[camera],
                joint_names=joint_names,
            )
            annotated_videos[camera] = destination.name

    tag_counts = {
        camera: {
            str(tag_id): sum(
                int(detection["tag_id"] == tag_id)
                for frame_detections in tags[camera]
                for detection in frame_detections
            )
            for tag_id in range(6)
        }
        for camera in CAMERAS
    }
    absolute_deltas = np.asarray([abs(pair[2]) for pair in pairs], dtype=np.float64)
    summary = {
        "status": "complete",
        "schema": "hearwristcam_pose2d_apriltag.v1",
        "pose_model": {
            "family": "EgoRear-style ResNet18 heatmap + stereo refinement",
            "stage1_checkpoint": str(args.stage1_checkpoint),
            "stage1_epoch": int(stage1_checkpoint.get("epoch", -1)),
            "stage2_checkpoint": str(args.stage2_checkpoint),
            "stage2_epoch": int(stage2_checkpoint.get("epoch", -1)),
            "joint_names": joint_names,
            "model_input": [args.model_width, args.model_height],
        },
        "tag_detector": {
            "family": "tag36h11",
            "accepted_ids": list(range(6)),
            "black_square_m": 0.08,
            "white_outer_square_m": 0.10,
            "quad_decimate": args.tag_decimate,
        },
        "videos": {
            camera: {
                "source": str(sources[camera]),
                "properties": properties[camera],
                "annotated_fps": float(args.fps),
                "decoded_frames": int(len(stage1[camera])),
                "timestamp_rows": len(timestamps[camera]),
                "stage2_refined_frames": len(stage2[camera]),
                "tag_detections_by_id": tag_counts[camera],
                "annotated_video": annotated_videos.get(camera),
            }
            for camera in CAMERAS
        },
        "stereo_pairs": {
            "threshold_ms": args.pair_threshold_ms,
            "count": len(pairs),
            "absolute_delta_ms": {
                "min": float(absolute_deltas.min() / 1000.0) if len(pairs) else None,
                "median": float(np.median(absolute_deltas) / 1000.0) if len(pairs) else None,
                "max": float(absolute_deltas.max() / 1000.0) if len(pairs) else None,
            },
        },
        "outputs": {
            "pose_csv": pose_csv.name,
            "tag_csv": tag_csv.name,
            "annotated_videos": annotated_videos,
        },
    }
    (args.output_dir / "processing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
