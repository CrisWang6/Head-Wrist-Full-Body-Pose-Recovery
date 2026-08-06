#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egorear_sim2d.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    MultiViewHeatmapDataset,
    discover_label_files,
    torch_collate,
)
from egorear_sim2d.pose3d import EgoRearPose3DNet, EgoRearStage3Pipeline
from egorear_sim2d.refinement import (
    HeadBCHeatmapRefinementNet,
    load_refiner_state,
    load_stage1_model,
)


SKELETON_EDGES = (
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
PRED_COLOR = (255, 110, 0)
GT_COLOR = (0, 210, 70)
CANVAS_SIZE = (1280, 720)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export continuous stage-3 train/test prediction videos."
    )
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--pose3d-labels", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--stage3-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--split", choices=("train", "test", "all"), default="all")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-frames-per-split",
        type=int,
        default=0,
        help="Debug limit; zero exports the complete chronological split.",
    )
    return parser.parse_args()


class FfmpegVideoWriter:
    def __init__(self, path: Path, fps: float, crf: int):
        self.path = path
        width, height = CANVAS_SIZE
        self.process = subprocess.Popen(
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
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(path),
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame: np.ndarray) -> None:
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is not available")
        self.process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}: {self.path}")


def denormalize_image(image: np.ndarray) -> np.ndarray:
    rgb = image.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def project_point(
    point: np.ndarray,
    axes: tuple[int, int],
    limits: tuple[tuple[float, float], tuple[float, float]],
    rect: tuple[int, int, int, int],
) -> tuple[int, int]:
    left, top, width, height = rect
    x_value = float(point[axes[0]])
    y_value = float(point[axes[1]])
    x_min, x_max = limits[0]
    y_min, y_max = limits[1]
    x = left + int(round((x_value - x_min) / (x_max - x_min) * width))
    y = top + int(round((y_max - y_value) / (y_max - y_min) * height))
    return x, y


def draw_pose_panel(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    title: str,
    axes: tuple[int, int],
    limits: tuple[tuple[float, float], tuple[float, float]],
    axis_labels: tuple[str, str],
    prediction: np.ndarray,
    ground_truth: np.ndarray | None,
    joint_names: list[str],
) -> None:
    left, top, width, height = rect
    cv2.rectangle(canvas, (left, top), (left + width, top + height), (75, 75, 75), 1)
    cv2.putText(
        canvas,
        title,
        (left, top - 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        axis_labels[0],
        (left + width - 20, top + height + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (175, 175, 175),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        axis_labels[1],
        (left - 18, top + 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (175, 175, 175),
        1,
        cv2.LINE_AA,
    )

    index = {name: joint_idx for joint_idx, name in enumerate(joint_names)}

    def draw_one(pose: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
        points = [project_point(point, axes, limits, rect) for point in pose]
        for start_name, end_name in SKELETON_EDGES:
            if start_name in index and end_name in index:
                cv2.line(
                    canvas,
                    points[index[start_name]],
                    points[index[end_name]],
                    color,
                    thickness,
                    cv2.LINE_AA,
                )
        for point in points:
            cv2.circle(canvas, point, 4 if thickness > 1 else 3, color, -1, cv2.LINE_AA)

    if ground_truth is not None:
        draw_one(ground_truth, GT_COLOR, 2)
    draw_one(prediction, PRED_COLOR, 3)


def render_frame(
    images: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray | None,
    joint_names: list[str],
    *,
    split_name: str,
    checkpoint_label: str,
    checkpoint_epoch: int,
    global_index: int,
    split_index: int,
    split_length: int,
) -> tuple[np.ndarray, float | None]:
    width, height = CANVAS_SIZE
    canvas = np.full((height, width, 3), 24, dtype=np.uint8)
    for view_index, camera_name in enumerate(("CAM_B", "CAM_C")):
        image = denormalize_image(images[view_index])
        top = 60 + view_index * 292
        canvas[top : top + 256, 18 : 18 + 456] = image
        cv2.putText(
            canvas,
            camera_name,
            (28, top + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    mpjpe_mm = None
    if ground_truth is not None:
        mpjpe_mm = float(
            np.linalg.norm(prediction - ground_truth, axis=-1).mean() * 1000.0
        )
    cv2.putText(
        canvas,
        f"Stage 3 | {checkpoint_label} (checkpoint epoch {checkpoint_epoch}) | {split_name}",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    status = (
        f"frame {split_index + 1}/{split_length} | source index {global_index} | "
        + (f"MPJPE {mpjpe_mm:.1f} mm" if mpjpe_mm is not None else "GT unavailable")
    )
    cv2.putText(
        canvas,
        status,
        (500, 695),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.57,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.line(canvas, (510, 58), (550, 58), PRED_COLOR, 4, cv2.LINE_AA)
    cv2.putText(
        canvas, "prediction", (560, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.53,
        (230, 230, 230), 1, cv2.LINE_AA
    )
    cv2.line(canvas, (680, 58), (720, 58), GT_COLOR, 3, cv2.LINE_AA)
    cv2.putText(
        canvas, "lifting GT", (730, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.53,
        (230, 230, 230), 1, cv2.LINE_AA
    )

    y_limits = (-1.55, 0.35)
    draw_pose_panel(
        canvas,
        (515, 105, 345, 540),
        "Front view (X-Y)",
        (0, 1),
        ((-0.8, 0.8), y_limits),
        ("X", "Y"),
        prediction,
        ground_truth,
        joint_names,
    )
    draw_pose_panel(
        canvas,
        (900, 105, 345, 540),
        "Side view (Z-Y)",
        (2, 1),
        ((-0.6, 1.35), y_limits),
        ("Z", "Y"),
        prediction,
        ground_truth,
        joint_names,
    )
    return canvas, mpjpe_mm


def main() -> int:
    import torch
    from torch.utils.data import DataLoader

    args = parse_args()
    dataset = MultiViewHeatmapDataset(
        discover_label_files(Path(args.label_root)),
        image_size=(456, 256),
        visible_only_loss=True,
    )
    pose_labels = np.load(args.pose3d_labels, allow_pickle=True)
    pose_frames = np.asarray(pose_labels["frame_indices"], dtype=np.int64)
    pose_values = np.asarray(pose_labels["pose_head_m"], dtype=np.float32)
    pose_valid = np.asarray(pose_labels["valid"], dtype=bool)
    joint_names = [str(value) for value in pose_labels["joint_names"]]
    if len(dataset) != len(pose_frames):
        raise ValueError(f"2D/3D label length mismatch: {len(dataset)} vs {len(pose_frames)}")
    dataset_frames = np.asarray(
        [
            int(dataset._load_label(label_idx)["frame_indices"][frame_idx])
            for label_idx, frame_idx in dataset.index
        ],
        dtype=np.int64,
    )
    if not np.array_equal(dataset_frames, pose_frames):
        raise ValueError("2D and 3D frame_indices are not exactly aligned")

    stage2_checkpoint = torch.load(
        args.stage2_checkpoint, map_location="cpu", weights_only=False
    )
    stage2_config = stage2_checkpoint.get("config", {})
    stage1 = load_stage1_model(
        args.stage1_checkpoint, num_head_heatmaps=len(joint_names)
    )
    stage2 = HeadBCHeatmapRefinementNet(
        stage1,
        num_joints=len(joint_names),
        heatmap_size=(114, 64),
        base_channels=int(stage2_config.get("base_channels", 64)),
        query_dim=int(stage2_config.get("query_dim", 256)),
        sampling_points=int(stage2_config.get("sampling_points", 8)),
        freeze_stage1=True,
    )
    load_refiner_state(stage2, stage2_checkpoint["refiner"])
    checkpoint = torch.load(
        args.stage3_checkpoint, map_location="cpu", weights_only=False
    )
    pose3d = EgoRearPose3DNet(num_joints=len(joint_names))
    pose3d.load_state_dict(checkpoint["pose3d"], strict=True)
    pipeline = EgoRearStage3Pipeline(stage2, pose3d)
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    pipeline = pipeline.to(device).eval()
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        pipeline = torch.nn.DataParallel(pipeline)

    split_at = max(
        1, min(len(dataset) - 1, int(round(len(dataset) * args.train_ratio)))
    )
    requested_splits = ("train", "test") if args.split == "all" else (args.split,)
    split_ranges = {
        "train": (0, split_at),
        "test": (split_at, len(dataset)),
    }
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    writers = {}
    statistics = {}
    for split_name in requested_splits:
        path = output_dir / f"{args.checkpoint_label}_{split_name}.mp4"
        writers[split_name] = FfmpegVideoWriter(path, args.fps, args.crf)
        start, end = split_ranges[split_name]
        statistics[split_name] = {
            "path": str(path),
            "source_frames": end - start,
            "written_frames": 0,
            "valid_gt_frames": 0,
            "mpjpe_mm_sum": 0.0,
        }

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=torch_collate,
    )
    global_index = 0
    with torch.inference_mode():
        for batch in loader:
            images = batch["img"].to(device, non_blocking=True).float()
            predictions = pipeline(images)["pose3d"].detach().cpu().numpy()
            image_batch = batch["img"].numpy()
            for batch_index in range(len(predictions)):
                current_index = global_index + batch_index
                split_name = "train" if current_index < split_at else "test"
                if split_name not in writers:
                    continue
                stats = statistics[split_name]
                if (
                    args.max_frames_per_split > 0
                    and stats["written_frames"] >= args.max_frames_per_split
                ):
                    continue
                split_start, split_end = split_ranges[split_name]
                ground_truth = pose_values[current_index] if pose_valid[current_index] else None
                canvas, error = render_frame(
                    image_batch[batch_index],
                    predictions[batch_index],
                    ground_truth,
                    joint_names,
                    split_name=split_name,
                    checkpoint_label=args.checkpoint_label,
                    checkpoint_epoch=int(checkpoint.get("epoch", -1)),
                    global_index=current_index,
                    split_index=current_index - split_start,
                    split_length=split_end - split_start,
                )
                writers[split_name].write(canvas)
                stats["written_frames"] += 1
                if error is not None:
                    stats["valid_gt_frames"] += 1
                    stats["mpjpe_mm_sum"] += error
            global_index += len(predictions)
            if args.max_frames_per_split > 0:
                if all(
                    statistics[name]["written_frames"] >= args.max_frames_per_split
                    for name in requested_splits
                ):
                    break

    for writer in writers.values():
        writer.close()
    for split_name, stats in statistics.items():
        valid_count = int(stats.pop("valid_gt_frames"))
        error_sum = float(stats.pop("mpjpe_mm_sum"))
        stats["valid_gt_frames"] = valid_count
        stats["mean_mpjpe_over_valid_frames_mm"] = (
            error_sum / valid_count if valid_count else None
        )
        stats["fps"] = args.fps
        stats["duration_seconds"] = stats["written_frames"] / args.fps
        stats["checkpoint"] = str(Path(args.stage3_checkpoint).expanduser().resolve())
        stats["checkpoint_epoch"] = int(checkpoint.get("epoch", -1))
        stats["joint_names"] = joint_names
        stats["fixed_axis_limits_m"] = {
            "x": [-0.8, 0.8],
            "y": [-1.55, 0.35],
            "z": [-0.6, 1.35],
        }
        metadata_path = output_dir / f"{args.checkpoint_label}_{split_name}.json"
        metadata_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(json.dumps(stats, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
