#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
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
from egorear_sim2d.splits import load_split_manifest


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
PRED_COLOR = (255, 100, 0)
GT_COLOR = (0, 210, 70)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the shared held-out test set and export stage 1/2/3 skeleton montages."
    )
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--pose3d-labels", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--stage2-checkpoint", required=True)
    parser.add_argument("--stage3-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--montage-samples", type=int, default=4)
    parser.add_argument(
        "--export-all-stage3",
        action="store_true",
        help="Write one front/side GT-vs-prediction PNG for every valid 3D test frame.",
    )
    return parser.parse_args()


def denormalize(image: np.ndarray) -> np.ndarray:
    rgb = image.transpose(1, 2, 0) * IMAGENET_STD + IMAGENET_MEAN
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def decode_heatmaps(heatmaps: np.ndarray, image_size=(456, 256)) -> np.ndarray:
    height, width = heatmaps.shape[-2:]
    indices = heatmaps.reshape(*heatmaps.shape[:-2], -1).argmax(axis=-1)
    x = indices % width
    y = indices // width
    return np.stack(
        (
            (x + 0.5) * image_size[0] / width,
            (y + 0.5) * image_size[1] / height,
        ),
        axis=-1,
    ).astype(np.float32)


def draw_2d_pose(
    image: np.ndarray,
    points: np.ndarray,
    joint_names: list[str],
    color: tuple[int, int, int],
    *,
    visible: np.ndarray | None = None,
    thickness: int = 2,
) -> None:
    index = {name: idx for idx, name in enumerate(joint_names)}
    valid = (
        np.ones(len(joint_names), dtype=bool)
        if visible is None
        else np.asarray(visible, dtype=bool)
    )
    for start_name, end_name in SKELETON_EDGES:
        if start_name not in index or end_name not in index:
            continue
        start, end = index[start_name], index[end_name]
        if valid[start] and valid[end]:
            cv2.line(
                image,
                tuple(np.rint(points[start]).astype(int)),
                tuple(np.rint(points[end]).astype(int)),
                color,
                thickness,
                cv2.LINE_AA,
            )
    for idx, point in enumerate(points):
        if valid[idx]:
            cv2.circle(
                image,
                tuple(np.rint(point).astype(int)),
                4,
                color,
                -1,
                cv2.LINE_AA,
            )


def add_header(image: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    output = np.full((image.shape[0] + 58, image.shape[1], 3), 24, dtype=np.uint8)
    output[58:] = image
    cv2.putText(
        output, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
        (245, 245, 245), 2, cv2.LINE_AA
    )
    cv2.putText(
        output, subtitle, (12, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.47,
        (210, 210, 210), 1, cv2.LINE_AA
    )
    return output


def make_2d_montage(
    samples: list[dict[str, object]],
    *,
    stage_name: str,
    checkpoint_epoch: int,
    joint_names: list[str],
) -> np.ndarray:
    rows = []
    for sample in samples:
        views = []
        for view_index, camera_name in enumerate(("CAM_B", "CAM_C")):
            image = denormalize(sample["images"][view_index]).copy()
            draw_2d_pose(
                image,
                sample["gt_xy"][view_index],
                joint_names,
                GT_COLOR,
                visible=sample["visible"][view_index],
                thickness=2,
            )
            draw_2d_pose(
                image,
                sample["pred_xy"][view_index],
                joint_names,
                PRED_COLOR,
                thickness=3,
            )
            cv2.putText(
                image, camera_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (255, 255, 255), 2, cv2.LINE_AA
            )
            views.append(image)
        row = np.concatenate(views, axis=1)
        rows.append(
            add_header(
                row,
                f"{stage_name} | checkpoint epoch {checkpoint_epoch} | test source frame {sample['frame_idx']}",
                f"blue: prediction   green: GT   visible-joint error: {sample['pixel_error']:.2f} px",
            )
        )
    return np.concatenate(rows, axis=0)


def project_3d(
    point: np.ndarray,
    axes: tuple[int, int],
    limits: tuple[tuple[float, float], tuple[float, float]],
    rect: tuple[int, int, int, int],
) -> tuple[int, int]:
    left, top, width, height = rect
    x_min, x_max = limits[0]
    y_min, y_max = limits[1]
    x = left + int((float(point[axes[0]]) - x_min) / (x_max - x_min) * width)
    y = top + int((y_max - float(point[axes[1]])) / (y_max - y_min) * height)
    return x, y


def draw_3d_projection(
    canvas: np.ndarray,
    pose: np.ndarray,
    joint_names: list[str],
    color: tuple[int, int, int],
    axes: tuple[int, int],
    limits: tuple[tuple[float, float], tuple[float, float]],
    rect: tuple[int, int, int, int],
    thickness: int,
) -> None:
    index = {name: idx for idx, name in enumerate(joint_names)}
    points = [project_3d(point, axes, limits, rect) for point in pose]
    for start_name, end_name in SKELETON_EDGES:
        if start_name in index and end_name in index:
            cv2.line(
                canvas, points[index[start_name]], points[index[end_name]],
                color, thickness, cv2.LINE_AA
            )
    for point in points:
        cv2.circle(canvas, point, 4, color, -1, cv2.LINE_AA)


def make_3d_montage(
    samples: list[dict[str, object]],
    *,
    checkpoint_epoch: int,
    joint_names: list[str],
) -> np.ndarray:
    panels = []
    y_limits = (-1.55, 0.35)
    for sample in samples:
        panel = np.full((470, 760, 3), 24, dtype=np.uint8)
        cv2.putText(
            panel,
            f"Stage 3 | epoch {checkpoint_epoch} | test frame {sample['frame_idx']}",
            (16, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"blue: prediction   green: lifting GT   MPJPE: {sample['mpjpe_mm']:.2f} mm",
            (16, 51),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
        front_rect = (30, 90, 310, 345)
        side_rect = (420, 90, 310, 345)
        for rect, title in ((front_rect, "Front (X-Y)"), (side_rect, "Side (Z-Y)")):
            left, top, width, height = rect
            cv2.rectangle(panel, (left, top), (left + width, top + height), (75, 75, 75), 1)
            cv2.putText(
                panel, title, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (230, 230, 230), 1, cv2.LINE_AA
            )
        for pose, color, thickness in (
            (sample["gt_pose"], GT_COLOR, 2),
            (sample["pred_pose"], PRED_COLOR, 3),
        ):
            draw_3d_projection(
                panel, pose, joint_names, color, (0, 1),
                ((-0.8, 0.8), y_limits), front_rect, thickness
            )
            draw_3d_projection(
                panel, pose, joint_names, color, (2, 1),
                ((-0.6, 1.35), y_limits), side_rect, thickness
            )
        panels.append(panel)
    if len(panels) == 1:
        return panels[0]
    rows = []
    for start in range(0, len(panels), 2):
        row_panels = panels[start : start + 2]
        if len(row_panels) == 1:
            row_panels.append(np.full_like(row_panels[0], 24))
        rows.append(np.concatenate(row_panels, axis=1))
    return np.concatenate(rows, axis=0)


def main() -> int:
    import torch
    from torch.utils.data import DataLoader, Subset

    args = parse_args()
    dataset = MultiViewHeatmapDataset(
        discover_label_files(Path(args.label_root)),
        image_size=(456, 256),
        visible_only_loss=True,
    )
    dataset_frames = np.asarray(
        [
            int(dataset._load_label(label_idx)["frame_indices"][frame_idx])
            for label_idx, frame_idx in dataset.index
        ],
        dtype=np.int64,
    )
    split = load_split_manifest(
        args.split_manifest,
        expected_length=len(dataset),
        expected_frame_indices=dataset_frames,
    )
    test_indices = split["test_indices"].astype(np.int64)
    pose_labels = np.load(args.pose3d_labels, allow_pickle=True)
    pose_values = np.asarray(pose_labels["pose_head_m"], dtype=np.float32)
    pose_valid = np.asarray(pose_labels["valid"], dtype=bool)
    joint_names = [str(name) for name in pose_labels["joint_names"]]
    if not np.array_equal(pose_labels["frame_indices"], dataset_frames):
        raise ValueError("The 3D labels do not align with the image/heatmap dataset")

    stage1_checkpoint = torch.load(
        args.stage1_checkpoint, map_location="cpu", weights_only=False
    )
    stage1 = load_stage1_model(
        args.stage1_checkpoint, num_head_heatmaps=len(joint_names)
    )
    stage2_checkpoint = torch.load(
        args.stage2_checkpoint, map_location="cpu", weights_only=False
    )
    stage2_config = stage2_checkpoint.get("config", {})
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
    stage3_checkpoint = torch.load(
        args.stage3_checkpoint, map_location="cpu", weights_only=False
    )
    pose3d = EgoRearPose3DNet(num_joints=len(joint_names))
    pose3d.load_state_dict(stage3_checkpoint["pose3d"], strict=True)
    pipeline = EgoRearStage3Pipeline(stage2, pose3d)

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    pipeline = pipeline.to(device).eval()
    loader = DataLoader(
        Subset(dataset, test_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=torch_collate,
    )

    sample_candidates = test_indices[pose_valid[test_indices]]
    sample_positions = np.linspace(
        0, len(sample_candidates) - 1, args.montage_samples
    ).round().astype(int)
    selected = set(sample_candidates[sample_positions].astype(int).tolist())
    stage1_samples: list[dict[str, object]] = []
    stage2_samples: list[dict[str, object]] = []
    stage3_samples: list[dict[str, object]] = []
    stage1_error_sum = stage2_error_sum = 0.0
    stage12_count = 0
    stage1_joint_error_sum = np.zeros(len(joint_names), dtype=np.float64)
    stage2_joint_error_sum = np.zeros(len(joint_names), dtype=np.float64)
    stage12_joint_count = np.zeros(len(joint_names), dtype=np.int64)
    stage3_error_sum = 0.0
    stage3_count = 0
    stage3_joint_error_sum = np.zeros(len(joint_names), dtype=np.float64)
    stage3_valid_frames = 0
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    all_stage3_dir = output_dir / "stage3_test_all_frames_front_side"
    if args.export_all_stage3:
        all_stage3_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        cursor = 0
        for batch in loader:
            images = batch["img"].to(device, non_blocking=True).float()
            refinement = pipeline.stage2(images)
            stage1_heatmaps = refinement["initial_stage1"]
            stage2_heatmaps = refinement["refined"]
            output3d = pipeline.pose3d(
                images, stage2_heatmaps, refinement["refined_features"]
            )["pose3d"]
            gt_heatmaps = batch["head_gt_heatmap"].numpy()
            visible = batch["head_loss_mask"].numpy() > 0.5
            pred1_xy = decode_heatmaps(stage1_heatmaps.cpu().numpy())
            pred2_xy = decode_heatmaps(stage2_heatmaps.cpu().numpy())
            gt_xy = decode_heatmaps(gt_heatmaps)
            image_values = batch["img"].numpy()
            pred3d = output3d.cpu().numpy()
            for batch_idx in range(len(pred3d)):
                dataset_idx = int(test_indices[cursor + batch_idx])
                valid_mask = visible[batch_idx]
                error1 = np.linalg.norm(pred1_xy[batch_idx] - gt_xy[batch_idx], axis=-1)
                error2 = np.linalg.norm(pred2_xy[batch_idx] - gt_xy[batch_idx], axis=-1)
                stage1_error_sum += float(error1[valid_mask].sum())
                stage2_error_sum += float(error2[valid_mask].sum())
                stage12_count += int(valid_mask.sum())
                stage1_joint_error_sum += np.where(valid_mask, error1, 0.0).sum(axis=0)
                stage2_joint_error_sum += np.where(valid_mask, error2, 0.0).sum(axis=0)
                stage12_joint_count += valid_mask.sum(axis=0)
                if pose_valid[dataset_idx]:
                    error3 = np.linalg.norm(
                        pred3d[batch_idx] - pose_values[dataset_idx], axis=-1
                    )
                    stage3_error_sum += float(error3.sum())
                    stage3_count += int(error3.size)
                    stage3_joint_error_sum += error3
                    stage3_valid_frames += 1
                    if args.export_all_stage3:
                        all_frame_sample = {
                            "frame_idx": int(dataset_frames[dataset_idx]),
                            "pred_pose": pred3d[batch_idx],
                            "gt_pose": pose_values[dataset_idx],
                            "mpjpe_mm": float(error3.mean() * 1000.0),
                        }
                        all_frame_path = all_stage3_dir / (
                            f"test_{stage3_valid_frames - 1:04d}_"
                            f"source_{int(dataset_frames[dataset_idx]):06d}.png"
                        )
                        cv2.imwrite(
                            str(all_frame_path),
                            make_3d_montage(
                                [all_frame_sample],
                                checkpoint_epoch=int(stage3_checkpoint.get("epoch", -1)),
                                joint_names=joint_names,
                            ),
                        )
                if dataset_idx not in selected:
                    continue
                common = {
                    "images": image_values[batch_idx],
                    "gt_xy": gt_xy[batch_idx],
                    "visible": visible[batch_idx],
                    "frame_idx": int(dataset_frames[dataset_idx]),
                }
                stage1_samples.append(
                    common
                    | {
                        "pred_xy": pred1_xy[batch_idx],
                        "pixel_error": float(error1[valid_mask].mean()),
                    }
                )
                stage2_samples.append(
                    common
                    | {
                        "pred_xy": pred2_xy[batch_idx],
                        "pixel_error": float(error2[valid_mask].mean()),
                    }
                )
                stage3_samples.append(
                    {
                        "frame_idx": int(dataset_frames[dataset_idx]),
                        "pred_pose": pred3d[batch_idx],
                        "gt_pose": pose_values[dataset_idx],
                        "mpjpe_mm": float(
                            np.linalg.norm(
                                pred3d[batch_idx] - pose_values[dataset_idx], axis=-1
                            ).mean()
                            * 1000.0
                        ),
                    }
                )
            cursor += len(pred3d)

    outputs = {
        "stage1": output_dir / "stage1_test_gt_vs_prediction.png",
        "stage2": output_dir / "stage2_test_gt_vs_prediction.png",
        "stage3": output_dir / "stage3_test_gt_vs_prediction_front_side.png",
    }
    cv2.imwrite(
        str(outputs["stage1"]),
        make_2d_montage(
            stage1_samples,
            stage_name="Stage 1",
            checkpoint_epoch=int(stage1_checkpoint.get("epoch", -1)),
            joint_names=joint_names,
        ),
    )
    cv2.imwrite(
        str(outputs["stage2"]),
        make_2d_montage(
            stage2_samples,
            stage_name="Stage 2",
            checkpoint_epoch=int(stage2_checkpoint.get("epoch", -1)),
            joint_names=joint_names,
        ),
    )
    cv2.imwrite(
        str(outputs["stage3"]),
        make_3d_montage(
            stage3_samples,
            checkpoint_epoch=int(stage3_checkpoint.get("epoch", -1)),
            joint_names=joint_names,
        ),
    )
    metrics = {
        "split_manifest": str(Path(args.split_manifest).expanduser().resolve()),
        "test_frames_2d": int(len(test_indices)),
        "test_frames_3d_valid": int(pose_valid[test_indices].sum()),
        "stage1": {
            "checkpoint_epoch": int(stage1_checkpoint.get("epoch", -1)),
            "mean_visible_joint_pixel_error": stage1_error_sum / max(stage12_count, 1),
            "per_joint_visible_pixel_error": {
                name: float(error_sum / max(int(count), 1))
                for name, error_sum, count in zip(
                    joint_names, stage1_joint_error_sum, stage12_joint_count
                )
            },
        },
        "stage2": {
            "checkpoint_epoch": int(stage2_checkpoint.get("epoch", -1)),
            "mean_visible_joint_pixel_error": stage2_error_sum / max(stage12_count, 1),
            "per_joint_visible_pixel_error": {
                name: float(error_sum / max(int(count), 1))
                for name, error_sum, count in zip(
                    joint_names, stage2_joint_error_sum, stage12_joint_count
                )
            },
        },
        "stage3": {
            "checkpoint_epoch": int(stage3_checkpoint.get("epoch", -1)),
            "mpjpe_mm": stage3_error_sum / max(stage3_count, 1) * 1000.0,
            "per_joint_mpjpe_mm": {
                name: float(error_sum / max(stage3_valid_frames, 1) * 1000.0)
                for name, error_sum in zip(joint_names, stage3_joint_error_sum)
            },
        },
        "montage_source_frames": [int(sample["frame_idx"]) for sample in stage3_samples],
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    if args.export_all_stage3:
        metrics["stage3_all_frames_output_dir"] = str(all_stage3_dir)
        metrics["stage3_all_frames_written"] = int(stage3_valid_frames)
    (output_dir / "test_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
