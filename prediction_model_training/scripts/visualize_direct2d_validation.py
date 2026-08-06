#!/usr/bin/env python3
"""Visualize CAM_B/C predictions on the chronological direct-2D validation split."""
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

from egorear_sim2d.dataset import MultiViewHeatmapDataset, discover_label_files
from egorear_sim2d.model import EgoRearStage1HeatmapNet


EDGES_BY_NAME = (
    ("LeftFoot", "LeftUpLeg"),
    ("RightFoot", "RightUpLeg"),
    ("LeftUpLeg", "RightUpLeg"),
    ("LeftUpLeg", "Spine"),
    ("RightUpLeg", "Spine"),
    ("Spine", "Spine2"),
    ("Spine2", "LeftArm"),
    ("Spine2", "RightArm"),
    ("LeftArm", "RightArm"),
    ("LeftArm", "LeftForeArm"),
    ("LeftForeArm", "LeftHand"),
    ("RightArm", "RightForeArm"),
    ("RightForeArm", "RightHand"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-width", type=int, default=456)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--contact-sheet-columns", type=int, default=2)
    parser.add_argument("--contact-sheet-rows", type=int, default=3)
    return parser.parse_args()


def heatmap_peaks(heatmaps: np.ndarray, output_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    heatmaps = np.asarray(heatmaps, dtype=np.float32)
    joints, heatmap_height, heatmap_width = heatmaps.shape
    flat = heatmaps.reshape(joints, -1)
    flat_index = np.argmax(flat, axis=1)
    y, x = np.divmod(flat_index, heatmap_width)
    output_width, output_height = output_size
    points = np.stack(
        (
            (x.astype(np.float32) + 0.5) * output_width / heatmap_width,
            (y.astype(np.float32) + 0.5) * output_height / heatmap_height,
        ),
        axis=1,
    )
    confidence = flat[np.arange(joints), flat_index]
    return points, confidence


def draw_pose(
    image: np.ndarray,
    points: np.ndarray,
    visible: np.ndarray,
    edges: tuple[tuple[int, int], ...],
    *,
    color: tuple[int, int, int],
    predicted: bool,
) -> None:
    for a, b in edges:
        if visible[a] and visible[b]:
            cv2.line(
                image,
                tuple(np.round(points[a]).astype(int)),
                tuple(np.round(points[b]).astype(int)),
                color,
                2,
                cv2.LINE_AA,
            )
    for joint_index in np.flatnonzero(visible):
        center = tuple(np.round(points[joint_index]).astype(int))
        if predicted:
            cv2.drawMarker(
                image, center, color, markerType=cv2.MARKER_CROSS,
                markerSize=9, thickness=2, line_type=cv2.LINE_AA,
            )
        else:
            cv2.circle(image, center, 4, color, -1, cv2.LINE_AA)


def make_panel(
    image_path: str,
    gt_source_points: np.ndarray,
    gt_visible: np.ndarray,
    pred_heatmaps: np.ndarray,
    source_size: tuple[int, int],
    edges: tuple[tuple[int, int], ...],
    camera_name: str,
) -> tuple[np.ndarray, float, list[float | None]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read validation image: {image_path}")
    height, width = image.shape[:2]
    source_width, source_height = source_size
    gt_display = np.asarray(gt_source_points, dtype=np.float32).copy()
    gt_display[:, 0] *= width / source_width
    gt_display[:, 1] *= height / source_height
    pred_display, confidence = heatmap_peaks(pred_heatmaps, (width, height))
    pred_source, _ = heatmap_peaks(pred_heatmaps, source_size)

    valid = np.asarray(gt_visible, dtype=bool)
    errors = np.full(len(valid), np.nan, dtype=np.float32)
    errors[valid] = np.linalg.norm(pred_source[valid] - gt_source_points[valid], axis=1)
    mean_error = float(np.nanmean(errors)) if np.any(valid) else float("nan")

    overlay = (image.astype(np.float32) * 0.72).astype(np.uint8)
    draw_pose(overlay, gt_display, valid, edges, color=(30, 40, 255), predicted=False)
    draw_pose(
        overlay,
        pred_display,
        np.ones(len(pred_display), dtype=bool),
        edges,
        color=(255, 255, 0),
        predicted=True,
    )
    cv2.rectangle(overlay, (0, 0), (width, 48), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        f"{camera_name}  mean error={mean_error:.1f}px",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        "GT: red dots/lines   Prediction: cyan crosses/lines",
        (8, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return overlay, mean_error, [float(value) if np.isfinite(value) else None for value in errors]


def build_contact_sheets(
    sample_paths: list[Path],
    output_dir: Path,
    *,
    columns: int,
    rows: int,
) -> list[str]:
    page_size = max(1, int(columns) * int(rows))
    page_names = []
    for page_index, start in enumerate(range(0, len(sample_paths), page_size), start=1):
        images = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in sample_paths[start : start + page_size]]
        if any(image is None for image in images):
            raise RuntimeError("Could not read an image while building contact sheets")
        assert images
        cell_height, cell_width = images[0].shape[:2]
        blank = np.zeros((cell_height, cell_width, 3), dtype=np.uint8)
        while len(images) < page_size:
            images.append(blank.copy())
        grid_rows = [
            np.hstack(images[row * columns : (row + 1) * columns])
            for row in range(rows)
        ]
        sheet = np.vstack(grid_rows)
        name = f"validation_contact_sheet_{page_index:02d}.jpg"
        cv2.imwrite(str(output_dir / name), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
        page_names.append(name)
    return page_names


def main() -> int:
    import torch

    args = parse_args()
    label_files = discover_label_files(args.label_root)
    if len(label_files) != 1:
        raise RuntimeError(f"Expected one direct-2D label file, found {label_files}")
    label_path = label_files[0]
    with np.load(label_path, allow_pickle=True) as labels:
        joint_names = tuple(str(name) for name in labels["head_camera_joints"])
        source_size = tuple(int(value) for value in labels["video_size"])

    dataset = MultiViewHeatmapDataset(
        label_files,
        image_size=(args.image_width, args.image_height),
        visible_only_loss=True,
    )
    split_at = max(
        1,
        min(len(dataset) - 1, int(round(len(dataset) * float(args.train_ratio)))),
    )
    validation_indices = np.arange(split_at, len(dataset), dtype=np.int64)
    sample_count = min(int(args.samples), len(validation_indices))
    selected = np.unique(
        np.linspace(validation_indices[0], validation_indices[-1], sample_count, dtype=np.int64)
    )

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = EgoRearStage1HeatmapNet(
        num_head_heatmaps=len(joint_names),
        base_channels=args.base_channels,
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    joint_index = {name: index for index, name in enumerate(joint_names)}
    edges = tuple((joint_index[a], joint_index[b]) for a, b in EDGES_BY_NAME)
    sample_summaries = []
    sample_paths = []
    with torch.no_grad():
        for output_index, dataset_index in enumerate(selected):
            item = dataset[int(dataset_index)]
            label_index, frame_index = dataset.index[int(dataset_index)]
            label_data = dataset._load_label(label_index)
            image = torch.as_tensor(item["img"][None]).to(device).float()
            prediction = model(image, "head")["head"][0].detach().cpu().numpy()
            panels = []
            camera_summaries = []
            for camera_index, camera_name in enumerate(item["camera_names"]):
                panel, mean_error, joint_errors = make_panel(
                    str(label_data["image_paths"][frame_index, camera_index]),
                    np.asarray(label_data["head_keypoints"][frame_index, camera_index]),
                    np.asarray(label_data["head_visible"][frame_index, camera_index]),
                    prediction[camera_index],
                    source_size,
                    edges,
                    str(camera_name),
                )
                panels.append(panel)
                camera_summaries.append(
                    {
                        "camera": str(camera_name),
                        "mean_pixel_error": mean_error,
                        "joint_pixel_errors": dict(zip(joint_names, joint_errors)),
                    }
                )
            combined = np.hstack(panels)
            output_name = (
                f"sample_{output_index:03d}_dataset_{int(dataset_index):05d}"
                f"_frame_{int(item['frame_idx']):05d}.jpg"
            )
            output_path = output_dir / output_name
            cv2.imwrite(str(output_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 95])
            sample_paths.append(output_path)
            sample_summaries.append(
                {
                    "sample": output_index,
                    "dataset_index": int(dataset_index),
                    "frame_index": int(item["frame_idx"]),
                    "cameras": camera_summaries,
                }
            )

    contact_sheets = build_contact_sheets(
        sample_paths,
        output_dir,
        columns=args.contact_sheet_columns,
        rows=args.contact_sheet_rows,
    )
    all_camera_errors = [
        camera["mean_pixel_error"]
        for sample in sample_summaries
        for camera in sample["cameras"]
    ]
    summary = {
        "schema": "egorear.direct2d_validation_visualization.v1",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_val_loss": float(checkpoint.get("val_loss", float("nan"))),
        "label_file": str(label_path.resolve()),
        "split": "chronological validation tail",
        "train_ratio": args.train_ratio,
        "validation_start": int(validation_indices[0]),
        "validation_end": int(validation_indices[-1]),
        "validation_frames": int(len(validation_indices)),
        "samples": int(len(selected)),
        "joint_names": list(joint_names),
        "mean_selected_pixel_error": float(np.mean(all_camera_errors)),
        "contact_sheets": contact_sheets,
        "sample_files": [path.name for path in sample_paths],
        "sample_results": sample_summaries,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "sample_results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
