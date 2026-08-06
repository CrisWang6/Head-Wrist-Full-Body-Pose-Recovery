#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and visualize stage-1 heatmap predictions.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label-root", default="")
    parser.add_argument(
        "--image-root",
        default="",
        help="Unlabelled paired-image root containing module01/CAM_B and module01/CAM_C.",
    )
    parser.add_argument(
        "--reference-csv",
        default="",
        help="Optional CAM_C shoulder/elbow CSV used as a partial external reference.",
    )
    parser.add_argument("--reference-score-threshold", type=float, default=0.3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--frame-root", default="")
    parser.add_argument("--render-root", default="")
    parser.add_argument("--output-dir", default="outputs/test_heatmap")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split", default="all", choices=("all", "train", "val"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-width", type=int, default=456)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--visible-only-loss", action="store_true")
    parser.add_argument("--prefer-wrist-visible", action="store_true")
    parser.add_argument("--min-wrist-cameras", type=int, default=2)
    parser.add_argument("--min-wrist-joints", type=int, default=2)
    parser.add_argument("--highres", action="store_true")
    parser.add_argument("--save-individual-panels", action="store_true")
    parser.add_argument(
        "--blue-skeleton",
        action="store_true",
        help="Draw all predicted joints and body connections in blue.",
    )
    return parser.parse_args()


def main() -> int:
    import torch

    args = parse_args()
    if bool(args.label_root) == bool(args.image_root):
        raise ValueError("Provide exactly one of --label-root or --image-root")
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    num_head_heatmaps = int(checkpoint_config.get("num_head_heatmaps", 16))
    head_joint_names = [
        str(name)
        for name in checkpoint_config.get(
            "head_joint_names", [f"joint_{index}" for index in range(num_head_heatmaps)]
        )
    ]
    model = EgoRearStage1HeatmapNet(
        num_head_heatmaps=num_head_heatmaps,
        base_channels=args.base_channels,
    ).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    if args.image_root:
        return run_unlabelled_head_test(
            args, model, device, head_joint_names=head_joint_names
        )

    label_files = discover_label_files(Path(args.label_root))
    dataset = MultiViewHeatmapDataset(
        label_files,
        frame_root=Path(args.frame_root) if args.frame_root else None,
        render_root=Path(args.render_root) if args.render_root else None,
        image_size=(args.image_width, args.image_height),
        visible_only_loss=args.visible_only_loss,
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    candidate_indices = split_indices(
        len(dataset),
        split=args.split,
        train_ratio=args.train_ratio,
        seed=args.split_seed,
    )
    sample_indices = select_sample_indices(
        dataset,
        candidate_indices,
        samples=args.samples,
        rng=rng,
        prefer_wrist_visible=args.prefer_wrist_visible,
        min_wrist_cameras=args.min_wrist_cameras,
        min_wrist_joints=args.min_wrist_joints,
    )
    metrics = []
    selected = []
    with torch.no_grad():
        for out_idx, dataset_idx in enumerate(sample_indices):
            item = dataset[int(dataset_idx)]
            label_idx, frame_idx = dataset.index[int(dataset_idx)]
            label_data = dataset._load_label(label_idx)
            img = torch.as_tensor(item["img"][None]).to(device).float()
            pred = model(img)
            head_mse = masked_heatmap_mse(
                pred["head"],
                torch.as_tensor(item["head_gt_heatmap"][None]).to(device).float(),
                torch.as_tensor(item["head_loss_mask"][None]).to(device).float(),
            )
            wrist_mse = masked_heatmap_mse(
                pred["wrist"],
                torch.as_tensor(item["wrist_gt_heatmap"][None]).to(device).float(),
                torch.as_tensor(item["wrist_loss_mask"][None]).to(device).float(),
            )
            mse = head_mse + wrist_mse
            metrics.append(float(mse.detach().cpu()))
            save_sample_visualization(
                output_dir=output_dir,
                sample_idx=out_idx,
                image=item["img"],
                head_target=item["head_gt_heatmap"],
                wrist_target=item["wrist_gt_heatmap"],
                pred={key: value[0].detach().cpu().numpy() for key, value in pred.items()},
                camera_names=item["camera_names"],
                frame_idx=int(item["frame_idx"]),
                label_path=str(item["label_path"]),
                label_data=label_data,
                highres=args.highres,
                save_individual_panels=args.save_individual_panels,
            )
            selected.append(
                {
                    "sample_idx": int(out_idx),
                    "dataset_idx": int(dataset_idx),
                    "label_path": str(item["label_path"]),
                    "frame_idx": int(item["frame_idx"]),
                    "wrist_visible_joints": int(wrist_visible_score(label_data, frame_idx)),
                    "mse": float(mse.detach().cpu()),
                    "head_mse": float(head_mse.detach().cpu()),
                    "wrist_mse": float(wrist_mse.detach().cpu()),
                }
            )

    summary = {
        "checkpoint": str(args.checkpoint),
        "label_root": str(args.label_root),
        "split": args.split,
        "candidate_frames": int(len(candidate_indices)),
        "samples": int(len(metrics)),
        "mse_mean": float(np.mean(metrics)) if metrics else 0.0,
        "mse_median": float(np.median(metrics)) if metrics else 0.0,
        "visualization_dir": str(output_dir),
        "highres": bool(args.highres),
        "prefer_wrist_visible": bool(args.prefer_wrist_visible),
        "selected": selected,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def run_unlabelled_head_test(args, model, device, *, head_joint_names: list[str]) -> int:
    """Run the shared CAM_B/C stage-1 branch without manufacturing missing GT."""
    import torch

    image_root = Path(args.image_root).expanduser().resolve()
    camera_dirs = {
        "CAM_B": image_root / "module01" / "CAM_B",
        "CAM_C": image_root / "module01" / "CAM_C",
    }
    indexed = {
        camera: {
            int(path.stem.rsplit("_", 1)[1]): path
            for path in sorted(directory.glob("frame_*.jpg"))
        }
        for camera, directory in camera_dirs.items()
    }
    frame_indices = sorted(set(indexed["CAM_B"]) & set(indexed["CAM_C"]))
    if not frame_indices:
        raise FileNotFoundError(f"No paired CAM_B/C frames under {image_root}")
    reference = (
        load_cam_c_reference(Path(args.reference_csv).expanduser().resolve())
        if args.reference_csv
        else {}
    )
    reference_mapping = {
        "left_shoulder": "LeftArm",
        "right_shoulder": "RightArm",
        "left_elbow": "LeftForeArm",
        "right_elbow": "RightForeArm",
    }
    joint_to_index = {name: index for index, name in enumerate(head_joint_names)}
    visual_count = min(max(0, int(args.samples)), len(frame_indices))
    visual_indices = set(
        np.linspace(0, len(frame_indices) - 1, visual_count, dtype=np.int64).tolist()
    )
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.csv"
    metric_distances = {name: [] for name in reference_mapping}
    metric_rows = 0
    prediction_fields = ["frame_index", "camera", "joint", "x", "y", "confidence"]

    with prediction_path.open("w", encoding="utf-8", newline="") as prediction_file:
        writer = csv.DictWriter(prediction_file, fieldnames=prediction_fields)
        writer.writeheader()
        for batch_start in range(0, len(frame_indices), max(1, int(args.batch_size))):
            batch_frame_indices = frame_indices[batch_start : batch_start + int(args.batch_size)]
            batch_images = []
            display_images = []
            for frame_index in batch_frame_indices:
                pair = []
                display_pair = []
                for camera in ("CAM_B", "CAM_C"):
                    bgr = cv2.imread(str(indexed[camera][frame_index]), cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise RuntimeError(f"Could not read {indexed[camera][frame_index]}")
                    bgr = cv2.resize(
                        bgr,
                        (int(args.image_width), int(args.image_height)),
                        interpolation=cv2.INTER_AREA,
                    )
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    display_pair.append(rgb)
                    pair.append(normalize_rgb(rgb))
                batch_images.append(pair)
                display_images.append(display_pair)
            tensor = torch.as_tensor(np.asarray(batch_images)).to(device).float()
            with torch.inference_mode():
                heatmaps = model(tensor, "head")["head"].detach().cpu().numpy()
            for local_index, frame_index in enumerate(batch_frame_indices):
                coordinates = heatmap_peak_coordinates(
                    heatmaps[local_index],
                    width=int(args.image_width),
                    height=int(args.image_height),
                )
                for camera_index, camera in enumerate(("CAM_B", "CAM_C")):
                    for joint_index, joint_name in enumerate(head_joint_names):
                        x, y, confidence = coordinates[camera_index, joint_index]
                        writer.writerow(
                            {
                                "frame_index": frame_index,
                                "camera": camera,
                                "joint": joint_name,
                                "x": f"{x:.6f}",
                                "y": f"{y:.6f}",
                                "confidence": f"{confidence:.8f}",
                            }
                        )
                reference_row = reference.get(frame_index)
                if reference_row is not None:
                    row_used = False
                    for reference_name, model_name in reference_mapping.items():
                        if model_name not in joint_to_index:
                            continue
                        score = float(reference_row[f"{reference_name}_score"])
                        gt_x = float(reference_row[f"{reference_name}_x"])
                        gt_y = float(reference_row[f"{reference_name}_y"])
                        if (
                            score < float(args.reference_score_threshold)
                            or not (0.0 <= gt_x < 1920.0 and 0.0 <= gt_y < 1200.0)
                        ):
                            continue
                        pred_x, pred_y = coordinates[1, joint_to_index[model_name], :2]
                        pred_source_x = pred_x * 1920.0 / float(args.image_width)
                        pred_source_y = pred_y * 1200.0 / float(args.image_height)
                        metric_distances[reference_name].append(
                            float(np.hypot(pred_source_x - gt_x, pred_source_y - gt_y))
                        )
                        row_used = True
                    metric_rows += int(row_used)
                global_position = batch_start + local_index
                if global_position in visual_indices:
                    save_unlabelled_visualization(
                        output_dir,
                        frame_index,
                        display_images[local_index],
                        coordinates,
                        head_joint_names,
                        reference_row,
                        reference_mapping,
                        joint_to_index,
                        float(args.reference_score_threshold),
                        bool(args.blue_skeleton),
                    )
            print(
                json.dumps(
                    {
                        "processed": min(batch_start + len(batch_frame_indices), len(frame_indices)),
                        "total": len(frame_indices),
                    }
                ),
                flush=True,
            )

    all_distances = [
        distance for distances in metric_distances.values() for distance in distances
    ]
    summary = {
        "mode": "unlabelled_head_bc",
        "checkpoint": str(args.checkpoint),
        "image_root": str(image_root),
        "paired_frames": len(frame_indices),
        "frame_range": [frame_indices[0], frame_indices[-1]],
        "joint_names": head_joint_names,
        "visualizations": visual_count,
        "predictions_csv": str(prediction_path),
        "reference_csv": str(args.reference_csv),
        "reference_note": (
            "Sapiens CAM_C shoulder/elbow predictions are a partial proxy, not ground truth."
            if reference
            else "No ground truth or external reference was provided."
        ),
        "reference_score_threshold": float(args.reference_score_threshold),
        "reference_frames_used": metric_rows,
        "reference_error_source_pixels": {
            name: {
                "count": len(values),
                "mean": float(np.mean(values)) if values else None,
                "median": float(np.median(values)) if values else None,
            }
            for name, values in metric_distances.items()
        },
        "reference_error_all_source_pixels": {
            "count": len(all_distances),
            "mean": float(np.mean(all_distances)) if all_distances else None,
            "median": float(np.median(all_distances)) if all_distances else None,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


def normalize_rgb(rgb: np.ndarray) -> np.ndarray:
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return ((rgb.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)


def heatmap_peak_coordinates(heatmaps: np.ndarray, *, width: int, height: int) -> np.ndarray:
    views, joints, heatmap_height, heatmap_width = heatmaps.shape
    flat = heatmaps.reshape(views, joints, -1)
    indices = flat.argmax(axis=-1)
    confidence = flat.max(axis=-1)
    x = (indices % heatmap_width + 0.5) * float(width) / float(heatmap_width)
    y = (indices // heatmap_width + 0.5) * float(height) / float(heatmap_height)
    return np.stack((x, y, confidence), axis=-1)


def load_cam_c_reference(path: Path) -> dict[int, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        return {
            int(row["decoded_frame_index"]): row
            for row in rows
            if row.get("status") == "ok" and row.get("decoded_frame_index", "").strip()
        }


def save_unlabelled_visualization(
    output_dir: Path,
    frame_index: int,
    images: list[np.ndarray],
    coordinates: np.ndarray,
    joint_names: list[str],
    reference_row: dict[str, str] | None,
    reference_mapping: dict[str, str],
    joint_to_index: dict[str, int],
    reference_score_threshold: float,
    blue_skeleton: bool,
) -> None:
    palette = [
        (255, 80, 80), (80, 80, 255), (255, 180, 40), (40, 180, 255),
        (255, 80, 200), (80, 255, 200), (180, 255, 80), (80, 255, 80),
        (255, 140, 180), (140, 180, 255), (255, 220, 80), (80, 220, 255),
    ]
    panels = []
    for camera_index, camera in enumerate(("CAM_B", "CAM_C")):
        panel = images[camera_index].copy()
        if blue_skeleton:
            blue = (30, 144, 255)
            skeleton_edges = (
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
            for start_name, end_name in skeleton_edges:
                if start_name not in joint_to_index or end_name not in joint_to_index:
                    continue
                start = coordinates[camera_index, joint_to_index[start_name], :2]
                end = coordinates[camera_index, joint_to_index[end_name], :2]
                cv2.line(
                    panel,
                    tuple(np.rint(start).astype(int)),
                    tuple(np.rint(end).astype(int)),
                    blue,
                    2,
                    cv2.LINE_AA,
                )
        for joint_index, joint_name in enumerate(joint_names):
            x, y = coordinates[camera_index, joint_index, :2]
            color = (30, 144, 255) if blue_skeleton else palette[joint_index % len(palette)]
            cv2.circle(panel, (int(round(x)), int(round(y))), 5, color, -1, cv2.LINE_AA)
            cv2.putText(
                panel,
                str(joint_index + 1),
                (int(round(x)) + 4, int(round(y)) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
        if camera == "CAM_C" and reference_row is not None:
            for reference_name, model_name in reference_mapping.items():
                score = float(reference_row[f"{reference_name}_score"])
                gt_x = float(reference_row[f"{reference_name}_x"]) * panel.shape[1] / 1920.0
                gt_y = float(reference_row[f"{reference_name}_y"]) * panel.shape[0] / 1200.0
                if (
                    score >= reference_score_threshold
                    and 0 <= gt_x < panel.shape[1]
                    and 0 <= gt_y < panel.shape[0]
                ):
                    cv2.drawMarker(
                        panel,
                        (int(round(gt_x)), int(round(gt_y))),
                        (255, 0, 0),
                        cv2.MARKER_CROSS,
                        14,
                        2,
                        cv2.LINE_AA,
                    )
        cv2.putText(
            panel,
            f"{camera} frame={frame_index} model=blue skeleton" if blue_skeleton
            else f"{camera} frame={frame_index} model=circle ref=cross",
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)
    grid = np.hstack(panels)
    cv2.imwrite(
        str(output_dir / f"frame_{frame_index:06d}_head_bc.jpg"),
        cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),
    )


def split_indices(count: int, *, split: str, train_ratio: float, seed: int) -> np.ndarray:
    indices = np.arange(int(count))
    if split == "all" or count <= 1:
        return indices
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    split_at = max(1, min(len(indices) - 1, int(round(len(indices) * float(train_ratio)))))
    return indices[:split_at] if split == "train" else indices[split_at:]


def select_sample_indices(
    dataset: MultiViewHeatmapDataset,
    candidate_indices: np.ndarray,
    *,
    samples: int,
    rng: np.random.Generator,
    prefer_wrist_visible: bool,
    min_wrist_cameras: int,
    min_wrist_joints: int,
) -> np.ndarray:
    if not prefer_wrist_visible:
        return rng.choice(candidate_indices, size=min(samples, len(candidate_indices)), replace=False)

    scored = []
    for dataset_idx in candidate_indices:
        label_idx, frame_idx = dataset.index[int(dataset_idx)]
        data = dataset._load_label(label_idx)
        wrist_counts = wrist_visible_counts(data, int(frame_idx))
        hit_cameras = int((wrist_counts >= int(min_wrist_joints)).sum())
        score = int(wrist_counts.sum())
        if hit_cameras >= int(min_wrist_cameras):
            scored.append((score, hit_cameras, int(label_idx), int(frame_idx), int(dataset_idx)))

    if len(scored) < samples:
        fallback = []
        for dataset_idx in candidate_indices:
            label_idx, frame_idx = dataset.index[int(dataset_idx)]
            data = dataset._load_label(label_idx)
            score = int(wrist_visible_counts(data, int(frame_idx)).sum())
            if score > 0:
                fallback.append((score, 0, int(label_idx), int(frame_idx), int(dataset_idx)))
        scored = sorted({item[-1]: item for item in scored + fallback}.values(), reverse=True)
    else:
        scored = sorted(scored, reverse=True)

    selected = []
    used_labels = set()
    for item in scored:
        if item[2] in used_labels:
            continue
        selected.append(item[-1])
        used_labels.add(item[2])
        if len(selected) >= samples:
            break
    if len(selected) < samples:
        for item in scored:
            if item[-1] in selected:
                continue
            selected.append(item[-1])
            if len(selected) >= samples:
                break
    if not selected:
        return rng.choice(candidate_indices, size=min(samples, len(candidate_indices)), replace=False)
    return np.asarray(selected[:samples], dtype=np.int64)


def wrist_visible_counts(label_data: dict[str, object], frame_idx: int) -> np.ndarray:
    visible = np.asarray(label_data["wrist_visible"], dtype=bool)[int(frame_idx)]
    camera_names = [str(name) for name in label_data["camera_names"]]
    wrist_indices = [idx for idx, name in enumerate(camera_names) if "wrist" in name]
    if not wrist_indices:
        return np.zeros(0, dtype=np.int32)
    return visible[wrist_indices].sum(axis=1).astype(np.int32)


def wrist_visible_score(label_data: dict[str, object], frame_idx: int) -> int:
    return int(wrist_visible_counts(label_data, frame_idx).sum())


def masked_heatmap_mse(pred, target, mask):
    weight = mask[..., None, None]
    denom = (weight.sum() * pred.shape[-1] * pred.shape[-2]).clamp_min(1.0)
    return ((pred - target) ** 2 * weight).sum() / denom


def save_sample_visualization(
    *,
    output_dir: Path,
    sample_idx: int,
    image: np.ndarray,
    head_target: np.ndarray,
    wrist_target: np.ndarray,
    pred: dict[str, np.ndarray],
    camera_names: list[str],
    frame_idx: int,
    label_path: str,
    label_data: dict[str, object],
    highres: bool,
    save_individual_panels: bool,
) -> None:
    panels = []
    highres_frames = load_original_frames(label_data, frame_idx) if highres else None
    for view_idx, camera_name in enumerate(camera_names):
        is_wrist = "wrist" in str(camera_name)
        branch = "wrist" if is_wrist else "head"
        target = wrist_target[view_idx] if is_wrist else head_target[view_idx]
        pred_view = pred[branch][view_idx]
        rgb = highres_frames[view_idx] if highres_frames is not None else denormalize(image[view_idx])
        gt = target.max(axis=0)
        pr = pred_view.max(axis=0)
        pr = (pr - pr.min()) / max(float(pr.max() - pr.min()), 1e-6)
        gt_up = cv2.resize(gt, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        pr_up = cv2.resize(pr, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        overlay = make_contrast_overlay(rgb, gt_up, pr_up)
        draw_prediction_peaks(overlay, pred_view, rgb.shape[1], rgb.shape[0], max_peaks=16)
        visible_key = "wrist_visible" if is_wrist else "head_visible"
        visible_count = int(np.asarray(label_data[visible_key], dtype=bool)[int(frame_idx), view_idx].sum())
        font_scale = 1.35 if highres else 0.55
        thickness = 3 if highres else 2
        cv2.putText(
            overlay,
            f"{camera_name}  visible={visible_count}",
            (18 if highres else 8, 44 if highres else 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            "GT red / pred cyan+cross",
            (18 if highres else 8, 96 if highres else 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale * 0.78,
            (255, 255, 255),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )
        if save_individual_panels:
            panel_path = output_dir / f"sample_{sample_idx:04d}_frame_{frame_idx:06d}_{camera_name}.jpg"
            cv2.imwrite(str(panel_path), cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR))
        panels.append(overlay.astype(np.uint8))
    top = np.hstack(panels[:4])
    bottom = np.hstack(panels[4:])
    grid = np.vstack([top, bottom])
    suffix = "highres" if highres else "trainres"
    out_path = output_dir / f"sample_{sample_idx:04d}_frame_{frame_idx:06d}_{suffix}.jpg"
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    meta = {
        "sample_idx": int(sample_idx),
        "frame_idx": int(frame_idx),
        "label_path": label_path,
        "camera_names": [str(name) for name in camera_names],
        "visible_per_camera": [
            int(
                np.asarray(label_data["wrist_visible" if "wrist" in str(name) else "head_visible"], dtype=bool)[
                    int(frame_idx), view_idx
                ].sum()
            )
            for view_idx, name in enumerate(camera_names)
        ],
    }
    (output_dir / f"sample_{sample_idx:04d}_frame_{frame_idx:06d}.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )


def make_contrast_overlay(rgb: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    overlay = (rgb.astype(np.float32) * 0.62).astype(np.uint8)
    red = np.asarray([255, 32, 32], dtype=np.float32)
    gt_mask = gt > 0.06
    if gt_mask.any():
        alpha = np.clip(gt[..., None] * 1.6, 0.0, 0.72)
        overlay = np.where(gt_mask[..., None], overlay * (1.0 - alpha) + red * alpha, overlay)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    draw_heatmap_contours(overlay, gt, (255, 32, 32), threshold=0.10)
    return overlay


def draw_heatmap_contours(image: np.ndarray, heatmap: np.ndarray, color: tuple[int, int, int], threshold: float) -> None:
    mask = (heatmap > float(threshold)).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(image, contours, -1, color, 1, cv2.LINE_AA)


def draw_prediction_peaks(image: np.ndarray, pred: np.ndarray, width: int, height: int, *, max_peaks: int) -> None:
    pred = np.asarray(pred, dtype=np.float32)
    max_values = pred.reshape(pred.shape[0], -1).max(axis=1)
    if max_values.size == 0 or float(max_values.max()) <= 1e-8:
        return
    keep = np.argsort(max_values)[-int(max_peaks) :]
    min_value = max(float(max_values.max()) * 0.03, 1e-8)
    scale_x = float(width) / float(pred.shape[-1])
    scale_y = float(height) / float(pred.shape[-2])
    for joint_idx in keep:
        if float(max_values[joint_idx]) < min_value:
            continue
        y, x = np.unravel_index(int(np.argmax(pred[joint_idx])), pred[joint_idx].shape)
        cx = int((float(x) + 0.5) * scale_x)
        cy = int((float(y) + 0.5) * scale_y)
        radius = 9 if width > 1000 else 4
        thickness = 3 if width > 1000 else 2
        color = (0, 255, 255)
        cv2.drawMarker(
            image,
            (cx, cy),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=radius * 3,
            thickness=thickness,
            line_type=cv2.LINE_AA,
        )
        cv2.circle(image, (cx, cy), radius, color, thickness, cv2.LINE_AA)


def load_original_frames(label_data: dict[str, object], frame_idx: int) -> list[np.ndarray]:
    frames = []
    for video_path in label_data["video_paths"]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return frames


def denormalize(img: np.ndarray) -> np.ndarray:
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]
    rgb = np.clip((img * std + mean), 0.0, 1.0)
    return (rgb.transpose(1, 2, 0) * 255.0).astype(np.uint8)


if __name__ == "__main__":
    raise SystemExit(main())
