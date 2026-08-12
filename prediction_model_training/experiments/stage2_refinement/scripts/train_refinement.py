#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egorear_sim2d.dataset import MultiViewHeatmapDataset, discover_label_files, torch_collate
from egorear_sim2d.splits import load_split_manifest
from egorear_sim2d.refinement import (
    HeadBCHeatmapRefinementNet,
    load_refiner_state,
    load_stage1_model,
    refiner_state_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CAM_B/C stage-2 heatmap refinement.")
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--output-dir", default="checkpoints/head_bc_stage2_refinement")
    parser.add_argument("--log-dir", default="logs/head_bc_stage2_refinement")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-mode", choices=("chronological", "random"), default="chronological")
    parser.add_argument(
        "--split-manifest",
        default="",
        help="Shared train/val/test NPZ. Overrides --train-ratio and --split-mode.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-width", type=int, default=456)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--heatmap-width", type=int, default=114)
    parser.add_argument("--heatmap-height", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--query-dim", type=int, default=256)
    parser.add_argument("--sampling-points", type=int, default=8)
    parser.add_argument("--proposal-source", choices=("stage1", "noisy_gt"), default="stage1")
    parser.add_argument("--proposal-shift-px", type=int, default=5)
    parser.add_argument("--proposal-noise-std", type=float, default=0.03)
    parser.add_argument("--proposal-joint-drop", type=float, default=0.08)
    parser.add_argument("--resume", default="")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--alignment-loss-weight", type=float, default=0.02)
    parser.add_argument("--selection-metric", choices=("loss", "refined_pixel_error"), default="refined_pixel_error")
    parser.add_argument("--min-epochs", type=int, default=8)
    parser.add_argument("--early-stop-patience", type=int, default=12)
    parser.add_argument("--overfit-patience", type=int, default=4)
    parser.add_argument(
        "--overfit-relative-delta",
        type=float,
        default=0.001,
        help="Relative validation degradation required for an overfit streak.",
    )
    parser.add_argument("--overfit-train-val-gap", type=float, default=0.05)
    parser.add_argument("--max-hours", type=float, default=0.0)
    return parser.parse_args()


def noisy_proposals(target, mask, *, max_shift: int, noise_std: float, joint_drop: float):
    import torch

    output = torch.empty_like(target)
    batch, views = target.shape[:2]
    for batch_idx in range(batch):
        for view_idx in range(views):
            shift_y = int(torch.randint(-max_shift, max_shift + 1, (1,), device=target.device).item())
            shift_x = int(torch.randint(-max_shift, max_shift + 1, (1,), device=target.device).item())
            shifted = torch.roll(target[batch_idx, view_idx], (shift_y, shift_x), dims=(-2, -1))
            if shift_y > 0:
                shifted[..., :shift_y, :] = 0
            elif shift_y < 0:
                shifted[..., shift_y:, :] = 0
            if shift_x > 0:
                shifted[..., :, :shift_x] = 0
            elif shift_x < 0:
                shifted[..., :, shift_x:] = 0
            output[batch_idx, view_idx] = shifted
    output = output + torch.randn_like(output) * float(noise_std)
    if joint_drop > 0:
        keep = torch.rand_like(mask) >= float(joint_drop)
        output = output * keep[..., None, None]
    return output.clamp(0.0, 1.0)


def masked_heatmap_mse(pred, target, mask):
    weight = mask[..., None, None]
    denom = (weight.sum() * pred.shape[-1] * pred.shape[-2]).clamp_min(1.0)
    return ((pred - target) ** 2 * weight).sum() / denom


def cross_view_alignment_loss(tokens, mask):
    """Align the two views' semantic joint tokens when both labels are visible."""
    import torch.nn.functional as F

    shared_visible = (mask[:, 0] > 0.5) & (mask[:, 1] > 0.5)
    if not bool(shared_visible.any()):
        return tokens.sum() * 0.0
    left = F.normalize(tokens[:, 0], dim=-1)
    right = F.normalize(tokens[:, 1], dim=-1)
    distance = 1.0 - (left * right).sum(dim=-1)
    return distance[shared_visible].mean()


def heatmap_pixel_error(pred, target, mask, image_size: tuple[int, int]) -> tuple[float, int]:
    import torch

    valid = mask > 0.5
    count = int(valid.sum().item())
    if count == 0:
        return 0.0, 0
    height, width = pred.shape[-2:]
    pred_idx = pred.flatten(-2).argmax(dim=-1)
    target_idx = target.flatten(-2).argmax(dim=-1)
    pred_xy = torch.stack((pred_idx % width, pred_idx // width), dim=-1).float()
    target_xy = torch.stack((target_idx % width, target_idx // width), dim=-1).float()
    scale = pred.new_tensor((image_size[0] / width, image_size[1] / height))
    distance = torch.linalg.vector_norm((pred_xy - target_xy) * scale, dim=-1)
    return float(distance[valid].sum().item()), count


def run_epoch(model, loader, device, args, *, optimizer=None, max_steps: int = 0):
    import torch

    training = optimizer is not None
    model.train(training)
    losses = []
    heatmap_losses = []
    alignment_losses = []
    initial_error = refined_error = 0.0
    metric_count = 0
    for step, batch in enumerate(loader, start=1):
        images = batch["img"].to(device, non_blocking=True).float()
        target = batch["head_gt_heatmap"].to(device, non_blocking=True).float()
        mask = batch["head_loss_mask"].to(device, non_blocking=True).float()
        proposal = None
        if args.proposal_source == "noisy_gt":
            proposal = noisy_proposals(
                target,
                mask,
                max_shift=args.proposal_shift_px,
                noise_std=args.proposal_noise_std,
                joint_drop=args.proposal_joint_drop,
            )
        with torch.set_grad_enabled(training):
            output = model(images, proposal)
            heatmap_loss = masked_heatmap_mse(output["refined"], target, mask)
            alignment_loss = cross_view_alignment_loss(output["joint_tokens"], mask)
            loss = heatmap_loss + float(args.alignment_loss_weight) * alignment_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
        heatmap_losses.append(float(heatmap_loss.detach().cpu()))
        alignment_losses.append(float(alignment_loss.detach().cpu()))
        before_sum, before_count = heatmap_pixel_error(
            output["proposal"].detach(), target, mask, (args.image_width, args.image_height)
        )
        after_sum, after_count = heatmap_pixel_error(
            output["refined"].detach(), target, mask, (args.image_width, args.image_height)
        )
        initial_error += before_sum
        refined_error += after_sum
        metric_count += min(before_count, after_count)
        if max_steps > 0 and step >= max_steps:
            break
    return {
        "loss": float(sum(losses) / max(len(losses), 1)),
        "heatmap_loss": float(sum(heatmap_losses) / max(len(heatmap_losses), 1)),
        "alignment_loss": float(sum(alignment_losses) / max(len(alignment_losses), 1)),
        "initial_pixel_error": float(initial_error / max(metric_count, 1)),
        "refined_pixel_error": float(refined_error / max(metric_count, 1)),
        "batches": len(losses),
    }


def main() -> int:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from torch.utils.tensorboard import SummaryWriter

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    label_files = discover_label_files(Path(args.label_root))
    dataset = MultiViewHeatmapDataset(
        label_files,
        image_size=(args.image_width, args.image_height),
        visible_only_loss=True,
    )
    first = dataset[0]
    camera_order = [str(name) for name in first["camera_names"]]
    if len(camera_order) != 2:
        raise ValueError(f"Expected 2 camera views, got {camera_order}")
    num_joints = int(first["head_gt_heatmap"].shape[1])
    joint_names = [str(name) for name in first.get("head_joint_names", [])]

    dataset_frames = np.asarray(
        [
            int(dataset._load_label(label_idx)["frame_indices"][frame_idx])
            for label_idx, frame_idx in dataset.index
        ],
        dtype=np.int64,
    )
    if args.split_manifest:
        split_manifest = load_split_manifest(
            args.split_manifest,
            expected_length=len(dataset),
            expected_frame_indices=dataset_frames,
        )
        train_indices = split_manifest["train_indices"].astype(np.int64)
        val_indices = split_manifest["val_indices"].astype(np.int64)
        test_indices = split_manifest["test_indices"].astype(np.int64)
    else:
        indices = np.arange(len(dataset))
        if args.split_mode == "random":
            np.random.default_rng(args.seed).shuffle(indices)
        split = max(1, min(len(indices) - 1, int(round(len(indices) * args.train_ratio))))
        train_indices, val_indices = indices[:split], indices[split:]
        test_indices = np.asarray([], dtype=np.int64)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "collate_fn": torch_collate,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(Subset(dataset, train_indices.tolist()), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(Subset(dataset, val_indices.tolist()), shuffle=False, **loader_kwargs)

    stage1 = load_stage1_model(
        args.stage1_checkpoint,
        base_channels=args.base_channels,
        num_head_heatmaps=num_joints,
    )
    model = HeadBCHeatmapRefinementNet(
        stage1,
        num_joints=num_joints,
        heatmap_size=(args.heatmap_width, args.heatmap_height),
        base_channels=args.base_channels,
        query_dim=args.query_dim,
        sampling_points=args.sampling_points,
        freeze_stage1=True,
    )
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data_parallel = device.type == "cuda" and torch.cuda.device_count() > 1
    if data_parallel:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    start_epoch = 1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        load_refiner_state(model, checkpoint["refiner"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args) | {
        "label_files": [str(path) for path in label_files],
        "train_frames": len(train_indices),
        "val_frames": len(val_indices),
        "test_frames": len(test_indices),
        "device_resolved": str(device),
        "camera_order": camera_order,
        "joint_names": joint_names,
        "num_joints": num_joints,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_device_count": torch.cuda.device_count(),
        "data_parallel": data_parallel,
    }
    (output_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    writer = SummaryWriter(str(log_dir))
    history_path = output_dir / "history.jsonl"
    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    overfit_streak = 0
    started_at = time.time()

    with torch.no_grad():
        baseline = run_epoch(model, val_loader, device, args, max_steps=args.max_steps)
    (output_dir / "stage1_baseline.json").write_text(
        json.dumps(baseline, indent=2), encoding="utf-8"
    )
    print(json.dumps({"stage1_baseline": baseline}, indent=2), flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, args, optimizer=optimizer, max_steps=args.max_steps)
        with torch.no_grad():
            val_metrics = run_epoch(model, val_loader, device, args, max_steps=args.max_steps)
        selection_value = float(val_metrics[args.selection_metric])
        improved = selection_value < best_val
        if improved:
            best_val = selection_value
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        train_value = float(train_metrics[args.selection_metric])
        obvious_gap = train_value < selection_value * (1.0 - args.overfit_train_val_gap)
        degraded = selection_value > best_val * (1.0 + args.overfit_relative_delta)
        overfit_streak = overfit_streak + 1 if obvious_gap and degraded else 0
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "selection_metric": args.selection_metric,
            "selection_value": selection_value,
            "best_epoch": best_epoch,
            "best_value": best_val,
            "epochs_without_improvement": epochs_without_improvement,
            "overfit_streak": overfit_streak,
        }
        print(json.dumps(record, indent=2), flush=True)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        for phase, metrics in (("train", train_metrics), ("val", val_metrics)):
            for name, value in metrics.items():
                if name != "batches":
                    writer.add_scalar(f"{phase}/{name}", value, epoch)
        checkpoint = {
            "epoch": epoch,
            "refiner": refiner_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "train": train_metrics,
            "val": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if improved:
            torch.save(checkpoint, output_dir / "best.pt")
        writer.flush()
        stop_reason = ""
        if epoch >= args.min_epochs and epochs_without_improvement >= args.early_stop_patience:
            stop_reason = "validation_plateau"
        elif args.max_hours > 0 and (time.time() - started_at) / 3600.0 >= args.max_hours:
            stop_reason = "max_hours"
        status = {
            "state": "stopped" if stop_reason else "running",
            "reason": stop_reason,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_value": best_val,
            "selection_metric": args.selection_metric,
        }
        (output_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
        if stop_reason:
            print(json.dumps({"early_stop": status}, indent=2), flush=True)
            break
    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
