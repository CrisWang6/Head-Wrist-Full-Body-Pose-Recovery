#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egorear_sim2d.dataset import MultiViewHeatmapDataset, discover_label_files, torch_collate
from egorear_sim2d.pose3d import (
    EgoRearPose3DNet,
    EgoRearStage3Pipeline,
    EgoRearStage3Stage1Pipeline,
)
from egorear_sim2d.refinement import HeadBCHeatmapRefinementNet, load_refiner_state, load_stage1_model
from egorear_sim2d.splits import load_split_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train EgoRear-style CAM_B/C stage-3 3D pose lifting.")
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--pose3d-labels", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument(
        "--stage2-checkpoint",
        default="",
        help="Stage-2 refiner checkpoint. Omit when using --skip-stage2.",
    )
    parser.add_argument(
        "--skip-stage2",
        action="store_true",
        help="Lift 3D directly from frozen stage-1 heatmaps (skip stage-2 refinement).",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument(
        "--split-manifest",
        default="",
        help="Shared train/val/test NPZ. Invalid 3D targets are filtered after membership is loaded.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--min-epochs", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--resume", default="", help="Resume pose3d and optimizer state from a checkpoint.")
    return parser.parse_args()


def mpjpe(prediction, target):
    return (prediction - target).square().sum(dim=-1).sqrt().mean()


def run_epoch(model, loader, pose_targets, device, joint_count, *, optimizer=None, max_steps=0):
    import torch

    training = optimizer is not None
    model.train(training)
    total_error = np.zeros(joint_count, dtype=np.float64)
    total_count = 0
    losses = []
    proposal_losses = []
    for step, batch in enumerate(loader, start=1):
        images = batch["img"].to(device, non_blocking=True).float()
        global_idx = batch["global_idx"].detach().cpu().numpy().astype(np.int64)
        target = torch.as_tensor(pose_targets[global_idx], device=device, dtype=torch.float32)
        with torch.set_grad_enabled(training):
            output = model(images)
            final_loss = mpjpe(output["pose3d"], target)
            proposal_loss = mpjpe(output["proposal"], target)
            loss = final_loss + 0.25 * proposal_loss
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        error = torch.linalg.vector_norm(output["pose3d"].detach() - target, dim=-1)
        total_error += error.sum(dim=0).cpu().numpy()
        total_count += int(error.shape[0])
        losses.append(float(final_loss.detach().cpu()))
        proposal_losses.append(float(proposal_loss.detach().cpu()))
        if max_steps > 0 and step >= max_steps:
            break
    per_joint_m = total_error / max(total_count, 1)
    return {
        "mpjpe_m": float(per_joint_m.mean()),
        "mpjpe_mm": float(per_joint_m.mean() * 1000.0),
        "proposal_mpjpe_m": float(np.mean(proposal_losses)) if proposal_losses else 0.0,
        "per_joint_mm": (per_joint_m * 1000.0).tolist(),
        "batches": len(losses),
    }


def main() -> int:
    import torch
    from torch.utils.data import DataLoader, Subset
    import torch
    from torch.utils.tensorboard import SummaryWriter

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = MultiViewHeatmapDataset(
        discover_label_files(Path(args.label_root)),
        image_size=(480, 300),
        visible_only_loss=True,
    )
    pose_labels = np.load(args.pose3d_labels, allow_pickle=True)
    pose_frames = np.asarray(pose_labels["frame_indices"], dtype=np.int64)
    pose_values = np.asarray(pose_labels["pose_head_m"], dtype=np.float32)
    pose_valid = np.asarray(pose_labels["valid"], dtype=bool)
    joint_names = [str(value) for value in pose_labels["joint_names"]]
    if len(pose_frames) != len(dataset):
        raise ValueError(f"2D/3D label length mismatch: {len(dataset)} vs {len(pose_frames)}")
    dataset_frames = np.asarray(
        [int(dataset._load_label(label_idx)["frame_indices"][frame_idx]) for label_idx, frame_idx in dataset.index]
    )
    if not np.array_equal(dataset_frames, pose_frames):
        raise ValueError("2D and 3D frame_indices are not exactly aligned")
    pose_targets = pose_values.astype(np.float32)

    class _IndexedSubset(torch.utils.data.Dataset):
        def __init__(self, base, indices):
            self.base = base
            self.indices = np.asarray(indices, dtype=np.int64)

        def __len__(self):
            return int(self.indices.shape[0])

        def __getitem__(self, idx):
            global_idx = int(self.indices[idx])
            sample = dict(self.base[global_idx])
            sample["global_idx"] = np.int64(global_idx)
            return sample

    def _collate_with_global_idx(batch):
        global_idx = torch.as_tensor([int(item.pop("global_idx")) for item in batch])
        collated = torch_collate(batch)
        collated["global_idx"] = global_idx
        return collated
    if args.split_manifest:
        split_manifest = load_split_manifest(
            args.split_manifest,
            expected_length=len(dataset),
            expected_frame_indices=dataset_frames,
        )
        train_indices = split_manifest["train_indices"][
            pose_valid[split_manifest["train_indices"]]
        ].astype(np.int64)
        val_indices = split_manifest["val_indices"][
            pose_valid[split_manifest["val_indices"]]
        ].astype(np.int64)
        test_indices = split_manifest["test_indices"][
            pose_valid[split_manifest["test_indices"]]
        ].astype(np.int64)
    else:
        split = max(1, min(len(dataset) - 1, int(round(len(dataset) * args.train_ratio))))
        train_indices = np.flatnonzero(pose_valid & (np.arange(len(dataset)) < split))
        val_indices = np.flatnonzero(pose_valid & (np.arange(len(dataset)) >= split))
        test_indices = np.asarray([], dtype=np.int64)
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
        "persistent_workers": args.workers > 0,
        "collate_fn": torch_collate,
    }
    train_loader = DataLoader(_IndexedSubset(dataset, train_indices), shuffle=True, **{k: v for k, v in loader_args.items() if k != 'collate_fn'}, collate_fn=_collate_with_global_idx)
    val_loader = DataLoader(_IndexedSubset(dataset, val_indices), shuffle=False, **{k: v for k, v in loader_args.items() if k != 'collate_fn'}, collate_fn=_collate_with_global_idx)

    stage1 = load_stage1_model(args.stage1_checkpoint, num_head_heatmaps=len(joint_names))
    pose3d = EgoRearPose3DNet(num_joints=len(joint_names))
    resume_checkpoint = None
    start_epoch = 1
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        pose3d.load_state_dict(resume_checkpoint["pose3d"], strict=True)
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
    if args.skip_stage2:
        if args.stage2_checkpoint:
            print(
                json.dumps(
                    {
                        "warning": "--stage2-checkpoint ignored because --skip-stage2 is set",
                        "stage2_checkpoint": args.stage2_checkpoint,
                    },
                    indent=2,
                ),
                flush=True,
            )
        pipeline = EgoRearStage3Stage1Pipeline(stage1, pose3d)
    else:
        if not args.stage2_checkpoint:
            raise ValueError("--stage2-checkpoint is required unless --skip-stage2 is set")
        stage2_checkpoint = torch.load(args.stage2_checkpoint, map_location="cpu", weights_only=False)
        stage2_config = stage2_checkpoint.get("config", {})
        stage2 = HeadBCHeatmapRefinementNet(
            stage1,
            num_joints=len(joint_names),
            heatmap_size=(int(stage2_config.get("heatmap_width", 114)), int(stage2_config.get("heatmap_height", 64))),
            base_channels=int(stage2_config.get("base_channels", 64)),
            query_dim=int(stage2_config.get("query_dim", 256)),
            sampling_points=int(stage2_config.get("sampling_points", 8)),
            freeze_stage1=True,
        )
        load_refiner_state(stage2, stage2_checkpoint["refiner"])
        pipeline = EgoRearStage3Pipeline(stage2, pose3d)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        pipeline = torch.nn.DataParallel(pipeline)
    trainable = (pipeline.module if hasattr(pipeline, "module") else pipeline).pose3d.parameters()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    if resume_checkpoint is not None and "optimizer" in resume_checkpoint:
        optimizer.load_state_dict(resume_checkpoint["optimizer"])

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = output_dir / "tensorboard"
    writer = SummaryWriter(str(tensorboard_dir))
    config = vars(args) | {
        "joint_names": joint_names,
        "skip_stage2": bool(args.skip_stage2),
        "coordinate_frame": "0806_nose_translation_offset_m",
        "target_unit": "meter",
        "train_frames": len(train_indices),
        "val_frames": len(val_indices),
        "test_frames": len(test_indices),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_device_count": torch.cuda.device_count(),
    }
    (output_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history_path = output_dir / "history.jsonl"
    if history_path.exists():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            historical = json.loads(line)
            historical_epoch = int(historical["epoch"])
            for phase in ("train", "val"):
                metrics = historical[phase]
                writer.add_scalar(f"{phase}/mpjpe_mm", metrics["mpjpe_mm"], historical_epoch)
                writer.add_scalar(
                    f"{phase}/proposal_mpjpe_mm",
                    metrics["proposal_mpjpe_m"] * 1000.0,
                    historical_epoch,
                )
                for name, value in zip(joint_names, metrics["per_joint_mm"]):
                    writer.add_scalar(
                        f"{phase}_joint_mm/{name}", value, historical_epoch
                    )
        writer.flush()
    best_checkpoint = resume_checkpoint
    existing_best_path = output_dir / "best.pt"
    if existing_best_path.exists():
        existing_best = torch.load(existing_best_path, map_location="cpu", weights_only=False)
        if (
            best_checkpoint is None
            or float(existing_best["val"]["mpjpe_mm"])
            <= float(best_checkpoint["val"]["mpjpe_mm"])
        ):
            best_checkpoint = existing_best
    best_value = (
        float(best_checkpoint["val"]["mpjpe_mm"])
        if best_checkpoint is not None
        else float("inf")
    )
    best_epoch = int(best_checkpoint.get("epoch", 0)) if best_checkpoint is not None else 0
    stale = 0
    if resume_checkpoint is not None:
        if not existing_best_path.exists():
            torch.save(best_checkpoint, existing_best_path)
        resumed_metrics = {
            "epoch": best_epoch,
            "mpjpe_mm": best_value,
            "joint_names": joint_names,
            "per_joint_mm": {
                name: value
                for name, value in zip(joint_names, best_checkpoint["val"]["per_joint_mm"])
            },
            "alignment": "prediction and GT are both expressed in the same mocap Head coordinate frame",
            "resumed_source": str(Path(args.resume).expanduser().resolve()),
        }
        (output_dir / "best_metrics.json").write_text(
            json.dumps(resumed_metrics, indent=2), encoding="utf-8"
        )
        writer.add_scalar("val/mpjpe_mm", best_value, best_epoch)
        for name, value in resumed_metrics["per_joint_mm"].items():
            writer.add_scalar(f"val_joint_mm/{name}", value, best_epoch)
    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = run_epoch(
            pipeline, train_loader, pose_targets, device, len(joint_names),
            optimizer=optimizer, max_steps=args.max_steps,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                pipeline, val_loader, pose_targets, device, len(joint_names),
                max_steps=args.max_steps,
            )
        current = float(val_metrics["mpjpe_mm"])
        improved = current < best_value
        if improved:
            best_value = current
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        record = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "best_epoch": best_epoch,
            "best_mpjpe_mm": best_value,
            "epochs_without_improvement": stale,
        }
        print(json.dumps(record, indent=2), flush=True)
        writer.add_scalar("train/mpjpe_mm", train_metrics["mpjpe_mm"], epoch)
        writer.add_scalar("val/mpjpe_mm", val_metrics["mpjpe_mm"], epoch)
        writer.add_scalar(
            "train/proposal_mpjpe_mm",
            train_metrics["proposal_mpjpe_m"] * 1000.0,
            epoch,
        )
        writer.add_scalar(
            "val/proposal_mpjpe_mm",
            val_metrics["proposal_mpjpe_m"] * 1000.0,
            epoch,
        )
        for phase, metrics in (("train", train_metrics), ("val", val_metrics)):
            for name, value in zip(joint_names, metrics["per_joint_mm"]):
                writer.add_scalar(f"{phase}_joint_mm/{name}", value, epoch)
        writer.flush()
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        target = pipeline.module if hasattr(pipeline, "module") else pipeline
        checkpoint = {
            "epoch": epoch,
            "pose3d": target.pose3d.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "train": train_metrics,
            "val": val_metrics,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if improved:
            torch.save(checkpoint, output_dir / "best.pt")
            comparison = {
                "epoch": epoch,
                "mpjpe_mm": current,
                "joint_names": joint_names,
                "per_joint_mm": {
                    name: value for name, value in zip(joint_names, val_metrics["per_joint_mm"])
                },
                "alignment": "prediction and GT are both expressed in the same mocap Head coordinate frame",
            }
            (output_dir / "best_metrics.json").write_text(
                json.dumps(comparison, indent=2), encoding="utf-8"
            )
        if epoch >= args.min_epochs and stale >= args.early_stop_patience:
            break
    status = {
        "state": "stopped",
        "reason": "validation_converged",
        "last_epoch": epoch,
        "best_epoch": best_epoch,
        "best_mpjpe_mm": best_value,
    }
    (output_dir / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
