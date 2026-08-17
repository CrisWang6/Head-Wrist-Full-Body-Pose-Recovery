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
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egorear_sim2d.dataset import MultiViewHeatmapDataset, discover_label_files, torch_collate
from egorear_sim2d.model import EgoRearStage1HeatmapNet
from egorear_sim2d.splits import load_split_manifest


LEGACY_HEAD_JOINTS_16 = (
    "Head", "Neck", "LeftArm", "RightArm", "LeftForeArm", "RightForeArm",
    "LeftHand", "RightHand", "LeftUpLeg", "RightUpLeg", "LeftLeg", "RightLeg",
    "LeftFoot", "RightFoot", "LeftToeBase", "RightToeBase",
)
HEAD_OUTPUT_KEYS = ("head_branch.head.3.weight", "head_branch.head.3.bias")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train stage-1 2D multiview heatmap estimator.")
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--frame-root", default="", help="Root containing extracted RGB frames. Falls back to video seek if omitted.")
    parser.add_argument("--render-root", default="", help="Render root used to map label source_render_dir to frame paths.")
    parser.add_argument("--output-dir", default="checkpoints/stage1_heatmap")
    parser.add_argument("--log-dir", default="logs/stage1_heatmap")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--split-mode", choices=("random", "chronological"), default="random")
    parser.add_argument(
        "--split-manifest",
        default="",
        help="Shared train/val/test NPZ. Overrides --train-ratio and --split-mode.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--image-width", type=int, default=456)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--visible-only-loss", action="store_true")
    parser.add_argument("--train-branch", default="all", choices=("all", "head", "wrist"))
    parser.add_argument("--resume", default="", help="Checkpoint to load before training.")
    parser.add_argument("--resume-optimizer", action="store_true", help="Also restore optimizer state from --resume.")
    parser.add_argument("--max-steps", type=int, default=0, help="Debug cap per epoch. 0 disables.")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--max-hours", type=float, default=0.0, help="Stop cleanly after this wall-clock duration. 0 disables.")
    parser.add_argument("--save-every", type=int, default=1, help="Keep a numbered checkpoint every N epochs. 0 disables.")
    parser.add_argument("--keep-last", type=int, default=3, help="Number of numbered checkpoints to retain.")
    parser.add_argument(
        "--joint-radius-config",
        default="",
        help="JSON file with per-joint Gaussian blob radius in source video pixels.",
    )
    parser.add_argument(
        "--default-joint-radius-px",
        type=float,
        default=10.0,
        help="Fallback blob radius (video px) for joints missing from --joint-radius-config.",
    )
    parser.add_argument("--early-stop-patience", type=int, default=15)
    return parser.parse_args()


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
    if not label_files:
        raise FileNotFoundError(f"No heatmap_labels_*.npz found under {args.label_root}")
    head_joint_names: tuple[str, ...] | None = None
    for label_file in label_files:
        with np.load(label_file, allow_pickle=True) as label_data:
            names = tuple(str(name) for name in label_data["head_camera_joints"])
        if head_joint_names is None:
            head_joint_names = names
        elif names != head_joint_names:
            raise ValueError(
                f"Head-joint layouts differ across label files: "
                f"{label_files[0]}={head_joint_names}, {label_file}={names}"
            )
    assert head_joint_names is not None
    joint_radius_px = {name: float(args.default_joint_radius_px) for name in head_joint_names}
    if args.joint_radius_config:
        radius_path = Path(args.joint_radius_config).expanduser()
        from joint_radius_config import load_joint_radius_video_px

        joint_radius_px.update(load_joint_radius_video_px(radius_path))

    dataset = MultiViewHeatmapDataset(
        label_files,
        frame_root=Path(args.frame_root) if args.frame_root else None,
        render_root=Path(args.render_root) if args.render_root else None,
        image_size=(args.image_width, args.image_height),
        visible_only_loss=args.visible_only_loss,
        joint_radius_px=joint_radius_px,
        default_joint_radius_px=args.default_joint_radius_px,
    )
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
        train_indices = split_manifest["train_indices"].astype(int).tolist()
        val_indices = split_manifest["val_indices"].astype(int).tolist()
        test_indices = split_manifest["test_indices"].astype(int).tolist()
    else:
        indices = np.arange(len(dataset))
        if args.split_mode == "random":
            rng = np.random.default_rng(args.seed)
            rng.shuffle(indices)
        split = max(1, min(len(indices) - 1, int(round(len(indices) * args.train_ratio)))) if len(indices) > 1 else len(indices)
        train_indices = indices[:split].tolist()
        val_indices = indices[split:].tolist() if split < len(indices) else indices[: min(8, len(indices))].tolist()
        test_indices = []

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=torch_collate,
        drop_last=False,
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=torch_collate,
        drop_last=False,
        persistent_workers=args.workers > 0,
        prefetch_factor=args.prefetch_factor if args.workers > 0 else None,
    )

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = EgoRearStage1HeatmapNet(
        num_head_heatmaps=len(head_joint_names),
        base_channels=args.base_channels,
    ).to(device)
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 1
    resume_info = {}
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        state = {key.removeprefix("module."): value for key, value in state.items()}
        target_model = model.module if hasattr(model, "module") else model
        source_config = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        source_head_joint_names = tuple(source_config.get("head_joint_names", ()))
        source_output_count = int(state[HEAD_OUTPUT_KEYS[0]].shape[0])
        if not source_head_joint_names:
            if source_output_count == len(LEGACY_HEAD_JOINTS_16):
                source_head_joint_names = LEGACY_HEAD_JOINTS_16
            elif source_output_count == len(head_joint_names):
                source_head_joint_names = head_joint_names
            else:
                raise ValueError(
                    f"Checkpoint has {source_output_count} head heatmaps but no joint-name metadata"
                )
        target_state = target_model.state_dict()
        remapped_channels: list[str] = []
        initialized_channels: list[str] = []
        for key, source_value in state.items():
            if key not in target_state:
                raise KeyError(f"Unexpected checkpoint tensor: {key}")
            if key in HEAD_OUTPUT_KEYS and source_head_joint_names != head_joint_names:
                if source_value.ndim != target_state[key].ndim or source_value.shape[1:] != target_state[key].shape[1:]:
                    raise ValueError(
                        f"Cannot remap checkpoint tensor {key}: "
                        f"{tuple(source_value.shape)} -> {tuple(target_state[key].shape)}"
                    )
                source_index = {name: index for index, name in enumerate(source_head_joint_names)}
                for target_index, name in enumerate(head_joint_names):
                    if name in source_index:
                        target_state[key][target_index].copy_(source_value[source_index[name]])
                continue
            if source_value.shape != target_state[key].shape:
                raise ValueError(
                    f"Checkpoint tensor shape mismatch for {key}: "
                    f"{tuple(source_value.shape)} != {tuple(target_state[key].shape)}"
                )
            target_state[key].copy_(source_value)
        if source_head_joint_names != head_joint_names:
            remapped_channels = [name for name in head_joint_names if name in source_head_joint_names]
            initialized_channels = [name for name in head_joint_names if name not in source_head_joint_names]
        target_model.load_state_dict(target_state, strict=True)
        if args.resume_optimizer and initialized_channels:
            raise ValueError(
                "--resume-optimizer cannot be used when the output layer gains new joint channels"
            )
        if args.resume_optimizer and isinstance(checkpoint, dict) and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        previous_epoch = int(checkpoint.get("epoch", 0)) if isinstance(checkpoint, dict) else 0
        # Loading pretrained weights starts a new run. Epoch/optimizer resume is
        # only requested explicitly with --resume-optimizer.
        start_epoch = max(1, previous_epoch + 1) if args.resume_optimizer else 1
        resume_info = {
            "resume": str(args.resume),
            "resume_epoch": previous_epoch,
            "resume_optimizer": bool(args.resume_optimizer),
            "checkpoint_head_joint_names": list(source_head_joint_names),
            "remapped_head_channels": remapped_channels,
            "randomly_initialized_head_channels": initialized_channels,
        }
        print(json.dumps({"loaded_checkpoint": resume_info}, indent=2), flush=True)

    output_dir = Path(args.output_dir).expanduser().resolve()
    log_dir = Path(args.log_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))
    config = vars(args) | {
        "label_files": [str(path) for path in label_files],
        "head_joint_names": list(head_joint_names),
        "joint_radius_px": joint_radius_px,
        "default_joint_radius_px": args.default_joint_radius_px,
        "num_head_heatmaps": len(head_joint_names),
        "view_weight_sharing": "one shared head_branch is applied to all camera views",
        "train_frames": len(train_indices),
        "val_frames": len(val_indices),
        "test_frames": len(test_indices),
        "device_resolved": str(device),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        **resume_info,
    }
    (output_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    writer.add_text("config/json", json.dumps(config, indent=2))

    best_val = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_path = output_dir / "best.pt"
    if args.resume_optimizer and best_path.exists():
        historical_best = torch.load(best_path, map_location="cpu", weights_only=False)
        historical_val = (
            historical_best.get("val_loss", float("inf"))
            if isinstance(historical_best, dict)
            else float("inf")
        )
        if np.isfinite(historical_val):
            best_val = float(historical_val)
            print(
                json.dumps(
                    {
                        "preserved_historical_best": str(best_path),
                        "best_val_loss": best_val,
                        "best_epoch": int(historical_best.get("epoch", 0)),
                    },
                    indent=2,
                ),
                flush=True,
            )
    started_wall = time.time()
    started_mono = time.monotonic()
    deadline = started_mono + args.max_hours * 3600.0 if args.max_hours > 0 else None
    status_path = output_dir / "training_status.json"
    write_status(
        status_path,
        config=config,
        state="running",
        reason="",
        started_wall=started_wall,
        deadline_wall=started_wall + args.max_hours * 3600.0 if args.max_hours > 0 else None,
        epoch=0,
    )
    stop_reason = "epochs_complete"
    for epoch in range(start_epoch, args.epochs + 1):
        if deadline is not None and time.monotonic() >= deadline:
            stop_reason = "max_hours"
            break
        t0 = time.perf_counter()
        model.train()
        train_result = run_epoch(
            model,
            train_loader,
            device,
            optimizer=optimizer,
            max_steps=args.max_steps,
            writer=writer,
            epoch=epoch,
            phase="train",
            log_every=args.log_every,
            train_branch=args.train_branch,
            deadline=deadline,
            image_size=(args.image_width, args.image_height),
        )
        if train_result["timed_out"]:
            val_result = {"loss": float("nan"), "pixel_error": float("nan"), "timed_out": True, "batches": 0}
        else:
            model.eval()
            with torch.no_grad():
                val_result = run_epoch(
                    model,
                    val_loader,
                    device,
                    optimizer=None,
                    max_steps=args.max_steps,
                    writer=writer,
                    epoch=epoch,
                    phase="val",
                    log_every=args.log_every,
                    train_branch=args.train_branch,
                    deadline=deadline,
                    image_size=(args.image_width, args.image_height),
                )
        train_loss = train_result["loss"]
        val_loss = val_result["loss"]
        elapsed = time.perf_counter() - t0
        writer.add_scalar("epoch/train_loss", train_loss, epoch)
        writer.add_scalar("epoch/train_pixel_error", train_result["pixel_error"], epoch)
        if np.isfinite(val_loss):
            writer.add_scalar("epoch/val_loss", val_loss, epoch)
            writer.add_scalar("epoch/val_pixel_error", val_result["pixel_error"], epoch)
        writer.add_scalar("epoch/seconds", elapsed, epoch)
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_pixel_error": train_result["pixel_error"],
                    "val_pixel_error": val_result["pixel_error"],
                    "seconds": elapsed,
                    "wall_hours": (time.monotonic() - started_mono) / 3600.0,
                },
                indent=2,
            ),
            flush=True,
        )
        checkpoint = {
            "epoch": epoch,
            "model": (model.module if hasattr(model, "module") else model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_pixel_error": train_result["pixel_error"],
            "val_pixel_error": val_result["pixel_error"],
            "config": config,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if args.save_every > 0 and epoch % args.save_every == 0:
            torch.save(checkpoint, output_dir / f"epoch_{epoch:04d}.pt")
            prune_numbered_checkpoints(output_dir, args.keep_last)
        improved = bool(np.isfinite(val_loss) and val_loss < best_val)
        if improved:
            best_val = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(checkpoint, best_path)
        elif np.isfinite(val_loss):
            epochs_without_improvement += 1
        timed_out = bool(train_result["timed_out"] or val_result["timed_out"])
        write_status(
            status_path,
            config=config,
            state="stopping" if timed_out else "running",
            reason="max_hours" if timed_out else "",
            started_wall=started_wall,
            deadline_wall=started_wall + args.max_hours * 3600.0 if args.max_hours > 0 else None,
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_pixel_error=train_result["pixel_error"],
            val_pixel_error=val_result["pixel_error"],
        )
        writer.flush()
        if timed_out:
            stop_reason = "max_hours"
            break
        if epochs_without_improvement >= args.early_stop_patience:
            stop_reason = "validation_plateau"
            break
    write_status(
        status_path,
        config=config,
        state="stopped",
        reason=stop_reason,
        started_wall=started_wall,
        deadline_wall=started_wall + args.max_hours * 3600.0 if args.max_hours > 0 else None,
        epoch=epoch if "epoch" in locals() else 0,
        best_epoch=best_epoch,
        best_val_loss=best_val,
        epochs_without_improvement=epochs_without_improvement,
    )
    writer.close()
    return 0


def run_epoch(
    model,
    loader,
    device,
    *,
    optimizer,
    max_steps: int,
    writer,
    epoch: int,
    phase: str,
    log_every: int,
    train_branch: str,
    deadline: float | None,
    image_size: tuple[int, int],
) -> dict[str, float | int | bool]:
    import torch

    losses = []
    pixel_error_sum = 0.0
    pixel_error_count = 0
    timed_out = False
    for step, batch in enumerate(loader, start=1):
        if deadline is not None and time.monotonic() >= deadline:
            timed_out = True
            break
        img = batch["img"].to(device, non_blocking=True).float()
        target_head = batch["head_gt_heatmap"].to(device, non_blocking=True).float()
        target_wrist = batch["wrist_gt_heatmap"].to(device, non_blocking=True).float()
        mask_head = batch["head_loss_mask"].to(device, non_blocking=True).float()
        mask_wrist = batch["wrist_loss_mask"].to(device, non_blocking=True).float()
        pred = model(img, train_branch)
        head_loss = masked_heatmap_mse(pred["head"], target_head, mask_head) if train_branch in {"all", "head"} else None
        wrist_loss = masked_heatmap_mse(pred["wrist"], target_wrist, mask_wrist) if train_branch in {"all", "wrist"} else None
        if train_branch == "head":
            loss = head_loss
        elif train_branch == "wrist":
            loss = wrist_loss
        else:
            loss = head_loss + wrist_loss
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        metric_pred = pred["head"] if train_branch in {"all", "head"} else pred["wrist"]
        metric_target = target_head if train_branch in {"all", "head"} else target_wrist
        metric_mask = mask_head if train_branch in {"all", "head"} else mask_wrist
        error_sum, error_count = heatmap_pixel_error(metric_pred.detach(), metric_target, metric_mask, image_size)
        pixel_error_sum += error_sum
        pixel_error_count += error_count
        global_step = (epoch - 1) * max(1, len(loader)) + step
        if writer is not None and (step == 1 or (log_every > 0 and step % log_every == 0)):
            writer.add_scalar(f"{phase}/loss", float(loss.detach().cpu()), global_step)
            if head_loss is not None:
                writer.add_scalar(f"{phase}/head_loss", float(head_loss.detach().cpu()), global_step)
            if wrist_loss is not None:
                writer.add_scalar(f"{phase}/wrist_loss", float(wrist_loss.detach().cpu()), global_step)
            if error_count > 0:
                writer.add_scalar(f"{phase}/pixel_error", error_sum / error_count, global_step)
            if step == 1:
                add_heatmap_summary(writer, batch, {key: value.detach() for key, value in pred.items()}, phase, epoch, train_branch)
        if max_steps > 0 and step >= max_steps:
            break
        if deadline is not None and time.monotonic() >= deadline:
            timed_out = True
            break
    return {
        "loss": float(sum(losses) / max(1, len(losses))),
        "pixel_error": float(pixel_error_sum / max(1, pixel_error_count)),
        "timed_out": timed_out,
        "batches": len(losses),
    }


def heatmap_pixel_error(pred, target, mask, image_size: tuple[int, int]) -> tuple[float, int]:
    import torch

    valid = mask > 0.5
    count = int(valid.sum().item())
    if count == 0:
        return 0.0, 0
    height, width = pred.shape[-2:]
    pred_index = pred.flatten(-2).argmax(dim=-1)
    target_index = target.flatten(-2).argmax(dim=-1)
    pred_xy = torch.stack((pred_index % width, pred_index // width), dim=-1).float()
    target_xy = torch.stack((target_index % width, target_index // width), dim=-1).float()
    scale = pred.new_tensor([image_size[0] / width, image_size[1] / height])
    distance = torch.linalg.vector_norm((pred_xy - target_xy) * scale, dim=-1)
    return float(distance[valid].sum().item()), count


def prune_numbered_checkpoints(output_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted(output_dir.glob("epoch_*.pt"))
    for path in checkpoints[:-keep_last]:
        path.unlink()


def write_status(path: Path, *, config: dict, state: str, reason: str, started_wall: float, deadline_wall: float | None, epoch: int, **metrics) -> None:
    import datetime as dt

    def iso(timestamp: float | None) -> str | None:
        return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).astimezone().isoformat() if timestamp is not None else None

    payload = {
        "state": state,
        "reason": reason,
        "pid": os.getpid(),
        "started_at": iso(started_wall),
        "deadline_at": iso(deadline_wall),
        "updated_at": iso(time.time()),
        "epoch": epoch,
        "output_dir": config["output_dir"],
        "log_dir": config["log_dir"],
        **metrics,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    temporary.replace(path)


def masked_heatmap_mse(pred, target, mask):
    if pred.shape[-2:] != target.shape[-2:]:
        raise RuntimeError(f"Prediction shape {tuple(pred.shape)} does not match target {tuple(target.shape)}")
    weight = mask[..., None, None]
    denom = (weight.sum() * pred.shape[-1] * pred.shape[-2]).clamp_min(1.0)
    return ((pred - target) ** 2 * weight).sum() / denom


def add_heatmap_summary(writer, batch, pred, phase: str, epoch: int, train_branch: str = "all") -> None:
    import torch

    branch = "head" if train_branch in {"all", "head"} else "wrist"
    camera_mask_key = "camera_is_head" if branch == "head" else "camera_is_wrist"
    camera_indices = torch.nonzero(batch[camera_mask_key][0].detach().cpu().bool(), as_tuple=False).flatten()
    if len(camera_indices) == 0 and branch == "head":
        camera_indices = torch.nonzero(batch["camera_is_wrist"][0].detach().cpu().bool(), as_tuple=False).flatten()
        branch = "wrist"
    view_idx = int(camera_indices[0]) if len(camera_indices) else 0

    img = batch["img"][0, view_idx].detach().cpu()
    target_key = f"{branch}_gt_heatmap"
    target = batch[target_key][0, view_idx].detach().cpu()
    pred0 = pred[branch][0, view_idx].detach().cpu()
    image = denormalize(img)
    target_max = target.max(dim=0).values.clamp(0, 1)
    pred_max = pred0.clamp(min=0).max(dim=0).values
    pred_vis = (pred_max / pred_max.max().clamp_min(1e-6)).clamp(0, 1)
    target_rgb = torch.stack([target_max, torch.zeros_like(target_max), torch.zeros_like(target_max)], dim=0)
    pred_rgb = torch.stack([torch.zeros_like(pred_vis), torch.zeros_like(pred_vis), pred_vis], dim=0)
    target_rgb = torch.nn.functional.interpolate(target_rgb[None], size=image.shape[-2:], mode="bilinear", align_corners=False)[0]
    pred_rgb = torch.nn.functional.interpolate(pred_rgb[None], size=image.shape[-2:], mode="bilinear", align_corners=False)[0]
    overlay = (0.55 * image + 0.25 * target_rgb + 0.35 * pred_rgb).clamp(0, 1)
    writer.add_image(f"{phase}/overlay_gt_red_pred_blue", overlay, epoch)
    writer.add_image(f"{phase}/gt_heatmap_red", target_rgb, epoch)
    writer.add_image(f"{phase}/pred_heatmap_blue", pred_rgb, epoch)


def denormalize(img):
    import torch

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)


if __name__ == "__main__":
    raise SystemExit(main())
