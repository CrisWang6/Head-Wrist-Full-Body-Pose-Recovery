"""Train the two-view heatmap refinement stage."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset import HeadStereoHeatmapDataset
from model import StereoHeatmapRefiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-joints", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def heatmap_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    positive_weight = 1.0 + 9.0 * target
    return (positive_weight * (prediction - target).square()).mean()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    for batch in loader:
        images = batch["images"].to(device, non_blocking=True)
        initial = batch["initial_heatmaps"].to(device, non_blocking=True)
        target = batch["gt_heatmaps"].to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            refined = model(images, initial)
            loss = heatmap_loss(refined, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total_loss += loss.item() * images.shape[0]
        total_samples += images.shape[0]
    return total_loss / max(total_samples, 1)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = HeadStereoHeatmapDataset(args.manifest)
    val_size = max(1, round(len(dataset) * args.val_ratio))
    train_size = len(dataset) - val_size
    if train_size < 1:
        raise ValueError("the manifest needs at least two samples")
    train_set, val_set = random_split(
        dataset,
        (train_size, val_size),
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(train_set, shuffle=True, drop_last=False, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_kwargs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StereoHeatmapRefiner(args.num_joints).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, (8, 10), gamma=0.1)
    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_loss = run_epoch(model, val_loader, device, None)
        scheduler.step()
        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(record)
        print(json.dumps(record), flush=True)
        checkpoint = {
            "model": model.state_dict(),
            "num_joints": args.num_joints,
            "epoch": epoch,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, output_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(checkpoint, output_dir / "best.pt")

    (output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
