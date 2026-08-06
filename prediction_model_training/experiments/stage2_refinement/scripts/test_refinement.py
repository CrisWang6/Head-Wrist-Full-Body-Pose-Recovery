#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egorear_sim2d.dataset import MultiViewHeatmapDataset, discover_label_files
from egorear_sim2d.refinement import HeadBCHeatmapRefinementNet, load_refiner_state, load_stage1_model
from train_refinement import heatmap_pixel_error, masked_heatmap_mse, noisy_proposals


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate CAM_B/C stage-2 refinement.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/head_bc_stage2_refinement")
    parser.add_argument("--samples", type=int, default=0, help="0 evaluates the complete validation split")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--proposal-source", choices=("stage1", "noisy_gt"), default="stage1")
    args = parser.parse_args()

    import torch

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", {})
    stage1 = load_stage1_model(
        args.stage1_checkpoint,
        base_channels=int(config.get("base_channels", 64)),
    )
    model = HeadBCHeatmapRefinementNet(
        stage1,
        num_joints=int(config.get("num_joints", 12)),
        heatmap_size=(int(config.get("heatmap_width", 114)), int(config.get("heatmap_height", 64))),
        base_channels=int(config.get("base_channels", 64)),
        query_dim=int(config.get("query_dim", 256)),
        sampling_points=int(config.get("sampling_points", 8)),
    )
    load_refiner_state(model, checkpoint["refiner"])
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    dataset = MultiViewHeatmapDataset(
        discover_label_files(Path(args.label_root)),
        image_size=(int(config.get("image_width", 456)), int(config.get("image_height", 256))),
        visible_only_loss=True,
    )
    all_indices = np.arange(len(dataset))
    if config.get("split_mode", "chronological") == "random":
        np.random.default_rng(int(config.get("seed", 42))).shuffle(all_indices)
    split = max(
        1,
        min(
            len(all_indices) - 1,
            int(round(len(all_indices) * float(config.get("train_ratio", 0.8)))),
        ),
    )
    val_indices = all_indices[split:]
    requested = int(args.samples)
    if requested > 0 and requested < len(val_indices):
        positions = np.linspace(0, len(val_indices) - 1, requested, dtype=np.int64)
        indices = val_indices[positions]
    else:
        indices = val_indices
    count = len(indices)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_error = refined_error = loss_sum = 0.0
    metric_count = 0

    with torch.inference_mode():
        for output_idx, dataset_idx in enumerate(indices):
            item = dataset[int(dataset_idx)]
            images = torch.as_tensor(item["img"][None]).to(device).float()
            target = torch.as_tensor(item["head_gt_heatmap"][None]).to(device).float()
            mask = torch.as_tensor(item["head_loss_mask"][None]).to(device).float()
            proposal = None
            if args.proposal_source == "noisy_gt":
                proposal = noisy_proposals(
                    target,
                    mask,
                    max_shift=int(config.get("proposal_shift_px", 5)),
                    noise_std=float(config.get("proposal_noise_std", 0.03)),
                    joint_drop=float(config.get("proposal_joint_drop", 0.08)),
                )
            result = model(images, proposal)
            loss_sum += float(masked_heatmap_mse(result["refined"], target, mask).cpu())
            before_sum, before_count = heatmap_pixel_error(
                result["proposal"], target, mask, (int(config.get("image_width", 456)), int(config.get("image_height", 256)))
            )
            after_sum, after_count = heatmap_pixel_error(
                result["refined"], target, mask, (int(config.get("image_width", 456)), int(config.get("image_height", 256)))
            )
            initial_error += before_sum
            refined_error += after_sum
            metric_count += min(before_count, after_count)
            if output_idx < 16:
                np.savez_compressed(
                    output_dir / f"seq_{int(item['frame_idx']):06d}.npz",
                    proposal=result["proposal"][0].cpu().numpy(),
                    refined=result["refined"][0].cpu().numpy(),
                    ground_truth=target[0].cpu().numpy(),
                    visible=mask[0].cpu().numpy(),
                )

    summary = {
        "samples": count,
        "split": "validation",
        "proposal_source": args.proposal_source,
        "mse": loss_sum / max(count, 1),
        "initial_pixel_error": initial_error / max(metric_count, 1),
        "refined_pixel_error": refined_error / max(metric_count, 1),
        "improvement_px": (initial_error - refined_error) / max(metric_count, 1),
        "improvement_percent": (
            100.0 * (initial_error - refined_error) / max(initial_error, 1e-12)
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
