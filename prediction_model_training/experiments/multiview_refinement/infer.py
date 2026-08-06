"""Run stage-two refinement and save CAM_B/C heatmaps as compressed NPZ files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import HeadStereoHeatmapDataset
from model import StereoHeatmapRefiner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = StereoHeatmapRefiner(checkpoint["num_joints"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dataset = HeadStereoHeatmapDataset(args.manifest)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for batch in loader:
            refined = model(
                batch["images"].to(device), batch["initial_heatmaps"].to(device)
            ).cpu().numpy()
            initial = batch["initial_heatmaps"].numpy()
            for index, sample_id in enumerate(batch["sample_id"]):
                np.savez_compressed(
                    output_dir / f"{sample_id}.npz",
                    initial_heatmap_b=initial[index, 0],
                    initial_heatmap_c=initial[index, 1],
                    refined_heatmap_b=refined[index, 0],
                    refined_heatmap_c=refined[index, 1],
                )


if __name__ == "__main__":
    main()
