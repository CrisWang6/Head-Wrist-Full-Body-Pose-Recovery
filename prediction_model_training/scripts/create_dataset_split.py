#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egorear_sim2d.dataset import MultiViewHeatmapDataset, discover_label_files
from egorear_sim2d.splits import (
    create_chronological_holdout_split,
    create_random_frame_split,
    create_strided_random_split,
    create_uniform_temporal_split,
    save_split_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a reproducible frame-level train/val/test split."
    )
    parser.add_argument("--label-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=("random", "uniform-temporal", "strided-random", "chronological-holdout"),
        default="random",
        help=(
            "random performs one global shuffle; uniform-temporal enforces quotas "
            "per block; strided-random keeps every Nth frame before shuffling; "
            "chronological-holdout uses the prefix for training and tail for evaluation."
        ),
    )
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = MultiViewHeatmapDataset(discover_label_files(Path(args.label_root)))
    frame_indices = np.asarray(
        [
            int(dataset._load_label(label_idx)["frame_indices"][frame_idx])
            for label_idx, frame_idx in dataset.index
        ],
        dtype=np.int64,
    )
    if args.mode == "random":
        split = create_random_frame_split(
            frame_indices,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
    elif args.mode == "chronological-holdout":
        split = create_chronological_holdout_split(
            frame_indices,
            train_ratio=args.train_ratio,
        )
    elif args.mode == "uniform-temporal":
        split = create_uniform_temporal_split(
            frame_indices,
            seed=args.seed,
            block_size=args.block_size,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
    else:
        split = create_strided_random_split(
            frame_indices,
            stride=args.stride,
            offset=args.offset,
            seed=args.seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
    save_split_manifest(args.output, split)
    print(
        json.dumps(
            {
                "output": str(Path(args.output).expanduser().resolve()),
                "total": len(frame_indices),
                "train": len(split["train_indices"]),
                "val": len(split["val_indices"]),
                "test": len(split["test_indices"]),
                "block_size": int(np.asarray(split.get("block_size", [0])).reshape(-1)[0]),
                "stride": int(np.asarray(split.get("stride", [1])).reshape(-1)[0]),
                "offset": int(np.asarray(split.get("offset", [0])).reshape(-1)[0]),
                "mode": args.mode,
                "seed": args.seed,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
