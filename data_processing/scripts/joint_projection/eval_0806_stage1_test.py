#!/usr/bin/env python3
"""Evaluate Stage1 checkpoint on 0806 test split (2D heatmap / pixel error)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from constants_0806_training import IMAGE_HEIGHT, IMAGE_WIDTH, JOINT_NAMES
from joint_radius_config import JOINT_RADIUS_CONFIG, load_joint_radius_video_px


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label-root", type=Path, required=True)
    p.add_argument("--frame-root", type=Path, required=True)
    p.add_argument("--split-npz", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--scheme", default="v31")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--joint-radius-config",
        type=Path,
        default=JOINT_RADIUS_CONFIG,
    )
    return p.parse_args()


def heatmap_to_uv(pred: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    h, w = pred.shape[-2:]
    flat = pred.reshape(*pred.shape[:-2], -1)
    idx = flat.argmax(axis=-1)
    x = idx % w
    y = idx // w
    iw, ih = image_size
    return np.stack(((x + 0.5) * iw / w, (y + 0.5) * ih / h), axis=-1).astype(np.float32)


def gt_heatmap_to_uv(target: np.ndarray, mask: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    out = heatmap_to_uv(target, image_size)
    out[~mask] = np.nan
    return out


def main() -> int:
    import sys

    ego_root = Path("/home/gaoweijian/EgoRear_w_hand")
    sys.path.insert(0, str(ego_root / "src"))
    jp = Path(__file__).resolve().parent
    if str(jp) not in sys.path:
        sys.path.insert(0, str(jp))

    from egorear_sim2d.dataset import MultiViewHeatmapDataset, discover_label_files, torch_collate
    from egorear_sim2d.model import EgoRearStage1HeatmapNet
    from egorear_sim2d.splits import load_split_manifest

    args = parse_args()
    label_files = discover_label_files(args.label_root.expanduser().resolve())
    joint_radius_px = load_joint_radius_video_px(args.joint_radius_config)

    dataset = MultiViewHeatmapDataset(
        label_files,
        frame_root=args.frame_root.expanduser().resolve(),
        image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        visible_only_loss=True,
        joint_radius_px=joint_radius_px,
        default_joint_radius_px=10.0,
    )
    dataset_frames = np.asarray(
        [int(dataset._load_label(li)["frame_indices"][fi]) for li, fi in dataset.index],
        dtype=np.int64,
    )
    split = load_split_manifest(
        args.split_npz,
        expected_length=len(dataset),
        expected_frame_indices=dataset_frames,
    )
    test_idx = np.asarray(split["test_indices"], dtype=int)
    loader = DataLoader(
        Subset(dataset, test_idx.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=torch_collate,
    )

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    joint_names = [str(x) for x in ckpt.get("joint_names", JOINT_NAMES)]
    model = EgoRearStage1HeatmapNet(
        num_head_heatmaps=len(joint_names),
        base_channels=int(ckpt.get("config", {}).get("base_channels", 64)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    image_size = (IMAGE_WIDTH, IMAGE_HEIGHT)
    per_joint_err = {name: {"sum": 0.0, "count": 0} for name in joint_names}
    total_sum = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            images = batch["img"].to(device)
            pred = model(images)["head"].cpu().numpy()
            target = batch["head_gt_heatmap"].numpy()
            mask = batch["head_loss_mask"].numpy().astype(bool)
            b, c, j, _, _ = pred.shape
            for bi in range(b):
                for ci in range(c):
                    pred_uv = heatmap_to_uv(pred[bi, ci], image_size)
                    gt_uv = gt_heatmap_to_uv(target[bi, ci], mask[bi, ci], image_size)
                    for ji, jname in enumerate(joint_names):
                        if not mask[bi, ci, ji]:
                            continue
                        if not (np.isfinite(gt_uv[ji]).all() and np.isfinite(pred_uv[ji]).all()):
                            continue
                        err = float(np.linalg.norm(pred_uv[ji] - gt_uv[ji]))
                        per_joint_err[jname]["sum"] += err
                        per_joint_err[jname]["count"] += 1
                        total_sum += err
                        total_count += 1

    per_joint = {
        name: {
            "mean_px": (v["sum"] / v["count"] if v["count"] else None),
            "count": v["count"],
        }
        for name, v in per_joint_err.items()
    }
    report = {
        "scheme": args.scheme,
        "checkpoint": str(args.checkpoint),
        "split_npz": str(args.split_npz),
        "test_frames": int(test_idx.size),
        "mean_pixel_error_px": total_sum / total_count if total_count else None,
        "evaluated_points": total_count,
        "joint_names": joint_names,
        "per_joint_mean_px": per_joint,
        "image_size": [IMAGE_WIDTH, IMAGE_HEIGHT],
        "note": "Stage1 test split 2D reprojection error (heatmap argmax vs GT)",
    }
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"stage1_test_{args.scheme}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
