#!/usr/bin/env python3
"""Render Stage1 test-split heatmap overlay videos (GT red / pred blue) for 0806."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

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
    p.add_argument("--pack-size", type=int, default=30)
    p.add_argument(
        "--pack-ids",
        default="",
        help="Comma-separated pack indices within test split; default picks 6 evenly spaced",
    )
    p.add_argument("--max-packs", type=int, default=6)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--joint-radius-config", type=Path, default=JOINT_RADIUS_CONFIG)
    p.add_argument("--eval-json", type=Path, default=None, help="Optional stage1_test JSON for overlay text")
    return p.parse_args()


def denormalize_rgb(img_chw: np.ndarray) -> np.ndarray:
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    rgb = np.clip(img_chw * std + mean, 0.0, 1.0)
    return (rgb.transpose(1, 2, 0) * 255.0).astype(np.uint8)


def heatmap_max_channel(heatmaps: np.ndarray, *, pred: bool) -> np.ndarray:
    """Return HxW float32 in [0, 1] for overlay."""
    hm = np.asarray(heatmaps, dtype=np.float32)
    peak = hm.max(axis=0)
    if pred:
        peak = np.clip(peak, 0.0, None)
        peak = peak / max(float(peak.max()), 1e-6)
    else:
        peak = np.clip(peak, 0.0, 1.0)
    return peak


def compose_overlay(
    bgr: np.ndarray,
    gt_peak: np.ndarray,
    pred_peak: np.ndarray,
    *,
    title: str,
) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    h, w = rgb.shape[:2]
    gt_rgb = np.stack([gt_peak, np.zeros_like(gt_peak), np.zeros_like(gt_peak)], axis=-1)
    pred_rgb = np.stack([np.zeros_like(pred_peak), np.zeros_like(pred_peak), pred_peak], axis=-1)
    overlay = np.clip(0.55 * rgb + 0.25 * gt_rgb + 0.35 * pred_rgb, 0.0, 1.0)
    out = (overlay * 255.0).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    cv2.putText(
        out,
        title,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "red=GT heatmap  blue=pred heatmap",
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return out


def pick_pack_ids(test_count: int, pack_size: int, max_packs: int, pack_ids: str) -> list[int]:
    n_packs = test_count // pack_size
    if pack_ids.strip():
        chosen = [int(x.strip()) for x in pack_ids.split(",") if x.strip()]
    else:
        if n_packs <= max_packs:
            chosen = list(range(n_packs))
        else:
            step = max(1, (n_packs - 1) // max(1, max_packs - 1))
            chosen = sorted({min(i * step, n_packs - 1) for i in range(max_packs)})
    return [p for p in chosen if 0 <= p < n_packs]


def main() -> int:
    import sys

    ego_root = Path("/home/gaoweijian/EgoRear_w_hand")
    sys.path.insert(0, str(ego_root / "src"))
    jp = Path(__file__).resolve().parent
    if str(jp) not in sys.path:
        sys.path.insert(0, str(jp))

    from egorear_sim2d.dataset import MultiViewHeatmapDataset, discover_label_files
    from egorear_sim2d.model import EgoRearStage1HeatmapNet
    from egorear_sim2d.splits import load_split_manifest

    args = parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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
    pack_ids = pick_pack_ids(int(test_idx.size), args.pack_size, args.max_packs, args.pack_ids)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    joint_names = [str(x) for x in ckpt.get("joint_names", JOINT_NAMES)]
    model = EgoRearStage1HeatmapNet(
        num_head_heatmaps=len(joint_names),
        base_channels=int(ckpt.get("config", {}).get("base_channels", 64)),
    )
    state = ckpt["model"]
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    mean_px = None
    eval_path = args.eval_json
    if eval_path is None:
        guess = args.output_dir.parent / "eval" / args.scheme / f"stage1_test_{args.scheme}.json"
        if guess.is_file():
            eval_path = guess
    if eval_path is not None and Path(eval_path).is_file():
        mean_px = json.loads(Path(eval_path).read_text(encoding="utf-8")).get("mean_pixel_error_px")

    manifest: dict[str, object] = {
        "scheme": args.scheme,
        "checkpoint": str(args.checkpoint),
        "split_npz": str(args.split_npz),
        "test_frames": int(test_idx.size),
        "pack_size": args.pack_size,
        "pack_ids": pack_ids,
        "mean_pixel_error_px": mean_px,
        "videos": [],
    }

    camera_labels = ("CAM_A", "CAM_D")
    for pack_id in pack_ids:
        start = pack_id * args.pack_size
        end = start + args.pack_size
        frame_indices = test_idx[start:end].tolist()
        if len(frame_indices) < args.pack_size:
            continue

        writers: dict[int, cv2.VideoWriter] = {}
        try:
            for local_i, ds_idx in enumerate(frame_indices):
                item = dataset[int(ds_idx)]
                img = torch.as_tensor(item["img"]).unsqueeze(0).to(device).float()
                with torch.no_grad():
                    pred = model(img, "head")["head"][0].cpu().numpy()
                gt = np.asarray(item["head_gt_heatmap"], dtype=np.float32)
                seq = int(item["frame_idx"])
                li, fi = dataset.index[int(ds_idx)]
                data = dataset._load_label(li)
                limb = Path(str(dataset.label_files[li])).parent.name

                for cam_i, cam_label in enumerate(camera_labels):
                    image_path = Path(str(data["image_paths"][fi, cam_i]))
                    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise RuntimeError(f"Could not read {image_path}")
                    bgr = cv2.resize(bgr, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)

                    gt_peak = heatmap_max_channel(gt[cam_i], pred=False)
                    pred_peak = heatmap_max_channel(pred[cam_i], pred=True)
                    gt_peak_up = cv2.resize(gt_peak, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
                    pred_peak_up = cv2.resize(pred_peak, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR)
                    title = f"v31 test {limb} pack{pack_id} {cam_label} seq={seq}"
                    frame = compose_overlay(bgr, gt_peak_up, pred_peak_up, title=title)

                    if cam_i not in writers:
                        out_path = out_dir / f"stage1_{args.scheme}_test_{limb}_pack{pack_id:03d}_{cam_label}_overlay.mp4"
                        writer = cv2.VideoWriter(
                            str(out_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            args.fps,
                            (IMAGE_WIDTH, IMAGE_HEIGHT),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not open video writer: {out_path}")
                        writers[cam_i] = writer
                        manifest["videos"].append(
                            {
                                "path": str(out_path),
                                "limb": limb,
                                "pack_id": pack_id,
                                "camera": cam_label,
                                "frames": args.pack_size,
                                "seq_start": seq if local_i == 0 else None,
                            }
                        )
                    writers[cam_i].write(frame)
        finally:
            for writer in writers.values():
                writer.release()

    manifest_path = out_dir / f"stage1_{args.scheme}_test_heatmap_videos.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
