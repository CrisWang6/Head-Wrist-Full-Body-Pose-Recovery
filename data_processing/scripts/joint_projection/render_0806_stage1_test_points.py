#!/usr/bin/env python3
"""Render Stage1 test videos: heatmap argmax joints at 1920x1200 with highlighted points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from constants_0806_training import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    JOINT_NAMES,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from delivery_keypoints import DELIVERY_EDGES
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
    p.add_argument("--pack-ids", default="")
    p.add_argument("--max-packs", type=int, default=6)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--joint-radius-config", type=Path, default=JOINT_RADIUS_CONFIG)
    p.add_argument("--eval-json", type=Path, default=None)
    p.add_argument("--point-radius", type=int, default=9)
    p.add_argument("--draw-skeleton", action="store_true", default=True)
    return p.parse_args()


def heatmap_to_uv(heatmaps: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    hm = np.asarray(heatmaps, dtype=np.float32)
    j, h, w = hm.shape
    flat = hm.reshape(j, -1)
    idx = flat.argmax(axis=-1)
    x = idx % w
    y = idx // w
    iw, ih = image_size
    return np.stack(((x + 0.5) * iw / w, (y + 0.5) * ih / h), axis=-1).astype(np.float32)


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


def load_native_bgr(image_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read {image_path}")
    h, w = bgr.shape[:2]
    if (w, h) != (VIDEO_WIDTH, VIDEO_HEIGHT):
        bgr = cv2.resize(bgr, (VIDEO_WIDTH, VIDEO_HEIGHT), interpolation=cv2.INTER_AREA)
    return bgr


def draw_highlight_point(
    canvas: np.ndarray,
    x: int,
    y: int,
    *,
    fill_bgr: tuple[int, int, int],
    radius: int,
) -> None:
    cv2.circle(canvas, (x, y), radius + 4, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), radius + 2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.circle(canvas, (x, y), radius, fill_bgr, -1, cv2.LINE_AA)


def draw_joints_and_skeleton(
    canvas: np.ndarray,
    uv: np.ndarray,
    mask: np.ndarray,
    joint_names: list[str],
    *,
    fill_bgr: tuple[int, int, int],
    edge_bgr: tuple[int, int, int],
    radius: int,
    draw_skeleton: bool,
) -> None:
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    if draw_skeleton:
        for a, b in DELIVERY_EDGES:
            ia = name_to_idx.get(a)
            ib = name_to_idx.get(b)
            if ia is None or ib is None:
                continue
            if not (mask[ia] and mask[ib]):
                continue
            pa = uv[ia]
            pb = uv[ib]
            if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
                continue
            p1 = (int(round(float(pa[0]))), int(round(float(pa[1]))))
            p2 = (int(round(float(pb[0]))), int(round(float(pb[1]))))
            cv2.line(canvas, p1, p2, edge_bgr, 2, cv2.LINE_AA)

    for ji, _name in enumerate(joint_names):
        if not mask[ji]:
            continue
        pt = uv[ji]
        if not np.isfinite(pt).all():
            continue
        draw_highlight_point(
            canvas,
            int(round(float(pt[0]))),
            int(round(float(pt[1]))),
            fill_bgr=fill_bgr,
            radius=radius,
        )


def compose_frame(
    bgr: np.ndarray,
    gt_uv: np.ndarray,
    pred_uv: np.ndarray,
    mask: np.ndarray,
    joint_names: list[str],
    *,
    title: str,
    point_radius: int,
    draw_skeleton: bool,
) -> np.ndarray:
    out = bgr.copy()
    draw_joints_and_skeleton(
        out,
        gt_uv,
        mask,
        joint_names,
        fill_bgr=(0, 0, 255),
        edge_bgr=(0, 0, 180),
        radius=point_radius,
        draw_skeleton=draw_skeleton,
    )
    draw_joints_and_skeleton(
        out,
        pred_uv,
        mask,
        joint_names,
        fill_bgr=(255, 120, 0),
        edge_bgr=(255, 200, 0),
        radius=max(4, point_radius - 2),
        draw_skeleton=False,
    )
    cv2.putText(out, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(
        out,
        "RED=GT argmax  CYAN=Pred argmax  (1920x1200)",
        (12, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )
    return out


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
    video_size = (VIDEO_WIDTH, VIDEO_HEIGHT)

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
    state = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state, strict=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    mean_px = None
    eval_path = args.eval_json
    if eval_path is None:
        guess = args.output_dir.parent / f"stage1_test_{args.scheme}.json"
        if not guess.is_file():
            guess = args.output_dir.parent.parent / "eval" / args.scheme / f"stage1_test_{args.scheme}.json"
        if guess.is_file():
            eval_path = guess
    if eval_path is not None and Path(eval_path).is_file():
        mean_px = json.loads(Path(eval_path).read_text(encoding="utf-8")).get("mean_pixel_error_px")

    manifest: dict[str, object] = {
        "scheme": args.scheme,
        "checkpoint": str(args.checkpoint),
        "video_size": [VIDEO_WIDTH, VIDEO_HEIGHT],
        "coordinate_note": "heatmap argmax mapped to video pixels via heatmap_to_uv(..., (1920,1200))",
        "mean_pixel_error_px": mean_px,
        "pack_ids": pack_ids,
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
                    pred_hm = model(img, "head")["head"][0].cpu().numpy()
                gt_hm = np.asarray(item["head_gt_heatmap"], dtype=np.float32)
                mask = np.asarray(item["head_loss_mask"], dtype=bool)
                seq = int(item["frame_idx"])
                li, fi = dataset.index[int(ds_idx)]
                data = dataset._load_label(li)
                limb = Path(str(dataset.label_files[li])).parent.name

                for cam_i, cam_label in enumerate(camera_labels):
                    image_path = Path(str(data["image_paths"][fi, cam_i]))
                    bgr = load_native_bgr(image_path)
                    gt_uv = heatmap_to_uv(gt_hm[cam_i], video_size)
                    pred_uv = heatmap_to_uv(pred_hm[cam_i], video_size)
                    title = f"v31 test {limb} pack{pack_id} {cam_label} seq={seq}"
                    frame = compose_frame(
                        bgr,
                        gt_uv,
                        pred_uv,
                        mask[cam_i],
                        joint_names,
                        title=title,
                        point_radius=args.point_radius,
                        draw_skeleton=args.draw_skeleton,
                    )

                    if cam_i not in writers:
                        out_path = (
                            out_dir
                            / f"stage1_{args.scheme}_test_{limb}_pack{pack_id:03d}_{cam_label}_points1920.mp4"
                        )
                        writer = cv2.VideoWriter(
                            str(out_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            args.fps,
                            (VIDEO_WIDTH, VIDEO_HEIGHT),
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
                                "resolution": [VIDEO_WIDTH, VIDEO_HEIGHT],
                            }
                        )
                    writers[cam_i].write(frame)
        finally:
            for writer in writers.values():
                writer.release()

    manifest_path = out_dir / f"stage1_{args.scheme}_test_points1920_videos.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
