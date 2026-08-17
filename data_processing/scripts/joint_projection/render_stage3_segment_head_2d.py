#!/usr/bin/env python3
"""Render Stage3 pipeline refined-heatmap 2D skeleton on head cameras for a 10s segment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from constants_0806_training import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    JOINT_NAMES,
    LABEL_NPZ_NAME,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from constants_0810_training import SESSION_ORDER
from delivery_keypoints import DELIVERY_EDGES
from joint_radius_config import JOINT_RADIUS_CONFIG, load_joint_radius_video_px


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label-root", type=Path, required=True)
    p.add_argument("--frame-root", type=Path, required=True)
    p.add_argument("--stage1-checkpoint", type=Path, required=True)
    p.add_argument("--stage2-checkpoint", type=Path, required=True)
    p.add_argument("--stage3-checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--session", default="line1", choices=SESSION_ORDER)
    p.add_argument("--segment-starts", default="0")
    p.add_argument("--segment-frames", type=int, default=300)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--joint-radius-config", type=Path, default=JOINT_RADIUS_CONFIG)
    p.add_argument("--line-thickness", type=int, default=4)
    p.add_argument("--point-radius", type=int, default=6)
    p.add_argument("--draw-gt", action="store_true")
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


def load_native_bgr(image_path: Path) -> np.ndarray:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read {image_path}")
    h, w = bgr.shape[:2]
    if (w, h) != (VIDEO_WIDTH, VIDEO_HEIGHT):
        bgr = cv2.resize(bgr, (VIDEO_WIDTH, VIDEO_HEIGHT), interpolation=cv2.INTER_AREA)
    return bgr


def draw_skeleton_2d(
    canvas: np.ndarray,
    uv: np.ndarray,
    mask: np.ndarray,
    joint_names: list[str],
    *,
    edge_bgr: tuple[int, int, int],
    joint_bgr: tuple[int, int, int],
    line_thickness: int,
    point_radius: int,
) -> None:
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    for a, b in DELIVERY_EDGES:
        ia, ib = name_to_idx.get(a), name_to_idx.get(b)
        if ia is None or ib is None or not (mask[ia] and mask[ib]):
            continue
        pa, pb = uv[ia], uv[ib]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        cv2.line(
            canvas,
            (int(round(float(pa[0]))), int(round(float(pa[1])))),
            (int(round(float(pb[0]))), int(round(float(pb[1])))),
            edge_bgr,
            line_thickness,
            cv2.LINE_AA,
        )
    for ji, _ in enumerate(joint_names):
        if not mask[ji]:
            continue
        pt = uv[ji]
        if not np.isfinite(pt).all():
            continue
        center = (int(round(float(pt[0]))), int(round(float(pt[1]))))
        cv2.circle(canvas, center, point_radius + 2, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.circle(canvas, center, point_radius, joint_bgr, -1, cv2.LINE_AA)


def compose_frame(
    bgr: np.ndarray,
    pred_uv: np.ndarray,
    gt_uv: np.ndarray | None,
    mask: np.ndarray,
    joint_names: list[str],
    *,
    title: str,
    line_thickness: int,
    point_radius: int,
    draw_gt: bool,
) -> np.ndarray:
    out = bgr.copy()
    if draw_gt and gt_uv is not None:
        draw_skeleton_2d(
            out, gt_uv, mask, joint_names,
            edge_bgr=(0, 180, 0), joint_bgr=(0, 255, 0),
            line_thickness=max(2, line_thickness - 1),
            point_radius=max(3, point_radius - 2),
        )
    draw_skeleton_2d(
        out, pred_uv, mask, joint_names,
        edge_bgr=(255, 0, 255), joint_bgr=(255, 120, 255),
        line_thickness=line_thickness, point_radius=point_radius,
    )
    cv2.putText(out, title, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    legend = "magenta=Stage3 refined 2D (0806 ckpt) @1920"
    if draw_gt:
        legend += "  green=GT heatmap"
    cv2.putText(out, legend, (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 2, cv2.LINE_AA)
    return out


def build_session_seq_index(dataset, session: str) -> dict[int, int]:
    label_idx = None
    for i, path in enumerate(dataset.label_files):
        if Path(str(path)).parent.name == session:
            label_idx = i
            break
    if label_idx is None:
        raise RuntimeError(f"No label file for session={session}")
    seq_to_ds: dict[int, int] = {}
    for ds_idx, (li, fi) in enumerate(dataset.index):
        if li != label_idx:
            continue
        data = dataset._load_label(li)
        seq_to_ds[int(data["frame_indices"][fi])] = ds_idx
    return seq_to_ds


def segment_dataset_indices(seq_to_ds: dict[int, int], start_seq: int, n: int) -> tuple[list[int], int, int]:
    end_seq = start_seq + n - 1
    missing = [s for s in range(start_seq, end_seq + 1) if s not in seq_to_ds]
    if missing:
        raise RuntimeError(f"Missing seq in segment: {missing[:5]} ... ({len(missing)} total)")
    return [seq_to_ds[s] for s in range(start_seq, end_seq + 1)], start_seq, end_seq


def load_pipeline(args, joint_names: list[str], device: torch.device):
    from egorear_sim2d.pose3d import EgoRearPose3DNet, EgoRearStage3Pipeline
    from egorear_sim2d.refinement import HeadBCHeatmapRefinementNet, load_refiner_state, load_stage1_model

    stage3_ckpt = torch.load(args.stage3_checkpoint, map_location="cpu", weights_only=False)
    stage2_ckpt = torch.load(args.stage2_checkpoint, map_location="cpu", weights_only=False)
    stage2_config = stage2_ckpt.get("config", {})
    stage1 = load_stage1_model(
        str(args.stage1_checkpoint),
        num_head_heatmaps=len(joint_names),
        base_channels=int(stage2_config.get("base_channels", 64)),
    )
    stage2 = HeadBCHeatmapRefinementNet(
        stage1,
        num_joints=len(joint_names),
        heatmap_size=(
            int(stage2_config.get("heatmap_width", 120)),
            int(stage2_config.get("heatmap_height", 75)),
        ),
        base_channels=int(stage2_config.get("base_channels", 64)),
        query_dim=int(stage2_config.get("query_dim", 256)),
        sampling_points=int(stage2_config.get("sampling_points", 8)),
        freeze_stage1=True,
    )
    load_refiner_state(stage2, stage2_ckpt["refiner"])
    pose3d = EgoRearPose3DNet(num_joints=len(joint_names))
    pose3d.load_state_dict(stage3_ckpt["pose3d"], strict=True)
    pipeline = EgoRearStage3Pipeline(stage2, pose3d).to(device).eval()
    return pipeline


def main() -> int:
    ego_root = Path("/home/gaoweijian/EgoRear_w_hand")
    sys.path.insert(0, str(ego_root / "src"))
    jp = Path(__file__).resolve().parent
    if str(jp) not in sys.path:
        sys.path.insert(0, str(jp))

    from egorear_sim2d.dataset import MultiViewHeatmapDataset, discover_label_files

    args = parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    segment_starts = [int(x.strip()) for x in args.segment_starts.split(",") if x.strip()]
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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
    seq_to_ds = build_session_seq_index(dataset, args.session)
    pipeline = load_pipeline(args, list(JOINT_NAMES), device)
    joint_names = list(JOINT_NAMES)
    camera_labels = ("CAM_A", "CAM_D")

    manifest: dict = {
        "session": args.session,
        "checkpoints": {
            "stage1": str(args.stage1_checkpoint),
            "stage2": str(args.stage2_checkpoint),
            "stage3": str(args.stage3_checkpoint),
        },
        "segment_frames": args.segment_frames,
        "segments": [],
    }

    for seg_i, start_seq in enumerate(segment_starts):
        ds_indices, seq_start, seq_end = segment_dataset_indices(
            seq_to_ds, start_seq, args.segment_frames
        )
        seg_info: dict = {"seq_start": seq_start, "seq_end": seq_end, "videos": []}
        writers: dict[str, cv2.VideoWriter] = {}
        try:
            for frame_i, ds_idx in enumerate(ds_indices):
                item = dataset[int(ds_idx)]
                img = torch.as_tensor(item["img"]).unsqueeze(0).to(device).float()
                with torch.no_grad():
                    out = pipeline(img)
                    pred_hm = out["refined_heatmaps"][0].cpu().numpy()
                gt_hm = np.asarray(item["head_gt_heatmap"], dtype=np.float32)
                mask_all = np.asarray(item["head_loss_mask"], dtype=bool)
                seq = int(item["frame_idx"])
                li, fi = dataset.index[int(ds_idx)]
                data = dataset._load_label(li)
                for cam_i, cam_label in enumerate(camera_labels):
                    bgr = load_native_bgr(Path(str(data["image_paths"][fi, cam_i])))
                    pred_uv = heatmap_to_uv(pred_hm[cam_i], (VIDEO_WIDTH, VIDEO_HEIGHT))
                    gt_uv = heatmap_to_uv(gt_hm[cam_i], (VIDEO_WIDTH, VIDEO_HEIGHT)) if args.draw_gt else None
                    title = f"{args.session} seg{seg_i} {cam_label} seq={seq} ({frame_i+1}/{len(ds_indices)})"
                    frame = compose_frame(
                        bgr, pred_uv, gt_uv, mask_all[cam_i], joint_names,
                        title=title,
                        line_thickness=args.line_thickness,
                        point_radius=args.point_radius,
                        draw_gt=args.draw_gt,
                    )
                    if cam_label not in writers:
                        out_path = out_dir / (
                            f"{args.session}_10s_seg{seg_i:02d}_seq{seq_start}_{seq_end}_"
                            f"{cam_label}_stage3_refined2d_1920.mp4"
                        )
                        writers[cam_label] = cv2.VideoWriter(
                            str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                            float(args.fps), (VIDEO_WIDTH, VIDEO_HEIGHT),
                        )
                        seg_info["videos"].append({"path": str(out_path), "camera": cam_label})
                    writers[cam_label].write(frame)
        finally:
            for w in writers.values():
                w.release()
        manifest["segments"].append(seg_info)

    (out_dir / "stage3_head_2d_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
