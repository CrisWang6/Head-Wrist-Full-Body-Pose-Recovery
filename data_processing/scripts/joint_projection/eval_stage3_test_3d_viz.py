#!/usr/bin/env python3
"""Run Stage3 inference on 0806 test split and export 3D skeleton playback + yaw MP4."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from constants_0806_training import IMAGE_HEIGHT, IMAGE_WIDTH, LABEL_NPZ_NAME, LIMB_ORDER
from delivery_keypoints import export_skeleton_playback


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label-root", type=Path, required=True)
    p.add_argument("--pose3d-labels", type=Path, required=True)
    p.add_argument("--stage1-checkpoint", type=Path, required=True)
    p.add_argument("--stage2-checkpoint", type=Path, default=None)
    p.add_argument("--skip-stage2", action="store_true", help="Use frozen stage1 heatmaps only (no stage2 refiner).")
    p.add_argument("--stage3-checkpoint", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--split-name", choices=("test", "val", "train"), default="test")
    p.add_argument(
        "--limb",
        choices=LIMB_ORDER,
        default=None,
        help="Optional limb filter (ankle/wrist/wu) within the chosen split",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-frames", type=int, default=0, help="0 = all frames in split")
    p.add_argument("--yaw-deg", type=float, default=100.0)
    p.add_argument("--pitch-deg", type=float, default=18.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--skip-yaw-render", action="store_true")
    return p.parse_args()


def record_from_pose_nose_m(pose_nose_m: np.ndarray, seq: int, joint_names: list[str]) -> dict:
    payload = {}
    for ji, name in enumerate(joint_names):
        xyz = pose_nose_m[ji]
        if not np.all(np.isfinite(xyz)):
            continue
        payload[name] = {"xyz_world_m": [float(x) for x in xyz.tolist()]}
    return {"seq": int(seq), "methods": {"filtered": {"multiview": payload}}}


def render_yaw_mp4(
    playback_path: Path,
    output_path: Path,
    *,
    yaw_deg: float,
    pitch_deg: float,
    fps: float,
    script_dir: Path,
    max_frames: int = 0,
) -> None:
    cmd = [
        sys.executable,
        str(script_dir / "render_skeleton_yaw_video.py"),
        "--data",
        str(playback_path),
        "--output",
        str(output_path),
        "--yaw-deg",
        str(yaw_deg),
        "--pitch-deg",
        str(pitch_deg),
        "--fps",
        str(fps),
    ]
    if max_frames > 0:
        cmd.extend(["--max-frames", str(max_frames)])
    subprocess.run(cmd, check=True)


def limb_global_ranges(label_root: Path) -> dict[str, tuple[int, int]]:
    offset = 0
    ranges: dict[str, tuple[int, int]] = {}
    for limb in LIMB_ORDER:
        npz = label_root / limb / LABEL_NPZ_NAME
        if not npz.is_file():
            raise FileNotFoundError(npz)
        count = int(np.load(npz, allow_pickle=True)["source_aligned_seq"].shape[0])
        ranges[limb] = (offset, offset + count)
        offset += count
    return ranges


def split_tag(split_name: str, limb: str | None) -> str:
    return f"{split_name}_{limb}" if limb else split_name


def main() -> int:
    import torch
    from torch.utils.data import DataLoader, Subset

    from egorear_sim2d.dataset import MultiViewHeatmapDataset, discover_label_files, torch_collate
    from egorear_sim2d.pose3d import (
        EgoRearPose3DNet,
        EgoRearStage3Pipeline,
        EgoRearStage3Stage1Pipeline,
    )
    from egorear_sim2d.refinement import HeadBCHeatmapRefinementNet, load_refiner_state, load_stage1_model
    from egorear_sim2d.splits import load_split_manifest

    args = parse_args()
    jp = Path(__file__).resolve().parent
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    label_files = discover_label_files(args.label_root.expanduser().resolve())
    dataset = MultiViewHeatmapDataset(
        label_files,
        image_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
        visible_only_loss=True,
    )
    pose_labels = np.load(args.pose3d_labels.expanduser().resolve(), allow_pickle=True)
    pose_frames = np.asarray(pose_labels["frame_indices"], dtype=np.int64)
    pose_values = np.asarray(pose_labels["pose_head_m"], dtype=np.float32)
    pose_valid = np.asarray(pose_labels["valid"], dtype=bool)
    joint_names = [str(v) for v in pose_labels["joint_names"]]

    dataset_frames = np.asarray(
        [
            int(dataset._load_label(label_idx)["frame_indices"][frame_idx])
            for label_idx, frame_idx in dataset.index
        ],
        dtype=np.int64,
    )
    if len(pose_frames) != len(dataset):
        raise ValueError(f"2D/3D length mismatch: {len(dataset)} vs {len(pose_frames)}")
    if not np.array_equal(dataset_frames, pose_frames):
        raise ValueError("2D and 3D frame_indices are not aligned")

    split_manifest = load_split_manifest(
        str(args.split_manifest),
        expected_length=len(dataset),
        expected_frame_indices=dataset_frames,
    )
    split_key = f"{args.split_name}_indices"
    split_indices = split_manifest[split_key].astype(np.int64)
    split_indices = split_indices[pose_valid[split_indices]]
    if args.limb:
        limb_ranges = limb_global_ranges(args.label_root.expanduser().resolve())
        lo, hi = limb_ranges[args.limb]
        split_indices = split_indices[(split_indices >= lo) & (split_indices < hi)]
        if split_indices.size == 0:
            raise ValueError(f"No frames for split={args.split_name} limb={args.limb}")
    if args.max_frames > 0:
        split_indices = split_indices[: int(args.max_frames)]

    tag = split_tag(args.split_name, args.limb)

    loader = DataLoader(
        Subset(dataset, split_indices.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=torch_collate,
        persistent_workers=args.workers > 0,
    )
    index_lookup = split_indices.astype(np.int64)

    stage3_ckpt = torch.load(
        args.stage3_checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False
    )
    stage1 = load_stage1_model(
        str(args.stage1_checkpoint),
        num_head_heatmaps=len(joint_names),
    )
    pose3d = EgoRearPose3DNet(num_joints=len(joint_names))
    pose3d.load_state_dict(stage3_ckpt["pose3d"], strict=True)
    if args.skip_stage2:
        pipeline = EgoRearStage3Stage1Pipeline(stage1, pose3d)
    else:
        if args.stage2_checkpoint is None:
            raise ValueError("--stage2-checkpoint is required unless --skip-stage2 is set")
        stage2_ckpt = torch.load(
            args.stage2_checkpoint.expanduser().resolve(), map_location="cpu", weights_only=False
        )
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
        pipeline = EgoRearStage3Pipeline(stage2, pose3d)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    pipeline = pipeline.to(device).eval()

    pred_records: list[dict] = []
    gt_records: list[dict] = []
    per_joint_err: dict[str, list[float]] = {name: [] for name in joint_names}
    seqs: list[int] = []
    global_indices: list[int] = []

    with torch.no_grad():
        cursor = 0
        for batch in loader:
            images = batch["img"].to(device, non_blocking=True).float()
            batch_n = int(images.shape[0])
            batch_global = index_lookup[cursor : cursor + batch_n]
            cursor += batch_n
            target = torch.as_tensor(
                pose_values[batch_global],
                device=device,
                dtype=torch.float32,
            )
            output = pipeline(images)
            pred = output["pose3d"].detach().cpu().numpy()
            err = torch.linalg.vector_norm(output["pose3d"] - target, dim=-1).cpu().numpy()
            frame_seqs = [int(pose_frames[int(gi)]) for gi in batch_global]
            for bi, (seq, gi) in enumerate(zip(frame_seqs, batch_global, strict=True)):
                gi = int(gi)
                seqs.append(seq)
                global_indices.append(gi)
                pred_records.append(record_from_pose_nose_m(pred[bi], seq, joint_names))
                gt_records.append(record_from_pose_nose_m(pose_values[gi], seq, joint_names))
                for ji, name in enumerate(joint_names):
                    per_joint_err[name].append(float(err[bi, ji]))

    pred_playback = out_dir / f"skeleton_playback_stage3_{tag}_pred.json"
    gt_playback = out_dir / f"skeleton_playback_stage3_{tag}_gt.json"
    export_skeleton_playback(
        pred_records,
        pred_playback,
        source=f"Stage3 v31 prediction ({tag}, nose-offset frame)",
        joint_names=joint_names,
    )
    export_skeleton_playback(
        gt_records,
        gt_playback,
        source=f"Stage3 GT from pre_limb ({tag}, nose-offset frame)",
        joint_names=joint_names,
    )

    pred_yaw = out_dir / f"skeleton_yaw_stage3_{tag}_pred.mp4"
    gt_yaw = out_dir / f"skeleton_yaw_stage3_{tag}_gt.mp4"
    if not args.skip_yaw_render:
        render_yaw_mp4(
            pred_playback,
            pred_yaw,
            yaw_deg=args.yaw_deg,
            pitch_deg=args.pitch_deg,
            fps=args.fps,
            script_dir=jp,
            max_frames=args.max_frames,
        )
        render_yaw_mp4(
            gt_playback,
            gt_yaw,
            yaw_deg=args.yaw_deg,
            pitch_deg=args.pitch_deg,
            fps=args.fps,
            script_dir=jp,
            max_frames=args.max_frames,
        )

    per_joint_mm = {
        name: float(np.mean(values) * 1000.0) if values else float("nan")
        for name, values in per_joint_err.items()
    }
    overall_mm = float(np.mean(list(per_joint_mm.values())))
    report = {
        "split_name": args.split_name,
        "limb": args.limb,
        "split_manifest": str(args.split_manifest),
        "stage3_checkpoint": str(args.stage3_checkpoint),
        "skip_stage2": bool(args.skip_stage2),
        "stage3_epoch": int(stage3_ckpt.get("epoch", -1)),
        "frames": len(seqs),
        "seq_start": int(min(seqs)) if seqs else None,
        "seq_end": int(max(seqs)) if seqs else None,
        "mpjpe_mm_overall": overall_mm,
        "mpjpe_mm_per_joint": per_joint_mm,
        "coordinate_frame": "nose translation offset (meters)",
        "alignment_note": "GT indexed by global dataset row, not per-limb seq key",
        "outputs": {
            "pred_playback": str(pred_playback),
            "gt_playback": str(gt_playback),
            "pred_yaw_mp4": str(pred_yaw) if not args.skip_yaw_render else None,
            "gt_yaw_mp4": str(gt_yaw) if not args.skip_yaw_render else None,
        },
    }
    report_path = out_dir / f"stage3_{tag}_eval.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
