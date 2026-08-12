#!/usr/bin/env python3
"""Build EgoRear stage-1 NPZ labels from head 2D reprojection CSV + 0806dataset frames."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from constants_0806_training import (
    HEATMAP_HEIGHT,
    HEATMAP_WIDTH,
    JOINT_NAMES,
    LABEL_NPZ_NAME,
    VIDEO_HEIGHT,
    VIDEO_WIDTH,
)
from delivery_keypoints import DELIVERY_JOINTS, TOE_ALIASES

CAMERA_ORDER = ("CAM_A", "CAM_D")
CSV_JOINT_ALIASES = {
    "left_toe": "left_big_toe",
    "right_toe": "right_big_toe",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--limb", required=True)
    p.add_argument("--head-dir-name", required=True, help="e.g. 0712_033709")
    p.add_argument(
        "--frame-root",
        type=Path,
        default=Path("/home/gaoweijian/0806dataset/frames"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"defaults to label dir / {LABEL_NPZ_NAME}",
    )
    p.add_argument("--video-width", type=int, default=VIDEO_WIDTH)
    p.add_argument("--video-height", type=int, default=VIDEO_HEIGHT)
    p.add_argument("--heatmap-width", type=int, default=HEATMAP_WIDTH)
    p.add_argument("--heatmap-height", type=int, default=HEATMAP_HEIGHT)
    return p.parse_args()


def normalize_csv_joint(name: str) -> str | None:
    name = CSV_JOINT_ALIASES.get(name, name)
    if name in DELIVERY_JOINTS:
        return name
    for dest, aliases in TOE_ALIASES.items():
        if name in aliases:
            return dest
    return None


def load_csv_rows(csv_path: Path) -> dict[tuple[int, str, str], tuple[float, float]]:
    """Map (seq, camera, delivery_joint) -> (u, v) in video pixels."""
    out: dict[tuple[int, str, str], tuple[float, float]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            joint = normalize_csv_joint(str(row["joint"]))
            if joint is None:
                continue
            camera = str(row["camera"])
            if camera not in CAMERA_ORDER:
                continue
            u = float(row["u_px"])
            v = float(row["v_px"])
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            out[(int(row["seq"]), camera, joint)] = (u, v)
    return out


def main() -> int:
    args = parse_args()
    csv_path = args.csv.resolve()
    lookup = load_csv_rows(csv_path)
    seqs = sorted({seq for seq, _, _ in lookup})
    if not seqs:
        raise RuntimeError(f"No usable rows in {csv_path}")

    frame_count = len(seqs)
    cam_count = len(CAMERA_ORDER)
    joint_count = len(JOINT_NAMES)
    head_keypoints = np.full((frame_count, cam_count, joint_count, 2), np.nan, dtype=np.float32)
    head_visible = np.zeros((frame_count, cam_count, joint_count), dtype=bool)

    source_render_dir = f"{args.limb}/{args.head_dir_name}"
    image_paths = np.empty((frame_count, cam_count), dtype=object)

    for fi, seq in enumerate(seqs):
        for ci, camera in enumerate(CAMERA_ORDER):
            for ji, jname in enumerate(JOINT_NAMES):
                uv = lookup.get((seq, camera, jname))
                if uv is None:
                    continue
                head_keypoints[fi, ci, ji] = uv
                head_visible[fi, ci, ji] = True
            image_paths[fi, ci] = str(
                args.frame_root
                / source_render_dir
                / camera
                / f"{seq:06d}.jpg"
            )

    head_joint_mask = np.ones((cam_count, joint_count), dtype=bool)
    wrist_keypoints = np.full((frame_count, cam_count, 7, 2), np.nan, dtype=np.float32)
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else csv_path.parent.parent.parent / "labels" / args.limb / LABEL_NPZ_NAME
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray(["egorear_0806_delivery15_nose2cam_v1"]),
        keypoints=head_keypoints.astype(np.float32),
        visible=head_visible.astype(bool),
        joint_mask=head_joint_mask.astype(bool),
        head_keypoints=head_keypoints.astype(np.float32),
        head_visible=head_visible.astype(bool),
        head_joint_mask=head_joint_mask.astype(bool),
        wrist_keypoints=wrist_keypoints.astype(np.float32),
        wrist_visible=np.zeros((frame_count, cam_count, 7), dtype=bool),
        wrist_joint_mask=np.zeros((cam_count, 7), dtype=bool),
        camera_is_head=np.ones(cam_count, dtype=bool),
        camera_is_wrist=np.zeros(cam_count, dtype=bool),
        camera_names=np.asarray([f"module01_{c}" for c in CAMERA_ORDER]),
        joints=np.asarray(JOINT_NAMES),
        head_camera_joints=np.asarray(JOINT_NAMES),
        head_source_smplx_joints=np.asarray(JOINT_NAMES),
        wrist_camera_joints=np.asarray(
            ("L_Ankle", "R_Ankle", "L_Knee", "R_Knee", "L_Hip", "R_Hip", "Spine1")
        ),
        video_paths=np.asarray(["", ""]),
        image_paths=image_paths,
        frame_indices=np.asarray(seqs, dtype=np.int64),
        source_aligned_seq=np.asarray(seqs, dtype=np.int64),
        video_size=np.asarray([args.video_width, args.video_height], dtype=np.int32),
        heatmap_size=np.asarray([args.heatmap_width, args.heatmap_height], dtype=np.int32),
        sigma=np.asarray([1.5], dtype=np.float32),
        projection_model=np.asarray(["0806_multiview_head_reprojection_nose_origin"]),
        fisheye_fov_deg=np.asarray([220.0], dtype=np.float32),
        source_csv=np.asarray([str(csv_path)]),
        source_render_dir=np.asarray([source_render_dir]),
        limb=np.asarray([args.limb]),
    )
    manifest = {
        "schema": "egorear.0806_delivery15_manifest.v1",
        "output": str(output),
        "limb": args.limb,
        "joints": list(JOINT_NAMES),
        "frames": frame_count,
        "heatmap_size": [args.heatmap_width, args.heatmap_height],
        "seq_range": [int(seqs[0]), int(seqs[-1])],
        "visible_points": int(head_visible.sum()),
        "source_render_dir": source_render_dir,
        "frame_root": str(args.frame_root),
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
