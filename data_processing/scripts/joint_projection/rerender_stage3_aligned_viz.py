#!/usr/bin/env python3
"""Rebuild aligned GT playback + pred/GT dual yaw video after seq-collision fix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from delivery_keypoints import export_skeleton_playback


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pose3d-labels", type=Path, required=True)
    p.add_argument("--split-manifest", type=Path, required=True)
    p.add_argument("--pred-playback", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--split-name", default="test")
    return p.parse_args()


def record_from_pose(pose: np.ndarray, seq: int, joint_names: list[str]) -> dict:
    payload = {}
    for ji, name in enumerate(joint_names):
        xyz = pose[ji]
        if not np.all(np.isfinite(xyz)):
            continue
        payload[name] = {"xyz_world_m": [float(x) for x in xyz.tolist()]}
    return {"seq": int(seq), "methods": {"filtered": {"multiview": payload}}}


def main() -> int:
    args = parse_args()
    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    jp = Path(__file__).resolve().parent

    pose = np.load(args.pose3d_labels, allow_pickle=True)
    joint_names = [str(x) for x in pose["joint_names"]]
    frame_indices = np.asarray(pose["frame_indices"], dtype=np.int64)
    pose_values = np.asarray(pose["pose_head_m"], dtype=np.float32)

    split = np.load(args.split_manifest, allow_pickle=True)
    indices = np.asarray(split[f"{args.split_name}_indices"], dtype=np.int64)

    gt_records = []
    for gi in indices:
        gi = int(gi)
        gt_records.append(record_from_pose(pose_values[gi], int(frame_indices[gi]), joint_names))

    gt_playback = out / f"skeleton_playback_stage3_{args.split_name}_gt_aligned.json"
    export_skeleton_playback(
        gt_records,
        gt_playback,
        source=f"Aligned GT by global index ({args.split_name})",
        joint_names=joint_names,
    )

    pred_filtered = args.pred_playback
    if not pred_filtered.is_file():
        pred_filtered = out.parent / "skeleton_playback_stage3_test_pred_filtered.json"

    dual_out = out / f"skeleton_yaw_stage3_{args.split_name}_pred_vs_gt_aligned.mp4"
    report_out = dual_out.with_suffix(".json")
    cmd = [
        sys.executable,
        str(jp / "render_stage3_dual_skeleton_yaw.py"),
        "--pred-playback",
        str(pred_filtered),
        "--gt-playback",
        str(gt_playback),
        "--output",
        str(dual_out),
        "--report",
        str(report_out),
    ]
    subprocess.run(cmd, check=True)
    print(json.dumps({"gt_playback": str(gt_playback), "dual_mp4": str(dual_out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
