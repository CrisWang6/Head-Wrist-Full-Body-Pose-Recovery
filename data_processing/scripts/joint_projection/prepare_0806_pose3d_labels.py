#!/usr/bin/env python3
"""Build stage-3 pose3d NPZ from 0806 pre_limb triangulation in nose-offset coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from constants_0806_training import (
    COORDINATE_CONVENTION,
    JOINT_NAMES,
    LABEL_NPZ_NAME,
    LIMB_ORDER,
    POSE3D_NPZ_NAME,
)
from delivery_keypoints import DELIVERY_JOINTS, resolve_joint_xyz


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--label-root",
        type=Path,
        default=Path("/home/gaoweijian/0806dataset/labels"),
    )
    p.add_argument(
        "--pre-limb-map",
        type=Path,
        required=True,
        help="JSON map limb->pre_limb jsonl path",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(f"/home/gaoweijian/0806dataset/labels/{POSE3D_NPZ_NAME}"),
    )
    return p.parse_args()


def load_jsonl_records(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            out[int(row["seq"])] = row
    return out


def world_to_nose_m(point_world_m: np.ndarray, nose_world_m: np.ndarray) -> np.ndarray:
    return (np.asarray(point_world_m, dtype=np.float64).reshape(3) - nose_world_m).astype(
        np.float32
    )


def fill_pose_from_record(pose: np.ndarray, record: dict) -> bool:
    joints = record["methods"]["filtered"]["multiview"]
    nose_xyz = resolve_joint_xyz(joints, "nose")
    if nose_xyz is None:
        return False
    nose_world = np.asarray(nose_xyz, dtype=np.float64).reshape(3)
    if not np.isfinite(nose_world).all():
        return False

    for ji, jname in enumerate(DELIVERY_JOINTS):
        xyz = resolve_joint_xyz(joints, jname)
        if xyz is None:
            return False
        pose[ji] = world_to_nose_m(np.asarray(xyz, dtype=np.float64), nose_world)
    return bool(np.isfinite(pose).all())


def main() -> int:
    args = parse_args()
    pre_limb_map = json.loads(args.pre_limb_map.read_text(encoding="utf-8"))

    all_frame_indices: list[int] = []
    all_poses: list[np.ndarray] = []
    all_valid: list[bool] = []

    for limb in LIMB_ORDER:
        npz_path = args.label_root / limb / LABEL_NPZ_NAME
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        label = np.load(npz_path, allow_pickle=True)
        seqs = np.asarray(label["frame_indices"], dtype=np.int64).reshape(-1)
        records = load_jsonl_records(Path(pre_limb_map[limb]))

        for seq in seqs:
            seq_i = int(seq)
            all_frame_indices.append(seq_i)
            pose = np.full((len(JOINT_NAMES), 3), np.nan, dtype=np.float32)
            valid = False
            record = records.get(seq_i)
            if record is not None:
                valid = fill_pose_from_record(pose, record)
            all_poses.append(pose)
            all_valid.append(valid)

    frame_indices = np.asarray(all_frame_indices, dtype=np.int64)
    pose_nose_m = np.stack(all_poses).astype(np.float32)
    valid = np.asarray(all_valid, dtype=bool)

    if not bool(valid.any()):
        raise RuntimeError("No valid 0806 pose3d supervision was produced")

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema_version=np.asarray(["egorear_0806_nose_pre_limb_stage3_v1"]),
        frame_indices=frame_indices,
        joint_names=np.asarray(JOINT_NAMES),
        pose_head_m=pose_nose_m,
        pose_nose_m=pose_nose_m,
        valid=valid,
        coordinate_convention=np.asarray([COORDINATE_CONVENTION]),
        limb_order=np.asarray(LIMB_ORDER),
    )
    summary = {
        "output": str(output),
        "frames_total": int(frame_indices.size),
        "frames_valid": int(valid.sum()),
        "joint_names": list(JOINT_NAMES),
        "limb_order": list(LIMB_ORDER),
        "unit": "meter",
        "coordinate_frame": "nose translation offset (world - nose)",
    }
    output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
