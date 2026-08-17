#!/usr/bin/env python3
"""Reproject filtered Stage3 3D skeletons onto head camera frames (pred vs GT)."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from constants_0806_training import LABEL_NPZ_NAME as LABEL_NPZ_0806
from constants_0806_training import LIMB_ORDER, VIDEO_HEIGHT, VIDEO_WIDTH
from constants_0810_training import LABEL_NPZ_NAME as LABEL_NPZ_0810
from constants_0810_training import PACK_SIZE as PACK_SIZE_0810
from constants_0810_training import SESSION_ORDER
from delivery_keypoints import DELIVERY_EDGES, DELIVERY_JOINTS, resolve_joint_xyz
from multiview_geometry import CameraPose, OmniCamera, load_json, rigid_world_transform
from render_multiview_to_head import (
    CAMERA_ORDER,
    REPO_ROOT,
    head_mocap_correction,
    open_writer,
    project_joints,
    resolve_repo_path,
    rigid_camera_mounts,
)
from render_stage3_dual_skeleton_yaw import load_playback_records
from skeleton_3d_filter import filter_skeleton_playback_records

PRED_EDGE = (255, 120, 0)
PRED_JOINT = (255, 180, 80)
GT_EDGE = (0, 0, 255)
GT_JOINT = (0, 0, 220)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred-playback", type=Path, required=True)
    p.add_argument("--gt-playback", type=Path, required=True)
    p.add_argument("--label-root", type=Path, required=True)
    p.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Pack split NPZ; required to map playback order -> limb/images",
    )
    p.add_argument("--split-name", choices=("test", "val", "train"), default="test")
    p.add_argument("--pre-limb-map", type=Path, required=True)
    p.add_argument(
        "--scope-configs",
        "--line-configs",
        type=Path,
        required=True,
        dest="scope_configs",
        help="JSON map scope key (line1/line2 or ankle/wrist/wu) -> mocap config",
    )
    p.add_argument(
        "--batch-root",
        type=Path,
        default=None,
        help="0810_batch root with line1/line2 data_root (0810 layout only)",
    )
    p.add_argument("--layout", choices=("0810", "0806"), default="0810")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--scope-filter",
        "--session",
        default="",
        dest="scope_filter",
        help="Optional scope filter: line2 (0810) or ankle (0806); empty = all",
    )
    p.add_argument("--pack-size", type=int, default=0, help="0 = auto (150/0810, 30/0806)")
    p.add_argument("--pack-ids", default="0,50", help="Comma-separated pack ids within playback order")
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument(
        "--continuous",
        action="store_true",
        help="Render one long MP4 for the whole split (ignore --pack-ids)",
    )
    p.add_argument(
        "--segment-tag",
        default="segment",
        help="Output filename tag when --continuous is set",
    )
    p.add_argument("--no-filter", action="store_true", help="Skip 3D temporal filter on pred")
    p.add_argument("--min-depth-m", type=float, default=0.01)
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


def layout_groups(layout: str) -> tuple[str, tuple[str, ...], str]:
    if layout == "0806":
        return "limb", LIMB_ORDER, LABEL_NPZ_0806
    return "session", SESSION_ORDER, LABEL_NPZ_0810


def load_global_frame_meta(label_root: Path, layout: str) -> dict[int, dict[str, str]]:
    scope_key, groups, npz_name = layout_groups(layout)
    meta_by_global: dict[int, dict[str, str]] = {}
    offset = 0
    for group in groups:
        npz = label_root / group / npz_name
        if not npz.is_file():
            continue
        data = np.load(npz, allow_pickle=True)
        seqs = np.asarray(data["source_aligned_seq"], dtype=np.int64).reshape(-1)
        paths = np.asarray(data["image_paths"])
        for fi, seq in enumerate(seqs):
            meta_by_global[offset + fi] = {
                scope_key: group,
                "seq": int(seq),
                "CAM_A": str(paths[fi, 0]),
                "CAM_D": str(paths[fi, 1]),
            }
        offset += int(seqs.shape[0])
    return meta_by_global


def load_split_indices(split_manifest: Path, split_name: str) -> np.ndarray:
    split = np.load(split_manifest, allow_pickle=True)
    key = f"{split_name}_indices"
    if key not in split:
        raise KeyError(f"{key} missing in {split_manifest}")
    return np.asarray(split[key], dtype=np.int64)


def load_seq_image_paths(label_root: Path, layout: str) -> dict[int, dict[str, str]]:
    """Legacy seq->paths map (0810 lines have unique seqs). Prefer global meta."""
    scope_key, groups, npz_name = layout_groups(layout)
    lookup: dict[int, dict[str, str]] = {}
    for group in groups:
        npz = label_root / group / npz_name
        if not npz.is_file():
            continue
        data = np.load(npz, allow_pickle=True)
        seqs = np.asarray(data["source_aligned_seq"], dtype=np.int64).reshape(-1)
        paths = np.asarray(data["image_paths"])
        for fi, seq in enumerate(seqs):
            lookup[int(seq)] = {
                scope_key: group,
                "seq": int(seq),
                "CAM_A": str(paths[fi, 0]),
                "CAM_D": str(paths[fi, 1]),
            }
    return lookup


def data_root_for_scope(layout: str, scope: str, pre_limb_map: dict[str, str], batch_root: Path | None) -> Path:
    if layout == "0810":
        if batch_root is None:
            raise ValueError("--batch-root is required for 0810 layout")
        return batch_root / scope / "data_root"
    jsonl = Path(pre_limb_map[scope])
    return jsonl.parent.parent.parent


def nose_world_from_pre_limb(record: dict) -> np.ndarray | None:
    joints = record["methods"]["filtered"]["multiview"]
    xyz = resolve_joint_xyz(joints, "nose")
    if xyz is None:
        return None
    nose = np.asarray(xyz, dtype=np.float64).reshape(3)
    return nose if np.isfinite(nose).all() else None


def record_to_world_joints(record: dict, nose_world: np.ndarray) -> dict[str, list[float]]:
    joints = record["methods"]["filtered"]["multiview"]
    nose = np.asarray(nose_world, dtype=np.float64).reshape(3)
    out: dict[str, list[float]] = {}
    for name in DELIVERY_JOINTS:
        local = resolve_joint_xyz(joints, name)
        if local is None:
            continue
        world = np.asarray(local, dtype=np.float64).reshape(3) + nose
        if not np.isfinite(world).all():
            continue
        out[name] = world.tolist()
    return out


def draw_skeleton_2d(
    canvas: np.ndarray,
    projected: dict[str, np.ndarray],
    *,
    edge_bgr: tuple[int, int, int],
    joint_bgr: tuple[int, int, int],
    radius: int = 5,
) -> None:
    for a, b in DELIVERY_EDGES:
        if a not in projected or b not in projected:
            continue
        p1 = tuple(np.rint(projected[a]).astype(int))
        p2 = tuple(np.rint(projected[b]).astype(int))
        cv2.line(canvas, p1, p2, edge_bgr, 2, cv2.LINE_AA)
    for name, uv in projected.items():
        pt = tuple(np.rint(uv).astype(int))
        cv2.circle(canvas, pt, radius + 2, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.circle(canvas, pt, radius, joint_bgr, -1, cv2.LINE_AA)


def load_aligned_by_seq(data_root: Path) -> dict[int, dict[str, str]]:
    aligned_path = data_root / "aligned_data" / "aligned_30hz.csv"
    with aligned_path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {int(row["seq"]): row for row in csv.DictReader(stream)}


def build_scope_projection(scope: str, config_path: Path, data_root: Path) -> dict:
    config = load_json(config_path)
    head_cfg = config.get("head", {})
    head_intrinsics = load_json(
        resolve_repo_path(head_cfg.get("intrinsics", REPO_ROOT / "test_code/calibrate/parameters/intrinsics/head/head_intrinsics_kalibr_omni_1920x1200.json"))
    )
    head_rigid = load_json(
        resolve_repo_path(head_cfg.get("rigid_extrinsics", REPO_ROOT / "test_code/joint_projection/head_stereo_rigid_extrinsics.json"))
    )
    mounts = rigid_camera_mounts(
        head_rigid,
        head_intrinsics,
        basis=str(head_cfg.get("head_camera_rotation_basis", "file")),
    )
    mocap_to_head = head_mocap_correction(head_cfg, config)
    models = {
        socket: OmniCamera.from_calibration(head_intrinsics, socket, name=socket)
        for socket in CAMERA_ORDER
    }
    return {
        "scope": scope,
        "config": config,
        "head_cfg": head_cfg,
        "aligned_by_seq": load_aligned_by_seq(data_root),
        "mounts": mounts,
        "mocap_to_head": mocap_to_head,
        "models": models,
        "rigid_prefix": str(head_cfg.get("rigid_prefix", "mocap_CH3_08")),
    }


def project_world_joints(
    line_ctx: dict,
    seq: int,
    joints_world: dict[str, list[float]],
    *,
    min_depth_m: float,
) -> dict[str, dict[str, np.ndarray]]:
    aligned = line_ctx["aligned_by_seq"].get(int(seq))
    if aligned is None:
        return {}
    rigid_prefix = line_ctx["rigid_prefix"]
    if int(float(aligned.get(f"{rigid_prefix}_status", "0"))) != 1:
        return {}
    world_rigid = rigid_world_transform(aligned, rigid_prefix) @ line_ctx["mocap_to_head"]
    projected_by_socket: dict[str, dict[str, np.ndarray]] = {}
    for socket in CAMERA_ORDER:
        world_camera = world_rigid @ line_ctx["mounts"][socket]
        pose = CameraPose(line_ctx["models"][socket], world_camera)
        projected = project_joints(
            pose,
            joints_world,
            min_depth_m=min_depth_m,
            width=line_ctx["models"][socket].width,
            height=line_ctx["models"][socket].height,
        )
        projected_by_socket[socket] = projected
    return projected_by_socket


def read_bgr(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Could not read {path}")
    h, w = bgr.shape[:2]
    if (w, h) != (VIDEO_WIDTH, VIDEO_HEIGHT):
        bgr = cv2.resize(bgr, (VIDEO_WIDTH, VIDEO_HEIGHT), interpolation=cv2.INTER_AREA)
    return bgr


def process_playback_frame(
    *,
    pred_rec: dict,
    gt_rec: dict,
    meta: dict[str, str],
    scope_key: str,
    scope_filter: str,
    pre_limb_by_scope: dict,
    scope_ctx_by_key: dict,
    min_depth_m: float,
    title: str,
) -> dict[str, np.ndarray] | None:
    scope = meta[scope_key]
    if scope_filter and scope != scope_filter:
        return None
    seq = int(pred_rec["seq"])
    pre_limb = pre_limb_by_scope.get(scope, {})
    line_ctx = scope_ctx_by_key.get(scope)
    if line_ctx is None:
        return None
    pre_row = pre_limb.get(seq)
    if pre_row is None:
        return None
    nose_world = nose_world_from_pre_limb(pre_row)
    if nose_world is None:
        return None

    pred_world = record_to_world_joints(pred_rec, nose_world)
    gt_world = record_to_world_joints(gt_rec, nose_world)
    pred_proj = project_world_joints(line_ctx, seq, pred_world, min_depth_m=min_depth_m)
    gt_proj = project_world_joints(line_ctx, seq, gt_world, min_depth_m=min_depth_m)

    frames: dict[str, np.ndarray] = {}
    for socket in CAMERA_ORDER:
        image_path = Path(meta[socket])
        if not image_path.is_file():
            image_path = Path(str(meta[socket]))
        canvas = read_bgr(image_path)
        draw_skeleton_2d(
            canvas,
            {k: v for k, v in gt_proj.get(socket, {}).items()},
            edge_bgr=GT_EDGE,
            joint_bgr=GT_JOINT,
            radius=5,
        )
        draw_skeleton_2d(
            canvas,
            {k: v for k, v in pred_proj.get(socket, {}).items()},
            edge_bgr=PRED_EDGE,
            joint_bgr=PRED_JOINT,
            radius=4,
        )
        cv2.putText(canvas, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            "RED=GT 3D->2D  ORANGE=Pred filtered 3D->2D",
            (12, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        frames[socket] = canvas
    return frames


def pick_pack_ids(pack_ids: str, max_pack: int) -> list[int]:
    if not pack_ids.strip():
        return [0]
    chosen = [int(x.strip()) for x in pack_ids.split(",") if x.strip()]
    return [p for p in chosen if 0 <= p < max_pack]


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    scope_key, groups, _ = layout_groups(args.layout)
    pack_size = args.pack_size or (PACK_SIZE_0810 if args.layout == "0810" else 30)
    scope_filter = args.scope_filter.strip()
    if not scope_filter and args.layout == "0810":
        scope_filter = "line2"

    _, joint_names, pred_records = load_playback_records(args.pred_playback)
    _, _, gt_records = load_playback_records(args.gt_playback)
    if len(pred_records) != len(gt_records):
        raise ValueError("pred/gt playback length mismatch")

    if args.no_filter:
        filtered_pred = pred_records
        filter_report = {"skipped": True}
    else:
        filtered_pred, filter_report = filter_skeleton_playback_records(
            pred_records,
            joint_names,
            min_volume_score=0.12,
            speed_mad_factor=4.0,
            min_speed_m=0.08,
            bone_length_deviation=0.42,
            median_window=5,
            temporal_sigma=1.0,
        )

    pre_limb_map = json.loads(args.pre_limb_map.read_text(encoding="utf-8"))
    scope_configs = json.loads(args.scope_configs.read_text(encoding="utf-8"))
    pre_limb_by_scope = {
        scope: load_jsonl_records(Path(pre_limb_map[scope]))
        for scope in groups
        if scope in pre_limb_map
    }
    scope_ctx_by_key = {}
    for scope in groups:
        if scope not in scope_configs:
            continue
        data_root = data_root_for_scope(args.layout, scope, pre_limb_map, args.batch_root)
        scope_ctx_by_key[scope] = build_scope_projection(
            scope,
            resolve_repo_path(scope_configs[scope]),
            data_root,
        )

    if args.split_manifest is None:
        raise ValueError("--split-manifest is required to align playback frames with label rows")

    split_indices = load_split_indices(args.split_manifest.expanduser().resolve(), args.split_name)
    if len(split_indices) != len(filtered_pred):
        raise ValueError(
            f"split length {len(split_indices)} != playback length {len(filtered_pred)}"
        )

    meta_by_global = load_global_frame_meta(args.label_root.expanduser().resolve(), args.layout)
    pack_ids = pick_pack_ids(args.pack_ids, max(1, len(filtered_pred) // pack_size))
    manifest: dict[str, object] = {
        "layout": args.layout,
        "pred_playback": str(args.pred_playback),
        "gt_playback": str(args.gt_playback),
        "filter_report": filter_report,
        "continuous": bool(args.continuous),
        "segment_tag": args.segment_tag if args.continuous else None,
        "pack_ids": None if args.continuous else pack_ids,
        "pack_size": pack_size,
        "scope_filter": scope_filter or None,
        "split_manifest": str(args.split_manifest),
        "split_name": args.split_name,
        "fps": args.fps,
        "videos": [],
    }

    if args.continuous:
        writers: dict[str, cv2.VideoWriter] = {}
        stereo_writer: cv2.VideoWriter | None = None
        frame_count = 0
        try:
            for playback_i, (pred_rec, gt_rec) in enumerate(zip(filtered_pred, gt_records, strict=True)):
                global_i = int(split_indices[playback_i])
                seq = int(pred_rec["seq"])
                if int(gt_rec["seq"]) != seq:
                    raise ValueError(f"seq mismatch at playback {playback_i}")
                meta = meta_by_global.get(global_i)
                if meta is None:
                    continue
                if int(meta["seq"]) != seq:
                    raise ValueError(
                        f"global/seq mismatch at playback {playback_i}: "
                        f"global {global_i} seq {meta['seq']} vs playback {seq}"
                    )
                socket_frames = process_playback_frame(
                    pred_rec=pred_rec,
                    gt_rec=gt_rec,
                    meta=meta,
                    scope_key=scope_key,
                    scope_filter=scope_filter,
                    pre_limb_by_scope=pre_limb_by_scope,
                    scope_ctx_by_key=scope_ctx_by_key,
                    min_depth_m=args.min_depth_m,
                    title=f"{args.layout} {args.segment_tag} {scope} seq={seq}",
                )
                if socket_frames is None:
                    continue
                panels = []
                for socket in CAMERA_ORDER:
                    canvas = socket_frames[socket]
                    if socket not in writers:
                        out_path = out_dir / f"stage3_reproj_{args.layout}_{args.segment_tag}_{socket}.mp4"
                        writers[socket] = open_writer(out_path, (VIDEO_WIDTH, VIDEO_HEIGHT), args.fps)
                        manifest["videos"].append(
                            {
                                "path": str(out_path),
                                "camera": socket,
                                "frames": 0,
                                "segment_tag": args.segment_tag,
                            }
                        )
                    writers[socket].write(canvas)
                    panels.append(cv2.resize(canvas, (960, 600), interpolation=cv2.INTER_AREA))
                if panels:
                    stereo = np.hstack(panels)
                    if stereo_writer is None:
                        stereo_path = out_dir / f"stage3_reproj_{args.layout}_{args.segment_tag}_stereo.mp4"
                        stereo_writer = open_writer(stereo_path, (1920, 600), args.fps)
                        manifest["videos"].append(
                            {
                                "path": str(stereo_path),
                                "camera": "stereo",
                                "frames": 0,
                                "segment_tag": args.segment_tag,
                            }
                        )
                    stereo_writer.write(stereo)
                    frame_count += 1
                    if frame_count % 100 == 0:
                        print(f"rendered {frame_count} frames...", flush=True)
        finally:
            for writer in writers.values():
                writer.release()
            if stereo_writer is not None:
                stereo_writer.release()
        for item in manifest["videos"]:
            if isinstance(item, dict):
                item["frames"] = frame_count
        manifest["rendered_frames"] = frame_count
    else:
        pack_ids = pick_pack_ids(args.pack_ids, max(1, len(filtered_pred) // pack_size))
        manifest["pack_ids"] = pack_ids
        for pack_id in pack_ids:
            start = pack_id * pack_size
            end = start + pack_size
            chunk_pred = filtered_pred[start:end]
            chunk_gt = gt_records[start:end]
            if len(chunk_pred) < pack_size:
                continue

            writers: dict[str, cv2.VideoWriter] = {}
            stereo_writer: cv2.VideoWriter | None = None
            try:
                for local_i, (pred_rec, gt_rec) in enumerate(zip(chunk_pred, chunk_gt, strict=True)):
                    playback_i = start + local_i
                    global_i = int(split_indices[playback_i])
                    seq = int(pred_rec["seq"])
                    if int(gt_rec["seq"]) != seq:
                        raise ValueError(f"seq mismatch at pack {pack_id} frame {local_i}")
                    meta = meta_by_global.get(global_i)
                    if meta is None:
                        continue
                    if int(meta["seq"]) != seq:
                        raise ValueError(
                            f"global/seq mismatch at playback {playback_i}: "
                            f"global {global_i} seq {meta['seq']} vs playback {seq}"
                        )
                    scope = meta[scope_key]
                    socket_frames = process_playback_frame(
                        pred_rec=pred_rec,
                        gt_rec=gt_rec,
                        meta=meta,
                        scope_key=scope_key,
                        scope_filter=scope_filter,
                        pre_limb_by_scope=pre_limb_by_scope,
                        scope_ctx_by_key=scope_ctx_by_key,
                        min_depth_m=args.min_depth_m,
                        title=f"{args.layout} test {scope} pack{pack_id} seq={seq}",
                    )
                    if socket_frames is None:
                        continue
                    panels = []
                    for socket in CAMERA_ORDER:
                        canvas = socket_frames[socket]
                        if socket not in writers:
                            out_path = out_dir / f"stage3_reproj_{args.layout}_{scope}_pack{pack_id:03d}_{socket}.mp4"
                            writers[socket] = open_writer(
                                out_path,
                                (VIDEO_WIDTH, VIDEO_HEIGHT),
                                args.fps,
                            )
                            manifest["videos"].append(
                                {
                                    "path": str(out_path),
                                    "pack_id": pack_id,
                                    "camera": socket,
                                    "frames": pack_size,
                                    "seq_start": seq if local_i == 0 else None,
                                }
                            )
                        writers[socket].write(canvas)
                        panels.append(cv2.resize(canvas, (960, 600), interpolation=cv2.INTER_AREA))
                    if panels:
                        stereo = np.hstack(panels)
                        if stereo_writer is None:
                            stereo_path = out_dir / f"stage3_reproj_{args.layout}_{scope}_pack{pack_id:03d}_stereo.mp4"
                            stereo_writer = open_writer(stereo_path, (1920, 600), args.fps)
                            manifest["videos"].append(
                                {
                                    "path": str(stereo_path),
                                    "pack_id": pack_id,
                                    "camera": "stereo",
                                    "frames": pack_size,
                                }
                            )
                        stereo_writer.write(stereo)
            finally:
                for writer in writers.values():
                    writer.release()
                if stereo_writer is not None:
                    stereo_writer.release()

    manifest_path = out_dir / "stage3_reproj_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
