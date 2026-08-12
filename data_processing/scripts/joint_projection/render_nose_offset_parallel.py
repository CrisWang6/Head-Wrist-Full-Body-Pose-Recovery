#!/usr/bin/env python3
"""Parallel chunked render for nose-offset before/after head videos.

Workflow:
  1) --prepare   build projection cache + report (fast, single process)
  2) --chunk I N render seq shard I of N into chunks/chunk_II/
  3) --merge     concat all chunk mp4s into final videos

Launch many --chunk workers on local + remote hosts to saturate CPUs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from multiview_geometry import OmniCamera, load_json
from optimize_multiview_head_nose_offset import (
    CAMERA_ORDER,
    DEFAULT_CONFIG,
    build_records,
    draw_body_skeleton,
    filter_face,
    fixed_nose_uv_from_csv,
    load_skeleton_playback,
    open_writer,
)
from render_multiview_to_head import (
    DEFAULT_HEAD_INTRINSICS,
    DEFAULT_HEAD_RIGID,
    H265CaptureReader,
    HeadTimestampIndex,
    discover_head_dir,
    draw_laterality_labels,
    find_head_video,
    head_mocap_correction,
    infer_head_module_from_video,
    resolve_repo_path,
    rigid_camera_mounts,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--head-a-nose-csv", type=Path, required=True)
    p.add_argument("--head-d-nose-csv", type=Path, required=True)
    p.add_argument("--report", type=Path, help="Existing report with final_offset_rigid_m")
    p.add_argument(
        "--before-playback",
        type=Path,
        help="Raw triangulated skeleton_playback for BEFORE panels "
        "(default: full/skeleton_playback_raw.json).",
    )
    p.add_argument(
        "--after-playback",
        type=Path,
        help="Optimized skeleton_playback for AFTER panels "
        "(default: full/skeleton_playback.json).",
    )
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--min-depth-m", type=float, default=0.01)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--chunk", type=int, nargs=2, metavar=("INDEX", "COUNT"))
    p.add_argument("--merge", action="store_true")
    p.add_argument(
        "--after-only",
        action="store_true",
        help="Render/merge only optimized (AFTER) panels; skip direct before + 2x2 grid.",
    )
    p.add_argument("--ffmpeg", type=Path)
    return p.parse_args()


def find_ffmpeg(explicit: Path | None) -> str:
    if explicit is not None:
        return str(explicit)
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def prepare(args: argparse.Namespace) -> Path:
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_json(args.config)
    head_cfg = config.get("head", {})
    report_path = args.report or (output_dir / "report.json")
    report = load_json(report_path)
    offset = np.asarray(report["final_offset_rigid_m"], dtype=np.float64)

    full = data_root / "multiview_3d_results" / "full"
    before_path = (
        args.before_playback.resolve()
        if args.before_playback is not None
        else full / "skeleton_playback_raw.json"
    )
    after_path = (
        args.after_playback.resolve()
        if args.after_playback is not None
        else full / "skeleton_playback.json"
    )
    if not before_path.is_file():
        before_path = after_path
    _, _, frames_before = load_skeleton_playback(before_path)
    _, _, frames_after = load_skeleton_playback(after_path)
    aligned_path = data_root / "aligned_data" / "aligned_30hz.csv"
    import csv

    with aligned_path.open("r", encoding="utf-8-sig", newline="") as stream:
        aligned_by_seq = {int(row["seq"]): row for row in csv.DictReader(stream)}

    head_dir = discover_head_dir(data_root, config)
    rigid_prefix = head_cfg.get("rigid_prefix", "mocap_CH3_08")
    head_intrinsics = load_json(
        resolve_repo_path(head_cfg.get("intrinsics", DEFAULT_HEAD_INTRINSICS))
    )
    head_rigid = load_json(
        resolve_repo_path(head_cfg.get("rigid_extrinsics", DEFAULT_HEAD_RIGID))
    )
    camera_basis = str(head_cfg.get("head_camera_rotation_basis", "xy_swap"))
    mounts = rigid_camera_mounts(head_rigid, head_intrinsics, basis=camera_basis)
    mocap_to_head = head_mocap_correction(head_cfg, config)
    models = {
        socket: OmniCamera.from_calibration(head_intrinsics, socket, name=socket)
        for socket in CAMERA_ORDER
    }
    videos = {
        socket: find_head_video(head_dir, socket, config) for socket in CAMERA_ORDER
    }
    indexes = {
        socket: HeadTimestampIndex(
            head_dir / "timestamps.csv",
            socket,
            module=infer_head_module_from_video(videos[socket]),
        )
        for socket in CAMERA_ORDER
    }
    tolerance_ms = float(head_cfg.get("timestamp_match_tolerance_ms", 1.0))
    fixed_a, _ = fixed_nose_uv_from_csv(args.head_a_nose_csv)
    fixed_d, _ = fixed_nose_uv_from_csv(args.head_d_nose_csv)
    fixed_uv = {"CAM_A": fixed_a, "CAM_D": fixed_d}

    # BEFORE: raw triangulated skeleton, no extra head-rigid offset.
    before = build_records(
        frames_before,
        aligned_by_seq,
        rigid_prefix,
        mocap_to_head,
        mounts,
        models,
        indexes,
        offset_rigid=np.zeros(3),
        raw_nose=None,
        min_depth_m=args.min_depth_m,
        tolerance_ms=tolerance_ms,
    )
    # AFTER: limb-GT / joint-opt skeleton + small head-rigid refine from report.
    after = build_records(
        frames_after,
        aligned_by_seq,
        rigid_prefix,
        mocap_to_head,
        mounts,
        models,
        indexes,
        offset_rigid=offset,
        raw_nose=None,
        fixed_nose_uv=fixed_uv,
        min_depth_m=args.min_depth_m,
        tolerance_ms=tolerance_ms,
    )

    def pack(records: list[dict]) -> list[dict]:
        packed = []
        for item in records:
            packed.append(
                {
                    "seq": int(item["seq"]),
                    "frames": {k: int(v) for k, v in item["frames"].items()},
                    "projected": {
                        socket: {
                            name: np.asarray(uv, dtype=np.float64).tolist()
                            for name, uv in item["projected"][socket].items()
                        }
                        for socket in CAMERA_ORDER
                    },
                    "rtmw": {
                        socket: np.asarray(uv, dtype=np.float64).tolist()
                        for socket, uv in item.get("rtmw", {}).items()
                    },
                }
            )
        return packed

    cache = {
        "videos": {socket: str(path) for socket, path in videos.items()},
        "timestamps": str(head_dir / "timestamps.csv"),
        "fps": args.fps,
        "sizes": {
            socket: [int(models[socket].width), int(models[socket].height)]
            for socket in CAMERA_ORDER
        },
        "before_playback": str(before_path),
        "after_playback": str(after_path),
        "before": pack(before),
        "after": pack(after),
        "final_offset_rigid_m": offset.tolist(),
        "fixed_rtmw_nose_uv": {
            socket: fixed_uv[socket].tolist() for socket in CAMERA_ORDER
        },
    }
    cache_path = output_dir / "render_cache.json"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")
    print(
        json.dumps(
            {
                "cache": str(cache_path),
                "frames": len(before),
                "offset_mm": float(np.linalg.norm(offset) * 1000.0),
            },
            ensure_ascii=False,
        )
    )
    return cache_path


def render_chunk(args: argparse.Namespace) -> None:
    index, count = args.chunk
    output_dir = args.output_dir.resolve()
    cache = load_json(output_dir / "render_cache.json")
    before = cache["before"]
    after = cache["after"]
    n = len(before)
    start = (index * n) // count
    end = ((index + 1) * n) // count
    if start >= end:
        print(json.dumps({"chunk": index, "frames": 0}))
        return
    chunk_dir = output_dir / "chunks" / f"chunk_{index:02d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    videos = {socket: Path(cache["videos"][socket]) for socket in CAMERA_ORDER}
    indexes = {
        socket: HeadTimestampIndex(
            Path(cache["timestamps"]),
            socket,
            module=infer_head_module_from_video(videos[socket]),
        )
        for socket in CAMERA_ORDER
    }
    readers = {
        socket: H265CaptureReader(videos[socket], indexes[socket].rows)
        for socket in CAMERA_ORDER
    }
    fps = float(cache["fps"])
    sizes = {socket: tuple(cache["sizes"][socket]) for socket in CAMERA_ORDER}
    writers = {
        "after_a": open_writer(chunk_dir / "after_a.mp4", sizes["CAM_A"], fps),
        "after_d": open_writer(chunk_dir / "after_d.mp4", sizes["CAM_D"], fps),
    }
    if not args.after_only:
        writers["before_a"] = open_writer(
            chunk_dir / "before_a.mp4", sizes["CAM_A"], fps
        )
        writers["before_d"] = open_writer(
            chunk_dir / "before_d.mp4", sizes["CAM_D"], fps
        )
        writers["grid"] = open_writer(chunk_dir / "grid.mp4", (1920, 1200), fps)
    try:
        for bi, ai in zip(before[start:end], after[start:end]):
            if args.after_only:
                for socket in ("CAM_A", "CAM_D"):
                    base = readers[socket].read(int(bi["frames"][socket]))
                    image_a = base.copy()
                    proj_a = {
                        name: np.asarray(uv, dtype=np.float64)
                        for name, uv in ai["projected"][socket].items()
                    }
                    draw_body_skeleton(image_a, proj_a, (0, 255, 255), radius=4)
                    draw_laterality_labels(image_a, filter_face(proj_a))
                    det = ai.get("rtmw", {}).get(socket)
                    if det is not None:
                        cv2.circle(
                            image_a,
                            tuple(np.rint(np.asarray(det)).astype(int)),
                            8,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                    cv2.putText(
                        image_a,
                        f"AFTER nose-opt {socket} seq={ai['seq']} cyan + green fixed RTMW",
                        (24, 42),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    if socket == "CAM_A":
                        writers["after_a"].write(image_a)
                    else:
                        writers["after_d"].write(image_a)
                continue
            images = {}
            for socket in CAMERA_ORDER:
                base = readers[socket].read(int(bi["frames"][socket]))
                image_b = base.copy()
                image_a = base.copy()
                proj_b = {
                    name: np.asarray(uv, dtype=np.float64)
                    for name, uv in bi["projected"][socket].items()
                }
                proj_a = {
                    name: np.asarray(uv, dtype=np.float64)
                    for name, uv in ai["projected"][socket].items()
                }
                draw_body_skeleton(image_b, proj_b, (255, 0, 255), radius=4)
                draw_body_skeleton(image_a, proj_a, (0, 255, 255), radius=4)
                draw_laterality_labels(image_b, filter_face(proj_b))
                draw_laterality_labels(image_a, filter_face(proj_a))
                det = ai.get("rtmw", {}).get(socket)
                if det is not None:
                    cv2.circle(
                        image_a,
                        tuple(np.rint(np.asarray(det)).astype(int)),
                        8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    image_b,
                    f"BEFORE direct {socket} seq={bi['seq']} magenta nose-only",
                    (24, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    image_a,
                    f"AFTER nose-opt {socket} seq={ai['seq']} cyan + green fixed RTMW",
                    (24, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
                images[socket] = (image_b, image_a)
            writers["before_a"].write(images["CAM_A"][0])
            writers["after_a"].write(images["CAM_A"][1])
            writers["before_d"].write(images["CAM_D"][0])
            writers["after_d"].write(images["CAM_D"][1])
            grid = np.vstack(
                (
                    np.hstack(
                        (
                            cv2.resize(images["CAM_A"][0], (960, 600)),
                            cv2.resize(images["CAM_A"][1], (960, 600)),
                        )
                    ),
                    np.hstack(
                        (
                            cv2.resize(images["CAM_D"][0], (960, 600)),
                            cv2.resize(images["CAM_D"][1], (960, 600)),
                        )
                    ),
                )
            )
            writers["grid"].write(grid)
    finally:
        for writer in writers.values():
            writer.release()
        for reader in readers.values():
            reader.close()
    meta = {"chunk": index, "count": count, "start": start, "end": end, "frames": end - start}
    (chunk_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    print(json.dumps(meta))


def merge(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.resolve()
    cache = load_json(output_dir / "render_cache.json")
    chunk_root = output_dir / "chunks"
    metas = sorted(chunk_root.glob("chunk_*/meta.json"))
    if not metas:
        raise RuntimeError(f"No chunks under {chunk_root}")
    order = []
    for meta_path in metas:
        meta = load_json(meta_path)
        order.append((int(meta["start"]), meta_path.parent))
    order.sort()
    ffmpeg = find_ffmpeg(args.ffmpeg)
    names = {
        "after_a": "head_CAM_A_nose_offset_opt.mp4",
        "after_d": "head_CAM_D_nose_offset_opt.mp4",
    }
    if not args.after_only:
        names.update(
            {
                "before_a": "head_CAM_A_direct_noseonly.mp4",
                "before_d": "head_CAM_D_direct_noseonly.mp4",
                "grid": "head_2x2_direct_vs_nose_offset_opt.mp4",
            }
        )
    for key, final_name in names.items():
        list_path = output_dir / f"concat_{key}.txt"
        lines = []
        for _, folder in order:
            clip = folder / f"{key}.mp4"
            if not clip.is_file():
                raise FileNotFoundError(clip)
            lines.append(f"file '{clip.as_posix()}'")
        list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        out = output_dir / final_name
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_path),
            "-c",
            "copy",
            str(out),
        ]
        subprocess.run(cmd, check=True)
        print({"merged": str(out), "bytes": out.stat().st_size, "parts": len(order)})
    report_path = output_dir / "report.json"
    if report_path.is_file():
        report = load_json(report_path)
        outputs = {
            "after_a": str(output_dir / names["after_a"]),
            "after_d": str(output_dir / names["after_d"]),
        }
        if not args.after_only:
            outputs.update(
                {
                    "before_a": str(output_dir / names["before_a"]),
                    "before_d": str(output_dir / names["before_d"]),
                    "grid": str(output_dir / names["grid"]),
                }
            )
        report["outputs"] = outputs
        report["rendered_frames"] = len(cache["before"])
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    modes = sum(bool(x) for x in (args.prepare, args.chunk, args.merge))
    if modes != 1:
        raise SystemExit("Specify exactly one of --prepare / --chunk I N / --merge")
    if args.prepare:
        prepare(args)
    elif args.chunk:
        render_chunk(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
