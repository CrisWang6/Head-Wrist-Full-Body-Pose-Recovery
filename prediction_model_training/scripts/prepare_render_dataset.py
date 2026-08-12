#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egorear_sim2d.labels import build_labels_for_render_dir, resolve_projection_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare frame-level RGB images and heatmap label caches.")
    parser.add_argument("--simulation-root", default="/home/gaoweijian/Simulation")
    parser.add_argument("--render-root", required=True)
    parser.add_argument("--label-dir", default="data/labels/humaneva_5app")
    parser.add_argument("--frame-dir", default="data/frames/humaneva_5app")
    parser.add_argument("--smplx-model", default="")
    parser.add_argument("--subjects", default="S1,S2", help="Comma-separated subject prefixes to include, e.g. S1,S2.")
    parser.add_argument("--heatmap-width", type=int, default=114)
    parser.add_argument("--heatmap-height", type=int, default=64)
    parser.add_argument("--frame-width", type=int, default=456)
    parser.add_argument("--frame-height", type=int, default=256)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--projection-model", default="fisheye_equidistant", choices=("auto", "fisheye_equidistant", "perspective_usd"))
    parser.add_argument("--fisheye-fov-deg", type=float, default=220.0)
    parser.add_argument("--exclude-actions", default="Box,ThrowCatch")
    parser.add_argument("--appearances-per-motion", type=int, default=2)
    parser.add_argument("--appearance-seed", type=int, default=20260612)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--extract-backend", default="ffmpeg", choices=("ffmpeg", "opencv"))
    parser.add_argument("--extract-workers", type=int, default=8)
    parser.add_argument("--ffmpeg-hwaccel", default="auto", choices=("auto", "cuda", "none"))
    parser.add_argument("--skip-existing-frames", action="store_true")
    parser.add_argument("--require-stats", action="store_true", help="Only use appearance dirs with 8 completed IsaacSim stats files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    simulation_root = Path(args.simulation_root).expanduser().resolve()
    render_root = Path(args.render_root).expanduser().resolve()
    label_dir = Path(args.label_dir).expanduser().resolve()
    frame_dir = Path(args.frame_dir).expanduser().resolve()
    smplx_model = Path(args.smplx_model).expanduser() if args.smplx_model else simulation_root / "smplx_models/SMPLX_NEUTRAL_2020.npz"
    subjects = tuple(part.strip() for part in args.subjects.split(",") if part.strip())

    label_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)
    exclude_actions = tuple(part.strip().lower() for part in args.exclude_actions.split(",") if part.strip())
    render_dirs = discover_complete_appearance_dirs(
        render_root,
        subjects,
        exclude_actions=exclude_actions,
        appearances_per_motion=args.appearances_per_motion,
        seed=args.appearance_seed,
        require_stats=args.require_stats,
    )
    if not render_dirs:
        raise FileNotFoundError(f"No complete appearance dirs found under {render_root}")

    label_items = []
    for render_dir in render_dirs:
        rel = render_dir.relative_to(render_root)
        label_path = label_dir / rel / f"heatmap_labels_{args.heatmap_width}x{args.heatmap_height}.npz"
        expected_projection = resolve_projection_model(render_dir, args.projection_model)
        if label_path.exists() and label_matches_projection(label_path, expected_projection, args.fisheye_fov_deg):
            result = load_existing_label_result(label_path)
        else:
            result = build_labels_for_render_dir(
                render_dir=render_dir,
                simulation_root=simulation_root,
                smplx_model_path=smplx_model,
                output_path=label_path,
                heatmap_size=(args.heatmap_width, args.heatmap_height),
                sigma=args.sigma,
                projection_model=args.projection_model,
                fisheye_fov_deg=args.fisheye_fov_deg,
            )
        label_items.append(
            {
                "label_path": label_path,
                "render_dir": render_dir,
                "frames": result.frames,
                "cameras": list(result.cameras),
                "valid_points": result.valid_points,
            }
        )

    extraction_results = extract_all_frames(
        label_items=label_items,
        render_root=render_root,
        frame_root=frame_dir,
        jpeg_quality=args.jpeg_quality,
        frame_size=(args.frame_width, args.frame_height),
        skip_existing=args.skip_existing_frames,
        backend=args.extract_backend,
        workers=args.extract_workers,
        ffmpeg_hwaccel=args.ffmpeg_hwaccel,
    )

    items = []
    for label_item, extracted in zip(label_items, extraction_results):
        item = {
            "render_dir": str(label_item["render_dir"]),
            "label_path": str(label_item["label_path"]),
            "frames": label_item["frames"],
            "cameras": label_item["cameras"],
            "valid_points": label_item["valid_points"],
            "extracted_frames": extracted,
        }
        print(json.dumps(item, indent=2), flush=True)
        items.append(item)

    manifest = {
        "simulation_root": str(simulation_root),
        "render_root": str(render_root),
        "label_dir": str(label_dir),
        "frame_dir": str(frame_dir),
        "smplx_model": str(smplx_model),
        "subjects": list(subjects),
        "heatmap_size": [args.heatmap_width, args.heatmap_height],
        "frame_size": [args.frame_width, args.frame_height],
        "extract_backend": args.extract_backend,
        "extract_workers": args.extract_workers,
        "ffmpeg_hwaccel": args.ffmpeg_hwaccel,
        "sigma": args.sigma,
        "projection_model": args.projection_model,
        "fisheye_fov_deg": args.fisheye_fov_deg,
        "exclude_actions": list(exclude_actions),
        "appearances_per_motion": int(args.appearances_per_motion),
        "appearance_seed": int(args.appearance_seed),
        "items": items,
    }
    manifest_path = label_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "items": len(items)}, indent=2), flush=True)
    return 0


def load_existing_label_result(label_path: Path):
    data = np.load(label_path, allow_pickle=True)
    if "head_keypoints" in data.files:
        frames = int(data["head_keypoints"].shape[0])
        valid_points = int(np.asarray(data["head_visible"], dtype=bool).sum() + np.asarray(data["wrist_visible"], dtype=bool).sum())
    else:
        frames = int(data["keypoints"].shape[0])
        valid_points = int(np.asarray(data["visible"], dtype=bool).sum())
    return SimpleNamespace(
        frames=frames,
        cameras=tuple(str(name) for name in data["camera_names"]),
        valid_points=valid_points,
    )


def label_matches_projection(label_path: Path, expected_projection: str, expected_fov_deg: float) -> bool:
    try:
        from egorear_sim2d.labels import LABEL_SCHEMA_VERSION

        data = np.load(label_path, allow_pickle=True)
        if "schema_version" not in data.files:
            return False
        if str(np.asarray(data["schema_version"]).reshape(-1)[0]) != LABEL_SCHEMA_VERSION:
            return False
        if "projection_model" not in data.files:
            return False
        actual = str(np.asarray(data["projection_model"]).reshape(-1)[0])
        if actual != str(expected_projection):
            return False
        if "fisheye_fov_deg" not in data.files:
            return False
        actual_fov = float(np.asarray(data["fisheye_fov_deg"]).reshape(-1)[0])
        return abs(actual_fov - float(expected_fov_deg)) < 1e-4
    except Exception:
        return False


def discover_complete_appearance_dirs(
    render_root: Path,
    subjects: tuple[str, ...],
    *,
    exclude_actions: tuple[str, ...] = (),
    appearances_per_motion: int = 2,
    seed: int = 20260612,
    require_stats: bool = False,
) -> list[Path]:
    dirs = []
    rng = np.random.default_rng(int(seed))
    for motion_dir in sorted(path for path in render_root.iterdir() if path.is_dir() and not path.name == "logs"):
        if subjects and not any(motion_dir.name.startswith(f"{subject}_") for subject in subjects):
            continue
        lowered = motion_dir.name.lower()
        if exclude_actions and any(action in lowered for action in exclude_actions):
            continue
        motion_appearances = []
        for appearance_dir in sorted(motion_dir.glob("appearance_*")):
            if not (appearance_dir / "blenderproc_motion_cache.npz").exists():
                continue
            if len(list(appearance_dir.glob("*.mp4"))) < 8:
                continue
            if require_stats and len(list(appearance_dir.glob("*_isaacsim_stats.json"))) < 8:
                continue
            motion_appearances.append(appearance_dir)
        if appearances_per_motion > 0 and len(motion_appearances) > appearances_per_motion:
            selected = rng.choice(len(motion_appearances), size=int(appearances_per_motion), replace=False)
            motion_appearances = [motion_appearances[int(idx)] for idx in sorted(selected)]
        dirs.extend(motion_appearances)
    return dirs


def extract_all_frames(
    *,
    label_items: list[dict[str, object]],
    render_root: Path,
    frame_root: Path,
    jpeg_quality: int,
    frame_size: tuple[int, int],
    skip_existing: bool,
    backend: str,
    workers: int,
    ffmpeg_hwaccel: str,
) -> list[int]:
    if workers <= 1:
        return [
            extract_frames_for_label(
                label_path=Path(item["label_path"]),
                render_root=render_root,
                frame_root=frame_root,
                jpeg_quality=jpeg_quality,
                frame_size=frame_size,
                skip_existing=skip_existing,
                backend=backend,
                ffmpeg_hwaccel=ffmpeg_hwaccel,
            )
            for item in label_items
        ]

    results = [0] * len(label_items)
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        future_to_idx = {
            executor.submit(
                extract_frames_for_label,
                label_path=Path(item["label_path"]),
                render_root=render_root,
                frame_root=frame_root,
                jpeg_quality=jpeg_quality,
                frame_size=frame_size,
                skip_existing=skip_existing,
                backend=backend,
                ffmpeg_hwaccel=ffmpeg_hwaccel,
            ): idx
            for idx, item in enumerate(label_items)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = int(future.result())
            print(json.dumps({"extract_done": idx + 1, "total": len(label_items), "frames": results[idx]}), flush=True)
    return results


def extract_frames_for_label(
    *,
    label_path: Path,
    render_root: Path,
    frame_root: Path,
    jpeg_quality: int,
    frame_size: tuple[int, int],
    skip_existing: bool,
    backend: str,
    ffmpeg_hwaccel: str,
) -> int:
    import numpy as np

    data = np.load(label_path, allow_pickle=True)
    source_render_dir = Path(str(data["source_render_dir"][0])).expanduser().resolve()
    rel = source_render_dir.relative_to(render_root)
    camera_names = [str(name) for name in data["camera_names"]]
    video_paths = [Path(str(path)) for path in data["video_paths"]]
    expected_frames = int(data["keypoints"].shape[0])
    written = 0
    for camera_name, video_path in zip(camera_names, video_paths):
        out_dir = frame_root / rel / camera_name
        out_dir.mkdir(parents=True, exist_ok=True)
        if skip_existing and len(list(out_dir.glob("*.jpg"))) >= expected_frames:
            continue
        if backend == "ffmpeg":
            written += extract_one_video_ffmpeg(
                video_path=video_path,
                out_dir=out_dir,
                expected_frames=expected_frames,
                frame_size=frame_size,
                jpeg_quality=jpeg_quality,
                hwaccel=ffmpeg_hwaccel,
            )
            continue
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx >= expected_frames:
                break
            frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
            out_path = out_dir / f"{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            written += 1
            frame_idx += 1
        cap.release()
        if frame_idx < expected_frames:
            raise RuntimeError(f"{video_path} has {frame_idx} frames, expected {expected_frames}")
    return written


def extract_one_video_ffmpeg(
    *,
    video_path: Path,
    out_dir: Path,
    expected_frames: int,
    frame_size: tuple[int, int],
    jpeg_quality: int,
    hwaccel: str,
) -> int:
    for old in out_dir.glob("*.jpg"):
        old.unlink()
    out_pattern = out_dir / "%06d.jpg"
    qscale = max(2, min(31, int(round((100 - jpeg_quality) / 3.2 + 2))))
    base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    vf = f"scale={frame_size[0]}:{frame_size[1]}"
    tried = []
    if hwaccel in {"auto", "cuda"}:
        tried.append(base + ["-hwaccel", "cuda", "-i", str(video_path), "-vf", vf, "-q:v", str(qscale), str(out_pattern)])
    if hwaccel in {"auto", "none"}:
        tried.append(base + ["-threads", "2", "-i", str(video_path), "-vf", vf, "-q:v", str(qscale), str(out_pattern)])
    last_error = ""
    for command in tried:
        proc = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0:
            count = len(list(out_dir.glob("*.jpg")))
            if count >= expected_frames:
                return count
            last_error = f"ffmpeg wrote {count}/{expected_frames} frames for {video_path}"
        else:
            last_error = proc.stderr.strip()
        for old in out_dir.glob("*.jpg"):
            old.unlink()
    raise RuntimeError(f"ffmpeg extraction failed for {video_path}: {last_error}")


if __name__ == "__main__":
    raise SystemExit(main())
