#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from egorear_sim2d.labels import build_labels_for_render_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 2D heatmap label caches from Simulation Isaac render outputs.")
    parser.add_argument("--simulation-root", default="/home/gaoweijian/Simulation")
    parser.add_argument("--render-root", required=True, help="Folder containing appearance_* render subfolders.")
    parser.add_argument("--output-dir", default="data/labels")
    parser.add_argument("--smplx-model", default="", help="Defaults to <simulation-root>/smplx_models/SMPLX_NEUTRAL_2020.npz.")
    parser.add_argument("--heatmap-width", type=int, default=114)
    parser.add_argument("--heatmap-height", type=int, default=64)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--projection-model", default="fisheye_equidistant", choices=("auto", "fisheye_equidistant", "perspective_usd"))
    parser.add_argument("--fisheye-fov-deg", type=float, default=220.0)
    parser.add_argument("--max-appearances", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    simulation_root = Path(args.simulation_root).expanduser().resolve()
    render_root = Path(args.render_root).expanduser().resolve()
    smplx_model = Path(args.smplx_model).expanduser() if args.smplx_model else simulation_root / "smplx_models/SMPLX_NEUTRAL_2020.npz"
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    render_dirs = sorted(path for path in render_root.glob("appearance_*") if (path / "blenderproc_motion_cache.npz").exists())
    if not render_dirs and (render_root / "blenderproc_motion_cache.npz").exists():
        render_dirs = [render_root]
    if args.max_appearances > 0:
        render_dirs = render_dirs[: args.max_appearances]
    if not render_dirs:
        raise FileNotFoundError(f"No render cache folders found under {render_root}")

    results = []
    for render_dir in render_dirs:
        out_path = output_dir / render_dir.name / f"heatmap_labels_{args.heatmap_width}x{args.heatmap_height}.npz"
        result = build_labels_for_render_dir(
            render_dir=render_dir,
            simulation_root=simulation_root,
            smplx_model_path=smplx_model,
            output_path=out_path,
            heatmap_size=(args.heatmap_width, args.heatmap_height),
            sigma=args.sigma,
            projection_model=args.projection_model,
            fisheye_fov_deg=args.fisheye_fov_deg,
        )
        item = {
            "render_dir": str(render_dir),
            "label_path": str(result.path),
            "frames": result.frames,
            "cameras": list(result.cameras),
            "valid_points": result.valid_points,
        }
        print(json.dumps(item, indent=2), flush=True)
        results.append(item)

    manifest = {
        "simulation_root": str(simulation_root),
        "render_root": str(render_root),
        "smplx_model": str(smplx_model),
        "heatmap_size": [args.heatmap_width, args.heatmap_height],
        "sigma": args.sigma,
        "projection_model": args.projection_model,
        "fisheye_fov_deg": args.fisheye_fov_deg,
        "items": results,
    }
    manifest_path = output_dir / "labels_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "items": len(results)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
