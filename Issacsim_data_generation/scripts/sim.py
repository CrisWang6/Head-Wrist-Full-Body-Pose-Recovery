#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geosim.config import load_config, override_config
from geosim.motion import load_motion_dir, load_motion_npz, synthetic_motion
from geosim.runner import run_motion_sequence, summarize_result_sets, summarize_results
from geosim.smplx_numpy import load_smplx_model
from geosim.visualization import render_motion_video


def parse_geometry_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run geometry-only wrist tag simulation.")
    parser.add_argument("--config", default=str(ROOT / "configs/default_geometry.json"))
    parser.add_argument("--smplx-model", default=str(ROOT / "smplx_models/SMPLX_NEUTRAL_2020.npz"))
    parser.add_argument("--motion", action="append", default=[], help="Prepared or AMASS motion .npz file. Can be repeated.")
    parser.add_argument("--motion-dir", default="", help="Directory containing prepared or AMASS motion .npz files.")
    parser.add_argument("--synthetic", action="store_true", help="Run the built-in synthetic reach sequence.")
    parser.add_argument("--tag-size-m", type=float, default=None, help="Override square tag side length in meters.")
    parser.add_argument("--pixel-noise-std", type=float, default=None, help="Gaussian pixel noise stddev.")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames per motion; 0 means all.")
    parser.add_argument("--no-visualization", action="store_true", help="Skip random 10Hz visualization video.")
    parser.add_argument("--visualization-dir", default=str(ROOT / "outputs/visualizations"))
    parser.add_argument("--visualization-seed", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary.")
    return parser.parse_args()


def geometry_main() -> int:
    args = parse_geometry_args()
    config = override_config(
        load_config(args.config),
        tag_size_m=args.tag_size_m,
        pixel_noise_std=args.pixel_noise_std,
        max_frames=args.max_frames,
    )

    default_humaneva = ROOT / "test_motion/HumanEva"
    motion_dir = args.motion_dir
    if not args.synthetic and not args.motion and not motion_dir and default_humaneva.exists():
        motion_dir = str(default_humaneva)

    motions = []
    if args.synthetic or (not args.motion and not motion_dir):
        motions.append(synthetic_motion())
    for motion_path in args.motion:
        motions.append(load_motion_npz(motion_path, smplx_model_path=args.smplx_model))
    if motion_dir:
        motions.extend(load_motion_dir(motion_dir, smplx_model_path=args.smplx_model))

    if not motions:
        print("No motion files found. Add prepared .npz files to test_motion/ or use --synthetic.", file=sys.stderr)
        return 2

    all_summaries = {}
    all_result_sets = []
    for motion in motions:
        results = run_motion_sequence(motion, config)
        summary = summarize_results(results)
        all_summaries[motion.name] = summary
        all_result_sets.append(results)
        if not args.json:
            print(f"\n{motion.name}", flush=True)
            print(f"  frames: {summary['frames']}")
            print(f"  success: {summary['success_frames']} / {summary['frames']} ({summary['success_rate']:.3f})")
            if summary["success_frames"]:
                print(f"  position error mean/p95/max: {summary['position_mean_m']:.6e} / {summary['position_p95_m']:.6e} / {summary['position_max_m']:.6e} m")
                print(f"  rotation error mean/p95/max: {summary['rotation_mean_deg']:.6e} / {summary['rotation_p95_deg']:.6e} / {summary['rotation_max_deg']:.6e} deg")
                print(f"  visible observations mean: {summary['visible_observations_mean']:.2f}")
    overall = summarize_result_sets(all_result_sets)
    all_summaries["_overall"] = overall

    visualization_path = ""
    visual_candidates = [motion for motion in motions if motion.source_path is not None]
    if visual_candidates and not args.no_visualization:
        rng = random.Random(args.visualization_seed)
        selected = rng.choice(visual_candidates)
        model = load_smplx_model(args.smplx_model)
        safe_name = selected.name.replace("/", "_").replace("\\", "_")
        visualization_path = str(Path(args.visualization_dir) / f"{safe_name}_geosim.avi")
        render_motion_video(
            selected,
            config,
            model,
            visualization_path,
            output_fps=10.0,
            max_frames=config.max_frames,
        )
        all_summaries["_visualization"] = {"motion": selected.name, "path": visualization_path}

    if not args.json:
        print("\nOverall", flush=True)
        print(f"  motions: {overall['motions']}")
        print(f"  frames: {overall['frames']}")
        print(f"  success: {overall['success_frames']} / {overall['frames']} ({overall['success_rate']:.3f})")
        if overall["success_frames"]:
            print(f"  position error mean/p95/max: {overall['position_mean_m']:.6e} / {overall['position_p95_m']:.6e} / {overall['position_max_m']:.6e} m")
            print(f"  rotation error mean/p95/max: {overall['rotation_mean_deg']:.6e} / {overall['rotation_p95_deg']:.6e} / {overall['rotation_max_deg']:.6e} deg")
            print(f"  visible observations mean: {overall['visible_observations_mean']:.2f}")
        if visualization_path:
            print(f"\nVisualization: {visualization_path}")
    if args.json:
        print(json.dumps(all_summaries, indent=2))
    return 0


COMMANDS = {
    'geometry': geometry_main,
}


def main() -> int:
    parser = argparse.ArgumentParser(description='Run simulation baselines.')
    parser.add_argument("command", choices=sorted(COMMANDS))
    args, rest = parser.parse_known_args()
    sys.argv = [f"{Path(sys.argv[0]).name} {args.command}", *rest]
    return int(COMMANDS[args.command]() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
