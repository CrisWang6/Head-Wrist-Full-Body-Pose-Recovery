#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import argparse
import os
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SIMULATION_ROOT = ROOT.parent / "Issacsim_data_generation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for S2 renders, prepare frame dataset, then start stage-1 training.")
    parser.add_argument("--simulation-root", default=str(DEFAULT_SIMULATION_ROOT))
    parser.add_argument("--render-root", default=str(DEFAULT_SIMULATION_ROOT / "outputs/isaacsim_humaneva_5app"))
    parser.add_argument("--label-dir", default=str(ROOT / "data/labels/humaneva_5app"))
    parser.add_argument("--frame-dir", default=str(ROOT / "data/frames/humaneva_5app"))
    parser.add_argument("--checkpoint-dir", default=str(ROOT / "checkpoints/humaneva_5app_stage1"))
    parser.add_argument("--log-dir", default=str(ROOT / "logs/humaneva_5app_stage1"))
    parser.add_argument("--gpu-ids", default="1,2,3,4")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--extract-workers", type=int, default=16)
    parser.add_argument("--poll-seconds", type=int, default=120)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = ROOT / "logs/wait_prepare_train.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(log_path, f"[stage1] start {now()}")
    simulation_root = Path(args.simulation_root)
    render_root = Path(args.render_root)
    while True:
        missing = missing_s2_appearances(simulation_root, render_root)
        active_render = has_active_render()
        log(log_path, f"[stage1] poll {now()} missing_s2={len(missing)} active_render={active_render}")
        if not missing and not active_render:
            break
        time.sleep(max(5, int(args.poll_seconds)))

    py = sys.executable
    prepare_cmd = [
        py,
        str(ROOT / "scripts/prepare_render_dataset.py"),
        "--simulation-root",
        str(simulation_root),
        "--render-root",
        str(render_root),
        "--label-dir",
        str(args.label_dir),
        "--frame-dir",
        str(args.frame_dir),
        "--subjects",
        "S1,S2",
        "--extract-backend",
        "ffmpeg",
        "--extract-workers",
        str(args.extract_workers),
        "--ffmpeg-hwaccel",
        "none",
        "--skip-existing-frames",
    ]
    log(log_path, "[stage1] prepare " + " ".join(prepare_cmd))
    run_logged(prepare_cmd, ROOT / "logs/prepare_render_dataset.log", env=os.environ.copy())

    train_env = os.environ.copy()
    train_env["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    train_cmd = [
        py,
        str(ROOT / "scripts/train_heatmap.py"),
        "--label-root",
        str(args.label_dir),
        "--frame-root",
        str(args.frame_dir),
        "--render-root",
        str(render_root),
        "--output-dir",
        str(args.checkpoint_dir),
        "--log-dir",
        str(args.log_dir),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--workers",
        str(args.workers),
        "--train-ratio",
        "0.8",
        "--device",
        "cuda",
        "--lr",
        "0.0008",
        "--weight-decay",
        "0.005",
        "--base-channels",
        "64",
        "--visible-only-loss",
        "--log-every",
        "25",
    ]
    log(log_path, "[stage1] train " + " ".join(train_cmd))
    run_logged(train_cmd, ROOT / "logs/train_heatmap.log", env=train_env)
    log(log_path, f"[stage1] done {now()}")
    return 0


def missing_s2_appearances(simulation_root: Path, render_root: Path) -> list[str]:
    sys.path.insert(0, str(simulation_root / "scripts"))
    import render

    all_motions = sorted((simulation_root / "test_motion/HumanEva").glob("*/*stageii.npz"))
    missing = []
    for motion in all_motions:
        if motion.parent.name != "S2" or motion.name == "Walking_3_stageii.npz":
            continue
        seed = 20260609 + all_motions.index(motion) * 101
        appearance_root = str(simulation_root / "smplx_models/icon_appearances")
        subjects = [path.name for path in render._select_appearance_subjects(appearance_root, 5, seed, "")]
        out_dir = render_root / f"{motion.parent.name}_{motion.stem}"
        for appearance_idx, subject in enumerate(subjects):
            appearance_dir = out_dir / f"appearance_{appearance_idx:02d}_{subject}"
            if len(list(appearance_dir.glob("*.mp4"))) < 8:
                missing.append(str(appearance_dir))
    return missing


def has_active_render() -> bool:
    result = subprocess.run(
        ["pgrep", "-f", "scripts/render.py isaacsim|src/geosim/isaacsim_runner.py|run_queue_8gpu_s2only.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    current = str(os.getpid())
    pids = [pid.strip() for pid in result.stdout.splitlines() if pid.strip() and pid.strip() != current]
    return bool(pids)


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as file:
        file.write(f"\n[{now()}] $ {' '.join(command)}\n")
        file.flush()
        subprocess.run(command, cwd=ROOT, env=env, stdout=file, stderr=subprocess.STDOUT, check=True)


def log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(message + "\n")
        file.flush()


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
