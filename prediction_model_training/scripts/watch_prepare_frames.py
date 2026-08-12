#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch IsaacSim renders and prepare frame/label data as appearances finish.")
    parser.add_argument("--simulation-root", default="/home/gaoweijian/Simulation")
    parser.add_argument("--render-root", default="/home/gaoweijian/Simulation/outputs/isaacsim_humaneva_2app_notags_fisheye220")
    parser.add_argument("--label-dir", default=str(ROOT / "data/labels/isaacsim_humaneva_2app_notags_fisheye220"))
    parser.add_argument("--frame-dir", default=str(ROOT / "data/frames/isaacsim_humaneva_2app_notags_fisheye220"))
    parser.add_argument("--subjects", default="S1,S2,S3")
    parser.add_argument("--exclude-actions", default="Box,Throw")
    parser.add_argument("--appearances-per-motion", type=int, default=2)
    parser.add_argument("--extract-workers", type=int, default=8)
    parser.add_argument("--poll-seconds", type=int, default=180)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = ROOT / "logs/watch_prepare_frames.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    render_root = Path(args.render_root).expanduser().resolve()
    label_dir = Path(args.label_dir).expanduser().resolve()
    frame_dir = Path(args.frame_dir).expanduser().resolve()
    label_dir.mkdir(parents=True, exist_ok=True)
    frame_dir.mkdir(parents=True, exist_ok=True)

    while True:
        complete_appearances = count_complete_appearances(render_root)
        prepared_appearances = count_prepared_labels(label_dir)
        log(log_path, {"event": "poll", "complete_appearances": complete_appearances, "prepared_labels": prepared_appearances})
        if complete_appearances > 0:
            command = [
                sys.executable,
                str(ROOT / "scripts/prepare_render_dataset.py"),
                "--simulation-root",
                str(Path(args.simulation_root).expanduser().resolve()),
                "--render-root",
                str(render_root),
                "--label-dir",
                str(label_dir),
                "--frame-dir",
                str(frame_dir),
                "--subjects",
                args.subjects,
                "--exclude-actions",
                args.exclude_actions,
                "--appearances-per-motion",
                str(args.appearances_per_motion),
                "--extract-backend",
                "ffmpeg",
                "--extract-workers",
                str(args.extract_workers),
                "--ffmpeg-hwaccel",
                "none",
                "--skip-existing-frames",
                "--require-stats",
            ]
            rc = run_logged(command, ROOT / "logs/prepare_render_dataset_watch.log")
            log(log_path, {"event": "prepare_return", "rc": rc})
        if args.once:
            return 0
        active_render = has_active_own_render()
        queue_done = render_queue_done(render_root)
        if queue_done and not active_render and count_prepared_labels(label_dir) >= count_complete_appearances(render_root):
            log(log_path, {"event": "done"})
            return 0
        time.sleep(max(10, int(args.poll_seconds)))


def count_complete_appearances(render_root: Path) -> int:
    count = 0
    for appearance_dir in sorted(render_root.glob("S*/appearance_*")):
        if len(list(appearance_dir.glob("*_isaacsim_stats.json"))) >= 8:
            count += 1
    return count


def count_prepared_labels(label_dir: Path) -> int:
    return sum(1 for _ in label_dir.rglob("heatmap_labels_*.npz"))


def render_queue_done(render_root: Path) -> bool:
    log_path = render_root / "logs/render_queue_master.log"
    return log_path.exists() and "queue_done" in log_path.read_text(encoding="utf-8", errors="ignore")


def has_active_own_render() -> bool:
    try:
        output = subprocess.check_output(["pgrep", "-af", "run_queue.py|scripts/render.py isaacsim|isaacsim_runner.py"], text=True)
    except subprocess.CalledProcessError:
        return False
    return any("/home/gaoweijian/Simulation" in line for line in output.splitlines())


def run_logged(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n[prepare] " + " ".join(command) + "\n")
        log_file.flush()
        proc = subprocess.run(command, cwd=str(ROOT), env=env, stdout=log_file, stderr=subprocess.STDOUT)
        return int(proc.returncode)


def log(path: Path, payload: dict[str, object]) -> None:
    payload = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **payload}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
