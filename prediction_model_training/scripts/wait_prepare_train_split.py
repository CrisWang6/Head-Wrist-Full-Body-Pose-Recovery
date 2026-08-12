#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for IsaacSim renders/frames, then train separate head and wrist heatmap models.")
    parser.add_argument("--render-root", default="/home/gaoweijian/Simulation/outputs/isaacsim_humaneva_2app_notags_fisheye220")
    parser.add_argument("--label-dir", default=str(ROOT / "data/labels/isaacsim_humaneva_2app_notags_fisheye220"))
    parser.add_argument("--frame-dir", default=str(ROOT / "data/frames/isaacsim_humaneva_2app_notags_fisheye220"))
    parser.add_argument("--expected-appearances", type=int, default=38)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-used-mb", type=int, default=30000, help="Exclude GPUs already above this memory use before each training job.")
    parser.add_argument("--min-gpus", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--base-channels", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    log_path = ROOT / "logs/wait_prepare_train_split.log"
    log(log_path, {"event": "start", "args": vars(args)})
    render_root = Path(args.render_root).expanduser().resolve()
    label_dir = Path(args.label_dir).expanduser().resolve()
    frame_dir = Path(args.frame_dir).expanduser().resolve()

    while True:
        complete = count_complete_appearances(render_root)
        prepared = count_prepared_appearances(label_dir, frame_dir, render_root)
        active_render = has_active_process("run_queue.py|scripts/render.py isaacsim|isaacsim_runner.py", "/home/gaoweijian/Simulation")
        active_prepare = has_active_process("watch_prepare_frames.py|prepare_render_dataset.py|ffmpeg", "/home/gaoweijian/EgoRear_w_hand")
        queue_done = render_queue_done(render_root)
        ready = queue_done and not active_render and not active_prepare and complete >= args.expected_appearances and prepared >= complete
        log(
            log_path,
            {
                "event": "poll",
                "complete_appearances": complete,
                "prepared_appearances": prepared,
                "active_render": active_render,
                "active_prepare": active_prepare,
                "queue_done": queue_done,
                "ready": ready,
            },
        )
        if ready:
            break
        time.sleep(max(30, int(args.poll_seconds)))

    rc = 0
    for branch in ("head", "wrist"):
        selected_gpus = wait_for_gpus(args.gpu_ids, args.max_used_mb, args.min_gpus, log_path, args.poll_seconds)
        branch_rc = run_training(branch, selected_gpus, args, label_dir, frame_dir, render_root, log_path)
        rc = rc or branch_rc
    log(log_path, {"event": "done", "rc": rc})
    return rc


def count_complete_appearances(render_root: Path) -> int:
    return sum(1 for appearance_dir in render_root.glob("S*/appearance_*") if len(list(appearance_dir.glob("*_isaacsim_stats.json"))) >= 8)


def count_prepared_appearances(label_dir: Path, frame_dir: Path, render_root: Path) -> int:
    count = 0
    for label_path in label_dir.rglob("heatmap_labels_*.npz"):
        if label_has_frames(label_path, frame_dir, render_root):
            count += 1
    return count


def label_has_frames(label_path: Path, frame_dir: Path, render_root: Path) -> bool:
    import numpy as np

    try:
        data = np.load(label_path, allow_pickle=True)
        source_render_dir = Path(str(data["source_render_dir"][0])).expanduser().resolve()
        rel = source_render_dir.relative_to(render_root)
        expected_frames = int(data["keypoints"].shape[0])
        camera_names = [str(name) for name in data["camera_names"]]
    except Exception:
        return False
    for camera_name in camera_names:
        if len(list((frame_dir / rel / camera_name).glob("*.jpg"))) < expected_frames:
            return False
    return True


def render_queue_done(render_root: Path) -> bool:
    path = render_root / "logs/render_queue_master.log"
    return path.exists() and "queue_done" in path.read_text(encoding="utf-8", errors="ignore")


def has_active_process(pattern: str, required_substring: str) -> bool:
    try:
        output = subprocess.check_output(["pgrep", "-af", pattern], text=True)
    except subprocess.CalledProcessError:
        return False
    return any(required_substring in line for line in output.splitlines())


def wait_for_gpus(gpu_ids: str, max_used_mb: int, min_gpus: int, log_path: Path, poll_seconds: int) -> str:
    requested = [part.strip() for part in gpu_ids.split(",") if part.strip()]
    while True:
        usage = query_gpu_memory()
        selected = [gpu for gpu in requested if usage.get(gpu, 10**9) <= int(max_used_mb)]
        if len(selected) >= int(min_gpus):
            chosen = ",".join(selected)
            log(log_path, {"event": "gpu_selected", "selected": chosen, "usage_mb": usage})
            return chosen
        log(log_path, {"event": "gpu_wait", "selected": ",".join(selected), "usage_mb": usage})
        time.sleep(max(30, int(poll_seconds)))


def query_gpu_memory() -> dict[str, int]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
    except Exception:
        return {}
    usage = {}
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 2:
            usage[parts[0]] = int(float(parts[1]))
    return usage


def run_training(branch: str, gpu_ids: str, args: argparse.Namespace, label_dir: Path, frame_dir: Path, render_root: Path, log_path: Path) -> int:
    output_dir = ROOT / f"checkpoints/isaacsim_humaneva_2app_notags_fisheye220_{branch}_stage1"
    tensorboard_dir = ROOT / f"logs/isaacsim_humaneva_2app_notags_fisheye220_{branch}_stage1"
    train_log = ROOT / f"logs/train_{branch}_stage1_long.log"
    command = [
        sys.executable,
        str(ROOT / "scripts/train_heatmap.py"),
        "--label-root",
        str(label_dir),
        "--frame-root",
        str(frame_dir),
        "--render-root",
        str(render_root),
        "--output-dir",
        str(output_dir),
        "--log-dir",
        str(tensorboard_dir),
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
        "0.0005",
        "--weight-decay",
        "0.005",
        "--base-channels",
        str(args.base_channels),
        "--visible-only-loss",
        "--train-branch",
        branch,
        "--log-every",
        "50",
    ]
    if shutil.which("nice"):
        command = ["nice", "-n", "15"] + command
    if shutil.which("ionice"):
        command = ["ionice", "-c", "2", "-n", "7"] + command
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_ids
    env["OMP_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    log(log_path, {"event": "train_start", "branch": branch, "gpu_ids": gpu_ids, "command": command})
    train_log.parent.mkdir(parents=True, exist_ok=True)
    with train_log.open("w", encoding="utf-8") as file:
        proc = subprocess.run(command, cwd=str(ROOT), env=env, stdout=file, stderr=subprocess.STDOUT)
    log(log_path, {"event": "train_done", "branch": branch, "rc": int(proc.returncode), "log": str(train_log)})
    return int(proc.returncode)


def log(path: Path, payload: dict[str, object]) -> None:
    payload = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **payload}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
