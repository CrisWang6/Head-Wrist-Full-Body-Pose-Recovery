#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
DATASET = "isaacsim_humaneva_2app_notags_fisheye220"
PYTHON = Path("/home/gaoweijian/miniforge3/envs/camtest/bin/python")
SIM_RENDER_ROOT = Path("/home/gaoweijian/Simulation/outputs") / DATASET
LABEL_ROOT = ROOT / "data" / "labels" / DATASET
FRAME_ROOT = ROOT / "data" / "frames" / DATASET
HEAD_CKPT = ROOT / "checkpoints" / f"{DATASET}_head_stage1"
WRIST_CKPT = ROOT / "checkpoints" / f"{DATASET}_wrist_stage1"
HEAD_LOG_DIR = ROOT / "logs" / f"{DATASET}_head_stage1"
WRIST_LOG_DIR = ROOT / "logs" / f"{DATASET}_wrist_stage1"
WRIST_TRAIN_LOG = ROOT / "logs" / "train_wrist_stage1_180.log"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stop head stage-1 at epoch 180, then train wrist to epoch 180.")
    parser.add_argument("--target-epoch", type=int, default=180)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-used-mb", type=int, default=30000)
    parser.add_argument("--min-gpus", type=int, default=2)
    return parser.parse_args()


def log(message: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S%z")
    print(f"[{timestamp}] {message}", flush=True)


def latest_epoch(ckpt_dir: Path) -> int:
    best = 0
    pattern = re.compile(r"epoch_(\d+)\.pt$")
    for path in ckpt_dir.glob("epoch_*.pt"):
        match = pattern.match(path.name)
        if match:
            best = max(best, int(match.group(1)))
    return best


def list_own_processes() -> list[tuple[int, int, int, str]]:
    output = subprocess.check_output(
        ["ps", "-u", str(os.getuid()), "-o", "pid=,ppid=,pgid=,args="],
        text=True,
    )
    rows: list[tuple[int, int, int, str]] = []
    for line in output.splitlines():
        parts = line.strip().split(None, 3)
        if len(parts) == 4:
            rows.append((int(parts[0]), int(parts[1]), int(parts[2]), parts[3]))
    return rows


def matching_train_pids(branch: str) -> list[tuple[int, int, int, str]]:
    needle = f"--train-branch {branch}"
    script = str(ROOT / "scripts" / "train_heatmap.py")
    return [
        row
        for row in list_own_processes()
        if script in row[3] and needle in row[3] and DATASET in row[3]
    ]


def stop_branch(branch: str) -> None:
    rows = matching_train_pids(branch)
    if not rows:
        log(f"no running {branch} train process found")
        return
    pgids = sorted({pgid for _, _, pgid, _ in rows})
    log(f"terminating {branch} train process groups: {pgids}")
    for pgid in pgids:
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.time() + 60
    while time.time() < deadline:
        if not matching_train_pids(branch):
            log(f"{branch} train stopped cleanly")
            return
        time.sleep(2)
    rows = matching_train_pids(branch)
    log(f"force killing remaining {branch} train pids: {[pid for pid, _, _, _ in rows]}")
    for pid, _, _, _ in rows:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def query_gpu_used_mb() -> dict[str, int]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    used: dict[str, int] = {}
    for line in output.splitlines():
        if not line.strip():
            continue
        idx, mem = [item.strip() for item in line.split(",", 1)]
        used[idx] = int(mem)
    return used


def choose_gpus(gpu_ids: str, max_used_mb: int, min_gpus: int) -> str:
    requested = [gpu.strip() for gpu in gpu_ids.split(",") if gpu.strip()]
    used = query_gpu_used_mb()
    available = [gpu for gpu in requested if used.get(gpu, 10**9) <= max_used_mb]
    selected = available if len(available) >= min_gpus else requested[: max(1, min_gpus)]
    log(f"gpu memory used MB: {used}; selected CUDA_VISIBLE_DEVICES={','.join(selected)}")
    return ",".join(selected)


def start_wrist(args: argparse.Namespace) -> int:
    WRIST_CKPT.mkdir(parents=True, exist_ok=True)
    WRIST_LOG_DIR.mkdir(parents=True, exist_ok=True)
    WRIST_TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = choose_gpus(args.gpu_ids, args.max_used_mb, args.min_gpus)
    command = [
        "ionice",
        "-c",
        "2",
        "-n",
        "7",
        "nice",
        "-n",
        "15",
        str(PYTHON),
        str(ROOT / "scripts" / "train_heatmap.py"),
        "--label-root",
        str(LABEL_ROOT),
        "--frame-root",
        str(FRAME_ROOT),
        "--render-root",
        str(SIM_RENDER_ROOT),
        "--output-dir",
        str(WRIST_CKPT),
        "--log-dir",
        str(WRIST_LOG_DIR),
        "--epochs",
        str(args.target_epoch),
        "--batch-size",
        "8",
        "--workers",
        "8",
        "--train-ratio",
        "0.8",
        "--device",
        "cuda",
        "--lr",
        "0.0005",
        "--weight-decay",
        "0.005",
        "--base-channels",
        "64",
        "--visible-only-loss",
        "--train-branch",
        "wrist",
        "--log-every",
        "50",
    ]
    log(f"starting wrist train: {' '.join(command)}")
    with WRIST_TRAIN_LOG.open("ab", buffering=0) as log_file:
        proc = subprocess.Popen(command, cwd=str(ROOT), env=env, stdout=log_file, stderr=subprocess.STDOUT)
        log(f"wrist train pid={proc.pid}; log={WRIST_TRAIN_LOG}")
        return proc.wait()


def main() -> int:
    args = parse_args()
    log(f"monitor started; waiting for head epoch {args.target_epoch}")
    while True:
        epoch = latest_epoch(HEAD_CKPT)
        log(f"latest head epoch={epoch}")
        if epoch >= args.target_epoch:
            break
        if not matching_train_pids("head"):
            log("head train is not running before target epoch; will keep waiting for checkpoints")
        time.sleep(args.poll_seconds)

    log(f"head target reached: epoch_{args.target_epoch:03d}.pt is available")
    stop_branch("head")
    rc = start_wrist(args)
    log(f"wrist train finished with rc={rc}; latest wrist epoch={latest_epoch(WRIST_CKPT)}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
