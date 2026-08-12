#!/usr/bin/env python3
"""Poll gwj until Stage1 v31 finishes; print eval path when ready."""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import paramiko

HOST = "192.168.20.221"
USER = "gaoweijian"
PASS = "gwj@#@2026"
DATASET = "/home/gaoweijian/0806dataset"
JP = "/home/gaoweijian/0806_batch/repo/test_code/joint_projection"
LOG_LOCAL = Path(__file__).resolve().parent / "logs" / "watch_stage1_v31.log"
POLL_SEC = 120
EVAL_REMOTE = f"{DATASET}/eval/v31/stage1_test_v31.json"


def log(msg: str) -> None:
    line = f"[{datetime.now():%F %T}] {msg}"
    print(line, flush=True)
    LOG_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    with LOG_LOCAL.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def ssh():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=30)
    return c


def run(c, cmd: str) -> str:
    _, o, _ = c.exec_command(cmd, timeout=60)
    return o.read().decode(errors="replace")


def main() -> None:
    log("watch start")
    while True:
        c = ssh()
        status = run(c, f"cat {DATASET}/WEEKEND_MASTER_STATUS.txt 2>/dev/null").strip()
        train = run(c, "pgrep -af 'train_heatmap.py' | grep stage1_v31 | grep -v pgrep || true").strip()
        eval_exists = run(c, f"test -f {EVAL_REMOTE} && echo yes || echo no").strip()
        best = run(c, f"test -f {DATASET}/checkpoints/stage1_v31/best.pt && echo yes || echo no").strip()
        c.close()

        log(f"status={status} train={'yes' if train else 'no'} best={best} eval={eval_exists}")

        if eval_exists == "yes":
            c = ssh()
            body = run(c, f"cat {EVAL_REMOTE}")
            c.close()
            log("STAGE1_V31_COMPLETE")
            print("=== TEST RESULT ===")
            print(body)
            out_local = LOG_LOCAL.parent / "stage1_test_v31.json"
            out_local.write_text(body, encoding="utf-8")
            return

        if status in ("stage1_v31_eval_done", "stage1_v31_done") and best == "yes" and not train:
            log("training stopped but eval missing - waiting for eval script")
        if status not in ("stage1_v31_running", "stage1_v31_done", "stage1_v31_eval_done", "resume_post_task2") and best == "yes" and eval_exists == "no":
            log("unexpected status with best.pt present")

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
