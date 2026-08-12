#!/usr/bin/env python3
"""24/7 monitor for 0806 weekend training on gwj — auto-fix, avoid killing main jobs."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import paramiko

HOST = "192.168.20.221"
USER = "gaoweijian"
PASS = "gwj@#@2026"
JP = "/home/gaoweijian/0806_batch/repo/test_code/joint_projection"
DATASET = "/home/gaoweijian/0806dataset"
PY = "/home/gaoweijian/miniforge3/envs/sapiens2/bin/python"
LOG_LOCAL = Path(__file__).resolve().parent / "logs" / "monitor_0806_weekend.log"
POLL_SEC = 120


def log(msg: str) -> None:
    line = f"[{datetime.now():strftime('%F %T')}] {msg}"
    print(line, flush=True)
    LOG_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    with LOG_LOCAL.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def ssh_client() -> paramiko.SSHClient:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=30)
    return c


def run(client: paramiko.SSHClient, cmd: str, timeout: int = 60) -> tuple[str, str, int]:
    _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    code = stdout.channel.recv_exit_status()
    return out, err, code


def check_label_schema(client: paramiko.SSHClient) -> list[str]:
    script = r"""
import numpy as np
from pathlib import Path
root = Path('/home/gaoweijian/0806dataset/labels')
issues = []
for limb in ('ankle','wrist','wu'):
    p = root / limb / 'heatmap_labels_120x75.npz'
    if not p.is_file():
        issues.append(f'missing:{limb}')
        continue
    d = np.load(p, allow_pickle=True)
    hs = tuple(int(x) for x in d['heatmap_size'])
    joints = [str(x) for x in d['joints']]
    if hs != (120, 75):
        issues.append(f'bad_heatmap:{limb}:{hs}')
    if len(joints) != 15 or joints[0] != 'nose':
        issues.append(f'bad_joints:{limb}:{len(joints)}')
print('\n'.join(issues))
"""
    out, _, _ = run(client, f"{PY} - <<'PY'\n{script}\nPY", timeout=60)
    return [x for x in out.splitlines() if x.strip()]


def reexport_limb(client: paramiko.SSHClient, limb: str) -> None:
    log(f"auto reexport heatmap for {limb}")
    run(client, f"bash {JP}/reexport_0806_heatmap_labels.sh {limb}", timeout=600)


def ensure_stage23_continuation(client: paramiko.SSHClient) -> None:
    out, _, _ = run(client, "pgrep -f run_weekend_stage23_0806.sh || true")
    if out.strip():
        return
    log("restart stage23 continuation (was not running)")
    run(
        client,
        "bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && "
        f"nohup bash {JP}/run_weekend_stage23_0806.sh "
        f">> {DATASET}/logs/weekend_stage23_nohup.log 2>&1 & echo PID=$!'",
        timeout=30,
    )


def tail_errors(client: paramiko.SSHClient) -> None:
    cmd = (
        f"tail -30 {DATASET}/logs/weekend_master.log 2>/dev/null | "
        r"grep -E 'Error|Traceback|FAILED|RuntimeError' | tail -5 || true"
    )
    out, _, _ = run(client, cmd)
    if out.strip():
        log(f"master errors tail: {out.strip()[:500]}")


def poll_once() -> None:
    client = ssh_client()
    try:
        # Keep toe-test pipeline from clobbering wu triangulation used for training.
        run(client, "pkill -u gaoweijian -f run_wu_full_10s_production_gwj.sh 2>/dev/null || true")

        master, _, _ = run(client, f"cat {DATASET}/WEEKEND_MASTER_STATUS.txt 2>/dev/null || echo unknown")
        stage23, _, _ = run(client, f"cat {DATASET}/WEEKEND_STAGE23_STATUS.txt 2>/dev/null || echo unknown")
        log(f"status master={master.strip()} stage23={stage23.strip()}")

        issues = check_label_schema(client)
        master_status = master.strip()
        if master_status in ("task2_done", "starting") or any(i.startswith("missing:") for i in issues):
            for issue in issues:
                if issue.startswith("missing:") or issue.startswith("bad_"):
                    limb = issue.split(":")[1]
                    if limb in ("wu", "wrist", "ankle"):
                        csv = (
                            f"/home/gaoweijian/0806_batch/{limb}/data_root/"
                            "multiview_3d_results/full/head_reprojection/"
                            "head_reprojection_2d_wo_calibration_proj.csv"
                        )
                        exists, _, _ = run(client, f"test -f {csv} && echo yes || echo no")
                        if exists.strip() == "yes":
                            reexport_limb(client, limb)

        ensure_stage23_continuation(client)
        tail_errors(client)

        if master_status == "done":
            log("MASTER DONE — monitor will keep polling for eval artifacts")
    finally:
        client.close()


def main() -> int:
    log("monitor start")
    while True:
        try:
            poll_once()
        except Exception as exc:
            log(f"poll exception (will retry): {exc}")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("monitor stopped by user")
        raise SystemExit(0)
