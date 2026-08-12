"""Upload v32 stage1 script, ensure split exists, launch training on gwj."""
from __future__ import annotations

import pathlib
import time

import paramiko

HOST = "192.168.20.221"
USER = "gaoweijian"
PASS = "gwj@#@2026"
JP = "/home/gaoweijian/0806_batch/repo/test_code/joint_projection"
DS = "/home/gaoweijian/0806dataset"
PY = "/home/gaoweijian/miniforge3/envs/sapiens2/bin/python"
LOCAL = pathlib.Path(__file__).resolve().parent


def run(c: paramiko.SSHClient, cmd: str, timeout: int = 120, check: bool = True) -> str:
    _, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    text = (out + err).strip()
    if check and code != 0:
        raise RuntimeError(f"cmd failed ({code}): {cmd}\n{text}")
    return text


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=30)
    sftp = c.open_sftp()

    local_sh = LOCAL / "run_stage1_v32_only_gwj.sh"
    remote_sh = f"{JP}/run_stage1_v32_only_gwj.sh"
    with sftp.file(remote_sh, "wb") as f:
        f.write(local_sh.read_bytes().replace(b"\r\n", b"\n"))
    run(c, f"chmod +x {remote_sh}")

    print("=== remote state before launch ===")
    probes = [
        f"ls -la {DS}/checkpoints/stage1_v31/best.pt 2>/dev/null || echo 'no v31 best'",
        f"ls -la {DS}/checkpoints/stage1_v32/best.pt 2>/dev/null || echo 'no v32 best'",
        f"ls -la {DS}/splits/pack30_v32.npz 2>/dev/null || echo 'no v32 split'",
        "pgrep -af 'train_heatmap|run_stage1' | grep -v pgrep || echo 'no training running'",
        f"tail -3 {DS}/logs/stage1_v31_only.log 2>/dev/null || true",
    ]
    for p in probes:
        print(f"$ {p}\n{run(c, p, timeout=30)}\n")

    split_path = f"{DS}/splits/pack30_v32.npz"
    check = run(c, f"test -f {split_path} && echo exists || echo missing")
    if "missing" in check:
        print("=== build v32 split ===")
        print(run(c, f"{PY} {JP}/build_0806_pack_splits.py --scheme v32", timeout=180))

    print("=== stop any stale stage1 v32 ===")
    run(
        c,
        "pkill -u gaoweijian -f 'run_stage1_v32_only_gwj.sh' 2>/dev/null || true; "
        "pkill -u gaoweijian -f 'train_heatmap.py.*stage1_v32' 2>/dev/null || true",
        timeout=30,
        check=False,
    )
    time.sleep(2)

    print("=== launch stage1 v32 ===")
    run(
        c,
        f"nohup bash {remote_sh} >> {DS}/logs/stage1_v32_only_nohup.log 2>&1 & echo LAUNCHED",
        timeout=30,
    )
    time.sleep(8)

    print("=== verify ===")
    verify = [
        "pgrep -af 'train_heatmap|run_stage1_v32' | grep -v pgrep || echo 'NOT RUNNING'",
        f"tail -20 {DS}/logs/stage1_v32_only.log 2>/dev/null || echo 'no log yet'",
        f"cat {DS}/WEEKEND_MASTER_STATUS.txt 2>/dev/null || true",
        "nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | head -5 || true",
    ]
    for v in verify:
        print(f"$ {v}\n{run(c, v, timeout=30)}\n")

    sftp.close()
    c.close()


if __name__ == "__main__":
    main()
