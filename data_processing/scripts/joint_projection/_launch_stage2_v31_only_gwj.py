"""Patch stage2 for 0806 CAM_A/D, relaunch v31 stage2, start TensorBoard."""
from __future__ import annotations

import json
import textwrap

import paramiko

HOST = "192.168.20.221"
USER = "gaoweijian"
PASS = "gwj@#@2026"
EGO = "/home/gaoweijian/EgoRear_w_hand"
JP = "/home/gaoweijian/0806_batch/repo/test_code/joint_projection"
DS = "/home/gaoweijian/0806dataset"
CAMTEST = "/home/gaoweijian/miniforge3/envs/camtest/bin/python"
TRAIN = f"{EGO}/experiments/stage2_refinement/scripts/train_refinement.py"
STAGE1 = f"{DS}/checkpoints/stage1_v31/best.pt"
OUT = f"{DS}/checkpoints/stage2_v31"
LOGD = f"{DS}/logs/stage2_v31"
LOG = f"{DS}/logs/stage2_v31_only.log"
SPLIT = f"{DS}/splits/pack30_v31.npz"
TB_LOG = f"{DS}/logs/tensorboard_stage2_v31.log"


def run(c, cmd: str, timeout: int = 120) -> tuple[str, str, int]:
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode(errors="replace")
    err = e.read().decode(errors="replace")
    code = o.channel.recv_exit_status()
    return out, err, code


REMOTE_SH = textwrap.dedent(
    f"""\
    set -euo pipefail
    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate camtest

    # Stop v32 stage1 and any stale stage2 attempts.
    pkill -u gaoweijian -f 'train_heatmap.py.*stage1_v32' 2>/dev/null || true
    pkill -u gaoweijian -f run_stage1_v32_only_gwj.sh 2>/dev/null || true
    pkill -u gaoweijian -f 'train_refinement.py.*stage2_v32' 2>/dev/null || true
    pkill -u gaoweijian -f 'tensorboard.*stage1_v32' 2>/dev/null || true
    sleep 2

    # Idempotent 0806 adapter: accept any 2-view head camera order (CAM_A/D or CAM_B/C).
    python3 - <<'PY'
from pathlib import Path
p = Path("{TRAIN}")
text = p.read_text(encoding="utf-8")
marker = 'camera_order = [str(name) for name in first["camera_names"]]'
if marker in text:
    print("already patched", p)
else:
    old = '''    first = dataset[0]
    if list(first["camera_names"]) != ["module01_CAM_B", "module01_CAM_C"]:
        raise ValueError(f"Expected head CAM_B/C labels, got {{first['camera_names']}}")
    num_joints = int(first["head_gt_heatmap"].shape[1])'''
    new = '''    first = dataset[0]
    camera_order = [str(name) for name in first["camera_names"]]
    if len(camera_order) != 2:
        raise ValueError(f"Expected 2 camera views, got {{camera_order}}")
    num_joints = int(first["head_gt_heatmap"].shape[1])'''
    if old not in text:
        raise SystemExit("camera check block not found")
    text = text.replace(old, new, 1)
    text = text.replace(
        '"camera_order": ["module01_CAM_B", "module01_CAM_C"],',
        '"camera_order": camera_order,',
        1,
    )
    p.write_text(text, encoding="utf-8")
    print("patched", p)
PY

    mkdir -p {OUT} {LOGD} {DS}/logs
    echo stage2_v31_running > {DS}/WEEKEND_STAGE23_STATUS.txt

    export PYTHONPATH={JP}:{EGO}/src
    cd {EGO}
    if pgrep -af 'train_refinement.py.*stage2_v31' >/dev/null; then
      echo stage2_already_running
    else
      nohup {CAMTEST} experiments/stage2_refinement/scripts/train_refinement.py \\
        --label-root {DS}/labels \\
        --stage1-checkpoint {STAGE1} \\
        --output-dir {OUT} --log-dir {LOGD} --split-manifest {SPLIT} \\
        --epochs 9999 --batch-size 32 --workers 12 --lr 0.001 --weight-decay 0.005 \\
        --selection-metric refined_pixel_error --min-epochs 1 --early-stop-patience 20 \\
        --heatmap-width 120 --heatmap-height 75 \\
        --image-width 480 --image-height 300 \\
        --base-channels 64 --device cuda --seed 42 --max-hours 72 \\
        >> {LOG} 2>&1 &
      echo launched_pid=$!
      sleep 8
    fi

    pgrep -af 'train_refinement.py.*stage2_v31' || (echo FAILED; tail -40 {LOG}; exit 1)

    if ! pgrep -af 'tensorboard.*stage2_v31' >/dev/null; then
      nohup {CAMTEST} -m tensorboard.main \\
        --logdir {LOGD} --host 0.0.0.0 --port 6031 --reload_interval 15 \\
        > {TB_LOG} 2>&1 &
      sleep 2
    fi
    pgrep -af 'tensorboard.*stage2_v31' || (echo TB_FAILED; tail -20 {TB_LOG}; exit 1)

    echo '--- gpu ---'
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    echo '--- log tail ---'
    tail -15 {LOG}
    """
)


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=30)
    sftp = c.open_sftp()
    remote_sh = "/tmp/launch_stage2_v31_only.sh"
    with sftp.file(remote_sh, "w") as f:
        f.write(REMOTE_SH)
    sftp.close()
    out, err, code = run(c, f"bash {remote_sh}", timeout=180)
    print(out)
    if err.strip():
        print("stderr:", err[-3000:])
    c.close()
    if code != 0:
        raise SystemExit(code)
    print(
        json.dumps(
            {
                "tensorboard_local": "http://127.0.0.1:6031",
                "tensorboard_note": "LAN http://192.168.20.221:6031 is blocked by firewall; run _tb_tunnel_stage2_v31.py first",
                "log_dir": LOGD,
                "train_log": LOG,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
