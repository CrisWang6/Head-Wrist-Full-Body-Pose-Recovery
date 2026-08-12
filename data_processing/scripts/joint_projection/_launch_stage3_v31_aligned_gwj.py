"""Launch Stage3 v31 retrain with global-index 3D supervision (alignment fix)."""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import paramiko

HOST = "192.168.20.221"
USER = "gaoweijian"
PASS = "gwj@#@2026"
EGO = "/home/gaoweijian/EgoRear_w_hand"
JP = "/home/gaoweijian/0806_batch/repo/test_code/joint_projection"
DS = "/home/gaoweijian/0806dataset"
CAMTEST = "/home/gaoweijian/miniforge3/envs/camtest/bin/python"
TRAIN_POSE3D = f"{EGO}/experiments/stage3_pose3d/scripts/train_pose3d.py"
POSE3D_LABELS = f"{DS}/labels/pose3d_nose_pre_limb_15j.npz"
STAGE1 = f"{DS}/checkpoints/stage1_v31/best.pt"
STAGE2 = f"{DS}/checkpoints/stage2_v31/best.pt"
OUT = f"{DS}/checkpoints/stage3_v31_aligned"
LOG = f"{DS}/logs/stage3_v31_aligned.log"
TB_LOG = f"{DS}/logs/tensorboard_stage3_v31_aligned.log"
SPLIT = f"{DS}/splits/pack30_v31.npz"
LOCAL_ROOT = Path(__file__).resolve().parent

REMOTE_SH = textwrap.dedent(
    f"""\
    set -euo pipefail
    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate camtest
    export PYTHONPATH={JP}:{EGO}/src

    mkdir -p {DS}/logs {OUT}

    if ! grep -q 'global_idx' {TRAIN_POSE3D}; then
      {CAMTEST} {JP}/_patch_train_pose3d_alignment.py
    else
      echo train_pose3d alignment patch already present
    fi

    pkill -u gaoweijian -f 'train_pose3d.py.*stage3_v31[^_]' 2>/dev/null || true
    sleep 2

    if pgrep -af 'train_pose3d.py.*stage3_v31_aligned' >/dev/null; then
      echo stage3_v31_aligned_already_running
    else
      cd {EGO}
      nohup {CAMTEST} experiments/stage3_pose3d/scripts/train_pose3d.py \\
        --label-root {DS}/labels \\
        --pose3d-labels {POSE3D_LABELS} \\
        --stage1-checkpoint {STAGE1} \\
        --stage2-checkpoint {STAGE2} \\
        --output-dir {OUT} \\
        --split-manifest {SPLIT} \\
        --epochs 9999 --batch-size 64 --workers 8 --lr 0.001 --weight-decay 0.0005 \\
        --min-epochs 1 --early-stop-patience 20 \\
        --device cuda --seed 42 \\
        >> {LOG} 2>&1 &
      echo launched_pid=$!
      sleep 15
    fi

    pgrep -af 'train_pose3d.py.*stage3_v31_aligned' | grep -v bash || (echo FAILED; tail -60 {LOG}; exit 1)

    if ! pgrep -af 'tensorboard.*stage3_v31_aligned' >/dev/null; then
      nohup {CAMTEST} -m tensorboard.main \\
        --logdir {OUT}/tensorboard \\
        --host 127.0.0.1 --port 6033 --reload_interval 15 \\
        > {TB_LOG} 2>&1 &
      sleep 2
    fi
    pgrep -af 'tensorboard.*stage3_v31_aligned' | head -2 || true

    echo stage3_v31_aligned_running > {DS}/WEEKEND_STAGE23_STATUS.txt
    echo '--- gpu ---'
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    echo '--- log tail ---'
    tail -25 {LOG}
    """
)


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=30)
    sftp = c.open_sftp()
    sftp.put(
        str(LOCAL_ROOT / "_patch_train_pose3d_alignment.py"),
        f"{JP}/_patch_train_pose3d_alignment.py",
    )
    remote_sh = "/tmp/launch_stage3_v31_aligned.sh"
    with sftp.file(remote_sh, "w") as f:
        f.write(REMOTE_SH)
    sftp.close()
    _, o, e = c.exec_command(f"bash {remote_sh}", timeout=600)
    out = o.read().decode(errors="replace")
    err = e.read().decode(errors="replace")
    code = o.channel.recv_exit_status()
    print(out)
    if err.strip():
        print("stderr:", err[-4000:])
    c.close()
    if code != 0:
        raise SystemExit(code)
    print(
        json.dumps(
            {
                "stage3_out": OUT,
                "train_log": LOG,
                "alignment": "global_idx supervision (fixed)",
                "tensorboard_tunnel": "http://127.0.0.1:6033 (run _tb_tunnel_stage3_v31_aligned.py)",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
