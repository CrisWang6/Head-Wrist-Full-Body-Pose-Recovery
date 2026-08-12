"""Prepare Stage3 labels, patch train_pose3d for 0806, launch v31 Stage3 + TensorBoard."""
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
BATCH = "/home/gaoweijian/0806_batch"
PY = "/home/gaoweijian/miniforge3/envs/sapiens2/bin/python"
CAMTEST = "/home/gaoweijian/miniforge3/envs/camtest/bin/python"
TRAIN_POSE3D = f"{EGO}/experiments/stage3_pose3d/scripts/train_pose3d.py"
POSE3D_LABELS = f"{DS}/labels/pose3d_nose_pre_limb_15j.npz"
PRE_LIMB_MAP = f"{DS}/pre_limb_map.json"
STAGE1 = f"{DS}/checkpoints/stage1_v31/best.pt"
STAGE2 = f"{DS}/checkpoints/stage2_v31/best.pt"
OUT = f"{DS}/checkpoints/stage3_v31"
LOG = f"{DS}/logs/stage3_v31_only.log"
TB_LOG = f"{DS}/logs/tensorboard_stage3_v31.log"
SPLIT = f"{DS}/splits/pack30_v31.npz"
LOCAL_ROOT = Path(__file__).resolve().parent

UPLOAD_FILES = [
    "constants_0806_training.py",
    "delivery_keypoints.py",
    "prepare_0806_pose3d_labels.py",
]


def run(c, cmd: str, timeout: int = 600) -> tuple[str, str, int]:
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode(errors="replace")
    err = e.read().decode(errors="replace")
    code = o.channel.recv_exit_status()
    return out, err, code


def upload_files(c: paramiko.SSHClient) -> None:
    sftp = c.open_sftp()
    for rel in UPLOAD_FILES:
        local = LOCAL_ROOT / rel
        remote = f"{JP}/{rel}"
        print(f"upload {rel}")
        sftp.put(str(local), remote)
    sftp.close()


REMOTE_SH = textwrap.dedent(
    f"""\
    set -euo pipefail
    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate camtest
    export PYTHONPATH={JP}:{EGO}/src

    mkdir -p {DS}/logs {OUT}

    if [[ ! -f {PRE_LIMB_MAP} ]]; then
      {PY} - <<'PY'
import json
from pathlib import Path
out = Path("{PRE_LIMB_MAP}")
batch = Path("{BATCH}")
mapping = {{
    "ankle": str(batch / "ankle/data_root/multiview_3d_results/full/multiview_3d_results_pre_limb.jsonl"),
    "wrist": str(batch / "wrist/data_root/multiview_3d_results/full/multiview_3d_results_pre_limb.jsonl"),
    "wu": str(batch / "wu/data_root/multiview_3d_results/full/multiview_3d_results_pre_limb.jsonl"),
}}
out.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
print("wrote", out)
PY
    fi

    echo "=== prepare pose3d labels ==="
    {PY} {JP}/prepare_0806_pose3d_labels.py \\
      --label-root {DS}/labels \\
      --pre-limb-map {PRE_LIMB_MAP} \\
      --output {POSE3D_LABELS}

    python3 - <<'PY'
from pathlib import Path
p = Path("{TRAIN_POSE3D}")
text = p.read_text(encoding="utf-8")
changed = False
if "image_size=(456, 256)" in text:
    text = text.replace("image_size=(456, 256)", "image_size=(480, 300)", 1)
    changed = True
old_coord = '"coordinate_frame": "mocap_head_full_rigid_transform",'
new_coord = '"coordinate_frame": "0806_nose_translation_offset_m",'
if old_coord in text:
    text = text.replace(old_coord, new_coord, 1)
    changed = True
if changed:
    p.write_text(text, encoding="utf-8")
    print("patched", p)
else:
    print("train_pose3d already patched", p)
PY

    echo stage3_v31_running > {DS}/WEEKEND_STAGE23_STATUS.txt

    if pgrep -af 'train_pose3d.py.*stage3_v31' >/dev/null; then
      echo stage3_already_running
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
      sleep 12
    fi

    pgrep -af 'train_pose3d.py.*stage3_v31' | grep -v bash || (echo FAILED; tail -50 {LOG}; exit 1)

    if ! pgrep -af 'tensorboard.*stage3_v31' >/dev/null; then
      nohup {CAMTEST} -m tensorboard.main \\
        --logdir {OUT}/tensorboard \\
        --host 127.0.0.1 --port 6032 --reload_interval 15 \\
        > {TB_LOG} 2>&1 &
      sleep 2
    fi
    pgrep -af 'tensorboard.*stage3_v31' | head -2 || true

    echo '--- gpu ---'
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    echo '--- log tail ---'
    tail -20 {LOG}
    """
)


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=30)
    upload_files(c)
    sftp = c.open_sftp()
    remote_sh = "/tmp/launch_stage3_v31.sh"
    with sftp.file(remote_sh, "w") as f:
        f.write(REMOTE_SH)
    sftp.close()
    out, err, code = run(c, f"bash {remote_sh}", timeout=600)
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
                "pose3d_labels": POSE3D_LABELS,
                "train_log": LOG,
                "tensorboard_tunnel": "http://127.0.0.1:6032 (run _tb_tunnel_stage3_v31.py)",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
