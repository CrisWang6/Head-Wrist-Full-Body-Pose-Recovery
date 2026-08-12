"""Run Stage3 aligned test-set eval + pred/GT dual skeleton viz on gwj."""
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
OUT_REMOTE = f"{DS}/eval/v31/stage3_aligned_test_3d_viz"
LOCAL_OUT = Path(__file__).resolve().parent / "output" / "stage3_v31_aligned_test_3d_viz"

UPLOAD = [
    "constants_0806_training.py",
    "delivery_keypoints.py",
    "eval_stage3_test_3d_viz.py",
    "render_skeleton_yaw_video.py",
    "render_stage3_dual_skeleton_yaw.py",
    "skeleton_3d_filter.py",
]

REMOTE_SH = textwrap.dedent(
    f"""\
    set -euo pipefail
    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate camtest
    export PYTHONPATH={JP}:{EGO}/src

    mkdir -p {OUT_REMOTE}
    cd {EGO}

    echo "=== Stage3 aligned test inference (ankle split) ==="
    {CAMTEST} {JP}/eval_stage3_test_3d_viz.py \\
      --label-root {DS}/labels \\
      --pose3d-labels {DS}/labels/pose3d_nose_pre_limb_15j.npz \\
      --stage1-checkpoint {DS}/checkpoints/stage1_v31/best.pt \\
      --stage2-checkpoint {DS}/checkpoints/stage2_v31/best.pt \\
      --stage3-checkpoint {DS}/checkpoints/stage3_v31_aligned/best.pt \\
      --split-manifest {DS}/splits/pack30_v31.npz \\
      --split-name test \\
      --output-dir {OUT_REMOTE} \\
      --batch-size 64 --workers 4 --device cuda

    echo "=== pred vs GT dual yaw (filtered pred) ==="
    {CAMTEST} {JP}/render_stage3_dual_skeleton_yaw.py \\
      --pred-playback {OUT_REMOTE}/skeleton_playback_stage3_test_pred.json \\
      --gt-playback {OUT_REMOTE}/skeleton_playback_stage3_test_gt.json \\
      --output {OUT_REMOTE}/skeleton_yaw_stage3_test_pred_vs_gt.mp4 \\
      --report {OUT_REMOTE}/skeleton_yaw_stage3_test_pred_vs_gt.json

    ls -lh {OUT_REMOTE}
    """
)


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=30)
    sftp = c.open_sftp()
    for rel in UPLOAD:
        sftp.put(str(Path(__file__).resolve().parent / rel), f"{JP}/{rel}")
    with sftp.file("/tmp/run_stage3_aligned_test_viz.sh", "w") as f:
        f.write(REMOTE_SH)
    sftp.close()

    _, o, e = c.exec_command("bash /tmp/run_stage3_aligned_test_viz.sh", timeout=7200)
    out = o.read().decode(errors="replace")
    err = e.read().decode(errors="replace")
    code = o.channel.recv_exit_status()
    print(out)
    if err.strip():
        print("stderr:", err[-8000:])
    if code != 0:
        c.close()
        raise SystemExit(code)

    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    sftp = c.open_sftp()
    names = [
        "stage3_test_eval.json",
        "skeleton_playback_stage3_test_pred.json",
        "skeleton_playback_stage3_test_gt.json",
        "skeleton_yaw_stage3_test_pred.mp4",
        "skeleton_yaw_stage3_test_gt.mp4",
        "skeleton_yaw_stage3_test_pred_vs_gt.mp4",
        "skeleton_yaw_stage3_test_pred_vs_gt.json",
    ]
    for name in names:
        remote = f"{OUT_REMOTE}/{name}"
        local = LOCAL_OUT / name
        try:
            sftp.get(remote, str(local))
            print(f"pulled {local}")
        except OSError as exc:
            print(f"skip {name}: {exc}")
    sftp.close()
    c.close()
    print(json.dumps({"local_out": str(LOCAL_OUT.resolve())}, indent=2))


if __name__ == "__main__":
    main()
