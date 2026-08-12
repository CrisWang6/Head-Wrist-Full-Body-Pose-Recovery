"""Stop stage3 training, run test-set 3D viz, pull MP4 locally."""
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
OUT_REMOTE = f"{DS}/eval/v31/stage3_test_3d_viz"
LOCAL_OUT = Path(__file__).resolve().parent / "output" / "stage3_v31_test_3d_viz"

UPLOAD = [
    "constants_0806_training.py",
    "delivery_keypoints.py",
    "eval_stage3_test_3d_viz.py",
    "render_skeleton_yaw_video.py",
]


def run(c, cmd: str, timeout: int = 3600) -> tuple[str, str, int]:
    _, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode(errors="replace")
    err = e.read().decode(errors="replace")
    return out, err, o.channel.recv_exit_status()


REMOTE_SH = textwrap.dedent(
    f"""\
    set -euo pipefail
    source ~/miniforge3/etc/profile.d/conda.sh
    conda activate camtest
    export PYTHONPATH={JP}:{EGO}/src

    pkill -u gaoweijian -f 'train_pose3d.py.*stage3_v31' 2>/dev/null || true
    sleep 2

    python3 - <<'PY'
import json
from pathlib import Path
status = {{
    "state": "stopped",
    "reason": "user_requested_snapshot",
    "checkpoint": "{DS}/checkpoints/stage3_v31/best.pt",
}}
Path("{DS}/checkpoints/stage3_v31/status.json").write_text(
    json.dumps(status, indent=2), encoding="utf-8"
)
Path("{DS}/WEEKEND_STAGE23_STATUS.txt").write_text("stage3_v31_stopped\\n")
print("stopped training, using best.pt")
PY

    mkdir -p {OUT_REMOTE}
    cd {EGO}
    {CAMTEST} {JP}/eval_stage3_test_3d_viz.py \\
      --label-root {DS}/labels \\
      --pose3d-labels {DS}/labels/pose3d_nose_pre_limb_15j.npz \\
      --stage1-checkpoint {DS}/checkpoints/stage1_v31/best.pt \\
      --stage2-checkpoint {DS}/checkpoints/stage2_v31/best.pt \\
      --stage3-checkpoint {DS}/checkpoints/stage3_v31/best.pt \\
      --split-manifest {DS}/splits/pack30_v31.npz \\
      --split-name test \\
      --output-dir {OUT_REMOTE} \\
      --batch-size 64 --workers 4 --device cuda

    ls -la {OUT_REMOTE}
    """
)


def pull_files(c: paramiko.SSHClient) -> None:
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    sftp = c.open_sftp()
    names = [
        "stage3_test_eval.json",
        "skeleton_playback_stage3_test_pred.json",
        "skeleton_playback_stage3_test_gt.json",
        "skeleton_yaw_stage3_test_pred.mp4",
        "skeleton_yaw_stage3_test_gt.mp4",
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


def main() -> None:
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PASS, timeout=30)
    sftp = c.open_sftp()
    for rel in UPLOAD:
        sftp.put(str(Path(__file__).resolve().parent / rel), f"{JP}/{rel}")
    with sftp.file("/tmp/run_stage3_test_viz.sh", "w") as f:
        f.write(REMOTE_SH)
    sftp.close()

    out, err, code = run(c, "bash /tmp/run_stage3_test_viz.sh", timeout=3600)
    print(out)
    if err.strip():
        print("stderr:", err[-5000:])
    if code != 0:
        c.close()
        raise SystemExit(code)

    pull_files(c)
    c.close()
    print(json.dumps({"local_out": str(LOCAL_OUT.resolve())}, indent=2))


if __name__ == "__main__":
    main()
