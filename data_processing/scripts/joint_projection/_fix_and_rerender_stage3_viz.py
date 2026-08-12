import paramiko
from pathlib import Path

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("192.168.20.221", username="gaoweijian", password="gwj@#@2026", timeout=30)
jp = "/home/gaoweijian/0806_batch/repo/test_code/joint_projection"
sftp = c.open_sftp()
for name in [
    "_patch_train_pose3d_alignment.py",
    "rerender_stage3_aligned_viz.py",
    "render_stage3_dual_skeleton_yaw.py",
    "eval_stage3_test_3d_viz.py",
    "skeleton_3d_filter.py",
    "delivery_keypoints.py",
]:
    sftp.put(str(Path(__file__).resolve().parent / name), f"{jp}/{name}")
sftp.close()

cmds = [
    f"python {jp}/_patch_train_pose3d_alignment.py",
    (
        f"bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate camtest && "
        f"export PYTHONPATH={jp}:/home/gaoweijian/EgoRear_w_hand/src && "
        f"python {jp}/rerender_stage3_aligned_viz.py "
        f"--pose3d-labels /home/gaoweijian/0806dataset/labels/pose3d_nose_pre_limb_15j.npz "
        f"--split-manifest /home/gaoweijian/0806dataset/splits/pack30_v31.npz "
        f"--pred-playback /home/gaoweijian/0806dataset/eval/v31/stage3_test_3d_viz/skeleton_playback_stage3_test_pred.json "
        f"--output-dir /home/gaoweijian/0806dataset/eval/v31/stage3_test_3d_viz'"
    ),
]
for cmd in cmds:
    print("===", cmd[:80], "===")
    _, o, e = c.exec_command(cmd, timeout=600)
    print(o.read().decode())
    err = e.read().decode()
    if err.strip():
        print("ERR:", err[-3000:])

# pull aligned outputs
local = Path(__file__).resolve().parent / "output" / "stage3_v31_test_3d_viz"
local.mkdir(parents=True, exist_ok=True)
sftp = c.open_sftp()
for name in [
    "skeleton_playback_stage3_test_gt_aligned.json",
    "skeleton_yaw_stage3_test_pred_vs_gt_aligned.mp4",
    "skeleton_yaw_stage3_test_pred_vs_gt_aligned.json",
]:
    try:
        sftp.get(
            f"/home/gaoweijian/0806dataset/eval/v31/stage3_test_3d_viz/{name}",
            str(local / name),
        )
        print("pulled", name)
    except OSError as exc:
        print("skip", name, exc)
sftp.close()
c.close()
