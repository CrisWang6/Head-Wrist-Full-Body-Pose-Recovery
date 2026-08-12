import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("192.168.20.221", username="gaoweijian", password="gwj@#@2026", timeout=30)
cmd = (
    "bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && conda activate camtest && "
    "python /home/gaoweijian/0806_batch/repo/test_code/joint_projection/_patch_train_pose3d_alignment.py'"
)
_, o, e = c.exec_command(cmd, timeout=60)
print(o.read().decode())
err = e.read().decode()
if err.strip():
    print("ERR:", err)
_, o, _ = c.exec_command(
    "grep -n global_idx /home/gaoweijian/EgoRear_w_hand/experiments/stage3_pose3d/scripts/train_pose3d.py | head",
    timeout=20,
)
print(o.read().decode())
c.close()
