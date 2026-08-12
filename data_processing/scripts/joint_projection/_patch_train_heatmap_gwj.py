import paramiko
from pathlib import Path

TRAIN = Path("/home/gaoweijian/EgoRear_w_hand/scripts/train_heatmap.py")
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("192.168.20.221", username="gaoweijian", password="gwj@#@2026", timeout=30)
_, o, _ = c.exec_command(f"cat {TRAIN}", timeout=60)
text = o.read().decode()
old = "    joint_radius_px = dict(DEFAULT_JOINT_HEATMAP_RADIUS_PX)"
new = "    joint_radius_px = {name: float(args.default_joint_radius_px) for name in head_joint_names}"
if old not in text:
    print("pattern missing", old in text)
else:
    text = text.replace(old, new, 1)
    sftp = c.open_sftp()
    with sftp.file(str(TRAIN), "w") as f:
        f.write(text)
    sftp.close()
    print("patched ok")
c.close()
