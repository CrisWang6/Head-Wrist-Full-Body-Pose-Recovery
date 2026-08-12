import paramiko
import time
from pathlib import Path

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("192.168.20.221", username="gaoweijian", password="gwj@#@2026", timeout=30)
JP = "/home/gaoweijian/0806_batch/repo/test_code/joint_projection"
local = Path(r"C:/Users/hand/Desktop/HearWristCam/test_code/joint_projection")
files = ["eval_0806_stage1_test.py", "run_stage1_v31_only_gwj.sh"]
sftp = c.open_sftp()
for f in files:
    data = (local / f).read_bytes().replace(b"\r\n", b"\n")
    with sftp.file(f"{JP}/{f}", "wb") as out:
        out.write(data)
    c.exec_command(f"chmod +x {JP}/{f}")
sftp.close()
c.exec_command("pkill -u gaoweijian -f train_heatmap.py 2>/dev/null || true")
c.exec_command("pkill -u gaoweijian -f run_weekend_master_resume.sh 2>/dev/null || true")
time.sleep(3)
launch = (
    "bash -lc 'source ~/miniforge3/etc/profile.d/conda.sh && "
    f"nohup bash {JP}/run_stage1_v31_only_gwj.sh "
    f">> {JP}/../0806dataset/logs/stage1_v31_only_nohup.log 2>&1 & echo PID=$!'"
)
_, o, _ = c.exec_command(launch, timeout=10)
print("launch:", o.read().decode().strip())
time.sleep(8)
_, o, _ = c.exec_command("pgrep -af train_heatmap | grep -v pgrep | head -1")
print("train:", o.read().decode()[:180])
_, o, _ = c.exec_command("cat /home/gaoweijian/0806dataset/WEEKEND_MASTER_STATUS.txt")
print("status:", o.read().decode().strip())
c.close()
