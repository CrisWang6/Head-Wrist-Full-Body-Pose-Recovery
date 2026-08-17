import paramiko
import time

HOST = "192.168.20.221"
USER = "gaoweijian"
PASS = "gwj@#@2026"
PORT = 6040
TB = "/home/gaoweijian/miniforge3/envs/camtest/bin/tensorboard"
LOGDIR_SPEC = (
    "stage1:/home/gaoweijian/0810dataset/logs/stage1_v31,"
    "stage2:/home/gaoweijian/0810dataset/logs/stage2_v31,"
    "stage3:/home/gaoweijian/0810dataset/checkpoints/stage3_v31_aligned/tensorboard"
)
LOG = "/home/gaoweijian/0810dataset/logs/tensorboard_0810_all_stages.log"
MARKER = "tensorboard.*0810dataset/logs.*6040"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, username=USER, password=PASS, timeout=30)

# kill prior 0810 all-stages TB on this port only
c.exec_command(f"pkill -u gaoweijian -f 'tensorboard.*--port {PORT}' 2>/dev/null || true")
time.sleep(1)

launch = (
    f"nohup {TB} --logdir_spec {LOGDIR_SPEC} --host 127.0.0.1 --port {PORT} --reload_interval 15 "
    f"> {LOG} 2>&1 & echo TB_PID=$!"
)
_, o, e = c.exec_command(launch, timeout=15)
print("launch:", o.read().decode().strip())
err = e.read().decode().strip()
if err:
    print("launch_err:", err)
time.sleep(4)

for cmd in [
    f"pgrep -af 'tensorboard.*{PORT}' || echo NO_TB",
    f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{PORT}/ || echo curl_fail",
    f"tail -20 {LOG} 2>/dev/null || echo NO_LOG",
]:
    print("---", cmd[:60])
    _, o, _ = c.exec_command(cmd, timeout=30)
    print(o.read().decode("utf-8", errors="replace"))

c.close()
print(f"REMOTE_PORT={PORT}")
