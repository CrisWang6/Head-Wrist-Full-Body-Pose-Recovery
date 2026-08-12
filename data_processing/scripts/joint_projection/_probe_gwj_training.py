#!/usr/bin/env python3
"""One-off probe of gwj training layout."""
from __future__ import annotations

import paramiko

HOST = "192.168.20.221"
USER = "gaoweijian"
PASSWORD = "gwj@#@2026"

CMDS = [
    ("ego_top", "ls -la /home/gaoweijian/EgoRear_w_hand/"),
    ("ego_scripts", "ls -la /home/gaoweijian/EgoRear_w_hand/scripts/ 2>/dev/null | head -50"),
    ("ego_experiments", "ls -laR /home/gaoweijian/EgoRear_w_hand/experiments/ 2>/dev/null | head -80"),
    ("ego_src", "find /home/gaoweijian/EgoRear_w_hand/src -type f -name '*.py' 2>/dev/null | head -60"),
    ("train_scripts", "find /home/gaoweijian/EgoRear_w_hand -type f \\( -name 'train_*.py' -o -name 'test_*.py' -o -name 'eval_*.py' \\) 2>/dev/null"),
    ("configs", "find /home/gaoweijian/EgoRear_w_hand/configs -type f 2>/dev/null | head -40"),
    ("patch_check", "grep -c global_idx /home/gaoweijian/EgoRear_w_hand/experiments/stage3_pose3d/scripts/train_pose3d.py 2>/dev/null || echo 0"),
    ("jp_0810", "ls /home/gaoweijian/0810_batch/repo/test_code/joint_projection/ 2>/dev/null | head -80"),
    ("jp_training", "ls /home/gaoweijian/0810_batch/repo/test_code/joint_projection/*stage* /home/gaoweijian/0810_batch/repo/test_code/joint_projection/*training* /home/gaoweijian/0810_batch/repo/test_code/joint_projection/eval_* /home/gaoweijian/0810_batch/repo/test_code/joint_projection/render_* /home/gaoweijian/0810_batch/repo/test_code/joint_projection/_tb_* 2>/dev/null"),
    ("0810dataset", "ls -la /home/gaoweijian/0810dataset/ 2>/dev/null | head -20"),
    ("0810_status", "cat /home/gaoweijian/0810dataset/0810_TRAINING_STATUS.txt 2>/dev/null || echo NO_STATUS"),
]


def main() -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=30)
    for title, cmd in CMDS:
        print(f"\n=== {title} ===")
        _, stdout, stderr = client.exec_command(cmd, timeout=120)
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        print(out[:12000])
        if err.strip():
            print("ERR:", err[:500])
    client.close()


if __name__ == "__main__":
    main()
