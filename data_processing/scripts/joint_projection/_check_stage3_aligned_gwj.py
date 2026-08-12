import json
import re
from pathlib import Path

import paramiko

LOG = "/home/gaoweijian/0806dataset/logs/stage3_v31_aligned.log"
CKPT_DIR = "/home/gaoweijian/0806dataset/checkpoints/stage3_v31_aligned"

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect("192.168.20.221", username="gaoweijian", password="gwj@#@2026", timeout=30)

cmds = [
    ("proc", "pgrep -af train_pose3d.py | grep stage3_v31_aligned | grep -v bash | head -3; echo ---; pgrep -c -f 'train_pose3d.py.*stage3_v31_aligned' || true"),
    ("gpu", "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader"),
    ("log_tail", f"tail -60 {LOG}"),
    ("status", f"cat {CKPT_DIR}/status.json 2>/dev/null || echo '{{}}'"),
    ("best", f"ls -lh {CKPT_DIR}/best.pt 2>/dev/null; ls -lh {CKPT_DIR}/last.pt 2>/dev/null"),
    ("epoch_count", f"grep -c '\"epoch\"' {LOG} || true"),
]
for name, cmd in cmds:
    _, o, _ = c.exec_command(cmd, timeout=30)
    print(f"=== {name} ===")
    print(o.read().decode(errors="replace"))

# parse last epoch json blocks from log
_, o, _ = c.exec_command(f"grep -E '^(\\{{|  )' {LOG} | tail -120", timeout=30)
text = o.read().decode(errors="replace")
blocks = []
buf = []
for line in text.splitlines():
    if line.startswith("{"):
        if buf:
            blocks.append("\n".join(buf))
        buf = [line]
    elif buf:
        buf.append(line)
if buf:
    blocks.append("\n".join(buf))

summaries = []
for block in blocks[-15:]:
    try:
        obj = json.loads(block)
    except json.JSONDecodeError:
        continue
    if "epoch" in obj and "val" in obj:
        summaries.append(
            {
                "epoch": obj.get("epoch"),
                "train_mpjpe_mm": obj.get("train", {}).get("mpjpe_mm"),
                "val_mpjpe_mm": obj.get("val", {}).get("mpjpe_mm"),
                "best_epoch": obj.get("best_epoch"),
                "best_mpjpe_mm": obj.get("best_mpjpe_mm"),
                "epochs_without_improvement": obj.get("epochs_without_improvement"),
                "early_stop": obj.get("early_stop_reason"),
            }
        )

print("=== epoch_summary ===")
for row in summaries[-8:]:
    print(row)
if summaries:
    last = summaries[-1]
    print("=== latest ===")
    print(json.dumps(last, indent=2))

# milestone curve from full log
_, o, _ = c.exec_command(f"cat {LOG}", timeout=60)
full = o.read().decode(errors="replace")
all_rows = []
buf = []
for line in full.splitlines():
    if line.startswith("{"):
        if buf:
            try:
                all_rows.append(json.loads("\n".join(buf)))
            except json.JSONDecodeError:
                pass
        buf = [line]
    elif buf:
        buf.append(line)
if buf:
    try:
        all_rows.append(json.loads("\n".join(buf)))
    except json.JSONDecodeError:
        pass

epoch_rows = [r for r in all_rows if "epoch" in r and "val" in r]
print("=== milestones ===")
for ep in [1, 10, 50, 100, 132, 152]:
    row = next((r for r in epoch_rows if r["epoch"] == ep), None)
    if row:
        print(
            f"epoch {ep}: train={row['train']['mpjpe_mm']:.1f}mm "
            f"val={row['val']['mpjpe_mm']:.1f}mm "
            f"best={row.get('best_mpjpe_mm', 0):.1f}mm @ ep{row.get('best_epoch')}"
        )

OLD_LOG = "/home/gaoweijian/0806dataset/logs/stage3_v31_only.log"
_, o, _ = c.exec_command(f"test -f {OLD_LOG} && tail -40 {OLD_LOG} || echo missing", timeout=20)
old_tail = o.read().decode(errors="replace")
print("=== old stage3_v31 (wrong GT) tail ===")
print(old_tail[-1200:])

c.close()
