"""Monitor 0810 training + pipeline on gwj."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta

import overnight_0806_limb_batch as ob

CST = timezone(timedelta(hours=8))


def q(cmd: str) -> str:
    _, out, err = ob.ssh_exec(cmd)
    return (out or err).strip()


def main() -> None:
    now = datetime.now(CST).strftime("%Y-%m-%d %H:%M:%S CST")
    print(f"=== 0810 monitor @ {now} ===\n")

    status = q("cat /home/gaoweijian/0810dataset/0810_TRAINING_STATUS.txt 2>/dev/null || echo unknown")
    print(f"Training status: {status}\n")

    print("--- training log (last 35 lines) ---")
    print(q("tail -n 35 /home/gaoweijian/0810dataset/logs/0810_training_master.log 2>/dev/null || echo no_log"))
    print()

    for name, expect in [("line1", 5857), ("line2", 13788)]:
        cam_a = q(f"ls /home/gaoweijian/0810dataset/frames/{name}/*/CAM_A/*.jpg 2>/dev/null | wc -l")
        cam_d = q(f"ls /home/gaoweijian/0810dataset/frames/{name}/*/CAM_D/*.jpg 2>/dev/null | wc -l")
        print(f"Frames {name}: CAM_A={cam_a}/{expect} CAM_D={cam_d}/{expect}")

    print()
    print("--- running jobs ---")
    print(q("ps aux | grep 0810 | grep -v grep | head -8 || echo none"))
    print()
    for key in ("line1", "line2"):
        print(f"Pipeline {key}: {q(f'cat /home/gaoweijian/0810_batch/{key}/STATUS.txt 2>/dev/null || echo unknown')}")

    print()
    print("--- checkpoints ---")
    print(q("ls -la /home/gaoweijian/0810dataset/checkpoints/ 2>/dev/null || echo none"))
    for stage in ("stage1_v31", "stage2_v31", "stage3_v31_aligned"):
        best = q(f"ls -la /home/gaoweijian/0810dataset/checkpoints/{stage}/best.pt 2>/dev/null || echo missing")
        print(f"  {stage}: {best}")

    split = q("ls -la /home/gaoweijian/0810dataset/splits/pack150_v31.npz 2>/dev/null || echo missing")
    print(f"\nSplit: {split}")

    if len(sys.argv) > 1 and sys.argv[1] == "--json":
        summary = {
            "time": now,
            "training_status": status,
            "pipeline": {
                "line1": q("cat /home/gaoweijian/0810_batch/line1/STATUS.txt 2>/dev/null"),
                "line2": q("cat /home/gaoweijian/0810_batch/line2/STATUS.txt 2>/dev/null"),
            },
        }
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
