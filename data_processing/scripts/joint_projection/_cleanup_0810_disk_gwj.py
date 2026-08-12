"""Free disk on gwj for 0810 training (inference done → drop MJPEG + viz renders)."""
from __future__ import annotations

import overnight_0806_limb_batch as ob

BATCH = "/home/gaoweijian/0810_batch"
DATASET = "/home/gaoweijian/0810dataset"


def q(cmd: str) -> str:
    _, out, _ = ob.ssh_exec(cmd, timeout=300)
    return out.strip()


def main() -> None:
    print("before:", q("df -h /home/gaoweijian | tail -1"))

    cleanup_cmds = [
        # Head render / viz (not needed for stage1/2/3 training).
        f"rm -rf {BATCH}/line1/data_root/multiview_3d_results/full/head_reprojection/nose_offset_opt",
        f"rm -rf {BATCH}/line2/data_root/multiview_3d_results/full/head_reprojection/nose_offset_opt",
        f"rm -f {BATCH}/line1/data_root/multiview_3d_results/full/visualization/*.mp4",
        f"rm -f {BATCH}/line2/data_root/multiview_3d_results/full/visualization/*.mp4",
        # External MJPEG already inferenced → safe to drop (~42G).
        f"rm -rf {BATCH}/line1/input/module01 {BATCH}/line1/input/module02",
        f"rm -rf {BATCH}/line2/input/module01 {BATCH}/line2/input/module02",
        # Duplicate jsonl copy (keep pre_limb only).
        f"rm -f {BATCH}/line1/data_root/multiview_3d_results/full/multiview_3d_results.jsonl",
        f"rm -f {BATCH}/line2/data_root/multiview_3d_results/full/multiview_3d_results.jsonl",
    ]
    for cmd in cleanup_cmds:
        print("run:", cmd[:90], "...")
        q(cmd)

    print("\nafter:", q("df -h /home/gaoweijian | tail -1"))
    print("0810_batch:", q(f"du -sh {BATCH}"))
    print("0810dataset:", q(f"du -sh {DATASET}"))

    # Resume full training master (reexport fast; extract uses --skip-existing).
    jp = f"{BATCH}/repo/test_code/joint_projection"
    log = f"{DATASET}/logs/nohup_training_master_resume.log"
    q(f"mkdir -p {DATASET}/logs")
    pid = q(
        f"nohup bash {jp}/run_0810_training_master.sh >{log} 2>&1 </dev/null & echo $!"
    )
    print("launched training master pid:", pid)


if __name__ == "__main__":
    main()
