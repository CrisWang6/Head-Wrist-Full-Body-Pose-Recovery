import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import overnight_0806_limb_batch as ob

ob.BATCH = "/home/gaoweijian/0810_batch"
ob.ssh_exec("pkill -u gaoweijian -f run_0810_training_master.sh 2>/dev/null || true")
for line in ("line1", "line2"):
    pre = f"{ob.BATCH}/{line}/data_root/multiview_3d_results/full/multiview_3d_results_pre_limb.jsonl"
    csv = f"{ob.BATCH}/{line}/data_root/multiview_3d_results/full/head_reprojection/head_reprojection_2d_wo_calibration_proj.csv"
    _, o, _ = ob.ssh_exec(f"wc -l {pre} {csv} 2>/dev/null || true")
    print(line, o.strip())
