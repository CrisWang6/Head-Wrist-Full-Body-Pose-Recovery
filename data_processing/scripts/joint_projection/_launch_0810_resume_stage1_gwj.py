from pathlib import Path
import overnight_0806_limb_batch as ob

JP = Path(__file__).resolve().parent
remote = "/home/gaoweijian/0810_batch/repo/test_code/joint_projection"
for name in ("prepare_0806_pose3d_labels.py", "run_0810_resume_stage1.sh"):
    ob.scp_to(JP / name, f"{remote}/{name}")
ob.ssh_exec(
    f"sed -i 's/\\r$//' {remote}/run_0810_resume_stage1.sh; "
    f"chmod +x {remote}/run_0810_resume_stage1.sh"
)
_, out, _ = ob.ssh_exec(
    f"nohup bash {remote}/run_0810_resume_stage1.sh "
    f"> /home/gaoweijian/0810dataset/logs/nohup_resume_stage1.log 2>&1 </dev/null & echo launched"
)
print(out.strip())
