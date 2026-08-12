"""Push H265/module fixes and start 0810 training prep on gwj."""
from __future__ import annotations

import json
from pathlib import Path

import overnight_0806_limb_batch as ob

ob.BATCH = "/home/gaoweijian/0810_batch"
ob.REPO_REMOTE = f"{ob.BATCH}/repo"
JP = Path(__file__).resolve().parent

PATCH_FILES = [
    "render_multiview_to_head.py",
    "detect_head_nose_rtmw.py",
    "extract_0806_head_frames.py",
    "optimize_multiview_head_nose_offset.py",
    "render_nose_offset_parallel.py",
    "export_0806_heatmap_labels.py",
    "run_0810_training_prep.sh",
    "run_0810_training_master.sh",
    "reexport_0810_heatmap_labels.sh",
    "build_0810_pack_splits.py",
    "prepare_0810_pose3d_labels.py",
    "constants_0810_training.py",
    "run_0810_line_dataset.sh",
    "_patch_train_pose3d_alignment.py",
]


def main() -> None:
    remote_jp = f"{ob.REPO_REMOTE}/test_code/joint_projection"
    for rel in PATCH_FILES:
        ob.scp_to(JP / rel, f"{remote_jp}/{rel}")
    ob.ssh_exec(
        f"sed -i 's/\\r$//' {remote_jp}/run_0810_training_prep.sh "
        f"{remote_jp}/run_0810_training_master.sh "
        f"{remote_jp}/run_0810_line_dataset.sh "
        f"{remote_jp}/reexport_0810_heatmap_labels.sh; "
        f"chmod +x {remote_jp}/run_0810_training_prep.sh "
        f"{remote_jp}/run_0810_training_master.sh "
        f"{remote_jp}/run_0810_line_dataset.sh "
        f"{remote_jp}/reexport_0810_heatmap_labels.sh"
    )

    py = "/home/gaoweijian/miniforge3/envs/sapiens2/bin/python"
    smoke = (
        f"cd {remote_jp} && PYTHONPATH={remote_jp} {py} - <<'PY'\n"
        "from pathlib import Path\n"
        "from render_multiview_to_head import H265CaptureReader, HeadTimestampIndex, infer_head_module_from_video\n"
        "for key, head in [('line1','0712_035226'),('line2','0712_035903')]:\n"
        f"    root = Path('{ob.BATCH}') / key / 'data_root' / head\n"
        "    vid = root / 'module01_D45D2E00_CAM_A.h265'\n"
        "    mod = infer_head_module_from_video(vid)\n"
        "    idx = HeadTimestampIndex(root / 'timestamps.csv', 'CAM_A', module=mod)\n"
        "    r = H265CaptureReader(vid, idx.rows)\n"
        "    r.read(len(idx.rows)-1)\n"
        "    r.close()\n"
        "    print(key, 'rows', len(idx.rows), 'ok')\n"
        "PY"
    )
    code, out, err = ob.ssh_exec(smoke, timeout=600)
    print(out)
    if code != 0:
        raise RuntimeError(err or "smoke failed")

    log = "/home/gaoweijian/0810dataset/logs/nohup_training_prep.log"
    ob.ssh_exec(f"mkdir -p /home/gaoweijian/0810dataset/logs")
    _, out, _ = ob.ssh_exec(
        f"nohup bash {remote_jp}/run_0810_training_prep.sh >{log} 2>&1 </dev/null & echo $!"
    )
    print("launched training prep pid", out.strip())

    # Resume full D→E viz pipeline in parallel (non-blocking).
    for key in ("line1", "line2"):
        plog = f"{ob.BATCH}/{key}/logs/nohup_pipeline_resume2.log"
        ob.ssh_exec(
            f"nohup bash {remote_jp}/run_0810_line_dataset.sh {key} >{plog} 2>&1 </dev/null & echo $!"
        )

    summary = {
        "training_prep_log": log,
        "training_status": "/home/gaoweijian/0810dataset/0810_TRAINING_STATUS.txt",
        "training_master_log": "/home/gaoweijian/0810dataset/logs/0810_training_master.log",
        "pipeline_resume_logs": {
            "line1": f"{ob.BATCH}/line1/logs/nohup_pipeline_resume2.log",
            "line2": f"{ob.BATCH}/line2/logs/nohup_pipeline_resume2.log",
        },
    }
    out_path = JP / "output" / "0810_training_prep_launch.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
