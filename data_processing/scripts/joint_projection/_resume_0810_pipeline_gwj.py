"""Resume 0810 pipeline on gwj after H265 decode fix (D→E) + training master.

Does not re-upload MJPEG/H265 (already on server). Uploads patched scripts only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import overnight_0806_limb_batch as ob

ob.BATCH = "/home/gaoweijian/0810_batch"
ob.REPO_REMOTE = f"{ob.BATCH}/repo"

ob.SCRIPT_FILES = ob.SCRIPT_FILES + [
    "run_0810_line_dataset.sh",
    "run_0810_training_master.sh",
    "reexport_0810_heatmap_labels.sh",
    "build_0810_pack_splits.py",
    "prepare_0810_pose3d_labels.py",
    "constants_0810_training.py",
    "export_0806_heatmap_labels.py",
    "extract_0806_head_frames.py",
    "configs/0810_line1_dual_external_mocap.json",
    "configs/0810_line2_dual_external_mocap.json",
    "configs/joint_radius_px_120x75_delivery15.json",
    "_patch_train_pose3d_alignment.py",
]


def remote_status(key: str) -> str:
    _, out, _ = ob.ssh_exec(f"cat {ob.BATCH}/{key}/STATUS.txt 2>/dev/null || echo unknown")
    return out.strip().splitlines()[-1] if out.strip() else "unknown"


def launch_pipeline(key: str) -> None:
    script = f"{ob.REPO_REMOTE}/test_code/joint_projection/run_0810_line_dataset.sh"
    log = f"{ob.BATCH}/{key}/logs/nohup_pipeline_resume.log"
    ob.ssh_exec(
        f"sed -i 's/\\r$//' {script}; chmod +x {script}; "
        f"mkdir -p {ob.BATCH}/{key}/logs"
    )
    cmd = f"nohup bash {script} {key} >{log} 2>&1 </dev/null & echo $!"
    _, out, err = ob.ssh_exec(cmd)
    print("launched", key, out.strip(), err.strip())


def launch_training_master() -> None:
    script = f"{ob.REPO_REMOTE}/test_code/joint_projection/run_0810_training_master.sh"
    log = "/home/gaoweijian/0810dataset/logs/nohup_training_master.log"
    ob.ssh_exec(
        f"mkdir -p /home/gaoweijian/0810dataset/logs; "
        f"sed -i 's/\\r$//' {script}; chmod +x {script}"
    )
    cmd = f"nohup bash {script} >{log} 2>&1 </dev/null & echo $!"
    _, out, err = ob.ssh_exec(cmd)
    print("launched training master", out.strip(), err.strip())


def verify_h265_decode(key: str) -> None:
    """Quick remote smoke test: read capture 0 and 100 from head CAM_A."""
    head_dirs = {"line1": "0712_035226", "line2": "0712_035903"}
    head = head_dirs[key]
    jp = f"{ob.REPO_REMOTE}/test_code/joint_projection"
    py = "/home/gaoweijian/miniforge3/envs/sapiens2/bin/python"
    cmd = (
        f"cd {jp} && PYTHONPATH={jp} {py} - <<'PY'\n"
        "from pathlib import Path\n"
        "from render_multiview_to_head import H265CaptureReader, HeadTimestampIndex\n"
        f"root = Path('{ob.BATCH}/{key}/data_root/{head}')\n"
        "idx = HeadTimestampIndex(root / 'timestamps.csv', 'CAM_A')\n"
        "r = H265CaptureReader(root / 'module01_D45D2E00_CAM_A.h265', idx.rows)\n"
        "for i in (0, 100, 1000):\n"
        "    img = r.read(i)\n"
        "    print(i, img.shape, 'fallbacks', r.missing_fallbacks)\n"
        "r.close()\n"
        "print('ok')\n"
        "PY"
    )
    code, out, err = ob.ssh_exec(cmd, timeout=120)
    print(f"=== h265 smoke {key} exit={code} ===")
    print(out)
    if err.strip():
        print("stderr:", err)
    if code != 0 or "ok" not in out:
        raise RuntimeError(f"H265 smoke test failed for {key}")


def main() -> None:
    datasets = ["line1", "line2"]
    print("=== upload patched repo scripts ===", flush=True)
    ob.upload_repo()
    ob.ssh_exec(
        f"sed -i 's/\\r$//' {ob.REPO_REMOTE}/test_code/joint_projection/run_0810_line_dataset.sh "
        f"{ob.REPO_REMOTE}/test_code/joint_projection/run_0810_training_master.sh "
        f"{ob.REPO_REMOTE}/test_code/joint_projection/reexport_0810_heatmap_labels.sh; "
        f"chmod +x {ob.REPO_REMOTE}/test_code/joint_projection/run_0810_line_dataset.sh "
        f"{ob.REPO_REMOTE}/test_code/joint_projection/run_0810_training_master.sh "
        f"{ob.REPO_REMOTE}/test_code/joint_projection/reexport_0810_heatmap_labels.sh"
    )

    for key in datasets:
        print(f"=== verify H265 decode {key} ===", flush=True)
        verify_h265_decode(key)

    before = {key: remote_status(key) for key in datasets}
    print("status before:", before, flush=True)

    for key in datasets:
        if before[key] == "done":
            print(f"skip pipeline {key} (already done)", flush=True)
            continue
        print(f"=== resume pipeline {key} ===", flush=True)
        launch_pipeline(key)
        time.sleep(3)

    results = {}
    for key in datasets:
        if before[key] == "done":
            results[key] = "done"
            continue
        status = ob.wait_pipeline(key, poll_s=120)
        results[key] = status
        print(time.strftime("%H:%M:%S"), key, "pipeline", status, flush=True)

    if all(results.get(k) == "done" for k in datasets):
        print("=== launch 3-stage training ===", flush=True)
        launch_training_master()
    else:
        print("WARNING: pipeline not complete; training master not started", flush=True)

    summary = {
        "batch": ob.BATCH,
        "status_before": before,
        "pipeline": results,
        "training_status_file": "/home/gaoweijian/0810dataset/0810_TRAINING_STATUS.txt",
        "training_log": "/home/gaoweijian/0810dataset/logs/0810_training_master.log",
    }
    out = Path(__file__).resolve().parent / "output" / "0810_pipeline_resume.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
