"""Upload 0810 data, run A→E pipeline on gwj, then 3-stage training (pack150)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import overnight_0806_limb_batch as ob

ob.BATCH = "/home/gaoweijian/0810_batch"
ob.REPO_REMOTE = f"{ob.BATCH}/repo"
ob.DATA_BASE = Path(r"C:\Users\hand\Desktop\双外部双目\0810")

ob.DATASETS = {
    "line1": {
        "local_name": "1",
        "external": "20260810_175143_465",
        "head": "0712_035226",
        "config": "0810_line1_dual_external_mocap.json",
    },
    "line2": {
        "local_name": "2",
        "external": "20260810_175842_396",
        "head": "0712_035903",
        "config": "0810_line2_dual_external_mocap.json",
    },
}

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


def upload_dataset_metadata(key: str) -> None:
    meta = ob.DATASETS[key]
    local_root = ob.DATA_BASE / meta["local_name"]
    remote_root = f"{ob.BATCH}/{key}/data_root"
    head = meta["head"]
    external = meta["external"]
    ob.ssh_exec(
        f"mkdir -p {remote_root}/aligned_data {remote_root}/{head} "
        f"{remote_root}/{external}/external_01 {remote_root}/{external}/external_02 "
        f"{remote_root}/multiview_3d_results "
        f"{ob.BATCH}/{key}/input/module01 {ob.BATCH}/{key}/input/module02 "
        f"{ob.BATCH}/{key}/logs {ob.BATCH}/{key}/inference"
    )
    strict = local_root / "aligned_data" / "aligned_30hz_strict.csv"
    aligned = strict if strict.is_file() else local_root / "aligned_data" / "aligned_30hz.csv"
    ob.upload_if_needed(aligned, f"{remote_root}/aligned_data/aligned_30hz.csv")
    ob.upload_if_needed(
        local_root / "aligned_data" / "aligned_30hz_report.json",
        f"{remote_root}/aligned_data/aligned_30hz_report.json",
    )
    if strict.is_file():
        ob.upload_if_needed(strict, f"{remote_root}/aligned_data/aligned_30hz_strict.csv")
    ob.upload_if_needed(
        local_root / "multiview_3d_results" / "aligned_manifest.jsonl",
        f"{remote_root}/multiview_3d_results/aligned_manifest.jsonl",
    )
    report = local_root / "multiview_3d_results" / "aligned_manifest_report.json"
    if report.is_file():
        ob.upload_if_needed(report, f"{remote_root}/multiview_3d_results/aligned_manifest_report.json")
    for name in (
        "timestamps.csv",
        "module01_D45D2E00_CAM_A.h265",
        "module01_D45D2E00_CAM_D.h265",
    ):
        ob.upload_if_needed(local_root / head / name, f"{remote_root}/{head}/{name}")
    local_ext = local_root / external
    for sub in ("external_01", "external_02"):
        for name in ("timestamps.csv", "recording.json", "trigger_events.csv"):
            path = local_ext / sub / name
            if path.is_file():
                ob.upload_if_needed(path, f"{remote_root}/{external}/{sub}/{name}")
    modules_json = local_ext / "external_modules.json"
    if modules_json.is_file():
        ob.upload_if_needed(modules_json, f"{remote_root}/{external}/external_modules.json")


def launch_pipeline(key: str) -> None:
    script = f"{ob.REPO_REMOTE}/test_code/joint_projection/run_0810_line_dataset.sh"
    log = f"{ob.BATCH}/{key}/logs/nohup_pipeline.log"
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


def main() -> None:
    datasets = ["line1", "line2"]
    ob.upload_repo()
    ob.ssh_exec(
        f"sed -i 's/\\r$//' {ob.REPO_REMOTE}/test_code/joint_projection/run_0810_line_dataset.sh "
        f"{ob.REPO_REMOTE}/test_code/joint_projection/run_0810_training_master.sh "
        f"{ob.REPO_REMOTE}/test_code/joint_projection/reexport_0810_heatmap_labels.sh; "
        f"chmod +x {ob.REPO_REMOTE}/test_code/joint_projection/run_0810_line_dataset.sh "
        f"{ob.REPO_REMOTE}/test_code/joint_projection/run_0810_training_master.sh "
        f"{ob.REPO_REMOTE}/test_code/joint_projection/reexport_0810_heatmap_labels.sh"
    )
    results = {}
    for key in datasets:
        print(f"=== {key}: upload metadata ===", flush=True)
        upload_dataset_metadata(key)
        print(f"=== {key}: upload videos (large) ===", flush=True)
        ob.upload_videos(key)
        print(f"=== {key}: launch pipeline ===", flush=True)
        launch_pipeline(key)
        status = ob.wait_pipeline(key, poll_s=120)
        results[key] = status
        print(time.strftime("%H:%M:%S"), key, "pipeline", status, flush=True)
        if status != "done":
            print(f"WARNING: {key} pipeline ended with {status}", flush=True)
    print("=== launch 3-stage training ===", flush=True)
    launch_training_master()
    summary = {
        "batch": ob.BATCH,
        "dataset": "/home/gaoweijian/0810dataset",
        "split": "pack150_v31 (line1 train / line2 test, 5s packs)",
        "pipeline": results,
        "training_status_file": "/home/gaoweijian/0810dataset/0810_TRAINING_STATUS.txt",
        "training_log": "/home/gaoweijian/0810dataset/logs/0810_training_master.log",
    }
    out = Path(__file__).resolve().parent / "output" / "0810_pipeline_launch.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
