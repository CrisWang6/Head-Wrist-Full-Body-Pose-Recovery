#!/usr/bin/env python3
"""Local controller: prepare, upload, launch, monitor, pull, write SUMMARY.

Datasets: 腕 (wrist, z=-60mm rigid-local) and 踝 (ankle, z=-80mm rigid-local).
"""

from __future__ import annotations

import argparse
import json
import shutil
import stat as pystat
import subprocess
import sys
import time
from pathlib import Path

import paramiko

HOST = "192.168.20.221"
USER = "gaoweijian"
PASSWORD = "gwj@#@2026"
BATCH = "/home/gaoweijian/0806_batch"
REPO_REMOTE = f"{BATCH}/repo"
LOCAL_REPO = Path(r"C:\Users\hand\Desktop\HearWristCam")
DATA_BASE = Path(r"C:\Users\hand\Desktop\双外部双目\0806")
JP = LOCAL_REPO / "test_code" / "joint_projection"

DATASETS = {
    "wrist": {
        "local_name": "腕",
        "external": "20260806_122429_952",
        "head": "0712_032704",
        "replace": "wrist",
        "z_offset_mm": -60,
        "config": "0806_wrist_dual_external_mocap.json",
    },
    "ankle": {
        "local_name": "踝",
        "external": "20260806_122808_469",
        "head": "0712_033034",
        "replace": "ankle",
        "z_offset_mm": -80,
        "config": "0806_ankle_dual_external_mocap.json",
    },
}

SCRIPT_FILES = [
    "infer_rtmpose_candidates.py",
    "process_external_multiview_3d.py",
    "merge_multiview_chunks.py",
    "multiview_geometry.py",
    "delivery_keypoints.py",
    "export_playback_from_jsonl.py",
    "replace_limb_mocap_gt.py",
    "detect_head_nose_rtmw.py",
    "optimize_multiview_head_nose_offset.py",
    "render_nose_offset_parallel.py",
    "render_multiview_to_head.py",
    "render_external_multiview_results.py",
    "render_skeleton_yaw_video.py",
    "external_stereo_rigid_k_extrinsics.json",
    "head_stereo_rigid_extrinsics.json",
    "run_0806_limb_dataset.sh",
    "configs/0806_dual_external_mocap.json",
    "configs/0806_wrist_dual_external_mocap.json",
    "configs/0806_ankle_dual_external_mocap.json",
]

CALIB_FILES = [
    "test_code/calibrate/parameters/intrinsics/handle01/handle01_ad_intrinsics_kalibr_omni_1920x1200_20260805.json",
    "test_code/calibrate/parameters/intrinsics/handle/handle_ac_intrinsics_kalibr_omni_1920x1200_20260729.json",
    "test_code/calibrate/parameters/intrinsics/head/head_intrinsics_kalibr_omni_1920x1200.json",
]


def ssh_exec(cmd: str, timeout: int | None = None) -> tuple[int, str, str]:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=20)
    try:
        _, stdout, stderr = client.exec_command(cmd, timeout=timeout)
        out = stdout.read().decode(errors="replace")
        err = stderr.read().decode(errors="replace")
        code = stdout.channel.recv_exit_status()
        return code, out, err
    finally:
        client.close()


def _sftp_client() -> tuple[paramiko.SSHClient, paramiko.SFTPClient]:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=30, banner_timeout=30)
    transport = client.get_transport()
    if transport is not None:
        transport.set_keepalive(30)
        # Larger windows help multi-GB mjpeg uploads.
        try:
            transport.default_window_size = paramiko.common.MAX_WINDOW_SIZE
            transport.packetizer.REKEY_BYTES = pow(2, 40)
            transport.packetizer.REKEY_PACKETS = pow(2, 40)
        except Exception:
            pass
    return client, client.open_sftp()


def _sftp_put_with_progress(sftp: paramiko.SFTPClient, local: Path, remote: str) -> None:
    size = local.stat().st_size
    last = {"t": time.time(), "n": 0}

    def _cb(transferred: int, total: int) -> None:
        now = time.time()
        if now - last["t"] < 5 and transferred < total:
            return
        dt = max(now - last["t"], 1e-3)
        mbps = (transferred - last["n"]) / dt / (1024 * 1024)
        pct = 100.0 * transferred / total if total else 0.0
        print(
            f"  {local.name}: {transferred / 1e6:.0f}/{total / 1e6:.0f} MB "
            f"({pct:.1f}%) {mbps:.1f} MB/s",
            flush=True,
        )
        last["t"] = now
        last["n"] = transferred

    print(f"SFTP put {local.name} ({size / 1e6:.0f} MB) -> {remote}", flush=True)
    sftp.put(str(local), remote, callback=_cb if size > 8_000_000 else None)


def scp_to(local: Path, remote: str) -> None:
    """Upload: prefer native OpenSSH scp (faster on this LAN), else paramiko SFTP."""
    remote_parent = remote.rsplit("/", 1)[0]
    ssh_exec(f"mkdir -p '{remote_parent}'")
    size = local.stat().st_size
    print(f"SCP put {local.name} ({size / 1e6:.0f} MB) -> {remote}", flush=True)
    cmd = [
        "scp",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ConnectTimeout=20",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=120",
        "-o",
        "TCPKeepAlive=yes",
        str(local),
        f"{USER}@{HOST}:{remote}",
    ]
    try:
        subprocess.run(cmd, check=True)
        return
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"native scp failed ({exc}); falling back to paramiko SFTP", flush=True)
    client, sftp = _sftp_client()
    try:
        _sftp_put_with_progress(sftp, local, remote)
    finally:
        sftp.close()
        client.close()


def scp_from(remote: str, local: Path) -> None:
    """Download file or directory via paramiko SFTP.

    If ``local`` ends with the remote basename (or does not exist yet), a remote
    directory is downloaded *into* ``local`` (not nested as local/basename).
    """
    local.parent.mkdir(parents=True, exist_ok=True)
    print("SFTP", remote, "->", local)
    client, sftp = _sftp_client()
    try:
        try:
            sftp.stat(remote)
        except FileNotFoundError:
            print("missing remote", remote)
            return
        code, out, _ = ssh_exec(f"if [ -d '{remote}' ]; then echo dir; else echo file; fi")
        kind = out.strip()
        remote_name = Path(remote.rstrip("/")).name
        if kind == "dir":
            dest = local
            if local.exists() and local.is_dir() and local.name != remote_name:
                dest = local / remote_name
            dest.mkdir(parents=True, exist_ok=True)
            _sftp_get_dir(sftp, remote, dest)
        else:
            dest = local / remote_name if local.is_dir() else local
            dest.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote, str(dest))
    finally:
        sftp.close()
        client.close()


def _sftp_get_dir(sftp: paramiko.SFTPClient, remote: str, local: Path) -> None:
    local.mkdir(parents=True, exist_ok=True)
    for entry in sftp.listdir_attr(remote):
        rpath = f"{remote.rstrip('/')}/{entry.filename}"
        lpath = local / entry.filename
        if pystat.S_ISDIR(entry.st_mode or 0):
            _sftp_get_dir(sftp, rpath, lpath)
        else:
            # Skip giant chunk intermediates under head_reprojection/chunks when possible
            if "chunks" in Path(rpath).parts and entry.filename.endswith(
                (".mp4", ".jpg", ".png", ".h265")
            ):
                continue
            lpath.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(rpath, str(lpath))


def build_manifest(key: str) -> Path:
    meta = DATASETS[key]
    local_root = DATA_BASE / meta["local_name"]
    out_dir = local_root / "multiview_3d_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "aligned_manifest.jsonl"
    if manifest.exists() and manifest.stat().st_size > 1000:
        print("manifest exists", manifest)
        return manifest
    config = JP / "configs" / meta["config"]
    cmd = [
        sys.executable,
        str(JP / "build_aligned_multiview_manifest.py"),
        "--data-root",
        str(local_root),
        "--config",
        str(config),
        "--output",
        str(manifest),
    ]
    print("building manifest", key)
    subprocess.run(cmd, check=True, cwd=str(LOCAL_REPO))
    return manifest


def upload_repo() -> None:
    ssh_exec(
        f"mkdir -p {REPO_REMOTE}/test_code/joint_projection/configs "
        f"{REPO_REMOTE}/test_code/calibrate/parameters/intrinsics/handle01 "
        f"{REPO_REMOTE}/test_code/calibrate/parameters/intrinsics/handle "
        f"{REPO_REMOTE}/test_code/calibrate/parameters/intrinsics/head"
    )
    for rel in SCRIPT_FILES:
        local = JP / rel
        remote = f"{REPO_REMOTE}/test_code/joint_projection/{rel.replace(chr(92), '/')}"
        scp_to(local, remote)
    for rel in CALIB_FILES:
        local = LOCAL_REPO / rel
        remote = f"{REPO_REMOTE}/{rel}"
        scp_to(local, remote)
    ssh_exec(
        f"sed -i 's/\\r$//' {REPO_REMOTE}/test_code/joint_projection/run_0806_limb_dataset.sh; "
        f"chmod +x {REPO_REMOTE}/test_code/joint_projection/run_0806_limb_dataset.sh"
    )


def _remote_size(remote: str) -> int:
    code, out, _ = ssh_exec(f"stat -c %s '{remote}' 2>/dev/null || echo 0")
    try:
        return int((out.strip() or "0").splitlines()[-1])
    except ValueError:
        return 0


def upload_if_needed(local: Path, remote: str) -> None:
    if not local.exists():
        raise FileNotFoundError(local)
    remote_size = _remote_size(remote)
    local_size = local.stat().st_size
    if remote_size == local_size:
        print("skip upload", local.name)
        return
    if remote_size > 0:
        print(f"size mismatch {local.name}: remote={remote_size} local={local_size}; re-uploading")
        ssh_exec(f"rm -f '{remote}'")
    scp_to(local, remote)


def upload_dataset_metadata(key: str) -> None:
    meta = DATASETS[key]
    local_root = DATA_BASE / meta["local_name"]
    remote_root = f"{BATCH}/{key}/data_root"
    head = meta["head"]
    ssh_exec(
        f"mkdir -p {remote_root}/aligned_data {remote_root}/{head} "
        f"{remote_root}/multiview_3d_results "
        f"{BATCH}/{key}/input/module01 {BATCH}/{key}/input/module02 "
        f"{BATCH}/{key}/logs {BATCH}/{key}/inference"
    )
    upload_if_needed(
        local_root / "aligned_data" / "aligned_30hz.csv",
        f"{remote_root}/aligned_data/aligned_30hz.csv",
    )
    upload_if_needed(
        local_root / "aligned_data" / "aligned_30hz_report.json",
        f"{remote_root}/aligned_data/aligned_30hz_report.json",
    )
    upload_if_needed(
        local_root / "multiview_3d_results" / "aligned_manifest.jsonl",
        f"{remote_root}/multiview_3d_results/aligned_manifest.jsonl",
    )
    report = local_root / "multiview_3d_results" / "aligned_manifest_report.json"
    if report.exists():
        upload_if_needed(report, f"{remote_root}/multiview_3d_results/aligned_manifest_report.json")
    for name in (
        "timestamps.csv",
        "module01_D45D2E00_CAM_A.h265",
        "module01_D45D2E00_CAM_D.h265",
    ):
        upload_if_needed(local_root / head / name, f"{remote_root}/{head}/{name}")


def upload_videos(key: str) -> None:
    """Upload external mjpeg videos; run up to 2 parallel SFTP puts for LAN throughput."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    meta = DATASETS[key]
    local_ext = DATA_BASE / meta["local_name"] / meta["external"]
    remote = f"{BATCH}/{key}/input"
    pairs = [
        (local_ext / "external_01" / "left_CAM_A_1920x1200_30fps.mjpeg", f"{remote}/module01/"),
        (local_ext / "external_01" / "right_CAM_D_1920x1200_30fps.mjpeg", f"{remote}/module01/"),
        (local_ext / "external_02" / "left_CAM_A_1920x1200_30fps.mjpeg", f"{remote}/module02/"),
        (local_ext / "external_02" / "right_CAM_D_1920x1200_30fps.mjpeg", f"{remote}/module02/"),
    ]
    ssh_exec(f"mkdir -p {remote}/module01 {remote}/module02")

    def _one(local: Path, rdir: str) -> str:
        rpath = rdir + local.name
        remote_size = _remote_size(rpath)
        local_size = local.stat().st_size
        if remote_size == local_size:
            return f"skip {key} {local.name}"
        if remote_size > 0:
            print(
                f"incomplete remote {key} {local.name}: "
                f"{remote_size}/{local_size}; re-uploading",
                flush=True,
            )
            ssh_exec(f"rm -f '{rpath}'")
        scp_to(local, rpath)
        got = _remote_size(rpath)
        if got != local_size:
            raise RuntimeError(
                f"upload size mismatch {rpath}: remote={got} local={local_size}"
            )
        return f"done {key} {local.name}"

    # Sequential scp: parallel writers race on the same remote paths when
    # multiple controllers restart, and also thrash LAN bandwidth.
    for local, rdir in pairs:
        print(_one(local, rdir), flush=True)


def launch_pipeline(key: str) -> None:
    status = remote_status(key)
    # Avoid double-launch if a prior controller/agent already started this dataset.
    runningish = {
        "starting",
        "rtmpose_running",
        "rtmpose_done",
        "triangulate_running",
        "triangulate_done",
        "replace_running",
        "replace_done",
        "nose_detect_running",
        "nose_detect_done",
        "optimize_running",
        "optimize_done",
        "render_running",
        "render_done",
    }
    if status == "done":
        print("skip launch", key, "(already done)")
        return
    if status in runningish:
        code, out, _ = ssh_exec(
            f"pgrep -af 'run_0806_limb_dataset.sh {key}' || true"
        )
        if out.strip():
            print("skip launch", key, f"(already {status})")
            return
    script = f"{REPO_REMOTE}/test_code/joint_projection/run_0806_limb_dataset.sh"
    log = f"{BATCH}/{key}/logs/nohup_pipeline.log"
    cmd = f"nohup bash {script} {key} >{log} 2>&1 </dev/null & echo $!"
    code, out, err = ssh_exec(cmd)
    print("launched", key, "pid", out.strip(), err)


def remote_status(key: str) -> str:
    code, out, _ = ssh_exec(f"cat {BATCH}/{key}/STATUS.txt 2>/dev/null || echo missing")
    return out.strip()


def wait_pipeline(key: str, poll_s: int = 60) -> str:
    while True:
        status = remote_status(key)
        print(time.strftime("%H:%M:%S"), key, status)
        if status in {
            "done",
            "rtmpose_failed",
            "rtmpose_missing_toes",
            "triangulate_failed",
            "nose_detect_failed",
            "render_failed",
            "missing_manifest",
            "missing_videos",
            "missing_head",
        }:
            return status
        if status.endswith("_failed"):
            return status
        time.sleep(poll_s)


def pull_results(key: str) -> Path:
    meta = DATASETS[key]
    local_root = DATA_BASE / meta["local_name"] / "multiview_3d_results"
    local_full = local_root / "full"
    if local_full.exists():
        try:
            shutil.rmtree(local_full)
        except PermissionError:
            # Another puller may hold files open; overwrite in place.
            print("rmtree busy; pulling into existing", local_full)
    local_full.mkdir(parents=True, exist_ok=True)
    remote_full = f"{BATCH}/{key}/data_root/multiview_3d_results/full"
    # Pull key artifacts (not giant chunk intermediates if avoidable)
    for rel in [
        "multiview_3d_results.jsonl",
        "multiview_3d_results_limb_gt.jsonl",
        "multiview_3d_report.json",
        "multiview_3d.csv",
        "skeleton_playback.json",
        f"replace_{meta['replace']}_report.json",
        "head_reprojection",
    ]:
        remote = f"{remote_full}/{rel}"
        code, _, _ = ssh_exec(f"test -e {remote}")
        if code != 0:
            print("missing remote", remote)
            continue
        dest = local_full / rel
        if rel == "head_reprojection":
            scp_from(remote, dest)
        else:
            scp_from(remote, dest)
    # Also pull inference jsonl files
    local_inf = local_root / "inference"
    local_inf.mkdir(parents=True, exist_ok=True)
    code, listing, _ = ssh_exec(f"ls {BATCH}/{key}/inference/*.jsonl 2>/dev/null || true")
    for remote_file in listing.split():
        scp_from(remote_file, local_inf / Path(remote_file).name)
    return local_full


def write_summary(results: dict) -> Path:
    path = DATA_BASE / "OVERNIGHT_SUMMARY_踝腕.md"
    meta_info = results.get("_meta") or {}
    tick_req = "status==1 AND raw_tick_valid==1"
    lines = [
        "# 0806 踝 / 腕 Overnight SUMMARY",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Physical convention (confirmed)",
        "",
        "Limb tip from mocap rigid is in **rigid/body local frame**, NOT world axes:",
        "",
        "```",
        "p_world = R_world_rigid @ [0, 0, z_offset_m] + t_world_rigid",
        "```",
        "",
        f"- tip_world requires `{tick_req}`",
        "- **腕 (wrist)**: `z_offset_mm = -60` (rigid-local -Z)",
        "- **踝 (ankle)**: `z_offset_mm = -80` (rigid-local -Z)",
        "- Joint fit anchors: nose + left + right GT (more than 1 nose-only)",
        "- Mapping: left -> `mocap_CH3_06`, right -> `mocap_CH3_07`",
        "",
        "## Toe / foot joints",
        "",
        "- Detector: external 4-cam **RTMW WholeBody** (COCO-WholeBody feet, indices 17–22).",
        "- Added joints: `left/right_big_toe`, `left/right_small_toe`, `left/right_heel`",
        "  (plus aliases `left_toe`/`right_toe` = big toe).",
        "- Toes stay from multiview triangulation (not replaced by mocap).",
        "",
        "## Dataset status",
        "",
    ]
    for key in ("wrist", "ankle"):
        if key not in results:
            continue
        info = results[key]
        meta = DATASETS[key]
        lines += [
            f"### {meta['local_name']} (`{key}`)",
            "",
            f"- Status: **{info.get('status', 'unknown')}**",
            f"- Frames (aligned): {info.get('frames', '?')}",
            f"- z_offset_mm (rigid-local): **{meta['z_offset_mm']}**",
            f"- Local results: `{info.get('local_full', '')}`",
            f"- Replace report: `{info.get('replace_report', '')}`",
            f"- Nose opt report: `{info.get('nose_report', '')}`",
            "",
        ]
        if info.get("replace_counts"):
            lines.append(
                "- Replace counts: `"
                + json.dumps(info["replace_counts"], ensure_ascii=False)
                + "`"
            )
            lines.append("")
        if info.get("nose_metrics"):
            lines.append(
                "- Nose metrics: `"
                + json.dumps(info["nose_metrics"], ensure_ascii=False)
                + "`"
            )
            lines.append("")
        if info.get("notes"):
            for note in info["notes"]:
                lines.append(f"- {note}")
            lines.append("")
        if info.get("artifacts"):
            lines.append("Key artifacts:")
            preferred = [
                a
                for a in info["artifacts"]
                if a.endswith((".mp4", ".html"))
                or a.endswith("report.json")
                or a.endswith("skeleton_playback.json")
                or a.endswith("multiview_3d_results_limb_gt.jsonl")
            ]
            for a in (preferred or info["artifacts"][:20])[:30]:
                lines.append(f"- `{a}`")
            lines.append("")
    lines += [
        "## Caveats / overnight continuity",
        "",
        "- GPU host: `gaoweijian@192.168.20.221`, env `sapiens2`, CUDA ONNX `LD_LIBRARY_PATH`.",
        "- Head video: elementary `.h265` + `timestamps.csv` (not remuxed mp4).",
        "- Face policy: nose only.",
        "- Local overnight controller died overnight (SSHException / host sleep); wrist finished remotely,",
        "  ankle MJPEG upload was interrupted mid-transfer and resumed next morning after killing",
        "  competing local `scp`/`finish_ankle*` processes that raced the same remote files.",
        "- Compute only on gwj; local machine used for upload/monitor/pull.",
        "",
    ]
    if meta_info.get("caveat"):
        lines.append(f"- Extra: {meta_info['caveat']}")
        lines.append("")
    if meta_info.get("elapsed_min") is not None:
        lines.append(f"- Finish-controller elapsed: {meta_info['elapsed_min']} min")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    print("SUMMARY ->", path)
    return path



def collect_local_metrics(key: str, local_full: Path) -> dict:
    meta = DATASETS[key]
    out: dict = {"local_full": str(local_full), "notes": [], "artifacts": []}
    rep = local_full / f"replace_{meta['replace']}_report.json"
    if rep.exists():
        data = json.loads(rep.read_text(encoding="utf-8"))
        out["replace_report"] = str(rep)
        counts = dict(data.get("counts") or {})
        fit = data.get("fit") or {}
        if "offset_norm_mm" in fit:
            counts["3anchor_offset_norm_mm"] = fit["offset_norm_mm"]
        out["replace_counts"] = counts
        out["frames"] = counts.get("frames") or data.get("frames")
        out["notes"].append(
            f"offset_frame={data.get('offset_frame')} z={data.get('z_offset_mm_rigid_local')} "
            f"formula={data.get('formula')} "
            f"fit_offset_norm_mm={fit.get('offset_norm_mm')} "
            f"fit_inliers={fit.get('inliers')}/{fit.get('samples')}"
        )
    nose = local_full / "head_reprojection" / "nose_offset_opt" / "report.json"
    if nose.exists():
        data = json.loads(nose.read_text(encoding="utf-8"))
        out["nose_report"] = str(nose)
        out["nose_metrics"] = {
            "chosen_offset_source": data.get("chosen_offset_source"),
            "final_offset_norm_mm": data.get("final_offset_norm_mm"),
            "final_offset_rigid_m": data.get("final_offset_rigid_m"),
            "layer1": data.get("layer1_3d_external_vs_rigid_gt"),
            "layer2": data.get("layer2_rtmw_2d_refine"),
        }
    for path in local_full.rglob("*"):
        if path.suffix.lower() in {".mp4", ".json", ".jsonl", ".csv"} and path.is_file():
            if "chunks" in path.parts:
                continue
            out["artifacts"].append(str(path))
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=["wrist", "ankle"])
    p.add_argument("--skip-upload-videos", action="store_true")
    p.add_argument("--skip-launch", action="store_true")
    p.add_argument("--only-pull", action="store_true")
    p.add_argument("--poll-s", type=int, default=90)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    results: dict = {}
    if not args.only_pull:
        for key in args.datasets:
            build_manifest(key)
        upload_repo()
        for key in args.datasets:
            upload_dataset_metadata(key)
            if not args.skip_upload_videos:
                upload_videos(key)
        if not args.skip_launch:
            # Launch sequentially: wrist then ankle (GPU-friendly).
            for key in args.datasets:
                launch_pipeline(key)
                status = wait_pipeline(key, poll_s=args.poll_s)
                results[key] = {"status": status}
                if status != "done":
                    results[key]["notes"] = [f"pipeline ended with {status}"]
                    # still try pull whatever exists
                local_full = pull_results(key)
                results[key].update(collect_local_metrics(key, local_full))
                results[key]["status"] = status
    else:
        for key in args.datasets:
            status = remote_status(key)
            local_full = pull_results(key)
            results[key] = {"status": status}
            results[key].update(collect_local_metrics(key, local_full))
    summary = write_summary(results)
    print(json.dumps({k: v.get("status") for k, v in results.items()}, ensure_ascii=False))
    print("SUMMARY", summary)


if __name__ == "__main__":
    main()
