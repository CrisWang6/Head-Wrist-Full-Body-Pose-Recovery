#!/usr/bin/env python3
"""Sync model-training code from gwj EgoRear + HearWristCam to Code repos.

Sources:
  - gwj:/home/gaoweijian/EgoRear_w_hand  (training scripts, src, configs)
  - HearWristCam test_code/joint_projection (orchestration, eval, viz, tunnels)
  - gwj:/home/gaoweijian/0810dataset/README.md (0810 training notes)

Destinations (both updated):
  - C:\\Users\\hand\\Desktop\\Code\\prediction_model_training
  - C:\\Users\\hand\\Desktop\\Code\\Head-Wrist-Full-Body-Pose-Recovery\\prediction_model_training
  - C:\\Users\\hand\\Desktop\\Code\\data_processing
  - C:\\Users\\hand\\Desktop\\Code\\Head-Wrist-Full-Body-Pose-Recovery\\data_processing
"""
from __future__ import annotations

import argparse
import stat
from pathlib import Path

import paramiko

HOST = "192.168.20.221"
USER = "gaoweijian"
PASSWORD = "gwj@#@2026"
EGO_REMOTE = "/home/gaoweijian/EgoRear_w_hand"
JP_REMOTE_0810 = "/home/gaoweijian/0810_batch/repo/test_code/joint_projection"
JP_REMOTE_0806 = "/home/gaoweijian/0806_batch/repo/test_code/joint_projection"

SRC_JP = Path(r"C:\Users\hand\Desktop\HearWristCam\test_code\joint_projection")

PRED_DEST_ROOTS = [
    Path(r"C:\Users\hand\Desktop\Code\prediction_model_training"),
    Path(
        r"C:\Users\hand\Desktop\Code\Head-Wrist-Full-Body-Pose-Recovery\prediction_model_training"
    ),
]

DATA_DEST_ROOTS = [
    Path(r"C:\Users\hand\Desktop\Code\data_processing"),
    Path(
        r"C:\Users\hand\Desktop\Code\Head-Wrist-Full-Body-Pose-Recovery\data_processing"
    ),
]

# EgoRear subset (exclude checkpoints/logs/outputs/data/.git)
EGO_PULL_PATHS = [
    "pyproject.toml",
    "README.md",
    "REAL_0717_HEAD2CAM.md",
    "REAL_0722_01_DIRECT2D.md",
    "scripts",
    "experiments/stage2_refinement",
    "experiments/stage3_pose3d",
    "experiments/multiview_refinement",
    "src/egorear_sim2d",
    "configs",
]

JP_TRAINING_FILES = [
    # 0810 orchestration (may overlap production sync; idempotent)
    "run_0810_training_master.sh",
    "run_0810_training_prep.sh",
    "run_0810_resume_stage1.sh",
    "reexport_0810_heatmap_labels.sh",
    "build_0810_pack_splits.py",
    "prepare_0810_pose3d_labels.py",
    "prepare_0806_pose3d_labels.py",
    "constants_0810_training.py",
    "constants_0806_training.py",
    "export_0806_heatmap_labels.py",
    "extract_0806_head_frames.py",
    "_patch_train_pose3d_alignment.py",
    "_launch_0810_full_pipeline.py",
    "_resume_0810_pipeline_gwj.py",
    "_launch_0810_training_prep_gwj.py",
    "_launch_0810_resume_stage1_gwj.py",
    "_cleanup_0810_disk_gwj.py",
    "_monitor_0810_gwj.py",
    "_stop_training_check_0810.py",
    # 0806 weekend orchestration
    "run_weekend_training_master.sh",
    "run_weekend_stage23_0806.sh",
    "run_weekend_master_resume.sh",
    "run_task2_bc_labels_gwj.sh",
    "monitor_0806_weekend_gwj.py",
    # stage1/2/3 launch + eval + viz
    "run_stage1_v31_only_gwj.sh",
    "run_stage1_v32_only_gwj.sh",
    "watch_stage1_v31_gwj.py",
    "eval_0806_stage1_test.py",
    "render_0806_stage1_test_heatmaps.py",
    "render_0806_stage1_test_points.py",
    "eval_stage3_test_3d_viz.py",
    "render_stage3_dual_skeleton_yaw.py",
    "rerender_stage3_aligned_viz.py",
    "skeleton_3d_filter.py",
    # gwj remote launch helpers (paramiko)
    "_apply_train_patch_gwj.py",
    "_patch_train_heatmap_gwj.py",
    "_fix_and_rerender_stage3_viz.py",
    "_launch_stage1_v31_watch_gwj.py",
    "_launch_stage1_v32_gwj.py",
    "_launch_stage2_v31_only_gwj.py",
    "_launch_stage3_v31_gwj.py",
    "_launch_stage3_v31_aligned_gwj.py",
    "_run_stage3_test_viz_gwj.py",
    "_run_stage3_aligned_test_viz_gwj.py",
    "_run_stage3_aligned_train_wu_viz_gwj.py",
    "_check_stage3_aligned_gwj.py",
    # TensorBoard SSH tunnels
    "_tb_tunnel_local.py",
    "_tb_tunnel_stage2_v31.py",
    "_tb_tunnel_stage3_v31.py",
    "_tb_tunnel_stage3_v31_aligned.py",
    "_probe_gwj_training.py",
    "_sync_training_to_code.py",
]

JP_CONFIG_FILES = [
    "configs/joint_radius_px_120x75_delivery15.json",
]

PATH_REPLACEMENTS = [
    (
        "test_code/calibrate/parameters/intrinsics/",
        "data_processing/configs/calibration/intrinsics/",
    ),
    (
        "test_code/joint_projection/",
        "data_processing/scripts/joint_projection/",
    ),
]

GWJ_UPLOAD_JP_FILES = [
    "eval_stage3_test_3d_viz.py",
    "render_stage3_dual_skeleton_yaw.py",
    "rerender_stage3_aligned_viz.py",
    "skeleton_3d_filter.py",
    "_fix_and_rerender_stage3_viz.py",
    "_run_stage3_aligned_test_viz_gwj.py",
    "_run_stage3_aligned_train_wu_viz_gwj.py",
    "_check_stage3_aligned_gwj.py",
    "_tb_tunnel_stage3_v31_aligned.py",
    "_stop_training_check_0810.py",
]


def patch_python_for_code_layout(text: str) -> str:
    text = text.replace(
        "REPO_ROOT = SCRIPT_DIR.parents[2]",
        "REPO_ROOT = SCRIPT_DIR.parents[2]",
    )
    return text


def patch_config_json(text: str) -> str:
    for old, new in PATH_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def connect() -> tuple[paramiko.SSHClient, paramiko.SFTPClient]:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(HOST, username=USER, password=PASSWORD, timeout=30)
    return client, client.open_sftp()


def sftp_is_dir(sftp: paramiko.SFTPClient, path: str) -> bool:
    try:
        return stat.S_ISDIR(sftp.stat(path).st_mode)
    except OSError:
        return False


def pull_egorear(sftp: paramiko.SFTPClient, dest_root: Path) -> list[str]:
    changed: list[str] = []

    def pull_file(remote: str, local: Path) -> None:
        local.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote, str(local))
        changed.append(str(local.relative_to(dest_root)))

    def pull_tree(remote_dir: str, local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        for entry in sftp.listdir_attr(remote_dir):
            name = entry.filename
            if name in {"__pycache__", ".git"}:
                continue
            remote_path = f"{remote_dir}/{name}"
            local_path = local_dir / name
            if stat.S_ISDIR(entry.st_mode):
                pull_tree(remote_path, local_path)
            else:
                pull_file(remote_path, local_path)

    for rel in EGO_PULL_PATHS:
        remote = f"{EGO_REMOTE}/{rel}"
        local = dest_root / rel
        try:
            if sftp_is_dir(sftp, remote):
                pull_tree(remote, local)
            else:
                pull_file(remote, local)
        except FileNotFoundError:
            print(f"  SKIP missing remote {remote}")
        except OSError as exc:
            print(f"  SKIP {remote}: {exc}")

    return changed


def sync_jp_training(dest_root: Path) -> list[str]:
    changed: list[str] = []
    jp_dest = dest_root / "scripts" / "joint_projection"
    jp_dest.mkdir(parents=True, exist_ok=True)

    for rel in JP_TRAINING_FILES:
        src = SRC_JP / rel
        if not src.is_file():
            print(f"  SKIP missing local {src}")
            continue
        dst = jp_dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        content = src.read_text(encoding="utf-8")
        if src.suffix == ".py":
            content = patch_python_for_code_layout(content)
        if src.suffix == ".sh":
            content = content.replace("\r\n", "\n")
        dst.write_text(content, encoding="utf-8", newline="\n")
        changed.append(str(dst.relative_to(dest_root.parent if dest_root.name == "data_processing" else dest_root)))

    for rel in JP_CONFIG_FILES:
        src = SRC_JP / rel
        if not src.is_file():
            continue
        dst = jp_dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        content = patch_config_json(src.read_text(encoding="utf-8"))
        dst.write_text(content, encoding="utf-8", newline="\n")
        changed.append(str(dst))

    return changed


def pull_0810_readme(sftp: paramiko.SFTPClient, dest_root: Path) -> list[str]:
    changed: list[str] = []
    docs = dest_root / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    remote = "/home/gaoweijian/0810dataset/README.md"
    local = docs / "TRAINING_0810_README.md"
    try:
        sftp.get(remote, str(local))
        changed.append(str(local.relative_to(dest_root.parent if dest_root.name == "data_processing" else dest_root)))
    except OSError as exc:
        print(f"  SKIP 0810 README: {exc}")
    return changed


def upload_jp_to_gwj(sftp: paramiko.SFTPClient, remote_jp: str) -> list[str]:
    uploaded: list[str] = []
    for name in GWJ_UPLOAD_JP_FILES:
        local = SRC_JP / name
        if not local.is_file():
            continue
        remote = f"{remote_jp}/{name}"
        sftp.put(str(local), remote)
        uploaded.append(remote)
    return uploaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-gwj-pull", action="store_true")
    parser.add_argument("--skip-gwj-upload", action="store_true")
    args = parser.parse_args()

    client = None
    sftp = None
    if not args.skip_gwj_pull or not args.skip_gwj_upload:
        client, sftp = connect()

    all_changed: dict[str, list[str]] = {}

    for root in PRED_DEST_ROOTS:
        print(f"\n=== EgoRear -> {root} ===")
        if args.skip_gwj_pull:
            print("  (skipped gwj pull)")
            all_changed[str(root)] = []
        else:
            assert sftp is not None
            all_changed[str(root)] = pull_egorear(sftp, root)
            print(f"  {len(all_changed[str(root)])} paths pulled")

    for root in DATA_DEST_ROOTS:
        print(f"\n=== joint_projection training -> {root} ===")
        jp_changed = sync_jp_training(root)
        if not args.skip_gwj_pull:
            assert sftp is not None
            jp_changed.extend(pull_0810_readme(sftp, root))
        all_changed[f"jp:{root}"] = jp_changed
        print(f"  {len(jp_changed)} paths updated")

    if not args.skip_gwj_upload:
        assert sftp is not None
        print("\n=== upload critical JP helpers -> gwj 0810_batch ===")
        up0810 = upload_jp_to_gwj(sftp, JP_REMOTE_0810)
        print(f"  uploaded {len(up0810)} files to 0810")
        print("\n=== upload eval/viz helpers -> gwj 0806_batch ===")
        up0806 = upload_jp_to_gwj(sftp, JP_REMOTE_0806)
        print(f"  uploaded {len(up0806)} files to 0806")

    if client is not None:
        sftp.close()
        client.close()

    total = sum(len(v) for v in all_changed.values())
    print(f"\nDone. {total} total paths updated across destinations.")


if __name__ == "__main__":
    main()
