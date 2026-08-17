#!/usr/bin/env python3
"""Full sync: gaoweijian host + HearWristCam -> Desktop/Code (+ monorepo mirror)."""
from __future__ import annotations

import argparse
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import paramiko

HOST = "192.168.20.221"
USER = "gaoweijian"
PASSWORD = "gwj@#@2026"

CODE = Path(r"C:\Users\hand\Desktop\Code")
HW = Path(r"C:\Users\hand\Desktop\HearWristCam")
MONO = CODE / "Head-Wrist-Full-Body-Pose-Recovery"

REMOTE = {
    "jp": "/home/gaoweijian/0810_batch/repo/test_code/joint_projection",
    "calibrate": "/home/gaoweijian/0810_batch/repo/test_code/calibrate",
    "egorear": "/home/gaoweijian/EgoRear_w_hand",
    "simulation": "/home/gaoweijian/Simulation",
}

SKIP_DIR_NAMES = {
    "__pycache__",
    ".git",
    ".codex_backups",
    "checkpoints",
    "logs",
    "outputs",
    "data",
    "venv",
    "weights",
    "smplx_models",
    "input",
    "results",
    "results_valid",
    "node_modules",
}

SKIP_FILE_SUFFIXES = {".pyc", ".pt", ".pth", ".ckpt", ".npz", ".mp4", ".h265", ".bag", ".jpg", ".png"}

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

EGO_PULL = [
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

# Remote Simulation uses `config/`; local repo standard is `configs/`.
SIM_PULL = ["pyproject.toml", "README.md", "config", "scripts", "src", "tests", "docs", ".gitignore"]

HW_JP_EXTRA = [
    "imu_mahony.py",
    "export_imu_wrist_overlay.py",
    "compare_head_imu_mocap_orientation.py",
    "validate_head_imu_mocap_sync.py",
    "scripts/build_skeleton_canvas.py",
]


def patch_python_for_code_layout(text: str) -> str:
    return text.replace(
        "REPO_ROOT = SCRIPT_DIR.parents[1]",
        "REPO_ROOT = SCRIPT_DIR.parents[2]",
    )


def patch_config_json(text: str) -> str:
    for old, new in PATH_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def should_skip(name: str, is_dir: bool) -> bool:
    if name in SKIP_DIR_NAMES:
        return True
    if "thuman" in name.lower():
        return True
    if not is_dir:
        low = name.lower()
        for suf in SKIP_FILE_SUFFIXES:
            if low.endswith(suf):
                return True
    return False


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


def pull_tree(sftp: paramiko.SFTPClient, remote_dir: str, local_dir: Path, changed: list[str]) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for entry in sftp.listdir_attr(remote_dir):
        name = entry.filename
        if should_skip(name, stat.S_ISDIR(entry.st_mode)):
            continue
        remote_path = f"{remote_dir}/{name}"
        local_path = local_dir / name
        if stat.S_ISDIR(entry.st_mode):
            pull_tree(sftp, remote_path, local_path, changed)
        else:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote_path, str(local_path))
            changed.append(str(local_path))


def pull_paths(sftp: paramiko.SFTPClient, remote_root: str, rel_paths: list[str], local_root: Path) -> list[str]:
    changed: list[str] = []
    for rel in rel_paths:
        remote = f"{remote_root}/{rel}"
        local = local_root / rel
        try:
            if sftp_is_dir(sftp, remote):
                pull_tree(sftp, remote, local, changed)
            else:
                local.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(remote, str(local))
                changed.append(str(local))
        except OSError as exc:
            print(f"  SKIP {remote}: {exc}")
    return changed


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    content = src.read_text(encoding="utf-8")
    if src.suffix == ".py":
        content = patch_python_for_code_layout(content)
    elif src.suffix == ".json":
        content = patch_config_json(content)
    elif src.suffix == ".sh":
        content = content.replace("\r\n", "\n")
    dst.write_text(content, encoding="utf-8", newline="\n")


def mirror_repo_subdir(src_root: Path, mono_sub: str) -> list[str]:
    dst_root = MONO / mono_sub
    changed: list[str] = []
    if not src_root.is_dir():
        return changed
    for path in src_root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src_root)
        parts = rel.parts
        if any(p in SKIP_DIR_NAMES for p in parts):
            continue
        if path.suffix.lower() in SKIP_FILE_SUFFIXES:
            continue
        dst = dst_root / rel
        if not dst.exists() or path.read_bytes() != dst.read_bytes():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dst)
            changed.append(str(dst))
    return changed


def run_local_sync_scripts() -> None:
    prod = HW / "test_code" / "joint_projection" / "_sync_production_pipeline_to_code.py"
    train = HW / "test_code" / "joint_projection" / "_sync_training_to_code.py"
    for script in (prod, train):
        if script.is_file():
            print(f"\n=== run {script.name} ===")
            subprocess.run([sys.executable, str(script)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-local-scripts", action="store_true")
    args = parser.parse_args()

    if not args.skip_local_scripts:
        run_local_sync_scripts()

    client, sftp = connect()
    summary: dict[str, int] = {}

    # Remote joint_projection -> data_processing
    jp_dest = CODE / "data_processing" / "scripts" / "joint_projection"
    print(f"\n=== pull remote joint_projection -> {jp_dest} ===")
    jp_changed: list[str] = []
    pull_tree(sftp, REMOTE["jp"], jp_dest, jp_changed)
    print(f"  {len(jp_changed)} files pulled")
    summary["remote_jp"] = len(jp_changed)

    # Remote calibrate parameters -> configs
    cal_remote = f"{REMOTE['calibrate']}/parameters/intrinsics"
    cal_dest = CODE / "data_processing" / "configs" / "calibration" / "intrinsics"
    print(f"\n=== pull remote calibrate -> {cal_dest} ===")
    cal_changed: list[str] = []
    if sftp_is_dir(sftp, cal_remote):
        pull_tree(sftp, cal_remote, cal_dest, cal_changed)
    print(f"  {len(cal_changed)} files pulled")
    summary["remote_cal"] = len(cal_changed)

    # 0810 training readme
    docs = CODE / "data_processing" / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    try:
        sftp.get("/home/gaoweijian/0810dataset/README.md", str(docs / "TRAINING_0810_README.md"))
        summary["0810_readme"] = 1
    except OSError as exc:
        print(f"  SKIP 0810 README: {exc}")
        summary["0810_readme"] = 0

    # HearWristCam IMU / canvas extras (may be newer than remote)
    hw_jp = HW / "test_code" / "joint_projection"
    hw_changed = 0
    print("\n=== overlay HearWristCam IMU/canvas extras ===")
    for rel in HW_JP_EXTRA:
        src = hw_jp / rel
        if not src.is_file():
            print(f"  SKIP missing {src}")
            continue
        dst = jp_dest / rel
        copy_file(src, dst)
        hw_changed += 1
    summary["hw_jp_extra"] = hw_changed

    # EgoRear -> prediction_model_training
    pred = CODE / "prediction_model_training"
    print(f"\n=== pull EgoRear -> {pred} ===")
    ego_changed = pull_paths(sftp, REMOTE["egorear"], EGO_PULL, pred)
    print(f"  {len(ego_changed)} paths pulled")
    summary["egorear"] = len(ego_changed)

    # Simulation -> Issacsim_data_generation (remote `config/` -> local `configs/`)
    sim = CODE / "Issacsim_data_generation"
    print(f"\n=== pull Simulation -> {sim} ===")
    sim_changed = pull_paths(sftp, REMOTE["simulation"], SIM_PULL, sim)
    # Rename pulled config/ to configs/ if needed
    pulled_config = sim / "config"
    configs_dir = sim / "configs"
    if pulled_config.is_dir():
        configs_dir.mkdir(parents=True, exist_ok=True)
        for item in pulled_config.iterdir():
            dest = configs_dir / item.name
            if item.is_dir():
                shutil.copytree(item, dest, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dest)
            sim_changed.append(str(dest))
        shutil.rmtree(pulled_config)
    print(f"  {len(sim_changed)} paths pulled")
    summary["simulation"] = len(sim_changed)

    sftp.close()
    client.close()

    # Mirror standalone repos into monorepo
    print("\n=== mirror into Head-Wrist-Full-Body-Pose-Recovery ===")
    for sub, name in [
        (CODE / "data_processing", "data_processing"),
        (CODE / "prediction_model_training", "prediction_model_training"),
        (CODE / "Issacsim_data_generation", "Issacsim_data_generation"),
        (CODE / "OAK_test", "OAK_test"),
    ]:
        n = len(mirror_repo_subdir(sub, name))
        summary[f"mirror_{name}"] = n
        print(f"  {name}: {n} files updated")

    print("\n=== summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\nDone.")


if __name__ == "__main__":
    main()
