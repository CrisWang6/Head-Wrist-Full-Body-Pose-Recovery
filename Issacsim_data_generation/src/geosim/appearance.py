from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class AppearanceAsset:
    subject: str
    source_dir: Path
    asset_dir: Path
    smplx_mesh: Path
    smplx_params: Path
    textured_mesh: Path | None
    render_status: str


def prepare_icon_appearance_library(
    *,
    smplx_root: Path,
    output_root: Path,
    count: int = 20,
    seed: int = 20260609,
    subjects: tuple[str, ...] = (),
    thuman_scans_root: Path | None = None,
    icon_root: Path | None = None,
    icon_python: str = "",
    run_icon_render: bool = False,
    overwrite: bool = False,
    strict_textures: bool = False,
) -> dict[str, object]:
    smplx_root = smplx_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not smplx_root.exists():
        raise FileNotFoundError(f"SMPL-X appearance root does not exist: {smplx_root}")

    selected = select_smplx_subjects(smplx_root, count=count, seed=seed, subjects=subjects)
    output_root.mkdir(parents=True, exist_ok=True)

    icon_render = None
    if run_icon_render:
        icon_render = run_icon_render_batch(
            icon_root=icon_root,
            output_root=output_root,
            thuman_scans_root=thuman_scans_root,
            icon_python=icon_python,
        )

    assets: list[AppearanceAsset] = []
    for source_dir in selected:
        asset = prepare_one_appearance_asset(
            source_dir=source_dir,
            output_root=output_root,
            thuman_scans_root=thuman_scans_root,
            overwrite=overwrite,
            strict_textures=strict_textures,
        )
        assets.append(asset)

    manifest = {
        "created_at_unix": time.time(),
        "smplx_root": str(smplx_root),
        "output_root": str(output_root),
        "requested_count": int(count),
        "seed": int(seed),
        "selected_subjects": [asset.subject for asset in assets],
        "icon_root": str(icon_root.expanduser().resolve()) if icon_root else "",
        "thuman_scans_root": str(thuman_scans_root.expanduser().resolve()) if thuman_scans_root else "",
        "icon_render": icon_render,
        "assets": [appearance_asset_to_manifest(asset) for asset in assets],
    }
    (output_root / "appearances_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def select_smplx_subjects(
    smplx_root: Path,
    *,
    count: int,
    seed: int,
    subjects: tuple[str, ...] = (),
) -> list[Path]:
    if subjects:
        selected = []
        for subject in subjects:
            source_dir = smplx_root / subject
            validate_smplx_subject_dir(source_dir)
            selected.append(source_dir)
        return selected

    candidates = sorted(path for path in smplx_root.iterdir() if path.is_dir() and is_smplx_subject_dir(path))
    if not candidates:
        raise FileNotFoundError(f"No SMPL-X subject folders found under {smplx_root}")
    rng = np.random.default_rng(int(seed))
    chosen = rng.choice(len(candidates), size=min(max(1, int(count)), len(candidates)), replace=False)
    return [candidates[int(idx)] for idx in chosen]


def is_smplx_subject_dir(path: Path) -> bool:
    return (path / "mesh_smplx.obj").exists() and (path / "smplx_param.pkl").exists()


def validate_smplx_subject_dir(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Subject folder does not exist: {path}")
    if not is_smplx_subject_dir(path):
        raise FileNotFoundError(f"Subject is missing mesh_smplx.obj or smplx_param.pkl: {path}")


def prepare_one_appearance_asset(
    *,
    source_dir: Path,
    output_root: Path,
    thuman_scans_root: Path | None,
    overwrite: bool,
    strict_textures: bool,
) -> AppearanceAsset:
    validate_smplx_subject_dir(source_dir)
    subject = source_dir.name
    asset_dir = output_root / subject
    if overwrite and asset_dir.exists():
        shutil.rmtree(asset_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)

    smplx_mesh = asset_dir / "mesh_smplx.obj"
    smplx_params = asset_dir / "smplx_param.pkl"
    shutil.copy2(source_dir / "mesh_smplx.obj", smplx_mesh)
    shutil.copy2(source_dir / "smplx_param.pkl", smplx_params)

    textured_source = find_textured_mesh_source(subject=subject, source_dir=source_dir, thuman_scans_root=thuman_scans_root)
    textured_mesh = None
    if textured_source is not None:
        textured_dir = asset_dir / "icon_textured"
        if overwrite and textured_dir.exists():
            shutil.rmtree(textured_dir)
        textured_dir.mkdir(parents=True, exist_ok=True)
        textured_mesh = copy_textured_mesh_bundle(textured_source, textured_dir)
        render_status = "textured_mesh_available"
    else:
        render_status = "fallback_no_texture_source"
        if strict_textures:
            raise FileNotFoundError(
                "No textured THuman/ICON mesh was found for "
                f"{subject}. Provide --thuman-scans-root with original scans or run without --strict-textures."
            )

    asset = AppearanceAsset(
        subject=subject,
        source_dir=source_dir,
        asset_dir=asset_dir,
        smplx_mesh=smplx_mesh,
        smplx_params=smplx_params,
        textured_mesh=textured_mesh,
        render_status=render_status,
    )
    (asset_dir / "asset_manifest.json").write_text(
        json.dumps(appearance_asset_to_manifest(asset), indent=2),
        encoding="utf-8",
    )
    return asset


def find_textured_mesh_source(
    *,
    subject: str,
    source_dir: Path,
    thuman_scans_root: Path | None,
) -> Path | None:
    search_dirs = [source_dir]
    if thuman_scans_root is not None:
        scans_root = thuman_scans_root.expanduser()
        search_dirs.extend(
            path
            for path in (
                scans_root / subject,
                scans_root / f"{int(subject):04d}" if subject.isdigit() else scans_root / subject,
            )
            if path.exists()
        )

    seen: set[Path] = set()
    for folder in search_dirs:
        folder = folder.resolve()
        if folder in seen or not folder.exists():
            continue
        seen.add(folder)
        for obj_path in sorted(folder.rglob("*.obj")):
            if obj_path.name == "mesh_smplx.obj":
                continue
            if obj_has_materials_or_uvs(obj_path):
                return obj_path
        for obj_path in sorted(folder.rglob("*.obj")):
            image_files = [path for path in obj_path.parent.iterdir() if path.suffix.lower() in IMAGE_EXTS]
            if image_files:
                return obj_path
    return None


def obj_has_materials_or_uvs(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            for idx, line in enumerate(file):
                if line.startswith(("mtllib ", "usemtl ", "vt ")):
                    return True
                if idx > 4096:
                    break
    except OSError:
        return False
    return False


def copy_textured_mesh_bundle(obj_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_obj = output_dir / obj_path.name
    shutil.copy2(obj_path, copied_obj)

    mtl_names = parse_mtl_names(obj_path)
    for mtl_name in mtl_names:
        mtl_path = obj_path.parent / mtl_name
        if not mtl_path.exists():
            continue
        shutil.copy2(mtl_path, output_dir / mtl_path.name)
        for texture_name in parse_texture_names(mtl_path):
            texture_path = mtl_path.parent / texture_name
            if texture_path.exists():
                shutil.copy2(texture_path, output_dir / texture_path.name)

    for texture_path in obj_path.parent.iterdir():
        if texture_path.is_file() and texture_path.suffix.lower() in IMAGE_EXTS:
            target = output_dir / texture_path.name
            if not target.exists():
                shutil.copy2(texture_path, target)
    return copied_obj


def parse_mtl_names(obj_path: Path) -> list[str]:
    names = []
    with obj_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            if line.startswith("mtllib "):
                names.extend(part.strip() for part in line.split(maxsplit=1)[1].split() if part.strip())
    return names


def parse_texture_names(mtl_path: Path) -> list[str]:
    names = []
    keys = {"map_Kd", "map_Ka", "map_Ks", "map_Bump", "bump", "disp", "decal"}
    with mtl_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2 and parts[0] in keys:
                names.append(parts[-1])
    return names


def run_icon_render_batch(
    *,
    icon_root: Path | None,
    output_root: Path,
    thuman_scans_root: Path | None,
    icon_python: str,
) -> dict[str, object]:
    if icon_root is None:
        raise FileNotFoundError("--run-icon-render needs --icon-root pointing to a local ICON checkout.")
    icon_root = icon_root.expanduser().resolve()
    if not (icon_root / "scripts" / "render_batch.py").exists():
        raise FileNotFoundError(f"ICON render_batch.py was not found under {icon_root}")
    if thuman_scans_root is None or not thuman_scans_root.expanduser().exists():
        raise FileNotFoundError("--run-icon-render needs --thuman-scans-root with original THuman scan folders.")

    command = [
        icon_python or sys.executable,
        "-m",
        "scripts.render_batch",
        "-headless",
        "-out_dir",
        str(output_root / "icon_render_data"),
    ]
    env = dict(**__import__("os").environ)
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    log_path = output_root / "icon_render_batch.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        subprocess.run(command, cwd=icon_root, env=env, stdout=log_file, stderr=subprocess.STDOUT, check=True)
    return {"command": command, "cwd": str(icon_root), "log": str(log_path)}


def appearance_asset_to_manifest(asset: AppearanceAsset) -> dict[str, object]:
    return {
        "subject": asset.subject,
        "source_dir": str(asset.source_dir),
        "asset_dir": str(asset.asset_dir),
        "smplx_mesh": str(asset.smplx_mesh),
        "smplx_params": str(asset.smplx_params),
        "textured_mesh": str(asset.textured_mesh) if asset.textured_mesh is not None else "",
        "render_status": asset.render_status,
        "animation_status": "smplx_topology" if asset.textured_mesh is None else "textured_source_registered_needs_binding",
    }
