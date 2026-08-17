from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import time
from typing import Iterable

from geosim.appearance import AppearanceAsset, appearance_asset_to_manifest, prepare_one_appearance_asset


SMPLX_REQUIRED_FILES = {"mesh_smplx.obj", "smplx_param.pkl"}
SCAN_MESH_EXTS = {".obj"}


@dataclass(frozen=True)
class SubjectArchiveEntry:
    subject: str
    root: str
    members: tuple[str, ...]


@dataclass(frozen=True)
class PreparedThumanSubject:
    subject: str
    smplx_source: Path
    scan_source: Path | None
    asset: AppearanceAsset


def list_7z_members(archive_path: Path, seven_zip_bin: str = "") -> list[str]:
    archive_path = archive_path.expanduser().resolve()
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)
    if seven_zip_bin:
        return _list_7z_members_cli(archive_path, seven_zip_bin)
    try:
        import py7zr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "py7zr is required when a 7z executable is not provided. "
            "Install with `python -m pip install py7zr` or pass --seven-zip-bin."
        ) from exc
    with py7zr.SevenZipFile(archive_path, mode="r") as archive:
        return sorted(name.replace("\\", "/") for name in archive.getnames())


def _list_7z_members_cli(archive_path: Path, seven_zip_bin: str) -> list[str]:
    result = subprocess.run(
        [seven_zip_bin, "l", "-ba", str(archive_path)],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    members = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 6:
            members.append(" ".join(parts[5:]).replace("\\", "/"))
    return sorted(members)


def build_smplx_subject_index(members: Iterable[str]) -> dict[str, SubjectArchiveEntry]:
    groups: dict[str, set[str]] = {}
    for member in members:
        path = PurePosixPath(member)
        if path.name not in SMPLX_REQUIRED_FILES:
            continue
        root = str(path.parent)
        subject = path.parent.name
        groups.setdefault(subject, set()).add(member)

    index: dict[str, SubjectArchiveEntry] = {}
    for subject, marker_members in groups.items():
        roots = {str(PurePosixPath(member).parent) for member in marker_members}
        for root in sorted(roots):
            names = {PurePosixPath(member).name for member in marker_members if str(PurePosixPath(member).parent) == root}
            if SMPLX_REQUIRED_FILES.issubset(names):
                index[subject] = SubjectArchiveEntry(subject=subject, root=root, members=())
                break
    return index


def build_scan_subject_index(members: Iterable[str]) -> dict[str, SubjectArchiveEntry]:
    roots: dict[str, set[str]] = {}
    for member in members:
        path = PurePosixPath(member)
        if path.suffix.lower() not in SCAN_MESH_EXTS:
            continue
        if path.name == "mesh_smplx.obj":
            continue
        if path.parent == PurePosixPath("."):
            continue
        subject = path.parent.name
        roots.setdefault(subject, set()).add(str(path.parent))

    index: dict[str, SubjectArchiveEntry] = {}
    for subject, subject_roots in roots.items():
        root = sorted(subject_roots, key=lambda value: (value.count("/"), value))[0]
        index[subject] = SubjectArchiveEntry(subject=subject, root=root, members=())
    return index


def select_subjects(
    smplx_index: dict[str, SubjectArchiveEntry],
    *,
    subjects: tuple[str, ...] = (),
    count: int = 1,
) -> list[str]:
    if subjects:
        missing = [subject for subject in subjects if subject not in smplx_index]
        if missing:
            raise FileNotFoundError(f"Subjects are missing from SMPL-X archive: {', '.join(missing)}")
        return list(subjects)
    return sorted(smplx_index)[: max(1, int(count))]


def extract_subject_roots(
    *,
    archive_path: Path,
    entries: Iterable[SubjectArchiveEntry],
    output_root: Path,
    seven_zip_bin: str = "",
) -> dict[str, Path]:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    entries = list(entries)
    if not entries:
        return {}

    if seven_zip_bin:
        return _extract_subject_roots_cli(
            archive_path=archive_path,
            entries=entries,
            output_root=output_root,
            seven_zip_bin=seven_zip_bin,
        )

    try:
        import py7zr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "py7zr is required when a 7z executable is not provided. "
            "Install with `python -m pip install py7zr` or pass --seven-zip-bin."
        ) from exc

    with py7zr.SevenZipFile(archive_path.expanduser().resolve(), mode="r") as archive:
        names = [name.replace("\\", "/") for name in archive.getnames()]
        targets = _members_under_roots(names, [entry.root for entry in entries])
        archive.extract(path=output_root, targets=targets)

    return {entry.subject: output_root / Path(*PurePosixPath(entry.root).parts) for entry in entries}


def _extract_subject_roots_cli(
    *,
    archive_path: Path,
    entries: list[SubjectArchiveEntry],
    output_root: Path,
    seven_zip_bin: str,
) -> dict[str, Path]:
    for entry in entries:
        subprocess.run(
            [seven_zip_bin, "x", "-y", f"-o{output_root}", str(archive_path), f"{entry.root}/*"],
            check=True,
        )
    return {entry.subject: output_root / Path(*PurePosixPath(entry.root).parts) for entry in entries}


def _members_under_roots(members: Iterable[str], roots: Iterable[str]) -> list[str]:
    normalized_roots = tuple(root.rstrip("/") + "/" for root in roots)
    exact_roots = {root.rstrip("/") for root in roots}
    return [
        member
        for member in members
        if member.rstrip("/") in exact_roots or any(member.startswith(root) for root in normalized_roots)
    ]


def prepare_thuman_appearance_subset(
    *,
    scan_archive: Path | None,
    scan_root: Path | None = None,
    smplx_archive: Path,
    extract_root: Path,
    appearance_root: Path,
    subjects: tuple[str, ...] = (),
    count: int = 1,
    seven_zip_bin: str = "",
    overwrite_extract: bool = False,
    overwrite_assets: bool = False,
    strict_textures: bool = False,
) -> dict[str, object]:
    extract_root = extract_root.expanduser().resolve()
    appearance_root = appearance_root.expanduser().resolve()
    smplx_members = list_7z_members(smplx_archive, seven_zip_bin=seven_zip_bin)
    scan_members = list_7z_members(scan_archive, seven_zip_bin=seven_zip_bin) if scan_archive else []
    smplx_index = build_smplx_subject_index(smplx_members)
    scan_index = build_scan_subject_index(scan_members)
    selected = select_subjects(smplx_index, subjects=subjects, count=count)

    if overwrite_extract and extract_root.exists():
        shutil.rmtree(extract_root)
    smplx_extract_root = extract_root / "smplx_paras"
    scan_extract_root = extract_root / "scans"
    smplx_dirs = extract_subject_roots(
        archive_path=smplx_archive,
        entries=[smplx_index[subject] for subject in selected],
        output_root=smplx_extract_root,
        seven_zip_bin=seven_zip_bin,
    )
    scan_dirs = {}
    if scan_root is not None:
        scan_root = scan_root.expanduser().resolve()
        scan_dirs = {subject: scan_root / subject for subject in selected if (scan_root / subject).exists()}
        if not scan_dirs:
            scan_dirs = {subject: scan_root / "model" / subject for subject in selected if (scan_root / "model" / subject).exists()}
    elif scan_archive is not None:
        scan_entries = [scan_index[subject] for subject in selected if subject in scan_index]
        scan_dirs = extract_subject_roots(
            archive_path=scan_archive,
            entries=scan_entries,
            output_root=scan_extract_root,
            seven_zip_bin=seven_zip_bin,
        )

    prepared: list[PreparedThumanSubject] = []
    for subject in selected:
        smplx_source = smplx_dirs[subject]
        scan_source = scan_dirs.get(subject)
        asset = prepare_one_appearance_asset(
            source_dir=smplx_source,
            output_root=appearance_root,
            thuman_scans_root=scan_source.parent if scan_source else scan_extract_root,
            overwrite=overwrite_assets,
            strict_textures=strict_textures,
        )
        prepared.append(
            PreparedThumanSubject(
                subject=subject,
                smplx_source=smplx_source,
                scan_source=scan_source,
                asset=asset,
            )
        )

    manifest = {
        "created_at_unix": time.time(),
        "scan_archive": str(scan_archive.expanduser().resolve()) if scan_archive else "",
        "scan_root": str(scan_root) if scan_root else "",
        "smplx_archive": str(smplx_archive.expanduser().resolve()),
        "extract_root": str(extract_root),
        "appearance_root": str(appearance_root),
        "selected_subjects": selected,
        "smplx_subject_count": len(smplx_index),
        "scan_subject_count": len(scan_index),
        "prepared": [
            {
                "subject": item.subject,
                "smplx_source": str(item.smplx_source),
                "scan_source": str(item.scan_source) if item.scan_source else "",
                "asset": appearance_asset_to_manifest(item.asset),
            }
            for item in prepared
        ],
    }
    appearance_root.mkdir(parents=True, exist_ok=True)
    (appearance_root / "thuman_prepare_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
