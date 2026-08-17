#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from geosim.thuman_assets import prepare_thuman_appearance_subset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a small THuman2.1 subset and prepare SMPL-X appearance assets for Blender/Isaac rendering."
    )
    parser.add_argument(
        "--scan-archive",
        default=str(ROOT / "data/thuman/archives/THuman2.1_Release_.7z"),
        help="THuman textured scan archive. Use empty string to prepare SMPL-X-only fallback assets.",
    )
    parser.add_argument(
        "--scan-root",
        default="",
        help="Already extracted THuman scan root containing subject folders or model/<subject>. Takes precedence over --scan-archive.",
    )
    parser.add_argument(
        "--smplx-archive",
        default=str(ROOT / "data/thuman/archives/THuman2.1_Release Smpl-X Paras_new.7z"),
        help="THuman SMPL-X parameter archive.",
    )
    parser.add_argument(
        "--extract-root",
        default=str(ROOT / "data/thuman/extracted_subset"),
        help="Directory for extracted subject folders. This is data, not source.",
    )
    parser.add_argument(
        "--appearance-root",
        default=str(ROOT / "smplx_models/thuman_appearances"),
        help="Prepared appearance library for render.py --appearance-root.",
    )
    parser.add_argument("--subjects", default="", help="Comma-separated subject ids. Empty selects the first --count ids.")
    parser.add_argument("--count", type=int, default=1, help="Number of subjects to prepare when --subjects is empty.")
    parser.add_argument("--seven-zip-bin", default="", help="Optional 7z/7zz executable. Empty uses Python py7zr.")
    parser.add_argument("--overwrite-extract", action="store_true")
    parser.add_argument("--overwrite-assets", action="store_true")
    parser.add_argument("--strict-textures", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subjects = tuple(part.strip() for part in args.subjects.split(",") if part.strip())
    manifest = prepare_thuman_appearance_subset(
        scan_archive=None if args.scan_root else (Path(args.scan_archive) if args.scan_archive else None),
        scan_root=Path(args.scan_root) if args.scan_root else None,
        smplx_archive=Path(args.smplx_archive),
        extract_root=Path(args.extract_root),
        appearance_root=Path(args.appearance_root),
        subjects=subjects,
        count=args.count,
        seven_zip_bin=args.seven_zip_bin,
        overwrite_extract=args.overwrite_extract,
        overwrite_assets=args.overwrite_assets,
        strict_textures=args.strict_textures,
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
