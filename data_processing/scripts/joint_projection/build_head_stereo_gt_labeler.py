#!/usr/bin/env python3
"""Inline synchronized labeler images and metadata into the conversation fragment."""

from __future__ import annotations

import base64
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = Path(r"C:\Users\hand\Desktop\0711_214559\head_stereo_manual_fullbody_gt_20260805")
TEMPLATE = HERE / "head_stereo_gt_labeler.template.html"
OUTPUT = Path(
    r"C:\Users\hand\.codex\visualizations\2026\07\22"
    r"\019f87f7-6961-7d51-8a57-2309e6505108\head-stereo-gt-selector.html"
)


def data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def main() -> None:
    manifest = json.loads((SOURCE / "labeler_manifest.json").read_text(encoding="utf-8"))
    for sample in manifest["samples"]:
        for view in sample["views"].values():
            image_path = Path(view.pop("image_path"))
            view["image_data"] = data_url(image_path)
        reference = sample["external_reference"]
        image_path = Path(reference.pop("image_path"))
        reference["image_data"] = data_url(image_path)
    source = TEMPLATE.read_text(encoding="utf-8")
    if source.count("__DATA__") != 1:
        raise RuntimeError("Template data placeholder count is not one")
    payload = json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))
    OUTPUT.write_text(source.replace("__DATA__", payload), encoding="utf-8")
    if OUTPUT.stat().st_size >= 2 * 1024 * 1024:
        raise RuntimeError(f"Visualization exceeds 2 MB: {OUTPUT.stat().st_size}")
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
