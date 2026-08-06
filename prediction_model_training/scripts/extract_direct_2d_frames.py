#!/usr/bin/env python3
"""Decode the CAM_B/C H.265 streams used by direct-2D heatmap training."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess


CAMERAS = ("CAM_B", "CAM_C")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--width", type=int, default=456)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=2, help="FFmpeg q:v value; 2 is high quality.")
    return parser.parse_args()


def load_required_indices(csv_path: Path) -> dict[str, set[int]]:
    required = {camera: set() for camera in CAMERAS}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for camera in CAMERAS:
                prefix = f"module01_{camera}"
                if row.get(f"{prefix}_status") != "ok":
                    continue
                value = row.get(f"{prefix}_decoded_frame_index", "").strip()
                if value:
                    required[camera].add(int(value))
    for camera, indices in required.items():
        if not indices:
            raise RuntimeError(f"No usable decoded frame indices found for {camera}")
    return required


def expected_decode_counts(dataset_root: Path) -> dict[str, int]:
    report_path = dataset_root / "aligned_data" / "module01_cam_bc_shoulder_elbow_2d_report.json"
    if not report_path.exists():
        return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return {
        item["camera"]: int(item["decode_mapping"]["decoded_frames"])
        for item in report.get("cameras", [])
        if "decode_mapping" in item
    }


def decode_camera(
    source: Path,
    destination: Path,
    required: set[int],
    *,
    width: int,
    height: int,
    jpeg_quality: int,
    expected_count: int | None,
) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    existing = list(destination.glob("frame_*.jpg"))
    if existing:
        raise RuntimeError(
            f"{destination} already contains {len(existing)} frames; "
            "refusing to mix a new decode with old files"
        )

    pattern = destination / "frame_%06d.jpg"
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-nostdin",
        "-i", str(source), "-map", "0:v:0", "-an", "-vsync", "0",
        "-vf", f"scale={width}:{height}:flags=area",
        "-q:v", str(jpeg_quality), "-start_number", "0", str(pattern),
    ]
    process = subprocess.run(command, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed for {source} with exit code {process.returncode}")

    decoded = sorted(destination.glob("frame_*.jpg"))
    decoded_indices = {int(path.stem.rsplit("_", 1)[1]) for path in decoded}
    missing = sorted(required - decoded_indices)
    if missing:
        raise RuntimeError(
            f"{source.name}: {len(missing)} required decoded frames are missing; "
            f"first missing indices: {missing[:20]}"
        )
    if expected_count is not None and len(decoded) != expected_count:
        raise RuntimeError(
            f"{source.name}: decoded {len(decoded)} frames, but the source 2D report "
            f"was built against {expected_count}; refusing a potentially shifted mapping"
        )
    return {
        "source": str(source.resolve()),
        "destination": str(destination.resolve()),
        "decoded_frames": len(decoded),
        "required_frames": len(required),
        "required_min": min(required),
        "required_max": max(required),
        "missing_required_frames": 0,
    }


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    csv_path = (
        args.csv.expanduser().resolve()
        if args.csv is not None
        else dataset_root / "aligned_data" / "module01_cam_bc_hybrid_skeleton_2d.csv"
    )
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else dataset_root / "images_direct_2d_456x256"
    )
    required = load_required_indices(csv_path)
    expected = expected_decode_counts(dataset_root)

    results = {}
    for camera in CAMERAS:
        source = dataset_root / f"module01_D45D2E00_{camera}.h265"
        if not source.exists():
            raise FileNotFoundError(source)
        results[camera] = decode_camera(
            source,
            output_root / "module01" / camera,
            required[camera],
            width=args.width,
            height=args.height,
            jpeg_quality=args.jpeg_quality,
            expected_count=expected.get(camera),
        )

    manifest = {
        "schema": "egorear.direct_2d_frame_decode.v1",
        "dataset_root": str(dataset_root),
        "source_csv": str(csv_path),
        "output_root": str(output_root),
        "decode_order": "sequential raw H.265 decode; frame_N is FFmpeg decoded frame index N",
        "vsync": 0,
        "image_size": [args.width, args.height],
        "jpeg_qv": args.jpeg_quality,
        "cameras": results,
    }
    manifest_path = output_root / "decode_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
