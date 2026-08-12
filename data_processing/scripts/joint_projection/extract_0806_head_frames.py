#!/usr/bin/env python3
"""Extract aligned head stereo RGB frames from 0806 limb datasets for EgoRear stage-1.

Output layout (EgoRear MultiViewHeatmapDataset compatible):
  {frame_root}/{limb}/{head_dir}/{CAM_A|CAM_D}/{aligned_seq:06d}.jpg

Decoding uses elementary .h265 + timestamps.csv capture indices matched from
aligned_30hz.csv head exposure timestamps (same as render_multiview_to_head.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from multiview_geometry import load_json  # noqa: E402
from render_multiview_to_head import (  # noqa: E402
    H265CaptureReader,
    HeadTimestampIndex,
    find_head_video,
    infer_head_module_from_video,
    resolve_repo_path,
)

CAMERAS = ("CAM_A", "CAM_D")
IMAGE_SIZE = (1920, 1200)

DATASETS: dict[str, dict[str, str]] = {
    "wu": {
        "batch_name": "wu",
        "head_dir": "0712_033709",
        "config": "0806_dual_external_mocap.json",
        "role": "none_limb_baseline",
    },
    "wrist": {
        "batch_name": "wrist",
        "head_dir": "0712_032704",
        "config": "0806_wrist_dual_external_mocap.json",
        "role": "wrist_limb",
    },
    "ankle": {
        "batch_name": "ankle",
        "head_dir": "0712_033034",
        "config": "0806_ankle_dual_external_mocap.json",
        "role": "ankle_limb",
    },
    "line1": {
        "batch_name": "line1",
        "head_dir": "0712_035226",
        "config": "0810_line1_dual_external_mocap.json",
        "role": "0810_line1",
    },
    "line2": {
        "batch_name": "line2",
        "head_dir": "0712_035903",
        "config": "0810_line2_dual_external_mocap.json",
        "role": "0810_line2",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-root",
        type=Path,
        default=Path("/home/gaoweijian/0806_batch"),
        help="Root containing wu/wrist/ankle batch folders",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/home/gaoweijian/0806dataset"),
        help="0806dataset root (frames/ + manifest/ written here)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=sorted(DATASETS),
    )
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip frames whose JPG already exists with non-zero size",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Count existing frames and validate manifest without decoding",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_aligned_rows(data_root: Path) -> list[dict[str, str]]:
    aligned_path = data_root / "aligned_data" / "aligned_30hz.csv"
    with aligned_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows.sort(key=lambda row: int(row["seq"]))
    return rows


def write_jpeg(path: Path, image, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError(f"Failed to encode JPEG for {path}")
    path.write_bytes(buffer.tobytes())


def frame_out_path(
    frame_root: Path, limb: str, head_dir: str, camera: str, seq: int
) -> Path:
    return frame_root / limb / head_dir / camera / f"{seq:06d}.jpg"


def build_decode_plan(
    aligned_rows: list[dict[str, str]],
    indexes: dict[str, HeadTimestampIndex],
    tolerance_ms: float,
) -> tuple[dict[str, list[tuple[int, int, dict, float]]], dict[str, int]]:
    per_camera: dict[str, list[tuple[int, int, dict, float]]] = {
        camera: [] for camera in CAMERAS
    }
    stats = {"timestamp_fallback_matches": 0}
    for row in aligned_rows:
        seq = int(row["seq"])
        for camera in CAMERAS:
            column = f"head_{camera}_exposure_end_timestamp_ms"
            capture_index, ts_row, match_error_ms = indexes[camera].nearest(
                float(row[column]), tolerance_ms
            )
            if match_error_ms:
                stats["timestamp_fallback_matches"] += 1
            per_camera[camera].append(
                (seq, capture_index, ts_row, float(match_error_ms))
            )
    return per_camera, stats


def extract_camera_frames(
    *,
    camera: str,
    tasks: list[tuple[int, int, dict, float]],
    reader: H265CaptureReader,
    frame_root: Path,
    limb: str,
    head_dir: str,
    jpeg_quality: int,
    skip_existing: bool,
) -> dict[str, int | list[int]]:
    sorted_tasks = sorted(tasks, key=lambda item: item[1])
    written = 0
    skipped = 0
    missing_seqs: list[int] = []
    width, height = IMAGE_SIZE

    for seq, capture_index, _ts_row, _match_error_ms in sorted_tasks:
        out_path = frame_out_path(frame_root, limb, head_dir, camera, seq)
        if skip_existing and out_path.is_file() and out_path.stat().st_size > 0:
            skipped += 1
            continue
        image = reader.read(capture_index)
        if image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        write_jpeg(out_path, image, jpeg_quality)
        written += 1

    if skip_existing:
        for seq, _, _, _ in sorted_tasks:
            out_path = frame_out_path(frame_root, limb, head_dir, camera, seq)
            if not out_path.is_file() or out_path.stat().st_size == 0:
                missing_seqs.append(seq)

    return {
        "written": written,
        "skipped": skipped,
        "missing_fallbacks": reader.missing_fallbacks,
        "missing_seqs_after_run": missing_seqs,
        "expected_frames": len(sorted_tasks),
    }


def verify_dataset_frames(
    frame_root: Path, limb: str, head_dir: str, expected_seqs: list[int]
) -> dict[str, object]:
    report: dict[str, object] = {"cameras": {}, "complete": True}
    for camera in CAMERAS:
        cam_dir = frame_root / limb / head_dir / camera
        existing = sorted(cam_dir.glob("*.jpg")) if cam_dir.is_dir() else []
        count = len(existing)
        missing = [
            seq
            for seq in expected_seqs
            if not (cam_dir / f"{seq:06d}.jpg").is_file()
        ]
        sample_ok = False
        if existing:
            sample = cv2.imread(str(existing[0]))
            sample_ok = sample is not None and sample.size > 0
        report["cameras"][camera] = {
            "frame_count": count,
            "expected_frames": len(expected_seqs),
            "missing_count": len(missing),
            "sample_readable": sample_ok,
            "sample_path": str(existing[0]) if existing else None,
        }
        if missing or not sample_ok:
            report["complete"] = False
    return report


def process_dataset(
    dataset_key: str,
    *,
    batch_root: Path,
    output_root: Path,
    jpeg_quality: int,
    skip_existing: bool,
    verify_only: bool,
) -> dict[str, object]:
    meta = DATASETS[dataset_key]
    batch_name = meta["batch_name"]
    head_dir_name = meta["head_dir"]
    data_root = batch_root / batch_name / "data_root"
    head_dir = data_root / head_dir_name
    frame_root = output_root / "frames"
    manifest_dir = output_root / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    result: dict[str, object] = {
        "dataset": dataset_key,
        "batch_name": batch_name,
        "data_root": str(data_root),
        "head_dir": head_dir_name,
        "role": meta["role"],
        "status": "pending",
    }

    for label, path in (
        ("aligned_csv", data_root / "aligned_data" / "aligned_30hz.csv"),
        ("head_a_h265", head_dir / "module01_D45D2E00_CAM_A.h265"),
        ("head_d_h265", head_dir / "module01_D45D2E00_CAM_D.h265"),
        ("head_timestamps", head_dir / "timestamps.csv"),
    ):
        if not path.is_file() or path.stat().st_size == 0:
            result["status"] = "missing_input"
            result["blocker"] = f"missing {label}: {path}"
            return result

    aligned_rows = load_aligned_rows(data_root)
    expected_seqs = [int(row["seq"]) for row in aligned_rows]
    sample_rel = f"{dataset_key}/{head_dir_name}"
    result["aligned_frame_count"] = len(expected_seqs)
    result["sequence_range"] = [expected_seqs[0], expected_seqs[-1]] if expected_seqs else None
    result["frame_root_rel"] = sample_rel
    result["video_size"] = list(IMAGE_SIZE)

    if verify_only:
        verify = verify_dataset_frames(frame_root, dataset_key, head_dir_name, expected_seqs)
        result.update(verify)
        result["status"] = "complete" if verify.get("complete") else "incomplete"
        return result

    config = load_json(resolve_repo_path(SCRIPT_DIR / "configs" / meta["config"]))
    head_cfg = config.get("head", {})
    tolerance_ms = float(head_cfg.get("timestamp_match_tolerance_ms", 1.0))

    videos = {
        camera: find_head_video(head_dir, camera, config) for camera in CAMERAS
    }
    indexes = {
        camera: HeadTimestampIndex(
            head_dir / "timestamps.csv",
            camera,
            module=infer_head_module_from_video(videos[camera]),
        )
        for camera in CAMERAS
    }
    readers = {
        camera: H265CaptureReader(videos[camera], indexes[camera].rows)
        for camera in CAMERAS
    }

    per_camera_tasks, plan_stats = build_decode_plan(aligned_rows, indexes, tolerance_ms)

    camera_stats: dict[str, dict] = {}
    frame_entries: list[dict] = []
    for camera in CAMERAS:
        camera_stats[camera] = extract_camera_frames(
            camera=camera,
            tasks=per_camera_tasks[camera],
            reader=readers[camera],
            frame_root=frame_root,
            limb=dataset_key,
            head_dir=head_dir_name,
            jpeg_quality=jpeg_quality,
            skip_existing=skip_existing,
        )
        readers[camera].close()

    for row in aligned_rows:
        seq = int(row["seq"])
        entry = {
            "seq": seq,
            "mocap_frame_index": int(float(row.get("mocap_frame_index", -1))),
            "camera_elapsed_sec": float(row.get("camera_elapsed_sec", seq / 30.0)),
            "cameras": {},
        }
        for camera in CAMERAS:
            capture_index, ts_row, match_error_ms = indexes[camera].nearest(
                float(row[f"head_{camera}_exposure_end_timestamp_ms"]), tolerance_ms
            )
            entry["cameras"][camera] = {
                "capture_index": int(capture_index),
                "capture_sequence": int(ts_row.get("seq", capture_index)),
                "exposure_end_ts_ms": float(ts_row["exposure_end_ts_ms"]),
                "timestamp_match_error_ms": float(match_error_ms),
                "frame_path": str(
                    frame_out_path(frame_root, dataset_key, head_dir_name, camera, seq)
                ),
            }
        frame_entries.append(entry)

    verify = verify_dataset_frames(frame_root, dataset_key, head_dir_name, expected_seqs)
    manifest = {
        "schema": "joint_projection.0806dataset_head_frames.v1",
        "created_at": utc_now(),
        "dataset": dataset_key,
        "role": meta["role"],
        "data_root": str(data_root),
        "head_dir": head_dir_name,
        "frame_root": str(frame_root),
        "sample_rel": sample_rel,
        "camera_names": list(CAMERAS),
        "video_size": list(IMAGE_SIZE),
        "decode": {
            "source": "elementary_h265_packet_by_timestamps_bytes",
            "timestamp_match_tolerance_ms": tolerance_ms,
            "timestamp_fallback_matches": plan_stats["timestamp_fallback_matches"],
        },
        "videos": {camera: str(videos[camera]) for camera in CAMERAS},
        "aligned_frame_count": len(expected_seqs),
        "sequence_range": result["sequence_range"],
        "camera_stats": camera_stats,
        "verification": verify,
        "frames": frame_entries,
    }
    manifest_path = manifest_dir / f"{dataset_key}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    result["manifest_path"] = str(manifest_path)
    result["camera_stats"] = camera_stats
    result["timestamp_fallback_matches"] = plan_stats["timestamp_fallback_matches"]
    result.update(verify)
    result["status"] = "complete" if verify.get("complete") else "incomplete"
    return result


def write_status(output_root: Path, reports: list[dict[str, object]]) -> None:
    lines = [
        f"0806dataset head frame extraction",
        f"updated: {utc_now()}",
        "",
    ]
    for report in reports:
        lines.append(f"[{report['dataset']}] status={report.get('status')}")
        if report.get("blocker"):
            lines.append(f"  blocker: {report['blocker']}")
            continue
        lines.append(
            f"  aligned_frames={report.get('aligned_frame_count')} "
            f"seq_range={report.get('sequence_range')}"
        )
        cameras = report.get("cameras") or {}
        for camera in CAMERAS:
            cam = cameras.get(camera, {})
            lines.append(
                f"  {camera}: frames={cam.get('frame_count')} "
                f"missing={cam.get('missing_count')} "
                f"sample_ok={cam.get('sample_readable')}"
            )
        lines.append("")
    (output_root / "STATUS.txt").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def write_readme(output_root: Path) -> None:
    readme = """# 0806dataset — head RGB frames for EgoRear stage-1

Extracted from 0806 batch head elementary `.h265` streams using aligned
`aligned_30hz.csv` exposure timestamps and `timestamps.csv` packet sizes
(`H265CaptureReader` in joint_projection).

## Layout

```
frames/
  wu/0712_033709/CAM_A/{seq:06d}.jpg
  wu/0712_033709/CAM_D/{seq:06d}.jpg
  wrist/0712_032704/CAM_A/{seq:06d}.jpg
  wrist/0712_032704/CAM_D/{seq:06d}.jpg
  ankle/0712_033034/CAM_A/{seq:06d}.jpg
  ankle/0712_033034/CAM_D/{seq:06d}.jpg
manifest/
  wu.json
  wrist.json
  ankle.json
```

`{seq}` is the aligned timeline index from `aligned_30hz.csv` (strict temporal order).

## EgoRear stage-1 usage

Point `MultiViewHeatmapDataset.frame_root` at `frames/` and set each label NPZ
`source_render_dir` to the sample path, e.g. `wu/0712_033709`. Camera names in
labels should be `CAM_A` and `CAM_D`. Training resizes to `(456, 256)` by default.

## Re-run extraction

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate sapiens2
python /home/gaoweijian/0806_batch/repo/test_code/joint_projection/extract_0806_head_frames.py \\
  --batch-root /home/gaoweijian/0806_batch \\
  --output-root /home/gaoweijian/0806dataset \\
  --skip-existing

# Verify only (no decode):
python .../extract_0806_head_frames.py --verify-only
```

Native resolution JPG (1920x1200); resize at train time.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    reports: list[dict[str, object]] = []
    for dataset_key in args.datasets:
        print(f"=== {dataset_key} ===", flush=True)
        report = process_dataset(
            dataset_key,
            batch_root=args.batch_root,
            output_root=args.output_root,
            jpeg_quality=args.jpeg_quality,
            skip_existing=args.skip_existing,
            verify_only=args.verify_only,
        )
        reports.append(report)
        print(json.dumps(report, indent=2), flush=True)

    write_status(args.output_root, reports)
    write_readme(args.output_root)

    failed = [r for r in reports if r.get("status") != "complete"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
