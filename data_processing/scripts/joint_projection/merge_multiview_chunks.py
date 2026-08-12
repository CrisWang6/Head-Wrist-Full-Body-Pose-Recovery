#!/usr/bin/env python3
"""Merge overlapped multiview chunks and recompute full-sequence metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from process_external_multiview_3d import (
    METHOD_CAMERAS,
    bone_statistics,
    method_statistics,
    stereo_disagreement,
    temporal_second_difference,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks-root", type=Path, required=True)
    parser.add_argument(
        "--cores",
        required=True,
        help="Comma-separated inclusive core ranges matching chunk_00, chunk_01, ...",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--context-frames", type=int, default=10)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    args = parse_args()
    cores = [
        tuple(map(int, item.split(":", 1)))
        for item in args.cores.split(",")
    ]
    records_by_seq: dict[int, dict] = {}
    chunk_sources = []
    for index, (core_start, core_end) in enumerate(cores):
        source = (
            args.chunks_root
            / f"chunk_{index:02d}"
            / "multiview_3d_results.jsonl"
        )
        chunk_sources.append(str(source.resolve()))
        for record in load_jsonl(source):
            seq = int(record["seq"])
            if core_start <= seq <= core_end:
                if seq in records_by_seq:
                    raise ValueError(f"Duplicate core seq {seq}")
                records_by_seq[seq] = record

    expected = set(range(cores[0][0], cores[-1][1] + 1))
    missing = sorted(expected - set(records_by_seq))
    if missing:
        raise ValueError(
            f"Missing {len(missing)} associated frames; first missing: {missing[:10]}"
        )
    records = [records_by_seq[seq] for seq in sorted(records_by_seq)]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / "multiview_3d_results.jsonl"
    with output_jsonl.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    output_csv = args.output_dir / "multiview_3d.csv"
    write_csv(records, output_csv)

    candidate_counts = {
        camera_name: sum(
            camera_name in record["observations"] for record in records
        )
        for camera_name in METHOD_CAMERAS["multiview"]
    }
    report = {
        "schema": "joint_projection.external_multiview_3d_report.v1",
        "manifest": str(args.manifest.resolve()),
        "config": str(args.config.resolve()),
        "candidates_dir": str(args.candidates_dir.resolve()),
        "outputs": {
            "jsonl": str(output_jsonl.resolve()),
            "csv": str(output_csv.resolve()),
        },
        "parallel_chunking": {
            "cores": [list(core) for core in cores],
            "context_frames_per_side": args.context_frames,
            "sources": chunk_sources,
        },
        "frames": {
            "manifest": len(expected),
            "associated": len(records),
            "cross_module_gate_pass": sum(
                bool(record["association"].get("cross_module_gate_pass"))
                for record in records
            ),
            "single_module_fallback": sum(
                "single_module_fallback" in record["association"]
                for record in records
            ),
        },
        "candidate_counts": candidate_counts,
        "methods": {
            stage: {
                method: {
                    **method_statistics(records, stage, method),
                    "bones": bone_statistics(records, stage, method),
                    "temporal_second_difference": temporal_second_difference(
                        records, stage, method
                    ),
                }
                for method in METHOD_CAMERAS
            }
            for stage in ("raw", "filtered")
        },
        "stereo_pair_disagreement_m": {
            stage: stereo_disagreement(records, stage)
            for stage in ("raw", "filtered")
        },
        "filter_2d": {
            "policy": (
                "Each parallel chunk was solved with overlapping context; only its "
                "core range was retained."
            ),
            "context_frames_per_side": args.context_frames,
        },
        "limitations": (
            "Mocap supplies camera-rigid pose truth, not human joint ground truth; "
            "reported accuracy is geometric consistency, coverage and stability."
        ),
    }
    report_path = args.output_dir / "multiview_3d_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "records": len(records),
                "jsonl": str(output_jsonl),
                "csv": str(output_csv),
                "report": str(report_path),
            }
        )
    )


if __name__ == "__main__":
    main()
