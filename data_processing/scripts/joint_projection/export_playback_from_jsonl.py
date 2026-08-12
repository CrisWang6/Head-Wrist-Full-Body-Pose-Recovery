#!/usr/bin/env python3
"""Export delivery-keypoint skeleton_playback.json from a multiview results jsonl."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from delivery_keypoints import export_skeleton_playback, prune_joints_inplace


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", default="methods.filtered.multiview raw triangulation")
    p.add_argument(
        "--prune",
        action="store_true",
        help="Also drop non-delivery face/foot keys inside a rewritten jsonl "
        "(does not rewrite --results; only affects playback).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    records = []
    with args.results.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            if args.prune:
                prune_joints_inplace(record["methods"]["filtered"]["multiview"])
            records.append(record)
    payload = export_skeleton_playback(records, args.output, source=args.source)
    print(
        json.dumps(
            {
                "frames": payload["frame_count"],
                "joints": payload["joints"],
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
