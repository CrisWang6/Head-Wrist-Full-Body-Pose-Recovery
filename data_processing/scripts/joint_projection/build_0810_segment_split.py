#!/usr/bin/env python3
"""Build a one-off split NPZ for a contiguous aligned-seq segment (e.g. line1 10s)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from constants_0810_training import LABEL_NPZ_NAME, SESSION_ORDER


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--label-root", type=Path, required=True)
    p.add_argument("--session", default="line1", choices=SESSION_ORDER)
    p.add_argument("--seq-start", type=int, default=0)
    p.add_argument("--seq-end", type=int, required=True, help="Inclusive aligned seq")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split-name", default="train", choices=("train", "val", "test"))
    return p.parse_args()


def session_offsets(label_root: Path) -> dict[str, tuple[int, np.ndarray]]:
    out: dict[str, tuple[int, np.ndarray]] = {}
    offset = 0
    for session in SESSION_ORDER:
        npz = label_root / session / LABEL_NPZ_NAME
        data = np.load(npz, allow_pickle=True)
        seqs = np.asarray(data["source_aligned_seq"], dtype=np.int64).reshape(-1)
        out[session] = (offset, seqs)
        offset += int(seqs.shape[0])
    return out


def main() -> int:
    args = parse_args()
    label_root = args.label_root.expanduser().resolve()
    meta = session_offsets(label_root)
    if args.session not in meta:
        raise KeyError(args.session)
    base_offset, seqs = meta[args.session]
    total = sum(int(meta[s][1].shape[0]) for s in SESSION_ORDER)
    all_frame_indices = np.concatenate([meta[s][1] for s in SESSION_ORDER]).astype(np.int64)

    want = set(range(int(args.seq_start), int(args.seq_end) + 1))
    seg_global: list[int] = []
    found_seqs: set[int] = set()
    for local_i, seq in enumerate(seqs.tolist()):
        seq_i = int(seq)
        if seq_i in want:
            found_seqs.add(seq_i)
            seg_global.append(base_offset + local_i)
    if len(seg_global) != len(want):
        missing = sorted(want - found_seqs)
        raise RuntimeError(
            f"Segment {args.session} seq {args.seq_start}-{args.seq_end} incomplete; "
            f"got {len(seg_global)}/{len(want)} missing e.g. {missing[:5]}"
        )
    seg_arr = np.asarray(sorted(seg_global), dtype=np.int64)
    all_indices = np.arange(total, dtype=np.int64)
    rest = np.asarray(sorted(set(all_indices.tolist()) - set(seg_arr.tolist())), dtype=np.int64)

    train = val = test = ignored = np.asarray([], dtype=np.int64)
    if args.split_name == "train":
        train = seg_arr
    elif args.split_name == "val":
        val = seg_arr
    else:
        test = seg_arr
    ignored = rest

    assignment = np.full(total, -1, dtype=np.int8)
    assignment[train] = 0
    assignment[val] = 1
    assignment[test] = 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        schema=np.asarray([f"0810_segment_{args.session}"]),
        schema_version=np.asarray(["0810_segment_split_v1"]),
        session=np.asarray([args.session]),
        seq_start=np.asarray([args.seq_start], dtype=np.int64),
        seq_end=np.asarray([args.seq_end], dtype=np.int64),
        frame_indices=all_frame_indices,
        assignment=assignment,
        train_indices=train,
        val_indices=val,
        test_indices=test,
        ignored_indices=ignored,
    )
    summary = {
        "output": str(args.output),
        "session": args.session,
        "seq_start": args.seq_start,
        "seq_end": args.seq_end,
        "segment_frames": int(seg_arr.size),
        "split_name": args.split_name,
        "total_frames": total,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
