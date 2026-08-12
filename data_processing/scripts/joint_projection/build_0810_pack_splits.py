#!/usr/bin/env python3
"""Build 5s pack train/val/test splits for 0810 (line1 train, line2 test)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from constants_0810_training import LABEL_NPZ_NAME, PACK_SIZE, SESSION_ORDER, SPLIT_SCHEME


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--label-root",
        type=Path,
        default=Path("/home/gaoweijian/0810dataset/labels"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/gaoweijian/0810dataset/splits"),
    )
    p.add_argument("--pack-size", type=int, default=PACK_SIZE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--scheme", default=SPLIT_SCHEME)
    p.add_argument(
        "--train-session",
        default="line1",
        help="Session name used for train packs (default line1)",
    )
    p.add_argument(
        "--test-session",
        default="line2",
        help="Session name used for test packs (default line2)",
    )
    return p.parse_args()


def load_label_meta(label_root: Path, pack_size: int) -> list[dict]:
    items = []
    offset = 0
    for session in SESSION_ORDER:
        npz = label_root / session / LABEL_NPZ_NAME
        if not npz.is_file():
            raise FileNotFoundError(npz)
        data = np.load(npz, allow_pickle=True)
        seqs = np.asarray(data["source_aligned_seq"], dtype=np.int64).reshape(-1)
        count = int(seqs.shape[0])
        packs = []
        for start in range(0, count, pack_size):
            end = min(count, start + pack_size)
            if end - start < pack_size:
                continue
            packs.append(list(range(offset + start, offset + end)))
        items.append(
            {
                "session": session,
                "offset": offset,
                "count": count,
                "seqs": seqs,
                "packs": packs,
            }
        )
        offset += count
    return items


def main() -> int:
    args = parse_args()
    items = load_label_meta(args.label_root, args.pack_size)
    total_frames = sum(item["count"] for item in items)
    all_packs: list[list[int]] = []
    pack_session: list[str] = []
    for item in items:
        for pack in item["packs"]:
            all_packs.append(pack)
            pack_session.append(item["session"])

    train_packs: list[int] = []
    val_packs: list[int] = []
    test_packs: list[int] = []

    for pi, session in enumerate(pack_session):
        if session == args.train_session:
            train_packs.append(pi)
        elif session == args.test_session:
            test_packs.append(pi)
        else:
            raise RuntimeError(f"Unexpected session in packs: {session}")

    rng = np.random.default_rng(args.seed)
    train_packs_arr = np.asarray(train_packs, dtype=int)
    rng.shuffle(train_packs_arr)
    n_val = max(1, int(round(len(train_packs_arr) * 0.1)))
    val_packs = train_packs_arr[:n_val].tolist()
    train_packs = train_packs_arr[n_val:].tolist()

    def pack_to_indices(pack_ids: list[int]) -> np.ndarray:
        idx: list[int] = []
        for pi in pack_ids:
            idx.extend(all_packs[pi])
        return np.asarray(sorted(idx), dtype=np.int64)

    train_indices = pack_to_indices(train_packs)
    val_indices = pack_to_indices(val_packs)
    test_indices = pack_to_indices(test_packs)

    packed = set()
    for pack in all_packs:
        packed.update(pack)
    ignored_indices = np.asarray(sorted(set(range(total_frames)) - packed), dtype=np.int64)

    dataset_frame_indices = np.concatenate(
        [np.asarray(item["seqs"], dtype=np.int64) for item in items]
    )

    assignment = np.full(total_frames, -1, dtype=np.int8)
    assignment[train_indices] = 0
    assignment[val_indices] = 1
    assignment[test_indices] = 2

    out_dir = args.output_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pack{args.pack_size}_{args.scheme}.npz"
    np.savez_compressed(
        out_path,
        schema=np.asarray([f"0810_pack_split_{args.scheme}"]),
        schema_version=np.asarray(["0810_pack_split_v1"]),
        pack_size=np.asarray([args.pack_size]),
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        ignored_indices=ignored_indices,
        frame_indices=dataset_frame_indices,
        assignment=assignment,
        train_packs=np.asarray(train_packs, dtype=np.int32),
        val_packs=np.asarray(val_packs, dtype=np.int32),
        test_packs=np.asarray(test_packs, dtype=np.int32),
        total_frames=np.asarray([total_frames]),
        seed=np.asarray([args.seed], dtype=np.int64),
        train_session=np.asarray([args.train_session]),
        test_session=np.asarray([args.test_session]),
    )
    summary = {
        "output": str(out_path),
        "scheme": args.scheme,
        "pack_size": args.pack_size,
        "pack_seconds_at_30hz": args.pack_size / 30.0,
        "total_frames": total_frames,
        "num_packs": len(all_packs),
        "train_frames": int(train_indices.size),
        "val_frames": int(val_indices.size),
        "test_frames": int(test_indices.size),
        "ignored_frames": int(ignored_indices.size),
        "train_packs": len(train_packs),
        "val_packs": len(val_packs),
        "test_packs": len(test_packs),
        "train_session": args.train_session,
        "test_session": args.test_session,
    }
    out_path.with_suffix(".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
