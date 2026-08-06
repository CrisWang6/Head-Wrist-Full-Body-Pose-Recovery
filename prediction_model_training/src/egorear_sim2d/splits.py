from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SPLIT_NAMES = ("train", "val", "test")


def create_random_frame_split(
    frame_indices: np.ndarray,
    *,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, np.ndarray]:
    """Globally shuffle individual frames without enforcing temporal quotas."""
    frames = np.asarray(frame_indices, dtype=np.int64)
    if frames.ndim != 1 or len(frames) < 3:
        raise ValueError("frame_indices must be a one-dimensional array with at least 3 frames")
    test_ratio = 1.0 - float(train_ratio) - float(val_ratio)
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("train, val and test ratios must all be positive")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(frames))
    train_end = int(round(len(frames) * train_ratio))
    val_end = train_end + int(round(len(frames) * val_ratio))
    train_indices = np.sort(shuffled[:train_end]).astype(np.int64)
    val_indices = np.sort(shuffled[train_end:val_end]).astype(np.int64)
    test_indices = np.sort(shuffled[val_end:]).astype(np.int64)
    assignment = np.empty(len(frames), dtype=np.int8)
    assignment[train_indices] = 0
    assignment[val_indices] = 1
    assignment[test_indices] = 2
    return {
        "frame_indices": frames,
        "assignment": assignment,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "seed": np.asarray([seed], dtype=np.int64),
        "block_size": np.asarray([0], dtype=np.int64),
        "ratios": np.asarray((train_ratio, val_ratio, test_ratio), dtype=np.float64),
        "schema_version": np.asarray(["global_random_frame_split_v1"]),
    }


def create_chronological_holdout_split(
    frame_indices: np.ndarray,
    *,
    train_ratio: float = 0.8,
) -> dict[str, np.ndarray]:
    """Use the chronological prefix for training and the tail as held-out data."""
    frames = np.asarray(frame_indices, dtype=np.int64)
    if frames.ndim != 1 or len(frames) < 2:
        raise ValueError("frame_indices must be a one-dimensional array with at least 2 frames")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between zero and one")
    split_at = max(1, min(len(frames) - 1, int(round(len(frames) * train_ratio))))
    train_indices = np.arange(split_at, dtype=np.int64)
    test_indices = np.arange(split_at, len(frames), dtype=np.int64)
    val_indices = np.asarray([], dtype=np.int64)
    assignment = np.full(len(frames), 2, dtype=np.int8)
    assignment[train_indices] = 0
    return {
        "frame_indices": frames,
        "assignment": assignment,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "seed": np.asarray([-1], dtype=np.int64),
        "block_size": np.asarray([0], dtype=np.int64),
        "ratios": np.asarray((train_ratio, 0.0, 1.0 - train_ratio), dtype=np.float64),
        "schema_version": np.asarray(["chronological_prefix_holdout_v1"]),
    }


def create_uniform_temporal_split(
    frame_indices: np.ndarray,
    *,
    seed: int = 42,
    block_size: int = 10,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, np.ndarray]:
    """Randomize within short temporal blocks so every split covers the full video."""
    frames = np.asarray(frame_indices, dtype=np.int64)
    if frames.ndim != 1 or len(frames) < 3:
        raise ValueError("frame_indices must be a one-dimensional array with at least 3 frames")
    if block_size < 3:
        raise ValueError("block_size must be at least 3")
    test_ratio = 1.0 - float(train_ratio) - float(val_ratio)
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("train, val and test ratios must all be positive")

    ratios = np.asarray((train_ratio, val_ratio, test_ratio), dtype=np.float64)
    assignment = np.empty(len(frames), dtype=np.int8)
    rng = np.random.default_rng(seed)
    for start in range(0, len(frames), block_size):
        end = min(start + block_size, len(frames))
        count = end - start
        expected = ratios * count
        counts = np.floor(expected).astype(np.int64)
        if count >= 3:
            counts = np.maximum(counts, 1)
        while counts.sum() > count:
            candidates = np.flatnonzero(counts > 1)
            remove_from = candidates[np.argmax(counts[candidates] - expected[candidates])]
            counts[remove_from] -= 1
        while counts.sum() < count:
            add_to = int(np.argmax(expected - counts))
            counts[add_to] += 1
        roles = np.repeat(np.arange(3, dtype=np.int8), counts)
        rng.shuffle(roles)
        assignment[start:end] = roles

    return {
        "frame_indices": frames,
        "assignment": assignment,
        "train_indices": np.flatnonzero(assignment == 0).astype(np.int64),
        "val_indices": np.flatnonzero(assignment == 1).astype(np.int64),
        "test_indices": np.flatnonzero(assignment == 2).astype(np.int64),
        "seed": np.asarray([seed], dtype=np.int64),
        "block_size": np.asarray([block_size], dtype=np.int64),
        "ratios": ratios,
        "schema_version": np.asarray(["uniform_temporal_split_v1"]),
    }


def create_strided_random_split(
    frame_indices: np.ndarray,
    *,
    stride: int,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    offset: int = 0,
) -> dict[str, np.ndarray]:
    """Subsample chronologically by ``stride``, then randomly split whole samples.

    Dataset indices not selected by the stride are recorded as ``ignored_indices``.
    Keeping the original full frame index array lets every training stage validate
    the manifest against the same source dataset while preventing adjacent frames
    from leaking across train, validation and test subsets.
    """
    frames = np.asarray(frame_indices, dtype=np.int64)
    if frames.ndim != 1 or len(frames) < 3:
        raise ValueError("frame_indices must be a one-dimensional array with at least 3 frames")
    if stride < 1:
        raise ValueError("stride must be positive")
    if offset < 0 or offset >= stride:
        raise ValueError("offset must satisfy 0 <= offset < stride")
    test_ratio = 1.0 - float(train_ratio) - float(val_ratio)
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("train, val and test ratios must all be positive")

    selected = np.arange(offset, len(frames), stride, dtype=np.int64)
    if len(selected) < 3:
        raise ValueError(
            f"stride={stride} and offset={offset} select only {len(selected)} frames"
        )
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(selected)
    train_end = int(round(len(selected) * train_ratio))
    val_end = train_end + int(round(len(selected) * val_ratio))
    train_end = max(1, min(train_end, len(selected) - 2))
    val_end = max(train_end + 1, min(val_end, len(selected) - 1))
    train_indices = np.sort(shuffled[:train_end]).astype(np.int64)
    val_indices = np.sort(shuffled[train_end:val_end]).astype(np.int64)
    test_indices = np.sort(shuffled[val_end:]).astype(np.int64)
    ignored_mask = np.ones(len(frames), dtype=bool)
    ignored_mask[selected] = False
    ignored_indices = np.flatnonzero(ignored_mask).astype(np.int64)
    assignment = np.full(len(frames), -1, dtype=np.int8)
    assignment[train_indices] = 0
    assignment[val_indices] = 1
    assignment[test_indices] = 2
    return {
        "frame_indices": frames,
        "assignment": assignment,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "ignored_indices": ignored_indices,
        "seed": np.asarray([seed], dtype=np.int64),
        "block_size": np.asarray([0], dtype=np.int64),
        "stride": np.asarray([stride], dtype=np.int64),
        "offset": np.asarray([offset], dtype=np.int64),
        "ratios": np.asarray((train_ratio, val_ratio, test_ratio), dtype=np.float64),
        "schema_version": np.asarray(["strided_random_frame_split_v1"]),
    }


def save_split_manifest(path: str | Path, split: dict[str, np.ndarray]) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **split)
    summary = {
        "schema_version": str(np.asarray(split["schema_version"]).reshape(-1)[0]),
        "seed": int(np.asarray(split["seed"]).reshape(-1)[0]),
        "block_size": int(np.asarray(split["block_size"]).reshape(-1)[0]),
        "stride": int(np.asarray(split.get("stride", [1])).reshape(-1)[0]),
        "offset": int(np.asarray(split.get("offset", [0])).reshape(-1)[0]),
        "ratios": np.asarray(split["ratios"], dtype=float).tolist(),
        "total_frames": int(len(split["frame_indices"])),
        "selected_frames": int(
            len(split["train_indices"])
            + len(split["val_indices"])
            + len(split["test_indices"])
        ),
        "ignored_frames": int(len(split.get("ignored_indices", []))),
        "train_frames": int(len(split["train_indices"])),
        "val_frames": int(len(split["val_indices"])),
        "test_frames": int(len(split["test_indices"])),
        "coverage": {
            name: (
                {
                    "first_source_frame": int(
                        split["frame_indices"][split[f"{name}_indices"][0]]
                    ),
                    "last_source_frame": int(
                        split["frame_indices"][split[f"{name}_indices"][-1]]
                    ),
                }
                if len(split[f"{name}_indices"])
                else None
            )
            for name in SPLIT_NAMES
        },
    }
    destination.with_suffix(".json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def load_split_manifest(
    path: str | Path,
    *,
    expected_length: int | None = None,
    expected_frame_indices: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as data:
        split = {key: np.asarray(data[key]) for key in data.files}
    required = {"frame_indices", "assignment", "train_indices", "val_indices", "test_indices"}
    missing = sorted(required - split.keys())
    if missing:
        raise ValueError(f"Split manifest is missing: {missing}")
    length = len(split["frame_indices"])
    if expected_length is not None and length != expected_length:
        raise ValueError(f"Split length mismatch: {length} != {expected_length}")
    if expected_frame_indices is not None and not np.array_equal(
        split["frame_indices"], np.asarray(expected_frame_indices, dtype=np.int64)
    ):
        raise ValueError("Split frame_indices do not exactly match the dataset")
    combined = np.concatenate(
        [
            split["train_indices"],
            split["val_indices"],
            split["test_indices"],
            split.get("ignored_indices", np.asarray([], dtype=np.int64)),
        ]
    )
    if not np.array_equal(np.sort(combined), np.arange(length)):
        raise ValueError("Split manifest does not contain each dataset index exactly once")
    return split
