#!/usr/bin/env python3
"""Canonical delivery keypoints / edges for 0806 dual-external skeletons.

Face: nose only (no eyes/ears).
Feet: one toe tip per foot (prefer left/right_big_toe; alias left/right_toe).
Do NOT deliver small_toe / heel in playback / head draw.
"""

from __future__ import annotations

import base64
from typing import Iterable

import numpy as np


# Delivery joint order (stable for playback schema v2).
DELIVERY_JOINTS: list[str] = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_big_toe",
    "right_big_toe",
]

# Alias accepted when reading triangulation records.
TOE_ALIASES = {
    "left_big_toe": ("left_big_toe", "left_toe"),
    "right_big_toe": ("right_big_toe", "right_toe"),
}

DELIVERY_EDGES: list[tuple[str, str]] = [
    ("nose", "left_shoulder"),
    ("nose", "right_shoulder"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("left_ankle", "left_big_toe"),
    ("right_ankle", "right_big_toe"),
]

FACE_HIDE = ("left_eye", "right_eye", "left_ear", "right_ear")
FOOT_HIDE = (
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
)

MISS = -32768


def resolve_joint_xyz(joints: dict, name: str) -> list[float] | None:
    """Return xyz_world_m for a delivery joint, resolving toe aliases."""
    names = TOE_ALIASES.get(name, (name,))
    for candidate in names:
        payload = joints.get(candidate)
        if not payload:
            continue
        xyz = payload.get("xyz_world_m") if isinstance(payload, dict) else payload
        if xyz is None:
            continue
        arr = np.asarray(xyz, dtype=np.float64)
        if arr.shape == (3,) and np.isfinite(arr).all():
            return arr.tolist()
    return None


def edge_indices(joints: Iterable[str] | None = None) -> list[list[int]]:
    names = list(joints) if joints is not None else list(DELIVERY_JOINTS)
    index = {name: i for i, name in enumerate(names)}
    edges = []
    for a, b in DELIVERY_EDGES:
        if a in index and b in index:
            edges.append([index[a], index[b]])
    return edges


def export_skeleton_playback(
    records: list[dict],
    path,
    *,
    source: str,
    joint_names: list[str] | None = None,
) -> dict:
    """Write skeleton_playback.json compatible with yaw MP4 + canvas viewer."""
    from pathlib import Path

    path = Path(path)
    names = list(joint_names) if joint_names is not None else list(DELIVERY_JOINTS)
    seqs = [int(r["seq"]) for r in records]
    array = np.full((len(records), len(names), 3), MISS, dtype=np.int16)
    for i, record in enumerate(records):
        joints = record["methods"]["filtered"]["multiview"]
        for j, name in enumerate(names):
            xyz = resolve_joint_xyz(joints, name)
            if xyz is None:
                continue
            array[i, j] = np.clip(np.rint(np.asarray(xyz) * 1000.0), -32767, 32767).astype(
                np.int16
            )
    # Foot height helper for canvas HUD (ankle world Y).
    left_i = names.index("left_ankle") if "left_ankle" in names else None
    right_i = names.index("right_ankle") if "right_ankle" in names else None
    left_above = []
    right_above = []
    # Ground ≈ ankle Y p5 across sequence (mocap Y-up).
    ankle_y = []
    for i in range(len(records)):
        for ji in (left_i, right_i):
            if ji is None:
                continue
            y = int(array[i, ji, 1])
            if y != MISS:
                ankle_y.append(y / 1000.0)
    ground = float(np.percentile(ankle_y, 5)) if ankle_y else 0.0
    for i in range(len(records)):
        if left_i is not None and int(array[i, left_i, 1]) != MISS:
            left_above.append(float(array[i, left_i, 1] / 1000.0 - ground) * 1000.0)
        else:
            left_above.append(None)
        if right_i is not None and int(array[i, right_i, 1]) != MISS:
            right_above.append(float(array[i, right_i, 1] / 1000.0 - ground) * 1000.0)
        else:
            right_above.append(None)

    payload = {
        "schema": "joint_projection.skeleton_playback.v2",
        "title": "0806 dual-external 3D skeleton",
        "joints": names,
        "edges": edge_indices(names),
        "seqs": seqs,
        "frame_count": len(seqs),
        "missing_sentinel": MISS,
        "xyz_i16_b64": base64.b64encode(array.tobytes()).decode("ascii"),
        "units": "millimetres_int16",
        "xyz_unit": "mm_i16",
        "ground_z_m": ground,  # actually world-Y ground; viewer remaps Y-up→Z-up
        "left_above_mm": left_above,
        "right_above_mm": right_above,
        "source": source,
        "face_policy": "nose_only",
        "foot_policy": "one_big_toe_per_foot",
        "constraint_note": (
            "Do NOT pull toe toward a foot direction derived from ankle mocap "
            "rigid frame (ankle rigid ≠ skeleton joint axes)."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        __import__("json").dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    return payload


def prune_joints_inplace(joints: dict) -> None:
    """Drop non-delivery face/foot keys from a methods.filtered.multiview dict."""
    for name in FACE_HIDE + FOOT_HIDE:
        joints.pop(name, None)
    # Normalize toe aliases → big_toe.
    for dest, aliases in TOE_ALIASES.items():
        xyz = resolve_joint_xyz(joints, dest)
        if xyz is None:
            continue
        base = joints.get(dest) if isinstance(joints.get(dest), dict) else {}
        joints[dest] = {**base, "xyz_world_m": xyz}
        for alias in aliases:
            if alias != dest:
                joints.pop(alias, None)
