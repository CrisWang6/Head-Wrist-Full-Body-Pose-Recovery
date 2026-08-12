#!/usr/bin/env python3
"""Stage E (part 1): per-frame nose align + limb tip GT replace.

Confirmed tip model (rigid-local, not world axes):
  p_world = R_world_rigid @ [0, 0, z_offset_m] + t_world_rigid

Per-frame (when tick_valid):
  1) Translate whole skeleton so external nose → nose_gt (CH3-08 tip)
  2) Hard-replace left/right limb tips with CH3-06/07 tips
       wrist dataset: replace wrists; keep triangulated ankles + toes
       ankle dataset: replace ankles; keep triangulated toes

Constraint ban (explicit):
  Do NOT pull toe / knee toward a foot direction derived from the ankle
  mocap rigid frame. Ankle rigid axes are NOT aligned with skeleton joint
  axes — never use R_ankle to invent a toe target.

Playback exports the delivery keypoint set (nose-only face, one big toe).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from delivery_keypoints import export_skeleton_playback, prune_joints_inplace
from multiview_geometry import rigid_world_transform


NOSE_OFFSET_M = np.asarray([0.0, -0.015, -0.125], dtype=np.float64)
LIMB_LOCAL = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
JOINTS = {
    "wrist": ("left_wrist", "right_wrist"),
    "ankle": ("left_ankle", "right_ankle"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--aligned", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--replace", choices=("wrist", "ankle"), required=True)
    p.add_argument(
        "--z-offset-mm",
        type=float,
        help="Rigid-local tip offset along Z (mm). Default: wrist=-60, ankle=-80.",
    )
    p.add_argument("--nose-prefix", default="mocap_CH3_08")
    p.add_argument("--left-prefix", default="mocap_CH3_06")
    p.add_argument("--right-prefix", default="mocap_CH3_07")
    p.add_argument("--playback-output", type=Path)
    p.add_argument(
        "--mode",
        choices=("per_frame", "fixed"),
        default="per_frame",
        help="per_frame: nose translation each frame; fixed: legacy multi-frame median.",
    )
    p.add_argument(
        "--temporal-smooth-sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma (frames) for light temporal filter after replace; 0=off.",
    )
    p.add_argument(
        "--skip-nose-align",
        action="store_true",
        help="Ablation: do not translate skeleton to per-frame nose GT.",
    )
    p.add_argument(
        "--skip-limb-replace",
        action="store_true",
        help="Ablation: do not hard-replace wrist/ankle tips with mocap GT.",
    )
    p.add_argument(
        "--bone-soft-config",
        type=Path,
        help="Ablation: JSON/YAML bone-length soft constraint weights (see ablation/).",
    )
    return p.parse_args()


def tip_world(row: dict, prefix: str, local_m: np.ndarray) -> np.ndarray | None:
    if int(float(row.get(f"{prefix}_status", "0"))) != 1:
        return None
    # Prefer fresh mocap ticks; status can stay 1 while raw_tick_valid drops.
    if int(float(row.get(f"{prefix}_raw_tick_valid", "0"))) != 1:
        return None
    try:
        world = rigid_world_transform(row, prefix)
    except (KeyError, TypeError, ValueError):
        return None
    return (world[:3, :3] @ local_m) + world[:3, 3]


def load_results(path: Path) -> list[dict]:
    records = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
    return records


def robust_median(values: np.ndarray) -> tuple[np.ndarray, dict]:
    center = np.median(values, axis=0)
    radii = np.linalg.norm(values - center, axis=1)
    mad = float(np.median(np.abs(radii - np.median(radii))))
    threshold = float(np.median(radii)) + max(0.02, 3.5 * 1.4826 * mad)
    keep = radii <= threshold
    offset = np.median(values[keep], axis=0)
    residual = np.linalg.norm(values[keep] - offset, axis=1)
    stats = {
        "samples": int(len(values)),
        "inliers": int(keep.sum()),
        "outlier_threshold_m": threshold,
        "offset_world_m": offset.tolist(),
        "offset_norm_mm": float(np.linalg.norm(offset) * 1000.0),
        "residual_median_mm": float(np.median(residual) * 1000.0),
        "residual_p90_mm": float(np.percentile(residual, 90) * 1000.0),
    }
    return offset, stats


def temporal_smooth(records: list[dict], sigma: float) -> None:
    if sigma <= 0:
        return
    radius = max(1, int(round(3 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    kernel /= kernel.sum()
    # Collect joint trajectories.
    names = sorted({n for r in records for n in r["methods"]["filtered"]["multiview"]})
    for name in names:
        traj = np.full((len(records), 3), np.nan, dtype=np.float64)
        for i, record in enumerate(records):
            payload = record["methods"]["filtered"]["multiview"].get(name)
            if not payload:
                continue
            xyz = np.asarray(payload["xyz_world_m"], dtype=np.float64)
            if xyz.shape == (3,) and np.isfinite(xyz).all():
                traj[i] = xyz
        # Skip hard GT joints that should stay pinned (source marker).
        # Still smooth them lightly only if not marked as mocap tip.
        for axis in range(3):
            col = traj[:, axis]
            valid = np.isfinite(col)
            if valid.sum() < 5:
                continue
            # Fill small gaps with nearest for convolution, then restore NaNs.
            filled = col.copy()
            idx = np.where(valid)[0]
            for i in range(len(filled)):
                if not np.isfinite(filled[i]):
                    j = idx[np.argmin(np.abs(idx - i))]
                    filled[i] = col[j]
            smooth = np.convolve(filled, kernel, mode="same")
            traj[:, axis] = np.where(valid, smooth, np.nan)
        for i, record in enumerate(records):
            if not np.isfinite(traj[i]).all():
                continue
            payload = record["methods"]["filtered"]["multiview"].get(name)
            if not payload:
                continue
            # Do not smooth hard mocap tip replacements.
            src = str(payload.get("source", ""))
            if "rigid_local" in src:
                continue
            payload["xyz_world_m"] = traj[i].tolist()


def main() -> None:
    args = parse_args()
    left_name, right_name = JOINTS[args.replace]
    if args.z_offset_mm is None:
        args.z_offset_mm = -60.0 if args.replace == "wrist" else -80.0
    z_m = float(args.z_offset_mm) / 1000.0
    limb_local = LIMB_LOCAL * z_m

    with args.aligned.open("r", encoding="utf-8-sig", newline="") as stream:
        aligned = {int(row["seq"]): row for row in csv.DictReader(stream)}
    records = load_results(args.results)

    # Optional legacy fixed offset for ablation / debug.
    fixed_offset = np.zeros(3, dtype=np.float64)
    fit_stats = None
    if args.mode == "fixed":
        deltas = []
        for record in records:
            seq = int(record["seq"])
            row = aligned.get(seq)
            if row is None:
                continue
            joints = record["methods"]["filtered"]["multiview"]
            pairs = []
            nose_gt = tip_world(row, args.nose_prefix, NOSE_OFFSET_M)
            if nose_gt is not None and "nose" in joints:
                nose_ext = np.asarray(joints["nose"]["xyz_world_m"], dtype=np.float64)
                pairs.append(nose_gt - nose_ext)
            left_gt = tip_world(row, args.left_prefix, limb_local)
            if left_gt is not None and left_name in joints:
                left_ext = np.asarray(joints[left_name]["xyz_world_m"], dtype=np.float64)
                pairs.append(left_gt - left_ext)
            right_gt = tip_world(row, args.right_prefix, limb_local)
            if right_gt is not None and right_name in joints:
                right_ext = np.asarray(joints[right_name]["xyz_world_m"], dtype=np.float64)
                pairs.append(right_gt - right_ext)
            if pairs:
                deltas.append(np.mean(pairs, axis=0))
        if not deltas:
            raise RuntimeError("No nose/limb GT↔external pairs available")
        fixed_offset, fit_stats = robust_median(np.asarray(deltas, dtype=np.float64))

    replaced_left = replaced_right = 0
    nose_aligned = 0
    output_records = []
    per_frame_nose_delta = []

    for record in records:
        seq = int(record["seq"])
        row = aligned.get(seq)
        out = json.loads(json.dumps(record))  # deep copy
        joints = out["methods"]["filtered"]["multiview"]

        if args.mode == "fixed" and not args.skip_nose_align:
            for payload in joints.values():
                xyz = np.asarray(payload["xyz_world_m"], dtype=np.float64) + fixed_offset
                payload["xyz_world_m"] = xyz.tolist()
        elif row is not None and not args.skip_nose_align:
            nose_gt = tip_world(row, args.nose_prefix, NOSE_OFFSET_M)
            if nose_gt is not None and "nose" in joints:
                nose_ext = np.asarray(joints["nose"]["xyz_world_m"], dtype=np.float64)
                delta = nose_gt - nose_ext
                for payload in joints.values():
                    xyz = np.asarray(payload["xyz_world_m"], dtype=np.float64) + delta
                    payload["xyz_world_m"] = xyz.tolist()
                nose_aligned += 1
                per_frame_nose_delta.append(delta)

        if row is not None and not args.skip_limb_replace:
            left_gt = tip_world(row, args.left_prefix, limb_local)
            right_gt = tip_world(row, args.right_prefix, limb_local)
            if left_gt is not None:
                joints[left_name] = {
                    **joints.get(left_name, {}),
                    "xyz_world_m": left_gt.tolist(),
                    "source": f"{args.left_prefix}+z{args.z_offset_mm}mm_rigid_local",
                }
                replaced_left += 1
            if right_gt is not None:
                joints[right_name] = {
                    **joints.get(right_name, {}),
                    "xyz_world_m": right_gt.tolist(),
                    "source": f"{args.right_prefix}+z{args.z_offset_mm}mm_rigid_local",
                }
                replaced_right += 1

        # Wrist: keep triangulated ankles+toes. Ankle: ankles replaced; toes stay.
        # Never invent toe targets from ankle rigid orientation.
        prune_joints_inplace(joints)
        out["limb_gt"] = {
            "replace": args.replace,
            "mode": args.mode,
            "z_offset_mm_rigid_local": args.z_offset_mm,
            "ankle_rigid_to_toe_constraint": False,
            "note": (
                "Toe/knee are NOT pulled using ankle mocap rigid axes "
                "(frames misaligned)."
            ),
        }
        if args.mode == "fixed":
            out["limb_gt"]["offset_world_m"] = fixed_offset.tolist()
        output_records.append(out)

    temporal_smooth(output_records, float(args.temporal_smooth_sigma))

    if args.bone_soft_config is not None:
        import sys

        ablation_dir = Path(__file__).resolve().parent / "ablation"
        if str(ablation_dir) not in sys.path:
            sys.path.insert(0, str(ablation_dir.parent))
        from ablation.ablation_bone_soft_constraints import apply_bone_soft_constraints

        apply_bone_soft_constraints(output_records, args.bone_soft_config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for record in output_records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.playback_output is not None:
        export_skeleton_playback(
            output_records,
            args.playback_output,
            source=(
                f"methods.filtered.multiview after {args.mode} nose align + "
                f"{args.replace} tip GT (delivery keypoints)"
            ),
        )

    report = {
        "schema": "joint_projection.replace_limb_mocap_gt.v2",
        "replace": args.replace,
        "mode": args.mode,
        "left_joint": left_name,
        "right_joint": right_name,
        "left_prefix": args.left_prefix,
        "right_prefix": args.right_prefix,
        "nose_prefix": args.nose_prefix,
        "z_offset_mm_rigid_local": args.z_offset_mm,
        "formula": "p_world = R @ [0,0,z_offset_m] + t  (rigid-local tip)",
        "ankle_rigid_to_toe_constraint": False,
        "constraint_ban": (
            "Do NOT derive toe/knee targets from ankle mocap rigid frame "
            "(踝刚体坐标系和骨架关节坐标系没对齐)."
        ),
        "dataset_policy": {
            "wrist": "replace wrists; keep triangulated ankles + toes",
            "ankle": "replace ankles; keep triangulated toes",
        }[args.replace],
        "fit": fit_stats,
        "per_frame_nose_delta_median_mm": (
            (np.median(np.asarray(per_frame_nose_delta), axis=0) * 1000.0).tolist()
            if per_frame_nose_delta
            else None
        ),
        "counts": {
            "frames": len(output_records),
            "nose_aligned_frames": nose_aligned,
            "replaced_left": replaced_left,
            "replaced_right": replaced_right,
        },
        "outputs": {
            "results": str(args.output),
            "playback": str(args.playback_output) if args.playback_output else None,
        },
    }
    report_path = args.output.with_name(f"replace_{args.replace}_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
