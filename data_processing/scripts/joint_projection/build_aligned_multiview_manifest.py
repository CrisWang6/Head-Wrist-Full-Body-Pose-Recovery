#!/usr/bin/env python3
"""Build an exact frame/pose manifest for the 0806 dual-external dataset."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from multiview_geometry import (
    camera_a_mount_transform,
    load_json,
    module_camera_world_transforms,
    rigid_world_transform,
    stereo_transform_d_a,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "0806_dual_external_mocap.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-seq", type=int)
    parser.add_argument("--end-seq", type=int, help="Inclusive aligned sequence")
    parser.add_argument(
        "--skip-inexact-timestamp",
        action="store_true",
        help="Drop aligned rows when any camera timestamp is not an exact match",
    )
    return parser.parse_args()


def resolve_repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def discover_external_root(data_root: Path) -> Path:
    matches = sorted(path.parent for path in data_root.glob("*/external_modules.json"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one external recording under {data_root}, found {matches}"
        )
    return matches[0]


class TimestampIndex:
    def __init__(self, path: Path, camera: str):
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = [
                row
                for row in csv.DictReader(stream)
                if row["camera"] == camera and int(row.get("jpeg_valid", "1")) == 1
            ]
        self.rows = {
            int(row["exposure_end_device_timestamp_us"]): row for row in rows
        }
        self.timestamps = sorted(self.rows)

    def nearest(self, timestamp_us: int, tolerance_us: int) -> tuple[dict, int]:
        exact = self.rows.get(timestamp_us)
        if exact is not None:
            return exact, 0
        position = bisect.bisect_left(self.timestamps, timestamp_us)
        candidates = self.timestamps[max(0, position - 1) : position + 1]
        if not candidates:
            raise KeyError(f"No timestamp rows near {timestamp_us}")
        nearest = min(candidates, key=lambda value: abs(value - timestamp_us))
        error = nearest - timestamp_us
        if abs(error) > tolerance_us:
            raise KeyError(
                f"Nearest timestamp to {timestamp_us} differs by {error} us"
            )
        return self.rows[nearest], error


def matrix_payload(transform: np.ndarray) -> list[list[float]]:
    return np.asarray(transform, dtype=np.float64).tolist()


def rigid_payload(row: Mapping[str, str], prefix: str) -> dict:
    return {
        "position_world_m": [float(row[f"{prefix}_{axis}"]) for axis in "xyz"],
        "quaternion_world_rigid_wxyz": [
            float(row[f"{prefix}_q{axis}"]) for axis in "wxyz"
        ],
        "status": int(float(row[f"{prefix}_status"])),
    }


def find_video(module_dir: Path, pattern: str) -> Path:
    matches = sorted(module_dir.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one video matching {module_dir / pattern}: {matches}")
    return matches[0]


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    config = load_json(args.config)
    external_root = discover_external_root(data_root)
    aligned_path = data_root / "aligned_data" / "aligned_30hz.csv"
    rigid_extrinsics_path = resolve_repo_path(config["rigid_extrinsics"])
    rigid_extrinsics = load_json(rigid_extrinsics_path)
    rigid_camera_a = camera_a_mount_transform(rigid_extrinsics)
    mechanical_d = rigid_extrinsics["cameras"]["right"]
    rigid_camera_d_mechanical = np.eye(4, dtype=np.float64)
    rigid_camera_d_mechanical[:3, :3] = np.asarray(
        mechanical_d["R_rigid_camera"], dtype=np.float64
    )
    rigid_camera_d_mechanical[:3, 3] = (
        np.asarray(mechanical_d["p_rigid_camera_mm"], dtype=np.float64) / 1000.0
    )
    mocap_rigid_rigid_k = np.asarray(
        config.get("T_mocap_rigid_rigid_k", np.eye(4)), dtype=np.float64
    ).reshape(4, 4)
    tolerance_us = int(config["quality"]["timestamp_match_tolerance_us"])

    module_runtime: dict[str, dict] = {}
    for module_name, module in config["modules"].items():
        directory = external_root / module["recording_directory"]
        calibration_path = resolve_repo_path(module["intrinsics"])
        calibration = load_json(calibration_path)
        indexes = {
            "CAM_A": TimestampIndex(directory / "timestamps.csv", "CAM_A"),
            "CAM_D": TimestampIndex(directory / "timestamps.csv", "CAM_D"),
        }
        module_runtime[module_name] = {
            "config": module,
            "directory": directory,
            "calibration_path": calibration_path,
            "camera_d_camera_a": stereo_transform_d_a(calibration),
            "indexes": indexes,
            "videos": {
                "CAM_A": find_video(directory, module["camera_a"]["video_glob"]),
                "CAM_D": find_video(directory, module["camera_d"]["video_glob"]),
            },
        }

    with aligned_path.open("r", encoding="utf-8-sig", newline="") as stream:
        aligned_rows = list(csv.DictReader(stream))
    selected_rows = [
        row
        for row in aligned_rows
        if (args.start_seq is None or int(row["seq"]) >= args.start_seq)
        and (args.end_seq is None or int(row["seq"]) <= args.end_seq)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"rows": 0, "timestamp_fallback_matches": 0, "skipped_inexact_rows": 0}
    baseline_values: dict[str, list[float]] = {
        name: [] for name in module_runtime
    }
    camera_d_mechanical_differences: dict[str, list[float]] = {
        name: [] for name in module_runtime
    }
    camera_d_wrong_direction_differences: dict[str, list[float]] = {
        name: [] for name in module_runtime
    }
    with args.output.open("w", encoding="utf-8") as output:
        for row in selected_rows:
            sequence = int(row["seq"])
            item = {
                "seq": sequence,
                "camera_elapsed_sec": float(row["camera_elapsed_sec"]),
                "mocap_frame_index": int(row["mocap_frame_index"]),
                "mocap_nearest_dt_ms": float(row["mocap_nearest_dt_ms"]),
                "alignment": {
                    "external01_clock_residual_ms": float(
                        row["external01_clock_residual_ms"]
                    ),
                    "external02_clock_residual_ms": float(
                        row["external02_clock_residual_ms"]
                    ),
                },
                "modules": {},
                "cameras": {},
            }
            skip_row = False
            for module_name, runtime in module_runtime.items():
                module = runtime["config"]
                prefix = module["rigid_prefix"]
                world_rigid = rigid_world_transform(row, prefix)
                world_a, world_d = module_camera_world_transforms(
                    world_rigid @ mocap_rigid_rigid_k,
                    rigid_camera_a,
                    runtime["camera_d_camera_a"],
                )
                world_d_mechanical = (
                    world_rigid @ mocap_rigid_rigid_k @ rigid_camera_d_mechanical
                )
                world_d_wrong_direction = (
                    world_a @ runtime["camera_d_camera_a"]
                )
                item["modules"][module_name] = {
                    "rigid_prefix": prefix,
                    **rigid_payload(row, prefix),
                }
                baseline_values[module_name].append(
                    float(np.linalg.norm(world_d[:3, 3] - world_a[:3, 3]))
                )
                camera_d_mechanical_differences[module_name].append(
                    float(np.linalg.norm(world_d[:3, 3] - world_d_mechanical[:3, 3]))
                )
                camera_d_wrong_direction_differences[module_name].append(
                    float(
                        np.linalg.norm(
                            world_d_wrong_direction[:3, 3]
                            - world_d_mechanical[:3, 3]
                        )
                    )
                )
                for socket, transform in (("CAM_A", world_a), ("CAM_D", world_d)):
                    column = (
                        f"external0{1 if module_name == 'module01' else 2}_"
                        f"{socket}_exposure_end_timestamp_ms"
                    )
                    target_us = int(round(float(row[column]) * 1000.0))
                    try:
                        timestamp_row, timestamp_error = runtime["indexes"][
                            socket
                        ].nearest(target_us, tolerance_us)
                    except KeyError:
                        skip_row = True
                        break
                    if args.skip_inexact_timestamp and timestamp_error != 0:
                        skip_row = True
                        break
                    if timestamp_error:
                        counts["timestamp_fallback_matches"] += 1
                    camera_name = f"{module_name}_{socket}"
                    item["cameras"][camera_name] = {
                        "module": module_name,
                        "socket": socket,
                        "frame_index": int(timestamp_row["frame_index"]),
                        "capture_sequence": int(timestamp_row["sequence"]),
                        "exposure_end_timestamp_us": int(
                            timestamp_row["exposure_end_device_timestamp_us"]
                        ),
                        "timestamp_match_error_us": timestamp_error,
                        "T_world_camera": matrix_payload(transform),
                    }
                if skip_row:
                    break
            if skip_row:
                counts["skipped_inexact_rows"] += 1
                continue
            output.write(json.dumps(item, separators=(",", ":")) + "\n")
            counts["rows"] += 1

    report = {
        "schema": "joint_projection.aligned_multiview_manifest_report.v1",
        "data_root": str(data_root),
        "aligned_csv": str(aligned_path),
        "config": str(args.config.resolve()),
        "rigid_extrinsics": str(rigid_extrinsics_path.resolve()),
        "output": str(args.output.resolve()),
        "counts": counts,
        "sequence_range": (
            [int(selected_rows[0]["seq"]), int(selected_rows[-1]["seq"])]
            if selected_rows
            else None
        ),
        "mapping": {
            name: {
                "recording_directory": runtime["config"]["recording_directory"],
                "rigid_prefix": runtime["config"]["rigid_prefix"],
                "intrinsics": str(runtime["calibration_path"].resolve()),
                "videos": {
                    socket: str(path.resolve())
                    for socket, path in runtime["videos"].items()
                },
                "baseline_m": {
                    "median": float(np.median(baseline_values[name])),
                    "minimum": float(np.min(baseline_values[name])),
                    "maximum": float(np.max(baseline_values[name])),
                },
                "kalibr_CAM_D_vs_mechanical_CAM_D_center_difference_m": {
                    "median": float(
                        np.median(camera_d_mechanical_differences[name])
                    ),
                    "maximum": float(
                        np.max(camera_d_mechanical_differences[name])
                    ),
                },
                "wrong_direct_T_CAM_D_CAM_A_vs_mechanical_difference_m": {
                    "median": float(
                        np.median(camera_d_wrong_direction_differences[name])
                    ),
                    "minimum": float(
                        np.min(camera_d_wrong_direction_differences[name])
                    ),
                },
            }
            for name, runtime in module_runtime.items()
        },
        "known_metadata_override": config["known_metadata_override"],
        "camera_pose_policy": (
            "CAM_A uses T_world_mocap_rigid times T_mocap_rigid_rigid_K times "
            "mechanical T_rigid_K_CAM_A; "
            "CAM_D uses T_world_CAM_A times inverse(Kalibr T_CAM_D_CAM_A)."
        ),
        "T_mocap_rigid_rigid_k": mocap_rigid_rigid_k.tolist(),
        "mocap_axis_note": config.get("mocap_axis_note"),
    }
    report_path = args.output.with_name(f"{args.output.stem}_report.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if args.skip_inexact_timestamp and counts["rows"]:
        kept_seqs = set()
        with args.output.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    kept_seqs.add(int(json.loads(line)["seq"]))
        strict_aligned = aligned_path.with_name("aligned_30hz_strict.csv")
        filtered = [row for row in aligned_rows if int(row["seq"]) in kept_seqs]
        with strict_aligned.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(filtered[0].keys()))
            writer.writeheader()
            writer.writerows(filtered)
        report["strict_aligned_csv"] = str(strict_aligned.resolve())
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    print(json.dumps({"manifest": str(args.output), "report": str(report_path), **counts}))


if __name__ == "__main__":
    main()
