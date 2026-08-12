#!/usr/bin/env python3
"""Recover a mocap-world body+foot skeleton from two external stereo modules.

Body = COCO-17. Feet = COCO-WholeBody (big/small toe + heel per side) from RTMW.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import itertools
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from multiview_geometry import (
    CameraPose,
    Observation,
    OmniCamera,
    TriangulationResult,
    load_json,
    triangulate_observations,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "0806_dual_external_mocap.json"
NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
    # COCO-WholeBody feet (RTMW indices 17-22); aliases left_toe/right_toe = big toe.
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
    "left_toe", "right_toe",
)
EDGES = (
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"), ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"), ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    # Foot direction: ankle -> toes/heel; toe span.
    ("left_ankle", "left_big_toe"), ("left_ankle", "left_small_toe"),
    ("left_ankle", "left_heel"), ("left_big_toe", "left_small_toe"),
    ("right_ankle", "right_big_toe"), ("right_ankle", "right_small_toe"),
    ("right_ankle", "right_heel"), ("right_big_toe", "right_small_toe"),
    ("left_ankle", "left_toe"), ("right_ankle", "right_toe"),
)
FILTER_2D = {
    "left_shoulder": (5, 0.12), "right_shoulder": (5, 0.12),
    "left_hip": (5, 0.12), "right_hip": (5, 0.12),
    # Limbs: slightly lighter than torso so fast hand/foot motion keeps up,
    # while keeping a short median + EMA to suppress jitter.
    "left_elbow": (3, 0.20), "right_elbow": (3, 0.20),
    "left_knee": (2, 0.26), "right_knee": (2, 0.26),
    "left_wrist": (2, 0.28), "right_wrist": (2, 0.28),
    "left_ankle": (2, 0.40), "right_ankle": (2, 0.40),
    "left_big_toe": (2, 0.45), "left_small_toe": (2, 0.45), "left_heel": (2, 0.42),
    "right_big_toe": (2, 0.45), "right_small_toe": (2, 0.45), "right_heel": (2, 0.42),
    "left_toe": (2, 0.45), "right_toe": (2, 0.45),
}
METHOD_CAMERAS = {
    "module01": ("module01_CAM_A", "module01_CAM_D"),
    "module02": ("module02_CAM_A", "module02_CAM_D"),
    "multiview": (
        "module01_CAM_A", "module01_CAM_D",
        "module02_CAM_A", "module02_CAM_D",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidates-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--start-seq", type=int)
    parser.add_argument("--end-seq", type=int, help="Inclusive aligned sequence")
    parser.add_argument("--max-candidates", type=int, default=3)
    return parser.parse_args()


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def load_jsonl(path: Path, *, indexed: bool = False):
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                rows.append(json.loads(line))
    return {int(row["frame_index"]): row for row in rows} if indexed else rows


def camera_models(config: dict) -> dict[str, OmniCamera]:
    output = {}
    for module_name, module in config["modules"].items():
        calibration = load_json(resolve_repo_path(module["intrinsics"]))
        for socket in ("CAM_A", "CAM_D"):
            name = f"{module_name}_{socket}"
            output[name] = OmniCamera.from_calibration(
                calibration, socket, name=name
            )
    return output


def result_payload(result: TriangulationResult) -> dict:
    return {
        "xyz_world_m": result.point_world.tolist(),
        "used_cameras": list(result.used_cameras),
        "rejected_cameras": list(result.rejected_cameras),
        "view_count": len(result.used_cameras),
        "reprojection_errors_px": result.reprojection_errors_px,
        "ray_misses_m": result.ray_misses_m,
        "condition_number": result.condition_number,
        "maximum_ray_angle_deg": result.maximum_ray_angle_deg,
    }


def quality_kwargs(config: dict) -> dict:
    quality = config["quality"]
    return {
        "minimum_confidence": float(quality["keypoint_confidence"]),
        "minimum_ray_angle_deg": float(quality["minimum_ray_angle_deg"]),
        "maximum_reprojection_error_px": float(
            quality["maximum_reprojection_error_px"]
        ),
        "robust_loss_scale_px": float(quality["robust_loss_scale_px"]),
    }


def candidate_observations(
    candidate_by_camera: dict[str, dict],
    poses: dict[str, CameraPose],
    joint: str,
    uv_key: str,
) -> list[Observation]:
    observations = []
    for camera_name, candidate in candidate_by_camera.items():
        keypoint = candidate["keypoints"].get(joint)
        if keypoint is None:
            continue
        uv = keypoint[uv_key] if isinstance(keypoint, dict) else keypoint[:2]
        confidence = (
            float(keypoint["confidence"])
            if isinstance(keypoint, dict)
            else float(keypoint[2])
        )
        observations.append(
            Observation(
                camera_name=camera_name,
                uv=np.asarray(uv, dtype=np.float64),
                confidence=confidence,
                pose=poses[camera_name],
            )
        )
    return observations


def stereo_hypotheses(
    module_name: str,
    candidates: dict[str, list[dict]],
    poses: dict[str, CameraPose],
    config: dict,
    maximum_candidates: int,
    previous_center: np.ndarray | None,
) -> list[dict]:
    camera_a, camera_d = METHOD_CAMERAS[module_name]
    answers = []
    for index_a, candidate_a in enumerate(candidates[camera_a][:maximum_candidates]):
        for index_d, candidate_d in enumerate(candidates[camera_d][:maximum_candidates]):
            selected = {camera_a: candidate_a, camera_d: candidate_d}
            joints = {}
            misses = []
            confidences = []
            for joint in NAMES:
                observations = candidate_observations(selected, poses, joint, "raw_uv")
                result = triangulate_observations(
                    observations, **quality_kwargs(config)
                )
                if result is None:
                    continue
                joints[joint] = result
                misses.extend(result.ray_misses_m.values())
                confidences.extend(item.confidence for item in observations)
            if len(joints) < 4:
                continue
            torso = [
                joints[name].point_world
                for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
                if name in joints
            ]
            center = np.mean(
                torso if len(torso) >= 2 else [item.point_world for item in joints.values()],
                axis=0,
            )
            temporal = (
                0.0
                if previous_center is None
                else min(float(np.linalg.norm(center - previous_center)), 2.0)
            )
            miss = float(np.median(misses)) if misses else math.inf
            confidence = float(np.mean(confidences)) if confidences else 0.0
            score = len(joints) * 2.0 + confidence * 4.0 - miss * 100.0 - temporal * 3.0
            answers.append(
                {
                    "indices": {camera_a: index_a, camera_d: index_d},
                    "candidates": selected,
                    "joints": joints,
                    "center": center,
                    "score": score,
                    "median_ray_miss_m": miss,
                }
            )
    return sorted(answers, key=lambda item: item["score"], reverse=True)


def choose_person(
    module_hypotheses: dict[str, list[dict]],
    previous_center: np.ndarray | None,
    center_gate_m: float,
) -> tuple[dict[str, dict], dict, np.ndarray] | None:
    first = module_hypotheses["module01"][:5]
    second = module_hypotheses["module02"][:5]
    combinations = []
    for left, right in itertools.product(first, second):
        common = set(left["joints"]) & set(right["joints"])
        disagreement = (
            float(
                np.median(
                    [
                        np.linalg.norm(
                            left["joints"][joint].point_world
                            - right["joints"][joint].point_world
                        )
                        for joint in common
                    ]
                )
            )
            if common
            else math.inf
        )
        center = (left["center"] + right["center"]) * 0.5
        center_distance = float(np.linalg.norm(left["center"] - right["center"]))
        temporal = (
            0.0
            if previous_center is None
            else float(np.linalg.norm(center - previous_center))
        )
        gate_penalty = max(0.0, center_distance - center_gate_m) * 100.0
        score = (
            left["score"]
            + right["score"]
            - disagreement * 25.0
            - temporal * 4.0
            - gate_penalty
        )
        combinations.append(
            (
                score,
                left,
                right,
                center,
                disagreement,
                center_distance <= center_gate_m,
            )
        )
    if combinations:
        score, left, right, center, disagreement, gate_pass = max(
            combinations, key=lambda item: item[0]
        )
        selected = {**left["candidates"], **right["candidates"]}
        association = {
            "score": score,
            "module01_indices": left["indices"],
            "module02_indices": right["indices"],
            "stereo_center_distance_m": float(
                np.linalg.norm(left["center"] - right["center"])
            ),
            "median_joint_disagreement_m": disagreement,
            "cross_module_gate_pass": gate_pass,
        }
        return selected, association, center

    available = [
        (name, hypotheses[0])
        for name, hypotheses in module_hypotheses.items()
        if hypotheses
    ]
    if not available:
        return None
    name, best = max(available, key=lambda item: item[1]["score"])
    return (
        dict(best["candidates"]),
        {
            "score": best["score"],
            f"{name}_indices": best["indices"],
            "single_module_fallback": name,
            "cross_module_gate_pass": False,
        },
        best["center"],
    )


def selected_keypoints(candidate_by_camera: dict[str, dict]) -> dict:
    output = {}
    for camera_name, candidate in candidate_by_camera.items():
        output[camera_name] = {
            joint: {
                "raw_uv": [float(values[0]), float(values[1])],
                "filtered_uv": [float(values[0]), float(values[1])],
                "confidence": float(values[2]),
            }
            for joint, values in candidate["keypoints"].items()
        }
    return output


def triangulate_methods(
    keypoints: dict[str, dict],
    poses: dict[str, CameraPose],
    config: dict,
    uv_key: str,
) -> dict:
    methods = {}
    for method, camera_names in METHOD_CAMERAS.items():
        selected = {
            camera_name: {"keypoints": keypoints[camera_name]}
            for camera_name in camera_names
            if camera_name in keypoints
        }
        joints = {}
        for joint in NAMES:
            observations = candidate_observations(selected, poses, joint, uv_key)
            result = triangulate_observations(
                observations, **quality_kwargs(config)
            )
            if result is not None:
                joints[joint] = result_payload(result)
        methods[method] = joints
    return methods


def zero_phase_ema(values: np.ndarray, alpha: float) -> np.ndarray:
    if len(values) < 2:
        return values.copy()
    forward = values.copy()
    for index in range(1, len(values)):
        forward[index] = alpha * values[index] + (1.0 - alpha) * forward[index - 1]
    backward = values.copy()
    for index in range(len(values) - 2, -1, -1):
        backward[index] = alpha * values[index] + (1.0 - alpha) * backward[index + 1]
    return (forward + backward) * 0.5


def robust_filter_2d(values: np.ndarray, radius: int, alpha: float) -> np.ndarray:
    if len(values) < 3:
        return values.copy()
    median = np.empty_like(values)
    for index in range(len(values)):
        median[index] = np.median(
            values[max(0, index - radius) : min(len(values), index + radius + 1)],
            axis=0,
        )
    residual = np.linalg.norm(values - median, axis=1)
    scale = max(float(np.median(residual)) * 3.5, 4.0)
    weight = np.minimum(1.0, scale / np.maximum(residual, 1e-9))[:, None]
    return zero_phase_ema(median + (values - median) * weight, alpha)


def filter_selected_2d(records: list[dict]) -> dict:
    statistics = {}
    for camera_name in METHOD_CAMERAS["multiview"]:
        for joint in NAMES:
            radius, alpha = FILTER_2D.get(joint, (2, 0.27))
            samples = [
                (
                    record_index,
                    int(record["seq"]),
                    np.asarray(
                        record["observations"][camera_name][joint]["raw_uv"],
                        dtype=np.float64,
                    ),
                )
                for record_index, record in enumerate(records)
                if camera_name in record["observations"]
                and joint in record["observations"][camera_name]
            ]
            raw_steps, filtered_steps = [], []
            start = 0
            while start < len(samples):
                end = start + 1
                while (
                    end < len(samples)
                    and samples[end][1] - samples[end - 1][1] <= 3
                ):
                    end += 1
                segment = samples[start:end]
                values = np.asarray([sample[2] for sample in segment])
                smoothed = robust_filter_2d(values, radius, alpha)
                for (record_index, _, _), uv in zip(segment, smoothed):
                    records[record_index]["observations"][camera_name][joint][
                        "filtered_uv"
                    ] = uv.tolist()
                if len(values) > 1:
                    raw_steps.extend(np.linalg.norm(np.diff(values, axis=0), axis=1))
                    filtered_steps.extend(
                        np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
                    )
                start = end
            statistics[f"{camera_name}:{joint}"] = {
                "samples": len(samples),
                "median_step_raw_px": (
                    float(np.median(raw_steps)) if raw_steps else 0.0
                ),
                "median_step_filtered_px": (
                    float(np.median(filtered_steps)) if filtered_steps else 0.0
                ),
                "median_radius_frames": radius,
                "zero_phase_ema_alpha": alpha,
            }
    return statistics


def point_statistics(values: Iterable[float]) -> dict:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0, "median": None, "p90": None}
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
    }


def method_statistics(records: list[dict], stage: str, method: str) -> dict:
    points = []
    reprojections = []
    angles = []
    conditions = []
    view_counts = Counter()
    frames = set()
    for record in records:
        for payload in record["methods"][stage][method].values():
            points.append(payload["xyz_world_m"])
            reprojections.extend(payload["reprojection_errors_px"].values())
            angles.append(payload["maximum_ray_angle_deg"])
            conditions.append(payload["condition_number"])
            view_counts[str(payload["view_count"])] += 1
            frames.add(record["seq"])
    return {
        "points": len(points),
        "frames_with_points": len(frames),
        "view_count_distribution": dict(sorted(view_counts.items())),
        "reprojection_error_px": point_statistics(reprojections),
        "maximum_ray_angle_deg": point_statistics(angles),
        "condition_number": point_statistics(conditions),
    }


def stereo_disagreement(records: list[dict], stage: str) -> dict:
    distances = []
    for record in records:
        first = record["methods"][stage]["module01"]
        second = record["methods"][stage]["module02"]
        for joint in set(first) & set(second):
            distances.append(
                np.linalg.norm(
                    np.asarray(first[joint]["xyz_world_m"])
                    - np.asarray(second[joint]["xyz_world_m"])
                )
            )
    return point_statistics(distances)


def bone_statistics(records: list[dict], stage: str, method: str) -> dict:
    lengths = defaultdict(list)
    for record in records:
        joints = record["methods"][stage][method]
        for first, second in EDGES:
            if first in joints and second in joints:
                lengths[f"{first}-{second}"].append(
                    float(
                        np.linalg.norm(
                            np.asarray(joints[first]["xyz_world_m"])
                            - np.asarray(joints[second]["xyz_world_m"])
                        )
                    )
                )
    edges = {}
    cvs = []
    for edge, values in lengths.items():
        array = np.asarray(values)
        mean = float(np.mean(array))
        coefficient = float(np.std(array) / mean) if mean > 1e-9 else math.nan
        if np.isfinite(coefficient):
            cvs.append(coefficient)
        edges[edge] = {
            "count": len(values),
            "median_m": float(np.median(array)),
            "coefficient_of_variation": coefficient,
        }
    return {
        "median_edge_coefficient_of_variation": (
            float(np.median(cvs)) if cvs else None
        ),
        "edges": edges,
    }


def temporal_second_difference(records: list[dict], stage: str, method: str) -> dict:
    by_joint = defaultdict(list)
    for record in records:
        for joint, payload in record["methods"][stage][method].items():
            by_joint[joint].append(
                (record["seq"], np.asarray(payload["xyz_world_m"], dtype=np.float64))
            )
    values = []
    for samples in by_joint.values():
        samples.sort(key=lambda item: item[0])
        for first, middle, last in zip(samples, samples[1:], samples[2:]):
            if middle[0] - first[0] == 1 and last[0] - middle[0] == 1:
                values.append(
                    1000.0
                    * float(np.linalg.norm(last[1] - 2.0 * middle[1] + first[1]))
                )
    result = point_statistics(values)
    result["unit"] = "mm_per_frame_squared"
    return result


def write_csv(records: list[dict], path: Path) -> None:
    fields = ["seq", "joint"]
    for method in METHOD_CAMERAS:
        for stage in ("raw", "filtered"):
            fields.extend(
                [
                    f"{method}_{stage}_x_m",
                    f"{method}_{stage}_y_m",
                    f"{method}_{stage}_z_m",
                    f"{method}_{stage}_view_count",
                    f"{method}_{stage}_median_reprojection_px",
                ]
            )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            all_joints = set()
            for stage in ("raw", "filtered"):
                for method in METHOD_CAMERAS:
                    all_joints.update(record["methods"][stage][method])
            for joint in sorted(
                all_joints, key=lambda name: NAMES.index(name) if name in NAMES else 999
            ):
                row = {"seq": record["seq"], "joint": joint}
                for method in METHOD_CAMERAS:
                    for stage in ("raw", "filtered"):
                        payload = record["methods"][stage][method].get(joint)
                        if payload is None:
                            continue
                        x, y, z = payload["xyz_world_m"]
                        row.update(
                            {
                                f"{method}_{stage}_x_m": x,
                                f"{method}_{stage}_y_m": y,
                                f"{method}_{stage}_z_m": z,
                                f"{method}_{stage}_view_count": payload["view_count"],
                                f"{method}_{stage}_median_reprojection_px": float(
                                    np.median(
                                        list(payload["reprojection_errors_px"].values())
                                    )
                                ),
                            }
                        )
                writer.writerow(row)


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    models = camera_models(config)
    manifest = [
        row
        for row in load_jsonl(args.manifest)
        if (args.start_seq is None or int(row["seq"]) >= args.start_seq)
        and (args.end_seq is None or int(row["seq"]) <= args.end_seq)
    ]
    candidates = {
        camera_name: load_jsonl(
            args.candidates_dir / f"{camera_name}.jsonl", indexed=True
        )
        for camera_name in METHOD_CAMERAS["multiview"]
    }

    records = []
    previous_center = None
    camera_candidate_counts = Counter()
    for frame in manifest:
        poses = {
            camera_name: CameraPose(
                camera=models[camera_name],
                world_camera=np.asarray(payload["T_world_camera"], dtype=np.float64),
            )
            for camera_name, payload in frame["cameras"].items()
        }
        frame_candidates = {}
        for camera_name, payload in frame["cameras"].items():
            source = candidates[camera_name].get(payload["frame_index"], {})
            frame_candidates[camera_name] = []
            for candidate in source.get("candidates", []):
                copied = dict(candidate)
                copied["keypoints"] = {
                    joint: {
                        "raw_uv": [float(values[0]), float(values[1])],
                        "filtered_uv": [float(values[0]), float(values[1])],
                        "confidence": float(values[2]),
                    }
                    for joint, values in candidate["keypoints"].items()
                }
                frame_candidates[camera_name].append(copied)
            camera_candidate_counts[camera_name] += len(frame_candidates[camera_name])

        hypotheses = {
            module: stereo_hypotheses(
                module,
                frame_candidates,
                poses,
                config,
                args.max_candidates,
                previous_center,
            )
            for module in ("module01", "module02")
        }
        chosen = choose_person(
            hypotheses,
            previous_center,
            float(config["quality"]["cross_module_center_gate_m"]),
        )
        if chosen is None:
            continue
        selected, association, previous_center = chosen
        observations = {
            camera_name: candidate["keypoints"]
            for camera_name, candidate in selected.items()
        }
        record = {
            "seq": int(frame["seq"]),
            "camera_elapsed_sec": frame["camera_elapsed_sec"],
            "frames": {
                camera_name: payload["frame_index"]
                for camera_name, payload in frame["cameras"].items()
            },
            "camera_poses": {
                camera_name: payload["T_world_camera"]
                for camera_name, payload in frame["cameras"].items()
            },
            "association": association,
            "observations": observations,
            "methods": {
                "raw": triangulate_methods(
                    observations, poses, config, "raw_uv"
                )
            },
        }
        records.append(record)

    filter_statistics = filter_selected_2d(records)
    for record in records:
        poses = {
            camera_name: CameraPose(
                camera=models[camera_name],
                world_camera=np.asarray(transform, dtype=np.float64),
            )
            for camera_name, transform in record["camera_poses"].items()
        }
        record["methods"]["filtered"] = triangulate_methods(
            record["observations"], poses, config, "filtered_uv"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = args.output_dir / "multiview_3d_results.jsonl"
    with output_jsonl.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    output_csv = args.output_dir / "multiview_3d.csv"
    write_csv(records, output_csv)

    report = {
        "schema": "joint_projection.external_multiview_3d_report.v1",
        "manifest": str(args.manifest.resolve()),
        "config": str(args.config.resolve()),
        "candidates_dir": str(args.candidates_dir.resolve()),
        "outputs": {
            "jsonl": str(output_jsonl.resolve()),
            "csv": str(output_csv.resolve()),
        },
        "frames": {
            "manifest": len(manifest),
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
        "candidate_counts": dict(camera_candidate_counts),
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
        "filter_2d": filter_statistics,
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
