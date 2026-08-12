#!/usr/bin/env python3
"""Detect body-nose and 68-point face-nose candidates with RTMW WholeBody.

Supports remuxed .mp4 via OpenCV and elementary .h265 via timestamps.csv packet sizes
(same capture-index convention as render_multiview_to_head.py).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from rtmlib import Wholebody

from render_multiview_to_head import H265CaptureReader, HeadTimestampIndex, infer_head_module_from_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--timestamps",
        type=Path,
        help="Required when --video is elementary .h265",
    )
    parser.add_argument(
        "--camera",
        help="Camera name in timestamps.csv (default: inferred from filename)",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--mode", default="performance")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--face-start-index", type=int, default=23)
    parser.add_argument(
        "--face-nose-index",
        type=int,
        default=30,
        help="68-point face index; 30 is the tip at the end of the nose bridge.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=48,
        help="Detect this many evenly spaced frames, robust-average to one fixed "
        "nose UV, then expand to the full timeline (production default). "
        "Set 0 to detect every frame (legacy / ablation).",
    )
    return parser.parse_args()


def choose_person(keypoints: np.ndarray, scores: np.ndarray) -> int | None:
    best_index, best_area = None, -1.0
    for index, (points, confidence) in enumerate(zip(keypoints, scores)):
        visible = np.isfinite(points[:17]).all(axis=1) & (confidence[:17] >= 0.1)
        if int(visible.sum()) < 4:
            continue
        xy = points[:17][visible]
        area = float(np.prod(np.maximum(xy.max(axis=0) - xy.min(axis=0), 0.0)))
        if area > best_area:
            best_index, best_area = index, area
    return best_index


def robust_fixed_point(
    points: dict[int, np.ndarray],
) -> tuple[np.ndarray, dict[int, bool], dict]:
    indices = sorted(points)
    values = np.asarray([points[index] for index in indices], dtype=np.float64)
    center = np.median(values, axis=0)
    radial = np.linalg.norm(values - center, axis=1)
    radial_center = float(np.median(radial))
    mad = float(np.median(np.abs(radial - radial_center)))
    threshold = radial_center + max(4.0, 4.5 * 1.4826 * mad)
    keep_array = radial <= threshold
    fixed = np.median(values[keep_array], axis=0)
    keep = {index: bool(flag) for index, flag in zip(indices, keep_array)}
    residual = np.linalg.norm(values[keep_array] - fixed, axis=1)
    stats = {
        "raw_samples": int(len(values)),
        "inlier_samples": int(keep_array.sum()),
        "outlier_samples": int((~keep_array).sum()),
        "outlier_threshold_px": float(threshold),
        "raw_to_fixed_median_px": float(np.median(residual)),
        "raw_to_fixed_p90_px": float(np.percentile(residual, 90)),
        "fixed_uv_px": fixed.tolist(),
    }
    return fixed, keep, stats


def sample_frame_indices(total: int, sample_count: int) -> list[int]:
    if sample_count <= 0 or sample_count >= total:
        return list(range(total))
    if sample_count == 1:
        return [total // 2]
    return [
        int(round(index * (total - 1) / (sample_count - 1)))
        for index in range(sample_count)
    ]


def row_from_detection(
    frame_index: int,
    capture_sequence: int | None,
    keypoints: np.ndarray,
    scores: np.ndarray,
    face_index: int,
) -> dict:
    person = choose_person(keypoints, scores) if keypoints.ndim == 3 else None
    row = {
        "frame_index": frame_index,
        "detected": int(person is not None),
        "body_nose_u_px": "",
        "body_nose_v_px": "",
        "body_nose_score": "",
        "face_nose_u_px": "",
        "face_nose_v_px": "",
        "face_nose_score": "",
    }
    if capture_sequence is not None:
        row["capture_sequence"] = capture_sequence
    if person is not None:
        row.update(
            {
                "body_nose_u_px": float(keypoints[person, 0, 0]),
                "body_nose_v_px": float(keypoints[person, 0, 1]),
                "body_nose_score": float(scores[person, 0]),
            }
        )
        if face_index < keypoints.shape[1]:
            row.update(
                {
                    "face_nose_u_px": float(keypoints[person, face_index, 0]),
                    "face_nose_v_px": float(keypoints[person, face_index, 1]),
                    "face_nose_score": float(scores[person, face_index]),
                }
            )
    return row


def expand_fixed_rows(
    *,
    total_frames: int,
    index_rows: list[dict],
    fixed_uv: np.ndarray,
    fixed_score: float,
    sample_keep: dict[int, bool],
) -> list[dict]:
    rows: list[dict] = []
    for frame_index in range(total_frames):
        capture_sequence = None
        if frame_index < len(index_rows):
            capture_sequence = int(index_rows[frame_index].get("seq", frame_index))
        rows.append(
            {
                "frame_index": frame_index,
                "capture_sequence": capture_sequence
                if capture_sequence is not None
                else frame_index,
                "detected": 1,
                "body_nose_u_px": float(fixed_uv[0]),
                "body_nose_v_px": float(fixed_uv[1]),
                "body_nose_score": fixed_score,
                "face_nose_u_px": float(fixed_uv[0]),
                "face_nose_v_px": float(fixed_uv[1]),
                "face_nose_score": fixed_score,
                "fixed_nose": 1,
                "sample_inlier": int(sample_keep.get(frame_index, False)),
            }
        )
    return rows


def infer_camera(video: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    name = video.name.upper()
    for camera in ("CAM_A", "CAM_D", "CAM_B", "CAM_C"):
        if camera in name:
            return camera
    raise ValueError(f"Could not infer camera from {video.name}; pass --camera")


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    model = Wholebody(
        mode=args.mode, to_openpose=False, backend="onnxruntime", device=args.device
    )
    face_index = args.face_start_index + args.face_nose_index
    rows: list[dict] = []
    output_shape = None
    fixed_meta: dict | None = None
    index_rows: list[dict] = []

    if args.video.suffix.lower() == ".h265":
        if args.timestamps is None:
            candidate = args.video.parent / "timestamps.csv"
            if not candidate.is_file():
                raise ValueError("--timestamps is required for .h265 input")
            args.timestamps = candidate
        camera = infer_camera(args.video, args.camera)
        module = infer_head_module_from_video(args.video)
        index = HeadTimestampIndex(args.timestamps, camera, module=module)
        index_rows = index.rows
        reader = H265CaptureReader(args.video, index.rows)
        try:
            end = len(index.rows) if args.max_frames is None else args.start_frame + args.max_frames
            end = min(end, len(index.rows))
            total_frames = end - args.start_frame
            if args.sample_count > 0:
                sample_indices = sample_frame_indices(total_frames, args.sample_count)
                sample_indices = [
                    args.start_frame + index for index in sample_indices
                ]
                raw_points: dict[int, np.ndarray] = {}
                raw_scores: dict[int, float] = {}
                for frame_index in sample_indices:
                    frame = reader.read(frame_index)
                    keypoints, scores = model(frame)
                    keypoints, scores = np.asarray(keypoints), np.asarray(scores)
                    output_shape = list(keypoints.shape)
                    row = row_from_detection(
                        frame_index,
                        int(index.rows[frame_index]["seq"]),
                        keypoints,
                        scores,
                        face_index,
                    )
                    rows.append(row)
                    if row["face_nose_u_px"] != "":
                        raw_points[frame_index] = np.asarray(
                            [row["face_nose_u_px"], row["face_nose_v_px"]],
                            dtype=np.float64,
                        )
                        raw_scores[frame_index] = float(row["face_nose_score"])
                if not raw_points:
                    raise RuntimeError("No valid face-nose detections in sampled frames")
                fixed_uv, sample_keep, stats = robust_fixed_point(raw_points)
                fixed_score = float(np.median(list(raw_scores.values())))
                rows = expand_fixed_rows(
                    total_frames=total_frames,
                    index_rows=index_rows[args.start_frame : end],
                    fixed_uv=fixed_uv,
                    fixed_score=fixed_score,
                    sample_keep=sample_keep,
                )
                fixed_meta = {
                    "schema": "joint_projection.head_rtmw_fixed_nose.v1",
                    "camera": camera,
                    "policy": "sample_evenly_then_robust_median_expand_full_timeline",
                    "sample_count_requested": int(args.sample_count),
                    "sample_frames_detected": len(sample_indices),
                    "total_frames": int(total_frames),
                    "fixed_uv_px": fixed_uv.tolist(),
                    "fixed_score": fixed_score,
                    "sample_stats": stats,
                }
            else:
                for frame_index in range(args.start_frame, end):
                    frame = reader.read(frame_index)
                    keypoints, scores = model(frame)
                    keypoints, scores = np.asarray(keypoints), np.asarray(scores)
                    output_shape = list(keypoints.shape)
                    rows.append(
                        row_from_detection(
                            frame_index,
                            int(index.rows[frame_index]["seq"]),
                            keypoints,
                            scores,
                            face_index,
                        )
                    )
        finally:
            reader.close()
    else:
        capture = cv2.VideoCapture(str(args.video))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open {args.video}")
        frame_index = 0
        while frame_index < args.start_frame:
            ok, _ = capture.read()
            if not ok:
                break
            frame_index += 1
        all_frames: list[tuple[int, np.ndarray]] = []
        while args.max_frames is None or len(all_frames) < args.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            all_frames.append((frame_index, frame))
            frame_index += 1
        capture.release()
        total_frames = len(all_frames)
        if args.sample_count > 0:
            sample_positions = sample_frame_indices(total_frames, args.sample_count)
            raw_points = {}
            raw_scores = {}
            for position in sample_positions:
                frame_index, frame = all_frames[position]
                keypoints, scores = model(frame)
                keypoints, scores = np.asarray(keypoints), np.asarray(scores)
                output_shape = list(keypoints.shape)
                row = row_from_detection(frame_index, None, keypoints, scores, face_index)
                rows.append(row)
                if row["face_nose_u_px"] != "":
                    raw_points[frame_index] = np.asarray(
                        [row["face_nose_u_px"], row["face_nose_v_px"]], dtype=np.float64
                    )
                    raw_scores[frame_index] = float(row["face_nose_score"])
            if not raw_points:
                raise RuntimeError("No valid face-nose detections in sampled frames")
            fixed_uv, sample_keep, stats = robust_fixed_point(raw_points)
            fixed_score = float(np.median(list(raw_scores.values())))
            rows = expand_fixed_rows(
                total_frames=total_frames,
                index_rows=[],
                fixed_uv=fixed_uv,
                fixed_score=fixed_score,
                sample_keep=sample_keep,
            )
            fixed_meta = {
                "schema": "joint_projection.head_rtmw_fixed_nose.v1",
                "camera": infer_camera(args.video, args.camera),
                "policy": "sample_evenly_then_robust_median_expand_full_timeline",
                "sample_count_requested": int(args.sample_count),
                "sample_frames_detected": len(sample_positions),
                "total_frames": int(total_frames),
                "fixed_uv_px": fixed_uv.tolist(),
                "fixed_score": fixed_score,
                "sample_stats": stats,
            }
        else:
            for frame_index, frame in all_frames:
                keypoints, scores = model(frame)
                keypoints, scores = np.asarray(keypoints), np.asarray(scores)
                output_shape = list(keypoints.shape)
                rows.append(
                    row_from_detection(frame_index, None, keypoints, scores, face_index)
                )

    with args.output_csv.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if fixed_meta is not None:
        meta_path = args.output_csv.with_suffix(".fixed.json")
        meta_path.write_text(
            json.dumps(fixed_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        {
            "frames": len(rows),
            "detected": sum(int(row["detected"]) for row in rows),
            "output_shape": output_shape,
            "face_global_index": face_index,
            "fixed_nose": fixed_meta is not None,
            "output": str(args.output_csv),
        }
    )


if __name__ == "__main__":
    main()
