#!/usr/bin/env python3
"""Keep only the multi-person pose whose head is associated with an AprilTag.

The recorded fisheye images are physically upside-down, so inference is run
after a 180-degree rotation and all coordinates are mapped back to the source
1920x1200 image. Direct tag-to-head matches are used as temporal anchors.
Short gaps where the tag is hidden are filled by interpolating the target
person box between neighbouring anchored frames. A frame without a reliable
anchor or person candidate is explicitly marked as no_target.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import json
import math
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from ultralytics import YOLO


CAMERAS = ("CAM_A", "CAM_C")
SIDES = {"CAM_A": "left", "CAM_C": "right"}
KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
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
)
SKELETON_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-video", type=Path, required=True)
    parser.add_argument("--right-video", type=Path, required=True)
    parser.add_argument("--tag-csv", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=50.0)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--person-confidence", type=float, default=0.12)
    parser.add_argument("--keypoint-confidence", type=float, default=0.15)
    parser.add_argument("--max-track-gap-frames", type=int, default=75)
    parser.add_argument("--max-track-cost", type=float, default=1.65)
    parser.add_argument("--device", default="0")
    parser.add_argument("--no-annotated-video", action="store_true")
    return parser.parse_args()


def load_tags(
    path: Path,
) -> tuple[
    dict[str, dict[int, list[dict[str, object]]]],
    dict[str, dict[int, dict[str, str]]],
]:
    detections: dict[str, dict[int, list[dict[str, object]]]] = {
        camera: {} for camera in CAMERAS
    }
    frame_meta: dict[str, dict[int, dict[str, str]]] = {
        camera: {} for camera in CAMERAS
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            camera = row["camera"]
            if camera not in detections:
                continue
            frame_index = int(row["frame_index"])
            frame_meta[camera].setdefault(frame_index, row)
            detections[camera].setdefault(frame_index, [])
            if row["detected"] != "1":
                continue
            corners = [
                [float(row[f"corner{index}_x"]), float(row[f"corner{index}_y"])]
                for index in range(4)
            ]
            detections[camera][frame_index].append(
                {
                    "tag_id": int(row["tag_id"]),
                    "center": np.asarray(
                        [float(row["center_x"]), float(row["center_y"])],
                        dtype=np.float32,
                    ),
                    "corners": np.asarray(corners, dtype=np.float32),
                    "decision_margin": float(row["decision_margin"]),
                }
            )
    return detections, frame_meta


def source_to_upright(points: np.ndarray, width: int, height: int) -> np.ndarray:
    mapped = np.asarray(points, dtype=np.float32).copy()
    mapped[..., 0] = float(width - 1) - mapped[..., 0]
    mapped[..., 1] = float(height - 1) - mapped[..., 1]
    return mapped


def upright_to_source(points: np.ndarray, width: int, height: int) -> np.ndarray:
    return source_to_upright(points, width, height)


def map_box_to_source(box: np.ndarray, width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(box, dtype=np.float32)
    return np.asarray(
        [
            float(width - 1) - x2,
            float(height - 1) - y2,
            float(width - 1) - x1,
            float(height - 1) - y1,
        ],
        dtype=np.float32,
    )


def candidate_head(candidate: dict[str, object], threshold: float) -> np.ndarray:
    points = candidate["keypoints"]
    confidence = candidate["keypoint_confidence"]
    valid = confidence[:5] >= float(threshold)
    if np.any(valid):
        return points[:5][valid].mean(axis=0)
    x1, y1, x2, y2 = candidate["box"]
    return np.asarray(
        [(x1 + x2) * 0.5, y1 + 0.12 * (y2 - y1)],
        dtype=np.float32,
    )


def result_candidates(result) -> list[dict[str, object]]:
    if (
        result.boxes is None
        or result.keypoints is None
        or len(result.boxes) == 0
    ):
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    keypoints = result.keypoints.xy.detach().cpu().numpy().astype(np.float32)
    confidence = result.keypoints.conf.detach().cpu().numpy().astype(np.float32)
    return [
        {
            "box": box,
            "person_confidence": float(score),
            "keypoints": points,
            "keypoint_confidence": point_confidence,
        }
        for box, score, points, point_confidence in zip(
            boxes, scores, keypoints, confidence
        )
    ]


def infer_candidates(
    *,
    video_path: Path,
    model: YOLO,
    args: argparse.Namespace,
) -> tuple[list[list[dict[str, object]]], dict[str, int | float]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    all_candidates: list[list[dict[str, object]]] = []
    batch: list[np.ndarray] = []
    started = time.perf_counter()
    decoded = 0

    def flush() -> None:
        nonlocal batch
        if not batch:
            return
        results = model.predict(
            source=batch,
            imgsz=int(args.imgsz),
            conf=float(args.person_confidence),
            iou=0.60,
            device=args.device,
            half=True,
            verbose=False,
            batch=len(batch),
        )
        all_candidates.extend(result_candidates(result) for result in results)
        batch = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        batch.append(cv2.rotate(frame, cv2.ROTATE_180))
        decoded += 1
        if len(batch) >= int(args.batch_size):
            flush()
            print(
                json.dumps(
                    {
                        "stage": "multiperson_pose",
                        "video": video_path.name,
                        "processed": decoded,
                        "seconds": round(time.perf_counter() - started, 2),
                    }
                ),
                flush=True,
            )
    flush()
    cap.release()
    return all_candidates, {
        "width": width,
        "height": height,
        "decoded_frames": decoded,
    }


def direct_tag_match(
    candidates: list[dict[str, object]],
    tags: list[dict[str, object]],
    *,
    width: int,
    height: int,
    keypoint_threshold: float,
) -> tuple[int, list[int], float] | None:
    if not candidates or not tags:
        return None
    upright_tags = [
        (
            int(tag["tag_id"]),
            source_to_upright(tag["center"], width, height),
        )
        for tag in tags
    ]
    best: tuple[float, int, int, float] | None = None
    for candidate_index, candidate in enumerate(candidates):
        box = candidate["box"]
        x1, y1, x2, y2 = box
        box_width = max(1.0, float(x2 - x1))
        box_height = max(1.0, float(y2 - y1))
        head = candidate_head(candidate, keypoint_threshold)
        threshold = max(
            90.0,
            min(230.0, 0.28 * box_width + 0.10 * box_height),
        )
        for tag_id, tag_center in upright_tags:
            in_head_region = (
                x1 - 0.18 * box_width
                <= tag_center[0]
                <= x2 + 0.18 * box_width
                and y1 - 0.20 * box_height
                <= tag_center[1]
                <= y1 + 0.45 * box_height
            )
            distance = float(np.linalg.norm(head - tag_center))
            if not in_head_region or distance > threshold:
                continue
            normalized = distance / threshold
            score = normalized - 0.05 * float(candidate["person_confidence"])
            if best is None or score < best[0]:
                best = (score, candidate_index, tag_id, distance)
    if best is None:
        return None

    _, candidate_index, nearest_tag_id, distance = best
    candidate = candidates[candidate_index]
    head = candidate_head(candidate, keypoint_threshold)
    box = candidate["box"]
    box_width = max(1.0, float(box[2] - box[0]))
    box_height = max(1.0, float(box[3] - box[1]))
    cluster_threshold = max(
        100.0,
        min(250.0, 0.32 * box_width + 0.10 * box_height),
    )
    matched_ids = sorted(
        {
            tag_id
            for tag_id, tag_center in upright_tags
            if float(np.linalg.norm(head - tag_center)) <= cluster_threshold
        }
    )
    if nearest_tag_id not in matched_ids:
        matched_ids.append(nearest_tag_id)
        matched_ids.sort()
    return candidate_index, matched_ids, distance


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(1.0, float(first[2] - first[0])) * max(
        1.0, float(first[3] - first[1])
    )
    area_second = max(1.0, float(second[2] - second[0])) * max(
        1.0, float(second[3] - second[1])
    )
    return intersection / max(1.0, area_first + area_second - intersection)


def tracking_cost(candidate_box: np.ndarray, target_box: np.ndarray) -> float:
    candidate_center = np.asarray(
        [
            (candidate_box[0] + candidate_box[2]) * 0.5,
            (candidate_box[1] + candidate_box[3]) * 0.5,
        ]
    )
    target_center = np.asarray(
        [
            (target_box[0] + target_box[2]) * 0.5,
            (target_box[1] + target_box[3]) * 0.5,
        ]
    )
    target_width = max(1.0, float(target_box[2] - target_box[0]))
    target_height = max(1.0, float(target_box[3] - target_box[1]))
    candidate_width = max(1.0, float(candidate_box[2] - candidate_box[0]))
    candidate_height = max(1.0, float(candidate_box[3] - candidate_box[1]))
    diagonal = math.hypot(target_width, target_height)
    center_cost = float(np.linalg.norm(candidate_center - target_center)) / diagonal
    size_cost = 0.25 * (
        abs(math.log(candidate_width / target_width))
        + abs(math.log(candidate_height / target_height))
    )
    overlap_cost = 0.55 * (1.0 - box_iou(candidate_box, target_box))
    return center_cost + size_cost + overlap_cost


def select_target_track(
    *,
    candidates_by_frame: list[list[dict[str, object]]],
    tags_by_frame: dict[int, list[dict[str, object]]],
    width: int,
    height: int,
    args: argparse.Namespace,
) -> list[dict[str, object] | None]:
    direct: dict[int, dict[str, object]] = {}
    for frame_index, candidates in enumerate(candidates_by_frame):
        match = direct_tag_match(
            candidates,
            tags_by_frame.get(frame_index, []),
            width=width,
            height=height,
            keypoint_threshold=args.keypoint_confidence,
        )
        if match is None:
            continue
        candidate_index, tag_ids, head_tag_distance = match
        direct[frame_index] = {
            "candidate_index": candidate_index,
            "association": "direct_head_tag",
            "matched_tag_ids": tag_ids,
            "head_tag_distance_px": head_tag_distance,
            "track_cost": 0.0,
        }

    anchor_frames = sorted(direct)
    selected: list[dict[str, object] | None] = [None] * len(candidates_by_frame)
    for frame_index, selection in direct.items():
        selected[frame_index] = selection
    if not anchor_frames:
        return selected

    max_gap = int(args.max_track_gap_frames)
    for frame_index, candidates in enumerate(candidates_by_frame):
        if selected[frame_index] is not None or not candidates:
            continue
        position = bisect_left(anchor_frames, frame_index)
        previous_frame = anchor_frames[position - 1] if position > 0 else None
        next_frame = anchor_frames[position] if position < len(anchor_frames) else None
        previous_valid = (
            previous_frame is not None and frame_index - previous_frame <= max_gap
        )
        next_valid = next_frame is not None and next_frame - frame_index <= max_gap
        if not previous_valid and not next_valid:
            continue

        if previous_valid:
            previous_selection = direct[previous_frame]
            previous_box = candidates_by_frame[previous_frame][
                previous_selection["candidate_index"]
            ]["box"]
        if next_valid:
            next_selection = direct[next_frame]
            next_box = candidates_by_frame[next_frame][
                next_selection["candidate_index"]
            ]["box"]

        if previous_valid and next_valid:
            alpha = (frame_index - previous_frame) / float(next_frame - previous_frame)
            target_box = (1.0 - alpha) * previous_box + alpha * next_box
            association = "interpolated_between_tag_anchors"
        elif previous_valid:
            target_box = previous_box
            association = "tracked_from_previous_tag"
        else:
            target_box = next_box
            association = "tracked_from_next_tag"

        costs = [
            tracking_cost(candidate["box"], target_box)
            - 0.05 * float(candidate["person_confidence"])
            for candidate in candidates
        ]
        candidate_index = int(np.argmin(costs))
        if costs[candidate_index] > float(args.max_track_cost):
            continue
        selected[frame_index] = {
            "candidate_index": candidate_index,
            "association": association,
            "matched_tag_ids": [],
            "head_tag_distance_px": None,
            "track_cost": float(costs[candidate_index]),
        }
    return selected


def selected_candidate(
    candidates: list[dict[str, object]],
    selection: dict[str, object] | None,
) -> dict[str, object] | None:
    if selection is None:
        return None
    index = int(selection["candidate_index"])
    return candidates[index] if 0 <= index < len(candidates) else None


def write_pose_csv(
    path: Path,
    *,
    candidates: dict[str, list[list[dict[str, object]]]],
    selected: dict[str, list[dict[str, object] | None]],
    frame_meta: dict[str, dict[int, dict[str, str]]],
    dimensions: dict[str, dict[str, int | float]],
) -> None:
    fields = [
        "camera",
        "side",
        "frame_index",
        "sequence",
        "device_timestamp_us",
        "status",
        "association",
        "matched_tag_ids",
        "head_tag_distance_px",
        "track_cost",
        "person_confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "joint",
        "x",
        "y",
        "confidence",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for camera in CAMERAS:
            width = int(dimensions[camera]["width"])
            height = int(dimensions[camera]["height"])
            for frame_index, frame_candidates in enumerate(candidates[camera]):
                selection = selected[camera][frame_index]
                candidate = selected_candidate(frame_candidates, selection)
                meta = frame_meta[camera].get(frame_index, {})
                common = {
                    "camera": camera,
                    "side": SIDES[camera],
                    "frame_index": frame_index,
                    "sequence": meta.get("sequence", ""),
                    "device_timestamp_us": meta.get("device_timestamp_us", ""),
                }
                if candidate is None:
                    writer.writerow(
                        common
                        | {
                            "status": "no_target",
                            "association": "",
                            "matched_tag_ids": "",
                        }
                    )
                    continue
                points = upright_to_source(
                    candidate["keypoints"], width, height
                )
                source_box = map_box_to_source(candidate["box"], width, height)
                details = common | {
                    "status": "ok",
                    "association": selection["association"],
                    "matched_tag_ids": ",".join(
                        str(tag_id) for tag_id in selection["matched_tag_ids"]
                    ),
                    "head_tag_distance_px": (
                        ""
                        if selection["head_tag_distance_px"] is None
                        else f"{selection['head_tag_distance_px']:.6f}"
                    ),
                    "track_cost": f"{selection['track_cost']:.8f}",
                    "person_confidence": f"{candidate['person_confidence']:.8f}",
                    "bbox_x1": f"{source_box[0]:.6f}",
                    "bbox_y1": f"{source_box[1]:.6f}",
                    "bbox_x2": f"{source_box[2]:.6f}",
                    "bbox_y2": f"{source_box[3]:.6f}",
                }
                for joint_index, joint_name in enumerate(KEYPOINT_NAMES):
                    writer.writerow(
                        details
                        | {
                            "joint": joint_name,
                            "x": f"{points[joint_index, 0]:.6f}",
                            "y": f"{points[joint_index, 1]:.6f}",
                            "confidence": (
                                f"{candidate['keypoint_confidence'][joint_index]:.8f}"
                            ),
                        }
                    )


def draw_tag(frame: np.ndarray, detection: dict[str, object]) -> None:
    corners = np.rint(detection["corners"]).astype(np.int32)
    center = tuple(np.rint(detection["center"]).astype(int))
    cv2.polylines(frame, [corners], True, (0, 255, 80), 4, cv2.LINE_AA)
    cv2.circle(frame, center, 6, (0, 255, 80), -1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"TAG {detection['tag_id']}",
        (center[0] + 10, center[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 80),
        2,
        cv2.LINE_AA,
    )


def draw_selected_pose(
    frame: np.ndarray,
    candidate: dict[str, object],
    *,
    width: int,
    height: int,
    confidence_threshold: float,
) -> None:
    points = upright_to_source(candidate["keypoints"], width, height)
    confidence = candidate["keypoint_confidence"]
    for first, second in SKELETON_EDGES:
        if (
            confidence[first] >= confidence_threshold
            and confidence[second] >= confidence_threshold
        ):
            point1 = tuple(np.rint(points[first]).astype(int))
            point2 = tuple(np.rint(points[second]).astype(int))
            cv2.line(frame, point1, point2, (255, 160, 20), 5, cv2.LINE_AA)
    for joint_index, point in enumerate(points):
        if confidence[joint_index] < confidence_threshold:
            continue
        pixel = tuple(np.rint(point).astype(int))
        cv2.circle(frame, pixel, 7, (255, 160, 20), -1, cv2.LINE_AA)
    source_box = np.rint(
        map_box_to_source(candidate["box"], width, height)
    ).astype(int)
    cv2.rectangle(
        frame,
        (source_box[0], source_box[1]),
        (source_box[2], source_box[3]),
        (255, 160, 20),
        3,
    )


def render_video(
    *,
    camera: str,
    source: Path,
    destination: Path,
    fps: float,
    candidates: list[list[dict[str, object]]],
    selected: list[dict[str, object] | None],
    tags: dict[int, list[dict[str, object]]],
    dimensions: dict[str, int | float],
    confidence_threshold: float,
) -> None:
    width = int(dimensions["width"])
    height = int(dimensions["height"])
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        stdin=subprocess.PIPE,
    )
    cap = cv2.VideoCapture(str(source))
    frame_index = 0
    try:
        while frame_index < len(candidates):
            ok, frame = cap.read()
            if not ok:
                break
            for detection in tags.get(frame_index, []):
                draw_tag(frame, detection)
            selection = selected[frame_index]
            candidate = selected_candidate(candidates[frame_index], selection)
            if candidate is not None:
                draw_selected_pose(
                    frame,
                    candidate,
                    width=width,
                    height=height,
                    confidence_threshold=confidence_threshold,
                )
                tag_text = (
                    ",".join(str(value) for value in selection["matched_tag_ids"])
                    or "tracked"
                )
                status = f"TARGET tag={tag_text} | {selection['association']}"
                color = (255, 160, 20)
            else:
                status = "NO TAG-GUIDED TARGET"
                color = (0, 80, 255)
            cv2.putText(
                frame,
                f"{camera} {SIDES[camera]} | frame {frame_index:06d}",
                (24, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.78,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                status,
                (24, 76),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                color,
                2,
                cv2.LINE_AA,
            )
            assert encoder.stdin is not None
            encoder.stdin.write(frame.tobytes())
            frame_index += 1
    finally:
        cap.release()
        if encoder.stdin is not None:
            encoder.stdin.close()
        return_code = encoder.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed for {destination}: {return_code}")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tags, frame_meta = load_tags(args.tag_csv)
    model = YOLO(str(args.model))
    sources = {
        "CAM_A": args.left_video,
        "CAM_C": args.right_video,
    }
    candidates: dict[str, list[list[dict[str, object]]]] = {}
    dimensions: dict[str, dict[str, int | float]] = {}
    selected: dict[str, list[dict[str, object] | None]] = {}

    for camera in CAMERAS:
        candidates[camera], dimensions[camera] = infer_candidates(
            video_path=sources[camera],
            model=model,
            args=args,
        )
        selected[camera] = select_target_track(
            candidates_by_frame=candidates[camera],
            tags_by_frame=tags[camera],
            width=int(dimensions[camera]["width"]),
            height=int(dimensions[camera]["height"]),
            args=args,
        )

    csv_path = args.output_dir / "tag_guided_pose_2d.csv"
    write_pose_csv(
        csv_path,
        candidates=candidates,
        selected=selected,
        frame_meta=frame_meta,
        dimensions=dimensions,
    )

    annotated: dict[str, str] = {}
    if not args.no_annotated_video:
        for camera in CAMERAS:
            destination = args.output_dir / (
                f"{SIDES[camera]}_{camera}_tag_guided_pose.mp4"
            )
            render_video(
                camera=camera,
                source=sources[camera],
                destination=destination,
                fps=args.fps,
                candidates=candidates[camera],
                selected=selected[camera],
                tags=tags[camera],
                dimensions=dimensions[camera],
                confidence_threshold=args.keypoint_confidence,
            )
            annotated[camera] = destination.name

    summary = {
        "status": "complete",
        "schema": "hearwristcam_tag_guided_multiperson_pose.v1",
        "method": (
            "YOLO multi-person COCO-17 pose on 180-degree rotated frames; "
            "select only a person whose head matches AprilTag 36h11 ID 0-5; "
            "fill short occlusion gaps by tag-anchor box interpolation"
        ),
        "model": str(args.model),
        "keypoint_names": list(KEYPOINT_NAMES),
        "parameters": {
            "fps": args.fps,
            "imgsz": args.imgsz,
            "person_confidence": args.person_confidence,
            "keypoint_confidence": args.keypoint_confidence,
            "max_track_gap_frames": args.max_track_gap_frames,
            "max_track_cost": args.max_track_cost,
        },
        "videos": {},
        "outputs": {
            "pose_csv": csv_path.name,
            "annotated_videos": annotated,
        },
    }
    for camera in CAMERAS:
        association_counts: dict[str, int] = {}
        matched_frames = 0
        people_counts = []
        for frame_candidates, selection in zip(
            candidates[camera], selected[camera]
        ):
            people_counts.append(len(frame_candidates))
            if selection is None:
                association = "no_target"
            else:
                association = str(selection["association"])
                matched_frames += 1
            association_counts[association] = (
                association_counts.get(association, 0) + 1
            )
        summary["videos"][camera] = {
            **dimensions[camera],
            "frames_with_any_tag": sum(
                bool(tags[camera].get(frame_index))
                for frame_index in range(len(candidates[camera]))
            ),
            "target_pose_frames": matched_frames,
            "no_target_frames": len(candidates[camera]) - matched_frames,
            "association_counts": association_counts,
            "mean_person_candidates": float(np.mean(people_counts)),
            "max_person_candidates": max(people_counts, default=0),
            "annotated_video": annotated.get(camera),
        }
    (args.output_dir / "tag_guided_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
