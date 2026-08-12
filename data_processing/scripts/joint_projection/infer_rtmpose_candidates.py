#!/usr/bin/env python3
"""Run RTMW WholeBody on a video and emit body+foot candidates for triangulation.

Uses RTMW (COCO-WholeBody): 17 body + 6 foot keypoints. Face/hands are ignored.
Foot joints give triangulable toe/heel tips for foot direction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from rtmlib import Wholebody


# COCO-17 body + COCO-WholeBody foot (indices 17-22 in RTMW output).
BODY_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)
FOOT_NAMES = (
    "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel",
)
NAMES = BODY_NAMES + FOOT_NAMES
# Keep legacy aliases used elsewhere in this repo (big toe).
ALIAS_TO_SOURCE = {
    "left_toe": "left_big_toe",
    "right_toe": "right_big_toe",
}
EMIT_NAMES = NAMES + tuple(ALIAS_TO_SOURCE)
FOOT_START = len(BODY_NAMES)  # 17
MIN_KEYPOINTS = FOOT_START + len(FOOT_NAMES)  # 23


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=("performance", "balanced", "lightweight"),
                   default="performance")
    p.add_argument("--device", default="cuda")
    p.add_argument("--backend", default="onnxruntime")
    p.add_argument("--rotate-180", action="store_true")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--start-frame", type=int, default=0,
                   help="First compact-video frame to process; emitted frame_index stays global.")
    p.add_argument(
        "--ranges",
        help=(
            "Optional comma-separated start:count ranges. The model is loaded once and "
            "each range keeps the source-video frame index, e.g. 104:103,1455:102."
        ),
    )
    p.add_argument("--visible-threshold", type=float, default=0.05)
    return p.parse_args()


def candidate(points: np.ndarray, scores: np.ndarray, width: int, height: int,
              rotated: bool, threshold: float) -> dict | None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(points) < MIN_KEYPOINTS or len(scores) < MIN_KEYPOINTS:
        return None
    points = points[:MIN_KEYPOINTS].copy()
    scores = scores[:MIN_KEYPOINTS].copy()
    if rotated:
        points[:, 0] = width - points[:, 0]
        points[:, 1] = height - points[:, 1]
    visible = np.isfinite(points).all(axis=1) & np.isfinite(scores) & (scores >= threshold)
    # Require a usable body; feet may be sparse when occluded.
    if int(visible[:FOOT_START].sum()) < 4:
        return None
    xy = points[visible]
    x1, y1 = xy.min(axis=0)
    x2, y2 = xy.max(axis=0)
    pad_x = max(12.0, 0.08 * (x2 - x1))
    pad_y = max(12.0, 0.08 * (y2 - y1))
    box = [max(0.0, x1-pad_x), max(0.0, y1-pad_y),
           min(float(width), x2+pad_x), min(float(height), y2+pad_y)]
    body_vis = visible[:FOOT_START]
    confidence = float(np.mean(np.sort(scores[:FOOT_START][body_vis])[-min(8, int(body_vis.sum())):]))
    keypoints = {
        name: [float(points[i, 0]), float(points[i, 1]), float(scores[i])]
        for i, name in enumerate(NAMES)
    }
    for alias, source in ALIAS_TO_SOURCE.items():
        keypoints[alias] = list(keypoints[source])
    return {"box_xyxy": box, "box_confidence": confidence, "keypoints": keypoints}


def main() -> None:
    a = parse_args()
    if a.ranges and (a.start_frame != 0 or a.max_frames is not None):
        raise ValueError("--ranges cannot be combined with --start-frame/--max-frames")
    ranges = (
        [
            (int(start), int(count))
            for item in a.ranges.split(",")
            for start, count in [item.split(":", 1)]
        ]
        if a.ranges
        else [(a.start_frame, a.max_frames)]
    )
    if any(start < 0 or count is not None and count <= 0 for start, count in ranges):
        raise ValueError("Frame starts must be nonnegative and counts must be positive")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    # RTMW WholeBody (performance) — same weights already used for head nose detect.
    model = Wholebody(mode=a.mode, to_openpose=False, backend=a.backend, device=a.device)
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {a.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows = people = 0
    current_index = -1
    with a.output.open("w", encoding="utf-8") as out:
        for start_frame, frame_count in ranges:
            # MJPEG CAP_PROP_POS_FRAMES seek is unreliable on these recordings.
            # Rewind by reopening, then advance with sequential reads.
            if current_index > start_frame or current_index < 0:
                cap.release()
                cap = cv2.VideoCapture(str(a.video))
                if not cap.isOpened():
                    raise RuntimeError(f"Cannot reopen {a.video}")
                current_index = -1
            while current_index + 1 < start_frame:
                ok, _ = cap.read()
                if not ok:
                    raise RuntimeError(
                        f"Could not advance {a.video} to frame {start_frame}"
                    )
                current_index += 1
            frame_index = start_frame
            range_rows = 0
            while frame_count is None or range_rows < frame_count:
                ok, frame = cap.read()
                if not ok:
                    break
                current_index = frame_index
                image = cv2.rotate(frame, cv2.ROTATE_180) if a.rotate_180 else frame
                keypoints, scores = model(image)
                candidates = []
                if keypoints is not None and scores is not None:
                    for pts, conf in zip(np.asarray(keypoints), np.asarray(scores)):
                        item = candidate(pts, conf, width, height, a.rotate_180,
                                         a.visible_threshold)
                        if item is not None:
                            candidates.append(item)
                candidates.sort(key=lambda item: item["box_confidence"], reverse=True)
                out.write(json.dumps({"frame_index": frame_index, "width": width, "height": height,
                                      "candidates": candidates,
                                      "model": "RTMW-WholeBody",
                                      "joints": list(EMIT_NAMES)},
                                     separators=(",", ":")) + "\n")
                people += len(candidates)
                rows += 1
                range_rows += 1
                frame_index += 1
                if rows % 25 == 0:
                    print(json.dumps({"frames": rows, "candidates": people}), flush=True)
    cap.release()
    print(json.dumps({"frames": rows, "candidates": people, "output": str(a.output),
                      "model": f"RTMW-WholeBody-{a.mode}",
                      "joints": list(EMIT_NAMES)}), flush=True)


if __name__ == "__main__":
    main()
