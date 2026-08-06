#!/usr/bin/env python3
"""Run RTMPose on a video and emit the candidate schema used by the stereo pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from rtmlib import Body


NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)


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
    p.add_argument("--visible-threshold", type=float, default=0.05)
    return p.parse_args()


def candidate(points: np.ndarray, scores: np.ndarray, width: int, height: int,
              rotated: bool, threshold: float) -> dict | None:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)[:len(NAMES)]
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)[:len(NAMES)]
    if len(points) != len(NAMES) or len(scores) != len(NAMES):
        return None
    if rotated:
        points[:, 0] = width - points[:, 0]
        points[:, 1] = height - points[:, 1]
    visible = np.isfinite(points).all(axis=1) & np.isfinite(scores) & (scores >= threshold)
    if int(visible.sum()) < 4:
        return None
    xy = points[visible]
    x1, y1 = xy.min(axis=0)
    x2, y2 = xy.max(axis=0)
    pad_x = max(12.0, 0.08 * (x2 - x1))
    pad_y = max(12.0, 0.08 * (y2 - y1))
    box = [max(0.0, x1-pad_x), max(0.0, y1-pad_y),
           min(float(width), x2+pad_x), min(float(height), y2+pad_y)]
    confidence = float(np.mean(np.sort(scores[visible])[-min(8, visible.sum()):]))
    keypoints = {
        name: [float(points[i, 0]), float(points[i, 1]), float(scores[i])]
        for i, name in enumerate(NAMES)
    }
    return {"box_xyxy": box, "box_confidence": confidence, "keypoints": keypoints}


def main() -> None:
    a = parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    model = Body(mode=a.mode, to_openpose=False, backend=a.backend, device=a.device)
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {a.video}")
    if a.start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, a.start_frame)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rows = people = 0
    frame_index = a.start_frame
    with a.output.open("w", encoding="utf-8") as out:
        while a.max_frames is None or rows < a.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
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
                                  "candidates": candidates}, separators=(",", ":")) + "\n")
            people += len(candidates)
            rows += 1
            frame_index += 1
            if rows % 25 == 0:
                print(json.dumps({"frames": rows, "candidates": people}), flush=True)
    cap.release()
    print(json.dumps({"frames": rows, "candidates": people, "output": str(a.output),
                      "model": f"RTMPose-{a.mode}"}), flush=True)


if __name__ == "__main__":
    main()
