#!/usr/bin/env python3
"""Detect body-nose and 68-point face-nose candidates with RTMW WholeBody."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from rtmlib import Wholebody


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--mode", default="performance")
    p.add_argument("--device", default="cuda")
    p.add_argument("--face-start-index", type=int, default=23)
    p.add_argument("--face-nose-index", type=int, default=30,
                   help="68-point face index; 30 is the tip at the end of the nose bridge.")
    return p.parse_args()


def choose_person(keypoints: np.ndarray, scores: np.ndarray) -> int | None:
    best_index, best_area = None, -1.0
    for index, (points, confidence) in enumerate(zip(keypoints, scores)):
        visible = np.isfinite(points[:17]).all(axis=1) & (confidence[:17] >= 0.1)
        if int(visible.sum()) < 4:
            continue
        xy = points[:17][visible]
        area = float(np.prod(np.maximum(xy.max(axis=0)-xy.min(axis=0), 0.0)))
        if area > best_area:
            best_index, best_area = index, area
    return best_index


def main() -> None:
    a = parse_args()
    a.output_csv.parent.mkdir(parents=True, exist_ok=True)
    model = Wholebody(mode=a.mode, to_openpose=False, backend="onnxruntime", device=a.device)
    capture = cv2.VideoCapture(str(a.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {a.video}")
    face_index = a.face_start_index + a.face_nose_index
    rows = []
    frame_index = 0
    output_shape = None
    while a.max_frames is None or frame_index < a.max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        keypoints, scores = model(frame)
        keypoints, scores = np.asarray(keypoints), np.asarray(scores)
        output_shape = list(keypoints.shape)
        person = choose_person(keypoints, scores) if keypoints.ndim == 3 else None
        row = {"frame_index": frame_index, "detected": int(person is not None),
               "body_nose_u_px": "", "body_nose_v_px": "", "body_nose_score": "",
               "face_nose_u_px": "", "face_nose_v_px": "", "face_nose_score": ""}
        if person is not None:
            row.update({"body_nose_u_px": float(keypoints[person, 0, 0]),
                        "body_nose_v_px": float(keypoints[person, 0, 1]),
                        "body_nose_score": float(scores[person, 0])})
            if face_index < keypoints.shape[1]:
                row.update({"face_nose_u_px": float(keypoints[person, face_index, 0]),
                            "face_nose_v_px": float(keypoints[person, face_index, 1]),
                            "face_nose_score": float(scores[person, face_index])})
        rows.append(row)
        frame_index += 1
    capture.release()
    with a.output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print({"frames": len(rows), "detected": sum(row["detected"] for row in rows),
           "output_shape": output_shape, "face_global_index": face_index,
           "output": str(a.output_csv)})


if __name__ == "__main__":
    main()
