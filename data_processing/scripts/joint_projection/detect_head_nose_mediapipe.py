#!/usr/bin/env python3
"""Detect the camera wearer's nose tip in head-camera video with MediaPipe."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--max-frames", type=int)
    p.add_argument("--nose-index", type=int, default=1)
    p.add_argument("--min-confidence", type=float, default=0.2)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    a.output_csv.parent.mkdir(parents=True, exist_ok=True)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(a.model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=2,
        min_face_detection_confidence=a.min_confidence,
        min_face_presence_confidence=a.min_confidence,
        min_tracking_confidence=a.min_confidence,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    capture = cv2.VideoCapture(str(a.video))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {a.video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 50.0
    rows = []
    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
        frame_index = 0
        while a.max_frames is None or frame_index < a.max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(image, int(round(frame_index * 1000.0 / fps)))
            chosen = None
            chosen_area = -1.0
            for face in result.face_landmarks:
                xs = [point.x for point in face]
                ys = [point.y for point in face]
                area = max(0.0, max(xs)-min(xs)) * max(0.0, max(ys)-min(ys))
                if area > chosen_area:
                    chosen, chosen_area = face, area
            if chosen is None or a.nose_index >= len(chosen):
                rows.append({"frame_index": frame_index, "detected": 0,
                             "nose_u_px": "", "nose_v_px": "", "face_bbox_area_norm": ""})
            else:
                nose = chosen[a.nose_index]
                rows.append({"frame_index": frame_index, "detected": 1,
                             "nose_u_px": float(nose.x * width),
                             "nose_v_px": float(nose.y * height),
                             "face_bbox_area_norm": float(chosen_area)})
            frame_index += 1
    capture.release()
    with a.output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    detected = sum(int(row["detected"]) for row in rows)
    print({"frames": len(rows), "detected": detected,
           "detection_rate": detected/max(1, len(rows)), "output": str(a.output_csv)})


if __name__ == "__main__":
    main()
