#!/usr/bin/env python3
"""Run YOLO pose on one recording and keep every person candidate per frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


KEYPOINT_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee",
    "right_knee", "left_ankle", "right_ankle",
)


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--conf", type=float, default=0.10)
    p.add_argument("--rotate180", action="store_true")
    return p.parse_args()


def main() -> None:
    a = args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(a.video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {a.video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model = YOLO(str(a.model))
    frame_index = 0
    with a.output.open("w", encoding="utf-8") as out:
        while True:
            frames = []
            for _ in range(a.batch):
                ok, image = cap.read()
                if not ok:
                    break
                frames.append(cv2.rotate(image, cv2.ROTATE_180) if a.rotate180 else image)
            if not frames:
                break
            results = model.predict(
                frames, device=a.device, imgsz=a.imgsz, conf=a.conf,
                classes=[0], verbose=False, half=True,
            )
            for result in results:
                candidates = []
                boxes = result.boxes
                kpts = result.keypoints
                if boxes is not None and kpts is not None:
                    xyxy = boxes.xyxy.detach().cpu().numpy()
                    bconf = boxes.conf.detach().cpu().numpy()
                    xy = kpts.xy.detach().cpu().numpy()
                    kc = kpts.conf.detach().cpu().numpy()
                    for box, score, pts, scores in zip(xyxy, bconf, xy, kc):
                        if a.rotate180:
                            box = [width - box[2], height - box[3], width - box[0], height - box[1]]
                            pts[:, 0] = width - pts[:, 0]
                            pts[:, 1] = height - pts[:, 1]
                        candidates.append({
                            "box_xyxy": [round(float(v), 4) for v in box],
                            "box_confidence": round(float(score), 6),
                            "keypoints": {
                                name: [round(float(p[0]), 4), round(float(p[1]), 4), round(float(c), 6)]
                                for name, p, c in zip(KEYPOINT_NAMES, pts, scores)
                            },
                        })
                out.write(json.dumps({
                    "frame_index": frame_index,
                    "width": width, "height": height,
                    "candidates": candidates,
                }, separators=(",", ":")) + "\n")
                frame_index += 1
            out.flush()
    cap.release()
    print(json.dumps({"video": str(a.video), "frames": frame_index, "output": str(a.output)}))


if __name__ == "__main__":
    main()
