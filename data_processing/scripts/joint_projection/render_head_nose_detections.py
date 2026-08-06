#!/usr/bin/env python3
"""Render body-nose and face-nose detections for stereo visual review."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def load(path: Path) -> dict[int, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {int(row["frame_index"]): row for row in csv.DictReader(f)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--a-video", type=Path, required=True)
    p.add_argument("--d-video", type=Path, required=True)
    p.add_argument("--a-csv", type=Path, required=True)
    p.add_argument("--d-csv", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    rows = {"A": load(a.a_csv), "D": load(a.d_csv)}
    captures = {"A": cv2.VideoCapture(str(a.a_video)), "D": cv2.VideoCapture(str(a.d_video))}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(a.output), cv2.VideoWriter_fourcc(*"mp4v"), 50, (1920, 600))
    frame_index = 0
    while True:
        images = {}
        for camera in ("A", "D"):
            ok, image = captures[camera].read()
            if not ok:
                image = None
            images[camera] = image
        if images["A"] is None or images["D"] is None:
            break
        panels = []
        for camera in ("A", "D"):
            image = images[camera]
            row = rows[camera].get(frame_index)
            if row and int(row["detected"]):
                body = tuple(np.rint([float(row["body_nose_u_px"]), float(row["body_nose_v_px"])]).astype(int))
                face = tuple(np.rint([float(row["face_nose_u_px"]), float(row["face_nose_v_px"])]).astype(int))
                cv2.circle(image, body, 14, (0, 165, 255), 3, cv2.LINE_AA)
                cv2.circle(image, face, 9, (255, 255, 0), -1, cv2.LINE_AA)
                cv2.putText(image, "orange=body nose  cyan=RTMW face nose tip", (24, 42),
                            cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(image, f"HEAD {camera} frame={frame_index}", (24, 76),
                        cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 255), 2, cv2.LINE_AA)
            panels.append(cv2.resize(image, (960, 600)))
        writer.write(np.hstack(panels))
        frame_index += 1
    for capture in captures.values():
        capture.release()
    writer.release()
    print({"frames": frame_index, "output": str(a.output)})


if __name__ == "__main__":
    main()
