#!/usr/bin/env python3
"""Restore source-time duration while retaining only complete aligned event frames."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2


p = argparse.ArgumentParser()
p.add_argument("--input", type=Path, required=True)
p.add_argument("--aligned", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--seconds", type=float, required=True)
p.add_argument("--timestamp-field", default="module01_CAM_A_device_ts_ms")
a = p.parse_args()

with a.aligned.open("r", encoding="utf-8-sig", newline="") as f:
    all_rows = list(csv.DictReader(f))
t0 = float(all_rows[0][a.timestamp_field])
rows = [r for r in all_rows if float(r[a.timestamp_field]) < t0 + a.seconds * 1000.0]
targets = [max(0, round((float(r[a.timestamp_field]) - t0) * 0.05)) for r in rows]

cap = cv2.VideoCapture(str(a.input))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
a.output.parent.mkdir(parents=True, exist_ok=True)
out = cv2.VideoWriter(str(a.output), cv2.VideoWriter_fourcc(*"mp4v"), 50, (width, height))

event = -1
current = None
next_target = targets[0]
for timeline in range(round(a.seconds * 50)):
    while event + 1 < len(targets) and targets[event + 1] <= timeline:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"event video ended at frame {event + 1}")
        event += 1
        current = frame
        next_target = targets[event + 1] if event + 1 < len(targets) else None
    if current is None:
        ok, current = cap.read()
        if not ok:
            raise RuntimeError("event video has no frames")
        event = 0
    out.write(current)

cap.release(); out.release()
print({"event_frames": len(rows), "output_frames": round(a.seconds * 50), "held_frames": round(a.seconds * 50) - len(set(targets))})
