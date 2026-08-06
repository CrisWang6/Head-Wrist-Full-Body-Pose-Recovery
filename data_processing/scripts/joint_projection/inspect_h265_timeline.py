#!/usr/bin/env python3
"""Inspect decoded H.265 PTS gaps without relying on the MP4 container."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


FFMPEG = Path(r"C:\Users\hand\miniconda3\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe")
ROOT = Path(r"C:\Users\hand\Desktop\Dataset\0722\record_9cam_0711_021044")
OUTPUT = Path(__file__).resolve().parent / "validation_0722_h265_timeline"
PATTERN = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)")


def inspect(camera: str) -> dict[str, object]:
    source = ROOT / f"module01_D45D2E00_{camera}.h265"
    command = [
        str(FFMPEG), "-hide_banner", "-i", str(source), "-vf", "showinfo",
        "-an", "-f", "null", "NUL",
    ]
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
    )
    pts: list[int] = []
    decoder_events: list[dict[str, object]] = []
    last_frame = -1
    assert process.stderr is not None
    for line in process.stderr:
        match = PATTERN.search(line)
        if match:
            pts.append(int(match.group(2)))
            last_frame = int(match.group(1))
        elif any(token in line for token in ("Could not find ref", "Duplicate POC", "Skipping invalid undecodable")):
            decoder_events.append({"near_decoded_index": last_frame, "message": line.strip()[-180:]})
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed for {camera}: {return_code}")
    positive_steps = [b - a for a, b in zip(pts, pts[1:]) if b > a]
    nominal = min(positive_steps)
    gaps = []
    cumulative_missing = 0
    decoded_to_capture: list[int] = []
    first_pts = pts[0]
    for decoded_index, value in enumerate(pts):
        capture_index = int(round((value - first_pts) / nominal))
        decoded_to_capture.append(capture_index)
        if decoded_index and capture_index > decoded_to_capture[-2] + 1:
            missing = capture_index - decoded_to_capture[-2] - 1
            cumulative_missing += missing
            gaps.append(
                {
                    "after_decoded_index": decoded_index - 1,
                    "before_decoded_index": decoded_index,
                    "after_capture_index": decoded_to_capture[-2],
                    "before_capture_index": capture_index,
                    "missing_frames": missing,
                    "cumulative_missing": cumulative_missing,
                }
            )
    return {
        "camera": camera,
        "source": str(source),
        "decoded_frames": len(pts),
        "nominal_pts_step": nominal,
        "first_pts": first_pts,
        "last_capture_index": decoded_to_capture[-1],
        "missing_from_pts_gaps": cumulative_missing,
        "gaps": gaps,
        "decoder_events": decoder_events,
        "decoded_to_capture_index": decoded_to_capture,
    }


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {camera: inspect(camera) for camera in ("CAM_B", "CAM_C")}
    (OUTPUT / "h265_timeline_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: {k: v for k, v in value.items() if k != "decoded_to_capture_index"} for key, value in report.items()}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
