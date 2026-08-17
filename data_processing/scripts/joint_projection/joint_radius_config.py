"""Load per-joint heatmap radii for 0806 delivery15 pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from constants_0806_training import HEATMAP_WIDTH, VIDEO_WIDTH

JOINT_RADIUS_CONFIG = (
    Path(__file__).resolve().parent / "configs/joint_radius_px_120x75_delivery15.json"
)
RADIUS_VIDEO_TO_HEATMAP_STRIDE = VIDEO_WIDTH / HEATMAP_WIDTH
DEFAULT_JOINT_RADIUS_HEATMAP_PX = 10.0 / RADIUS_VIDEO_TO_HEATMAP_STRIDE  # 0.625 @ 120x75


def load_joint_radius_video_px(
    config_path: Path | str,
    *,
    video_width: int = VIDEO_WIDTH,
    heatmap_width: int = HEATMAP_WIDTH,
) -> dict[str, float]:
    """Return joint radii in source-video pixels for EgoRear generate_heatmaps.

    Files named *120x75* store 3σ radii in heatmap pixels (1920→120 scale 1/16).
    Legacy *1920x1200* files are already in video pixels.
    """
    path = Path(config_path).expanduser().resolve()
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"joint radius config must be a JSON object: {path}")
    radii = {str(k): float(v) for k, v in loaded.items()}
    if "120x75" in path.name:
        stride = float(video_width) / float(heatmap_width)
        radii = {name: value * stride for name, value in radii.items()}
    return radii
