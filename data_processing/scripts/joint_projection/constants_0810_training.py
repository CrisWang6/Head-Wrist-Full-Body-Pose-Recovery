"""0810 dual-line training constants (5s packs @ 30fps, same 15-joint delivery as 0806)."""

from __future__ import annotations

from constants_0806_training import *  # noqa: F403

PACK_SIZE = 150  # 5 seconds @ 30 Hz
SESSION_ORDER = ("line1", "line2")
SPLIT_SCHEME = "v31"
SPLIT_NAME = f"pack{PACK_SIZE}_{SPLIT_SCHEME}"
