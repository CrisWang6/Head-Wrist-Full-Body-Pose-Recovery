#!/usr/bin/env bash
set -euo pipefail
DATASET=/home/gaoweijian/0810dataset
ARCHIVE="$DATASET/checkpoints/archive_20260812_loss_revert"
TS="$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$ARCHIVE"
if [[ -d "$DATASET/checkpoints/stage1_v31" ]]; then
  mv "$DATASET/checkpoints/stage1_v31" "$ARCHIVE/stage1_v31_${TS}"
  echo "archived stage1_v31 -> stage1_v31_${TS}"
fi
mkdir -p "$DATASET/checkpoints/stage1_v31"
mkdir -p "$DATASET/logs/tb_archive_20260812_loss_revert/stage1_v31"
find "$DATASET/logs/stage1_v31" -name 'events.out.tfevents*' -exec mv {} "$DATASET/logs/tb_archive_20260812_loss_revert/stage1_v31/" \; 2>/dev/null || true
ls -la "$DATASET/checkpoints/"
