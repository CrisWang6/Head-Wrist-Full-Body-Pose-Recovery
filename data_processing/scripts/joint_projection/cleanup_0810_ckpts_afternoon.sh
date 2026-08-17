#!/usr/bin/env bash
set -euo pipefail
DATASET=/home/gaoweijian/0810dataset
ARCHIVE="$DATASET/checkpoints/archive_20260812_afternoon"
mkdir -p "$ARCHIVE"
shopt -s nullglob
for path in "$DATASET/checkpoints"/stage1_v31 "$DATASET/checkpoints"/stage2_v31 "$DATASET/checkpoints"/stage3_v31_aligned "$DATASET/checkpoints"/stage2_v31_prev_* "$DATASET/checkpoints"/stage3_v31_aligned_prev_*; do
  if [[ -e "$path" ]]; then
    mv "$path" "$ARCHIVE/"
    echo "archived $(basename "$path")"
  fi
done
mkdir -p "$DATASET/checkpoints/stage1_v31"
mkdir -p "$DATASET/logs/tb_archive_20260812_afternoon/stage1_v31"
mkdir -p "$DATASET/logs/tb_archive_20260812_afternoon/stage2_v31"
mkdir -p "$DATASET/checkpoints/stage3_v31_aligned/tensorboard"
find "$DATASET/logs/stage1_v31" -name 'events.out.tfevents*' -exec mv {} "$DATASET/logs/tb_archive_20260812_afternoon/stage1_v31/" \; 2>/dev/null || true
find "$DATASET/logs/stage2_v31" -name 'events.out.tfevents*' -exec mv {} "$DATASET/logs/tb_archive_20260812_afternoon/stage2_v31/" \; 2>/dev/null || true
find "$DATASET/checkpoints/stage3_v31_aligned/tensorboard" -name 'events.out.tfevents*' -delete 2>/dev/null || true
ln -sfn "$DATASET/checkpoints/stage3_v31_aligned/tensorboard" "$DATASET/logs/stage3_v31_aligned"
ls -la "$DATASET/checkpoints/"
