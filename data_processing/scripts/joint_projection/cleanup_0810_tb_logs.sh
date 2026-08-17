#!/usr/bin/env bash
set -euo pipefail
ARCHIVE=/home/gaoweijian/0810dataset/logs/tb_archive_20260812
STAGE2_LOG=/home/gaoweijian/0810dataset/logs/stage2_v31
STAGE3_TB=/home/gaoweijian/0810dataset/checkpoints/stage3_v31_aligned/tensorboard
STAGE3_LINK=/home/gaoweijian/0810dataset/logs/stage3_v31_aligned
OLD_STAGE2="$STAGE2_LOG/events.out.tfevents.1786506816.gpu222.1594804.0"
OLD_STAGE3="$STAGE3_TB/events.out.tfevents.1786508174.gpu222.1678139.0"

mkdir -p "$ARCHIVE/stage2_v31" "$ARCHIVE/stage3_v31_aligned"
echo "=== before ==="
find "$STAGE2_LOG" "$STAGE3_TB" -name 'events.out.tfevents*' 2>/dev/null | sort || true

if [[ -f "$OLD_STAGE2" ]]; then
  mv "$OLD_STAGE2" "$ARCHIVE/stage2_v31/"
  echo "archived old stage2 TB (35-epoch early stop run)"
fi
if [[ -f "$OLD_STAGE3" ]]; then
  mv "$OLD_STAGE3" "$ARCHIVE/stage3_v31_aligned/"
  echo "archived old stage3 TB (58-epoch early stop run)"
fi

mkdir -p "$STAGE3_TB"
ln -sfn "$STAGE3_TB" "$STAGE3_LINK"

echo "=== after ==="
find "$STAGE2_LOG" "$STAGE3_TB" -name 'events.out.tfevents*' 2>/dev/null | sort || true
ls -la "$STAGE3_LINK"
