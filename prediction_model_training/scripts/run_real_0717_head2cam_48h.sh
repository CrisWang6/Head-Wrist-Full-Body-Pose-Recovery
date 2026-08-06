#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT}/data}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT}}"
LABELS="${LABELS:-${DATA_ROOT}/labels/real_0717_head2cam/heatmap_labels_114x64.npz}"
PRETRAINED="${PRETRAINED:-${ARTIFACT_ROOT}/checkpoints/isaacsim_humaneva_2app_notags_fisheye220_head_stage1/best.pt}"
OUTPUT="${OUTPUT:-${ARTIFACT_ROOT}/checkpoints/real_0717_head2cam_stage1_48h}"
LOGS="${LOGS:-${ARTIFACT_ROOT}/logs/real_0717_head2cam_stage1_48h}"

cd "$PROJECT"
test -f "$LABELS"
test -f "$PRETRAINED"
mkdir -p "$OUTPUT" "$LOGS"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

# 47.75 h lets Python write last.pt/status before the 48 h OS hard limit.
exec timeout --signal=INT --kill-after=300s 48h \
  "$PYTHON_BIN" scripts/train_heatmap.py \
  --label-root "$LABELS" \
  --output-dir "$OUTPUT" \
  --log-dir "$LOGS" \
  --epochs 9999 \
  --batch-size 16 \
  --workers 8 \
  --lr 0.0001 \
  --weight-decay 0.005 \
  --train-ratio 0.8 \
  --split-mode chronological \
  --seed 42 \
  --device cuda \
  --image-width 456 \
  --image-height 256 \
  --base-channels 64 \
  --visible-only-loss \
  --train-branch head \
  --resume "$PRETRAINED" \
  --max-hours 47.75 \
  --save-every 5 \
  --keep-last 3 \
  --log-every 50 \
  --prefetch-factor 4
