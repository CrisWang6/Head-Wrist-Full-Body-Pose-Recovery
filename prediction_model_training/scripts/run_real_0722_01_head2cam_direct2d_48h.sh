#!/usr/bin/env bash
set -euo pipefail

PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-${PROJECT}/data}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${PROJECT}}"
cd "${PROJECT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

resume_checkpoint="${RESUME_CHECKPOINT:-${ARTIFACT_ROOT}/checkpoints/isaacsim_humaneva_2app_notags_fisheye220_head_stage1/best.pt}"
max_hours="${MAX_HOURS:-47.75}"
resume_optimizer_args=()
if [[ "${RESUME_OPTIMIZER:-0}" == "1" ]]; then
  resume_optimizer_args+=(--resume-optimizer)
fi

"${PYTHON_BIN}" scripts/train_heatmap.py \
  --label-root "${DATA_ROOT}/labels/real_0722_01_head2cam_direct2d/heatmap_labels_114x64.npz" \
  --output-dir "${ARTIFACT_ROOT}/checkpoints/real_0722_01_head2cam_direct2d_12j_stage1_48h" \
  --log-dir "${ARTIFACT_ROOT}/logs/real_0722_01_head2cam_direct2d_12j_stage1_48h" \
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
  --resume "$resume_checkpoint" \
  "${resume_optimizer_args[@]}" \
  --max-hours "$max_hours" \
  --save-every 5 \
  --keep-last 3 \
  --log-every 50 \
  --prefetch-factor 4
