#!/usr/bin/env bash
# Stage3 from stage1 only (skip stage2), up to 200 epochs.
set -euo pipefail

BATCH=/home/gaoweijian/0810_batch
JP="$BATCH/repo/test_code/joint_projection"
CAMTEST=/home/gaoweijian/miniforge3/envs/camtest/bin/python
EGO=/home/gaoweijian/EgoRear_w_hand
DATASET=/home/gaoweijian/0810dataset
LOG="$DATASET/logs/0810_stage3_skip_stage2_200ep.log"
STATUS="$DATASET/0810_TRAINING_STATUS.txt"
SPLIT="$DATASET/splits/pack150_v31.npz"
POSE3D="$DATASET/labels/pose3d_nose_pre_limb_15j.npz"
STAGE1_BEST="$DATASET/checkpoints/stage1_v31/best.pt"
STAGE3_OUT="$DATASET/checkpoints/stage3_v31_aligned"
TS="$(date '+%Y%m%d_%H%M%S')"

export PYTHONPATH="$JP:$EGO/src"
unset CUDA_VISIBLE_DEVICES

mkdir -p "$DATASET/logs" "$STAGE3_OUT/tensorboard"
ln -sfn "$STAGE3_OUT/tensorboard" "$DATASET/logs/stage3_v31_aligned"

stamp() { date '+%F %T'; }
log() { echo "[$(stamp)] $*" | tee -a "$LOG"; }
set_status() { echo "$1" | tee "$STATUS"; }

exec >>"$LOG" 2>&1
log "===== 0810 STAGE3 200-EPOCH (skip stage2, stage1-only 2D) START ====="

if [[ ! -f "$STAGE1_BEST" ]]; then
  log "ERROR: missing stage1 best checkpoint: $STAGE1_BEST"
  exit 1
fi

if [[ -f "$STAGE3_OUT/best.pt" ]]; then
  backup="$DATASET/checkpoints/stage3_v31_aligned_prev_${TS}"
  log "backup old stage3 -> $backup"
  mv "$STAGE3_OUT" "$backup"
  mkdir -p "$STAGE3_OUT/tensorboard"
  ln -sfn "$STAGE3_OUT/tensorboard" "$DATASET/logs/stage3_v31_aligned"
fi

set_status "stage3_v31_aligned_running"
cd "$EGO"
"$CAMTEST" experiments/stage3_pose3d/scripts/train_pose3d.py \
  --label-root "$DATASET/labels" \
  --pose3d-labels "$POSE3D" \
  --stage1-checkpoint "$STAGE1_BEST" \
  --skip-stage2 \
  --output-dir "$STAGE3_OUT" \
  --split-manifest "$SPLIT" \
  --epochs 200 --batch-size 64 --workers 8 --lr 0.001 --weight-decay 0.0005 \
  --min-epochs 200 --early-stop-patience 9999 --device cuda --seed 42
set_status "stage3_v31_aligned_done"
log "===== 0810 STAGE3 200-EPOCH (skip stage2) DONE ====="
