#!/usr/bin/env bash
# 0810 Stage3: train 150 epochs, pick best; backup previous run.
set -euo pipefail

BATCH=/home/gaoweijian/0810_batch
JP="$BATCH/repo/test_code/joint_projection"
CAMTEST=/home/gaoweijian/miniforge3/envs/camtest/bin/python
EGO=/home/gaoweijian/EgoRear_w_hand
DATASET=/home/gaoweijian/0810dataset
LOG="$DATASET/logs/0810_stage3_150ep.log"
STATUS="$DATASET/0810_TRAINING_STATUS.txt"
SPLIT="$DATASET/splits/pack150_v31.npz"
POSE3D="$DATASET/labels/pose3d_nose_pre_limb_15j.npz"
STAGE2_OUT="$DATASET/checkpoints/stage2_v31"
STAGE3_OUT="$DATASET/checkpoints/stage3_v31_aligned"
TS="$(date '+%Y%m%d_%H%M%S')"

export PYTHONPATH="$JP:$EGO/src"

mkdir -p "$DATASET/logs" "$STAGE3_OUT/tensorboard"
ln -sfn "$STAGE3_OUT/tensorboard" "$DATASET/logs/stage3_v31_aligned"

stamp() { date '+%F %T'; }
log() { echo "[$(stamp)] $*" | tee -a "$LOG"; }
set_status() { echo "$1" | tee "$STATUS"; }

exec >>"$LOG" 2>&1
log "===== 0810 STAGE3 150-EPOCH RETRAIN START ====="

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
  --stage1-checkpoint "$DATASET/checkpoints/stage1_v31/best.pt" \
  --stage2-checkpoint "$STAGE2_OUT/best.pt" \
  --output-dir "$STAGE3_OUT" \
  --split-manifest "$SPLIT" \
  --epochs 150 --batch-size 64 --workers 8 --lr 0.001 --weight-decay 0.0005 \
  --min-epochs 150 --early-stop-patience 9999 --device cuda --seed 42
set_status "done"
log "===== 0810 STAGE3 150-EPOCH RETRAIN DONE ====="
