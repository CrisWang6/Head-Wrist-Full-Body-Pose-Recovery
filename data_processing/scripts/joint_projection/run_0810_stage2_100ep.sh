#!/usr/bin/env bash
# Re-run 0810 Stage2 for 100 epochs (best tracked), then Stage3; fix Stage3 TB path.
set -euo pipefail

BATCH=/home/gaoweijian/0810_batch
JP="$BATCH/repo/test_code/joint_projection"
CAMTEST=/home/gaoweijian/miniforge3/envs/camtest/bin/python
EGO=/home/gaoweijian/EgoRear_w_hand
DATASET=/home/gaoweijian/0810dataset
LOG="$DATASET/logs/0810_stage2_100ep.log"
STATUS="$DATASET/0810_TRAINING_STATUS.txt"
SPLIT="$DATASET/splits/pack150_v31.npz"
POSE3D="$DATASET/labels/pose3d_nose_pre_limb_15j.npz"
STAGE2_OUT="$DATASET/checkpoints/stage2_v31"
STAGE2_LOGD="$DATASET/logs/stage2_v31"
STAGE3_OUT="$DATASET/checkpoints/stage3_v31_aligned"
STAGE3_TB="$STAGE3_OUT/tensorboard"
TS="$(date '+%Y%m%d_%H%M%S')"

export PYTHONPATH="$JP:$EGO/src"

mkdir -p "$DATASET/logs" "$STAGE2_LOGD" "$STAGE3_TB"
ln -sfn "$STAGE3_TB" "$DATASET/logs/stage3_v31_aligned"

stamp() { date '+%F %T'; }
log() { echo "[$(stamp)] $*" | tee -a "$LOG"; }
set_status() { echo "$1" | tee "$STATUS"; }

exec >>"$LOG" 2>&1
log "===== 0810 STAGE2 100-EPOCH RETRAIN START ====="

if [[ -f "$STAGE2_OUT/best.pt" ]]; then
  backup="$DATASET/checkpoints/stage2_v31_prev_${TS}"
  log "backup old stage2 -> $backup"
  mv "$STAGE2_OUT" "$backup"
  mkdir -p "$STAGE2_OUT"
fi

set_status "stage2_v31_running"
cd "$EGO"
"$CAMTEST" experiments/stage2_refinement/scripts/train_refinement.py \
  --label-root "$DATASET/labels" \
  --stage1-checkpoint "$DATASET/checkpoints/stage1_v31/best.pt" \
  --output-dir "$STAGE2_OUT" --log-dir "$STAGE2_LOGD" --split-manifest "$SPLIT" \
  --epochs 100 --batch-size 16 --workers 8 --lr 0.001 --weight-decay 0.005 \
  --selection-metric refined_pixel_error --min-epochs 100 --early-stop-patience 9999 \
  --heatmap-width 120 --heatmap-height 75 --image-width 480 --image-height 300 \
  --base-channels 64 --device cuda --seed 42 --max-hours 72
set_status "stage2_v31_done"
log "stage2 finished; stage3 will be launched by run_0810_stage3_150ep_watchdog.sh"
log "===== 0810 STAGE2 100-EPOCH RETRAIN DONE (stage3 delegated) ====="
