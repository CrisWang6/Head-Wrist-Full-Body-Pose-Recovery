#!/usr/bin/env bash
# Run only Stage1 v31 (480x300 input, 120x75 heatmap) then eval test split.
set -euo pipefail

BATCH=/home/gaoweijian/0806_batch
JP="$BATCH/repo/test_code/joint_projection"
PY=/home/gaoweijian/miniforge3/envs/sapiens2/bin/python
CAMTEST_PY=/home/gaoweijian/miniforge3/envs/camtest/bin/python
EGO=/home/gaoweijian/EgoRear_w_hand
DATASET=/home/gaoweijian/0806dataset
LOG="$DATASET/logs/stage1_v31_only.log"
STATUS="$DATASET/WEEKEND_MASTER_STATUS.txt"
SCHEME=v31
SPLIT="$DATASET/splits/pack30_${SCHEME}.npz"
OUT="$DATASET/checkpoints/stage1_${SCHEME}"
LOGD="$DATASET/logs/stage1_${SCHEME}"
RADIUS="$JP/configs/joint_radius_px_120x75_delivery15.json"

export PYTHONPATH="$JP"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/home/gaoweijian/miniforge3/envs/sapiens2/lib/python3.12/site-packages/nvidia/cu13/lib:/home/gaoweijian/miniforge3/envs/sapiens2/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}

mkdir -p "$DATASET/logs" "$OUT" "$LOGD" "$DATASET/eval/${SCHEME}"

stamp() { date '+%F %T'; }
log() { echo "[$(stamp)] $*" | tee -a "$LOG"; }
set_status() { echo "$1" | tee "$STATUS"; }

exec >>"$LOG" 2>&1
log "===== STAGE1 v31 ONLY START (480x300 input, batch=64, 2x4090 DP) ====="
set_status "stage1_v31_running"

RESUME_ARGS=()
if [[ -f "$OUT/best.pt" ]]; then
  RESUME_ARGS=(--resume "$OUT/best.pt" --resume-optimizer)
  log "resume from $OUT/best.pt"
fi

cd "$EGO"
"$CAMTEST_PY" scripts/train_heatmap.py \
  --label-root "$DATASET/labels" \
  --frame-root "$DATASET/frames" \
  --output-dir "$OUT" \
  --log-dir "$LOGD" \
  --split-manifest "$SPLIT" \
  --epochs 9999 \
  --batch-size 64 \
  --workers 12 \
  --lr 0.0001 \
  --weight-decay 0.005 \
  --seed 42 \
  --device cuda \
  --image-width 480 \
  --image-height 300 \
  --base-channels 64 \
  --visible-only-loss \
  --train-branch head \
  --joint-radius-config "$RADIUS" \
  --default-joint-radius-px 10 \
  --early-stop-patience 20 \
  --save-every 5 \
  --keep-last 3 \
  --log-every 50 \
  --prefetch-factor 6 \
  --max-hours 72 \
  "${RESUME_ARGS[@]}"

set_status "stage1_v31_done"
log "eval test split"
"$PY" "$JP/eval_0806_stage1_test.py" \
  --label-root "$DATASET/labels" \
  --frame-root "$DATASET/frames" \
  --split-npz "$SPLIT" \
  --checkpoint "$OUT/best.pt" \
  --output-dir "$DATASET/eval/${SCHEME}" \
  --scheme "$SCHEME" \
  --batch-size 32 \
  --workers 4

set_status "stage1_v31_eval_done"
log "===== STAGE1 v31 DONE + TEST EVAL ====="
