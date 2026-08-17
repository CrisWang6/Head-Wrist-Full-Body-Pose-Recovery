#!/usr/bin/env bash
set -euo pipefail
BATCH=/home/gaoweijian/0810_batch
JP="$BATCH/repo/test_code/joint_projection"
CAMTEST=/home/gaoweijian/miniforge3/envs/camtest/bin/python
EGO=/home/gaoweijian/EgoRear_w_hand
DATASET=/home/gaoweijian/0810dataset
LOG="$DATASET/logs/0810_fresh_stage1.log"
STATUS="$DATASET/0810_TRAINING_STATUS.txt"
SPLIT="$DATASET/splits/pack150_v31.npz"
RADIUS="$JP/configs/joint_radius_px_120x75_delivery15_limbs_half.json"
RESUME="$DATASET/checkpoints/stage1_v31/last.pt"

export PYTHONPATH="$JP:$EGO/src"
unset CUDA_VISIBLE_DEVICES

exec >>"$LOG" 2>&1
echo "[$(date '+%F %T')] ===== STAGE1 RESUME (2xGPU DP, batch32, fastest) ====="
echo "stage1_v31_running" | tee "$STATUS"

cd "$EGO"
"$CAMTEST" scripts/train_heatmap.py \
  --label-root "$DATASET/labels" \
  --frame-root "$DATASET/frames" \
  --output-dir "$DATASET/checkpoints/stage1_v31" \
  --log-dir "$DATASET/logs/stage1_v31" \
  --split-manifest "$SPLIT" \
  --epochs 100 --batch-size 32 --workers 8 --lr 0.0001 --weight-decay 0.005 \
  --seed 42 --device cuda --image-width 480 --image-height 300 \
  --base-channels 64 --visible-only-loss --train-branch head \
  --joint-radius-config "$RADIUS" --default-joint-radius-px 10 \
  --early-stop-patience 9999 --save-every 5 --keep-last 3 --log-every 50 \
  --prefetch-factor 4 --max-hours 72 \
  --resume "$RESUME" --resume-optimizer
