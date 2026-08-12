#!/usr/bin/env bash
set -euo pipefail
BATCH=/home/gaoweijian/0810_batch
JP="$BATCH/repo/test_code/joint_projection"
PY=/home/gaoweijian/miniforge3/envs/sapiens2/bin/python
CAMTEST=/home/gaoweijian/miniforge3/envs/camtest/bin/python
DATASET=/home/gaoweijian/0810dataset
EGO=/home/gaoweijian/EgoRear_w_hand
export PYTHONPATH="$JP"
export LD_LIBRARY_PATH=/home/gaoweijian/miniforge3/envs/sapiens2/lib/python3.12/site-packages/nvidia/cu13/lib:/home/gaoweijian/miniforge3/envs/sapiens2/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
LOG="$DATASET/logs/0810_training_master.log"
STATUS="$DATASET/0810_TRAINING_STATUS.txt"
SPLIT="$DATASET/splits/pack150_v31.npz"
RADIUS="$JP/configs/joint_radius_px_120x75_delivery15.json"
POSE3D="$DATASET/labels/pose3d_nose_pre_limb_15j.npz"
PREMAP="$DATASET/pre_limb_map.json"

echo "prepare_pose3d_running" | tee "$STATUS"
echo "[$(date '+%F %T')] resume: prepare pose3d labels" | tee -a "$LOG"
"$PY" "$JP/prepare_0810_pose3d_labels.py" \
  --label-root "$DATASET/labels" \
  --pre-limb-map "$PREMAP" \
  --output "$POSE3D" | tee -a "$LOG"

if ! grep -q 'global_idx' "$EGO/experiments/stage3_pose3d/scripts/train_pose3d.py"; then
  "$CAMTEST" "$JP/_patch_train_pose3d_alignment.py"
fi

echo "stage1_v31_running" | tee "$STATUS"
cd "$EGO"
"$CAMTEST" scripts/train_heatmap.py \
  --label-root "$DATASET/labels" \
  --frame-root "$DATASET/frames" \
  --output-dir "$DATASET/checkpoints/stage1_v31" \
  --log-dir "$DATASET/logs/stage1_v31" \
  --split-manifest "$SPLIT" \
  --epochs 9999 --batch-size 32 --workers 8 --lr 0.0001 --weight-decay 0.005 \
  --seed 42 --device cuda --image-width 480 --image-height 300 \
  --base-channels 64 --visible-only-loss --train-branch head \
  --joint-radius-config "$RADIUS" --default-joint-radius-px 10 \
  --early-stop-patience 20 --save-every 5 --keep-last 3 --log-every 50 \
  --prefetch-factor 4 --max-hours 72 \
  2>&1 | tee -a "$LOG"
echo "stage1_v31_done" | tee "$STATUS"

echo "stage2_v31_running" | tee "$STATUS"
"$CAMTEST" experiments/stage2_refinement/scripts/train_refinement.py \
  --label-root "$DATASET/labels" \
  --stage1-checkpoint "$DATASET/checkpoints/stage1_v31/best.pt" \
  --output-dir "$DATASET/checkpoints/stage2_v31" \
  --log-dir "$DATASET/logs/stage2_v31" \
  --split-manifest "$SPLIT" \
  --epochs 9999 --batch-size 16 --workers 8 --lr 0.001 --weight-decay 0.005 \
  --selection-metric refined_pixel_error --min-epochs 1 --early-stop-patience 20 \
  --heatmap-width 120 --heatmap-height 75 --image-width 480 --image-height 300 \
  --base-channels 64 --device cuda --seed 42 --max-hours 72 \
  2>&1 | tee -a "$LOG"
echo "stage2_v31_done" | tee "$STATUS"

echo "stage3_v31_aligned_running" | tee "$STATUS"
"$CAMTEST" experiments/stage3_pose3d/scripts/train_pose3d.py \
  --label-root "$DATASET/labels" \
  --pose3d-labels "$POSE3D" \
  --stage1-checkpoint "$DATASET/checkpoints/stage1_v31/best.pt" \
  --stage2-checkpoint "$DATASET/checkpoints/stage2_v31/best.pt" \
  --output-dir "$DATASET/checkpoints/stage3_v31_aligned" \
  --split-manifest "$SPLIT" \
  --epochs 9999 --batch-size 64 --workers 8 --lr 0.001 --weight-decay 0.0005 \
  --min-epochs 1 --early-stop-patience 20 --device cuda --seed 42 \
  2>&1 | tee -a "$LOG"
echo "done" | tee "$STATUS"
