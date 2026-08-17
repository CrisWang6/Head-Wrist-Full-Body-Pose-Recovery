#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/home/gaoweijian/0810_batch/repo/test_code/joint_projection:/home/gaoweijian/EgoRear_w_hand/src"
export CUDA_VISIBLE_DEVICES=1
LOG="/home/gaoweijian/0810dataset/logs/0810_stage3_skip_stage2_200ep.log"
exec >>"$LOG" 2>&1
echo "[$(date '+%F %T')] relaunch stage3 batch32 GPU1"
echo stage3_v31_aligned_running > /home/gaoweijian/0810dataset/0810_TRAINING_STATUS.txt
cd /home/gaoweijian/EgoRear_w_hand
/home/gaoweijian/miniforge3/envs/camtest/bin/python experiments/stage3_pose3d/scripts/train_pose3d.py \
  --label-root /home/gaoweijian/0810dataset/labels \
  --pose3d-labels /home/gaoweijian/0810dataset/labels/pose3d_nose_pre_limb_15j.npz \
  --stage1-checkpoint /home/gaoweijian/0810dataset/checkpoints/stage1_v31/best.pt \
  --skip-stage2 \
  --output-dir /home/gaoweijian/0810dataset/checkpoints/stage3_v31_aligned \
  --split-manifest /home/gaoweijian/0810dataset/splits/pack150_v31.npz \
  --epochs 200 --batch-size 32 --workers 8 --lr 0.001 --weight-decay 0.0005 \
  --min-epochs 200 --early-stop-patience 9999 --device cuda --seed 42
