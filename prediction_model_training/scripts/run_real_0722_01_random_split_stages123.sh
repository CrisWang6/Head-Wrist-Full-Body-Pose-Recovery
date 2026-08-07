#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-${PYTHON_BIN:-python}}
LABEL_ROOT=${LABEL_ROOT:-data/labels/real_0722_01_head2cam_direct2d}
POSE3D_LABELS=${POSE3D_LABELS:-${LABEL_ROOT}/pose3d_head_stereo_lifted_12j.npz}
SPLIT_MANIFEST=${SPLIT_MANIFEST:-data/splits/real_0722_01_global_random_80_10_10_seed42.npz}
STAGE1_DIR=checkpoints/real_0722_01_randomsplit_seed42_stage1
STAGE2_DIR=checkpoints/real_0722_01_randomsplit_seed42_stage2
STAGE3_DIR=checkpoints/real_0722_01_randomsplit_seed42_stage3
RUN_ROOT=logs/real_0722_01_randomsplit_seed42_stages123

mkdir -p "${RUN_ROOT}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
START_STAGE=${START_STAGE:-1}

if (( START_STAGE <= 1 )); then
echo "stage1_started $(date --iso-8601=seconds)" > "${RUN_ROOT}/pipeline_status.txt"
"${PYTHON}" scripts/train_heatmap.py \
  --label-root "${LABEL_ROOT}" \
  --output-dir "${STAGE1_DIR}" \
  --log-dir "${RUN_ROOT}/stage1_tensorboard" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --epochs 500 \
  --batch-size 32 \
  --workers 8 \
  --lr 0.0001 \
  --weight-decay 0.005 \
  --visible-only-loss \
  --train-branch head \
  --save-every 0 \
  --early-stop-patience 15 \
  > "${RUN_ROOT}/stage1_train.log" 2>&1
fi

if (( START_STAGE <= 2 )); then
echo "stage1_complete stage2_started $(date --iso-8601=seconds)" > "${RUN_ROOT}/pipeline_status.txt"
"${PYTHON}" experiments/stage2_refinement/scripts/train_refinement.py \
  --label-root "${LABEL_ROOT}" \
  --stage1-checkpoint "${STAGE1_DIR}/best.pt" \
  --output-dir "${STAGE2_DIR}" \
  --log-dir "${RUN_ROOT}/stage2_tensorboard" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --epochs 500 \
  --batch-size 32 \
  --workers 8 \
  --lr 0.001 \
  --weight-decay 0.005 \
  --selection-metric refined_pixel_error \
  --min-epochs 1 \
  --early-stop-patience 15 \
  > "${RUN_ROOT}/stage2_train.log" 2>&1
fi

if (( START_STAGE <= 3 )); then
echo "stage2_complete stage3_started $(date --iso-8601=seconds)" > "${RUN_ROOT}/pipeline_status.txt"
"${PYTHON}" experiments/stage3_pose3d/scripts/train_pose3d.py \
  --label-root "${LABEL_ROOT}" \
  --pose3d-labels "${POSE3D_LABELS}" \
  --stage1-checkpoint "${STAGE1_DIR}/best.pt" \
  --stage2-checkpoint "${STAGE2_DIR}/best.pt" \
  --output-dir "${STAGE3_DIR}" \
  --split-manifest "${SPLIT_MANIFEST}" \
  --epochs 500 \
  --batch-size 64 \
  --workers 8 \
  --lr 0.001 \
  --weight-decay 0.0005 \
  --min-epochs 1 \
  --early-stop-patience 15 \
  > "${RUN_ROOT}/stage3_train.log" 2>&1
fi

echo "all_stages_complete $(date --iso-8601=seconds)" > "${RUN_ROOT}/pipeline_status.txt"
