#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-${PYTHON_BIN:-python}}
LABEL_ROOT=${LABEL_ROOT:-data/labels/real_0722_01_head2cam_direct2d}
POSE3D_LABELS=${LABEL_ROOT}/pose3d_head_stereo_lifted_12j.npz
SEED=${SEED:-42}
PATIENCE=${PATIENCE:-15}
MAX_EPOCHS=${MAX_EPOCHS:-500}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

run_stride() {
  local stride=$1
  local tag="real_0722_01_stride${stride}_random_seed${SEED}"
  local split="data/splits/${tag}.npz"
  local stage1_dir="checkpoints/${tag}_stage1"
  local stage2_dir="checkpoints/${tag}_stage2"
  local stage3_dir="checkpoints/${tag}_stage3"
  local run_root="logs/${tag}_stages123"

  mkdir -p "${run_root}"
  {
    echo "state=preparing_split"
    echo "stride=${stride}"
    echo "seed=${SEED}"
    echo "started=$(date --iso-8601=seconds)"
  } > "${run_root}/pipeline_status.txt"

  "${PYTHON}" scripts/create_dataset_split.py \
    --label-root "${LABEL_ROOT}" \
    --output "${split}" \
    --mode strided-random \
    --stride "${stride}" \
    --seed "${SEED}" \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    > "${run_root}/split.log" 2>&1

  echo "state=stage1 stride=${stride} started=$(date --iso-8601=seconds)" \
    > "${run_root}/pipeline_status.txt"
  "${PYTHON}" scripts/train_heatmap.py \
    --label-root "${LABEL_ROOT}" \
    --output-dir "${stage1_dir}" \
    --log-dir "${run_root}/stage1_tensorboard" \
    --split-manifest "${split}" \
    --epochs "${MAX_EPOCHS}" \
    --batch-size 32 \
    --workers 8 \
    --lr 0.0001 \
    --weight-decay 0.005 \
    --visible-only-loss \
    --train-branch head \
    --save-every 0 \
    --early-stop-patience "${PATIENCE}" \
    > "${run_root}/stage1_train.log" 2>&1

  echo "state=stage2 stride=${stride} started=$(date --iso-8601=seconds)" \
    > "${run_root}/pipeline_status.txt"
  "${PYTHON}" experiments/stage2_refinement/scripts/train_refinement.py \
    --label-root "${LABEL_ROOT}" \
    --stage1-checkpoint "${stage1_dir}/best.pt" \
    --output-dir "${stage2_dir}" \
    --log-dir "${run_root}/stage2_tensorboard" \
    --split-manifest "${split}" \
    --epochs "${MAX_EPOCHS}" \
    --batch-size 32 \
    --workers 8 \
    --lr 0.001 \
    --weight-decay 0.005 \
    --selection-metric refined_pixel_error \
    --min-epochs 1 \
    --early-stop-patience "${PATIENCE}" \
    > "${run_root}/stage2_train.log" 2>&1

  echo "state=stage3 stride=${stride} started=$(date --iso-8601=seconds)" \
    > "${run_root}/pipeline_status.txt"
  "${PYTHON}" experiments/stage3_pose3d/scripts/train_pose3d.py \
    --label-root "${LABEL_ROOT}" \
    --pose3d-labels "${POSE3D_LABELS}" \
    --stage1-checkpoint "${stage1_dir}/best.pt" \
    --stage2-checkpoint "${stage2_dir}/best.pt" \
    --output-dir "${stage3_dir}" \
    --split-manifest "${split}" \
    --epochs "${MAX_EPOCHS}" \
    --batch-size 64 \
    --workers 8 \
    --lr 0.001 \
    --weight-decay 0.0005 \
    --min-epochs 1 \
    --early-stop-patience "${PATIENCE}" \
    > "${run_root}/stage3_train.log" 2>&1

  echo "state=complete stride=${stride} finished=$(date --iso-8601=seconds)" \
    > "${run_root}/pipeline_status.txt"
}

MASTER_LOG=logs/real_0722_01_strided_three_runs_status.txt
mkdir -p "$(dirname "${MASTER_LOG}")"
echo "state=running started=$(date --iso-8601=seconds)" > "${MASTER_LOG}"
for stride in 10 30 90; do
  echo "current_stride=${stride} started=$(date --iso-8601=seconds)" >> "${MASTER_LOG}"
  run_stride "${stride}"
  echo "completed_stride=${stride} finished=$(date --iso-8601=seconds)" >> "${MASTER_LOG}"
done
echo "state=complete finished=$(date --iso-8601=seconds)" >> "${MASTER_LOG}"
