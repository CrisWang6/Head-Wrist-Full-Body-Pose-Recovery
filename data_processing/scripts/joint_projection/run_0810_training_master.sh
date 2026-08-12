#!/usr/bin/env bash
# 0810 master: labels -> pack150 splits -> Stage1/2/3 (aligned global_idx), same hyperparams as 0806 v31.
set -euo pipefail

BATCH=/home/gaoweijian/0810_batch
JP="$BATCH/repo/test_code/joint_projection"
PY=/home/gaoweijian/miniforge3/envs/sapiens2/bin/python
CAMTEST_PY=/home/gaoweijian/miniforge3/envs/camtest/bin/python
EGO=/home/gaoweijian/EgoRear_w_hand
DATASET=/home/gaoweijian/0810dataset
LOG="$DATASET/logs/0810_training_master.log"
STATUS="$DATASET/0810_TRAINING_STATUS.txt"
HM_W=120
HM_H=75
IMG_W=480
IMG_H=300
PACK=150
SCHEME=v31
SPLIT="$DATASET/splits/pack${PACK}_${SCHEME}.npz"
RADIUS_CFG="$JP/configs/joint_radius_px_120x75_delivery15.json"
POSE3D_LABELS="$DATASET/labels/pose3d_nose_pre_limb_15j.npz"
PRE_LIMB_MAP="$DATASET/pre_limb_map.json"

export PYTHONPATH="$JP"
export LD_LIBRARY_PATH=/home/gaoweijian/miniforge3/envs/sapiens2/lib/python3.12/site-packages/nvidia/cu13/lib:/home/gaoweijian/miniforge3/envs/sapiens2/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}

mkdir -p "$DATASET/logs" "$DATASET/labels" "$DATASET/splits" "$DATASET/checkpoints" "$DATASET/eval" "$DATASET/frames"

stamp() { date '+%F %T'; }
log() { echo "[$(stamp)] $*" | tee -a "$LOG"; }
set_status() { echo "$1" | tee "$STATUS"; }

exec >>"$LOG" 2>&1
log "===== 0810 TRAINING MASTER START (pack${PACK}=5s, line1 train / line2 test) ====="
set_status "starting"

for NAME in line1 line2; do
  set_status "reexport_${NAME}"
  bash "$JP/reexport_0810_heatmap_labels.sh" "$NAME"
done

log "extract head RGB frames"
"$PY" "$JP/extract_0806_head_frames.py" \
  --batch-root "$BATCH" \
  --output-root "$DATASET" \
  --datasets line1 line2 \
  --skip-existing

log "build pack splits (5s packs)"
"$PY" "$JP/build_0810_pack_splits.py" --label-root "$DATASET/labels" --output-dir "$DATASET/splits"

"$PY" - <<'PY' "$PRE_LIMB_MAP" "$BATCH"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
batch = Path(sys.argv[2])
mapping = {
    "line1": str(batch / "line1/data_root/multiview_3d_results/full/multiview_3d_results_pre_limb.jsonl"),
    "line2": str(batch / "line2/data_root/multiview_3d_results/full/multiview_3d_results_pre_limb.jsonl"),
}
out.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
print(json.dumps(mapping, indent=2))
PY

log "prepare pose3d labels"
"$PY" "$JP/prepare_0810_pose3d_labels.py" \
  --label-root "$DATASET/labels" \
  --pre-limb-map "$PRE_LIMB_MAP" \
  --output "$POSE3D_LABELS"

if ! grep -q 'global_idx' "$EGO/experiments/stage3_pose3d/scripts/train_pose3d.py"; then
  "$CAMTEST_PY" "$JP/_patch_train_pose3d_alignment.py"
fi

run_stage1() {
  local out="$DATASET/checkpoints/stage1_${SCHEME}"
  local logd="$DATASET/logs/stage1_${SCHEME}"
  set_status "stage1_${SCHEME}_running"
  cd "$EGO"
  "$CAMTEST_PY" scripts/train_heatmap.py \
    --label-root "$DATASET/labels" \
    --frame-root "$DATASET/frames" \
    --output-dir "$out" \
    --log-dir "$logd" \
    --split-manifest "$SPLIT" \
    --epochs 9999 --batch-size 32 --workers 8 --lr 0.0001 --weight-decay 0.005 \
    --seed 42 --device cuda --image-width "$IMG_W" --image-height "$IMG_H" \
    --base-channels 64 --visible-only-loss --train-branch head \
    --joint-radius-config "$RADIUS_CFG" --default-joint-radius-px 10 \
    --early-stop-patience 20 --save-every 5 --keep-last 3 --log-every 50 \
    --prefetch-factor 4 --max-hours 72
  set_status "stage1_${SCHEME}_done"
}

run_stage2() {
  local out="$DATASET/checkpoints/stage2_${SCHEME}"
  local logd="$DATASET/logs/stage2_${SCHEME}"
  set_status "stage2_${SCHEME}_running"
  cd "$EGO"
  "$CAMTEST_PY" experiments/stage2_refinement/scripts/train_refinement.py \
    --label-root "$DATASET/labels" \
    --stage1-checkpoint "$DATASET/checkpoints/stage1_${SCHEME}/best.pt" \
    --output-dir "$out" --log-dir "$logd" --split-manifest "$SPLIT" \
    --epochs 9999 --batch-size 16 --workers 8 --lr 0.001 --weight-decay 0.005 \
    --selection-metric refined_pixel_error --min-epochs 1 --early-stop-patience 20 \
    --heatmap-width "$HM_W" --heatmap-height "$HM_H" \
    --image-width "$IMG_W" --image-height "$IMG_H" \
    --base-channels 64 --device cuda --seed 42 --max-hours 72
  set_status "stage2_${SCHEME}_done"
}

run_stage3() {
  local out="$DATASET/checkpoints/stage3_${SCHEME}_aligned"
  set_status "stage3_${SCHEME}_aligned_running"
  cd "$EGO"
  "$CAMTEST_PY" experiments/stage3_pose3d/scripts/train_pose3d.py \
    --label-root "$DATASET/labels" \
    --pose3d-labels "$POSE3D_LABELS" \
    --stage1-checkpoint "$DATASET/checkpoints/stage1_${SCHEME}/best.pt" \
    --stage2-checkpoint "$DATASET/checkpoints/stage2_${SCHEME}/best.pt" \
    --output-dir "$out" \
    --split-manifest "$SPLIT" \
    --epochs 9999 --batch-size 64 --workers 8 --lr 0.001 --weight-decay 0.0005 \
    --min-epochs 1 --early-stop-patience 20 --device cuda --seed 42
  set_status "stage3_${SCHEME}_aligned_done"
}

run_stage1
run_stage2
run_stage3
set_status "done"
log "===== 0810 TRAINING MASTER DONE ====="
