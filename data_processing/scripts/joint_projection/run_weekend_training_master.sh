#!/usr/bin/env bash
# Weekend master: Task2 -> relabel 15j/120x75 -> splits -> Stage1/2/3
set -euo pipefail

BATCH=/home/gaoweijian/0806_batch
JP="$BATCH/repo/test_code/joint_projection"
PY=/home/gaoweijian/miniforge3/envs/sapiens2/bin/python
CAMTEST_PY=/home/gaoweijian/miniforge3/envs/camtest/bin/python
EGO=/home/gaoweijian/EgoRear_w_hand
DATASET=/home/gaoweijian/0806dataset
LOG="$DATASET/logs/weekend_master.log"
STATUS="$DATASET/WEEKEND_MASTER_STATUS.txt"
HM_W=120
HM_H=75
IMG_W=480
IMG_H=300
RADIUS_CFG="$JP/configs/joint_radius_px_120x75_delivery15.json"
POSE3D_LABELS="$DATASET/labels/pose3d_nose_pre_limb_15j.npz"

export PYTHONPATH="$JP"
export LD_LIBRARY_PATH=/home/gaoweijian/miniforge3/envs/sapiens2/lib/python3.12/site-packages/nvidia/cu13/lib:/home/gaoweijian/miniforge3/envs/sapiens2/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}

mkdir -p "$DATASET/logs" "$DATASET/labels" "$DATASET/splits" "$DATASET/checkpoints" "$DATASET/eval"

stamp() { date '+%F %T'; }
log() { echo "[$(stamp)] $*" | tee -a "$LOG"; }
set_status() { echo "$1" | tee "$STATUS"; }

exec >>"$LOG" 2>&1
log "===== WEEKEND MASTER START (delivery15 480x300 input / 120x75 heatmap) ====="
set_status "starting"

log "stop run_wu_full_10s if running"
pkill -u gaoweijian -f run_wu_full_10s_production_gwj.sh 2>/dev/null || true
sleep 3

for LIMB in wu wrist ankle; do
  set_status "task2_${LIMB}_running"
  log "Task2 $LIMB"
  bash "$JP/run_task2_bc_labels_gwj.sh" "$LIMB" || {
    log "Task2 $LIMB failed - retry once after 60s"
    sleep 60
    bash "$JP/run_task2_bc_labels_gwj.sh" "$LIMB"
  }
done
set_status "task2_done"

log "ensure pre_limb exists for all limbs (repair if external job wiped outputs)"
for LIMB in wu wrist ankle; do
  PRE="$BATCH/$LIMB/data_root/multiview_3d_results/full/multiview_3d_results_pre_limb.jsonl"
  if [[ ! -s "$PRE" ]]; then
    log "missing pre_limb for $LIMB - re-run Task2"
    bash "$JP/run_task2_bc_labels_gwj.sh" "$LIMB" || {
      log "Task2 repair $LIMB failed - retry once"
      sleep 60
      bash "$JP/run_task2_bc_labels_gwj.sh" "$LIMB"
    }
  fi
done

log "re-export heatmap labels (15 joints, 120x75) from Task2 CSV"
for LIMB in wu wrist ankle; do
  bash "$JP/reexport_0806_heatmap_labels.sh" "$LIMB"
done
rm -f "$DATASET/labels"/*/"heatmap_labels_114x64.npz" "$DATASET/labels"/*/"heatmap_labels_480x300.npz" 2>/dev/null || true
rm -rf "$DATASET/checkpoints/stage1_v31" "$DATASET/checkpoints/stage1_v32" \
       "$DATASET/checkpoints/stage2_v31" "$DATASET/checkpoints/stage2_v32" \
       "$DATASET/checkpoints/stage3_v31" "$DATASET/checkpoints/stage3_v32" 2>/dev/null || true

log "build pack splits"
"$PY" "$JP/build_0806_pack_splits.py" --scheme v31
"$PY" "$JP/build_0806_pack_splits.py" --scheme v32

PRE_LIMB_MAP="$DATASET/pre_limb_map.json"
"$PY" - <<'PY' "$PRE_LIMB_MAP" "$BATCH"
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
batch = Path(sys.argv[2])
mapping = {
    "ankle": str(batch / "ankle/data_root/multiview_3d_results/full/multiview_3d_results_pre_limb.jsonl"),
    "wrist": str(batch / "wrist/data_root/multiview_3d_results/full/multiview_3d_results_pre_limb.jsonl"),
    "wu": str(batch / "wu/data_root/multiview_3d_results/full/multiview_3d_results_pre_limb.jsonl"),
}
out.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
print(json.dumps(mapping, indent=2))
PY

run_stage1() {
  local scheme=$1
  local split_npz="$DATASET/splits/pack30_${scheme}.npz"
  local out="$DATASET/checkpoints/stage1_${scheme}"
  local logd="$DATASET/logs/stage1_${scheme}"
  set_status "stage1_${scheme}_running"
  log "Stage1 train scheme=$scheme"
  cd "$EGO"
  "$CAMTEST_PY" scripts/train_heatmap.py \
    --label-root "$DATASET/labels" \
    --frame-root "$DATASET/frames" \
    --output-dir "$out" \
    --log-dir "$logd" \
    --split-manifest "$split_npz" \
    --epochs 9999 \
    --batch-size 32 \
    --workers 8 \
    --lr 0.0001 \
    --weight-decay 0.005 \
    --seed 42 \
    --device cuda \
    --image-width "$IMG_W" \
    --image-height "$IMG_H" \
    --base-channels 64 \
    --visible-only-loss \
    --train-branch head \
    --joint-radius-config "$RADIUS_CFG" \
    --default-joint-radius-px 10 \
    --early-stop-patience 20 \
    --save-every 5 \
    --keep-last 3 \
    --log-every 50 \
    --prefetch-factor 4 \
    --max-hours 72
  set_status "stage1_${scheme}_done"
}

run_stage1 v31
run_stage1 v32

log "prepare pose3d labels (nose offset, 15 joints)"
"$PY" "$JP/prepare_0806_pose3d_labels.py" \
  --label-root "$DATASET/labels" \
  --pre-limb-map "$PRE_LIMB_MAP" \
  --output "$POSE3D_LABELS"

run_stage2() {
  local scheme=$1
  local split_npz="$DATASET/splits/pack30_${scheme}.npz"
  local out="$DATASET/checkpoints/stage2_${scheme}"
  local logd="$DATASET/logs/stage2_${scheme}"
  if [[ -f "$out/best.pt" ]]; then
    log "Stage2 $scheme already has best.pt - skip"
    return 0
  fi
  set_status "stage2_${scheme}_running"
  log "Stage2 refine scheme=$scheme"
  cd "$EGO"
  "$CAMTEST_PY" experiments/stage2_refinement/scripts/train_refinement.py \
    --label-root "$DATASET/labels" \
    --stage1-checkpoint "$DATASET/checkpoints/stage1_${scheme}/best.pt" \
    --output-dir "$out" \
    --log-dir "$logd" \
    --split-manifest "$split_npz" \
    --epochs 9999 \
    --batch-size 16 \
    --workers 8 \
    --lr 0.001 \
    --weight-decay 0.005 \
    --selection-metric refined_pixel_error \
    --min-epochs 1 \
    --early-stop-patience 20 \
    --heatmap-width "$HM_W" \
    --heatmap-height "$HM_H" \
    --image-width "$IMG_W" \
    --image-height "$IMG_H" \
    --base-channels 64 \
    --device cuda \
    --seed 42 \
    --max-hours 72
  set_status "stage2_${scheme}_done"
}

run_stage3() {
  local scheme=$1
  local split_npz="$DATASET/splits/pack30_${scheme}.npz"
  local out="$DATASET/checkpoints/stage3_${scheme}"
  if [[ -f "$out/best.pt" ]]; then
    log "Stage3 $scheme already has best.pt - skip"
    return 0
  fi
  set_status "stage3_${scheme}_running"
  log "Stage3 pose3d train scheme=$scheme"
  cd "$EGO"
  "$CAMTEST_PY" experiments/stage3_pose3d/scripts/train_pose3d.py \
    --label-root "$DATASET/labels" \
    --pose3d-labels "$POSE3D_LABELS" \
    --stage1-checkpoint "$DATASET/checkpoints/stage1_${scheme}/best.pt" \
    --stage2-checkpoint "$DATASET/checkpoints/stage2_${scheme}/best.pt" \
    --output-dir "$out" \
    --split-manifest "$split_npz" \
    --epochs 9999 \
    --batch-size 64 \
    --workers 8 \
    --lr 0.001 \
    --weight-decay 0.0005 \
    --min-epochs 1 \
    --early-stop-patience 20 \
    --device cuda \
    --seed 42
  set_status "stage3_${scheme}_done"
}

for scheme in v31 v32; do run_stage2 "$scheme"; done
for scheme in v31 v32; do run_stage3 "$scheme"; done

for scheme in v31 v32; do
  set_status "stage3_eval_${scheme}_running"
  "$PY" "$JP/eval_0806_test_3d.py" \
    --split-npz "$DATASET/splits/pack30_${scheme}.npz" \
    --pre-limb-map "$PRE_LIMB_MAP" \
    --label-root "$DATASET/labels" \
    --output-dir "$DATASET/eval/${scheme}" \
    --split-name test
done
set_status "done"
log "===== WEEKEND MASTER DONE ====="
