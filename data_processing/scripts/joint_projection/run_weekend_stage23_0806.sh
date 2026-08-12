#!/usr/bin/env bash
# Continuation: waits for Stage1, runs Stage2+3 with delivery15 / 480x300 input / 120x75 heatmap.
set -euo pipefail

BATCH=/home/gaoweijian/0806_batch
JP="$BATCH/repo/test_code/joint_projection"
PY=/home/gaoweijian/miniforge3/envs/sapiens2/bin/python
CAMTEST_PY=/home/gaoweijian/miniforge3/envs/camtest/bin/python
EGO=/home/gaoweijian/EgoRear_w_hand
DATASET=/home/gaoweijian/0806dataset
LOG="$DATASET/logs/weekend_stage23.log"
STATUS="$DATASET/WEEKEND_STAGE23_STATUS.txt"
PRE_LIMB_MAP="$DATASET/pre_limb_map.json"
POSE3D_LABELS="$DATASET/labels/pose3d_nose_pre_limb_15j.npz"
LABEL_NPZ="heatmap_labels_120x75.npz"
HM_W=120
HM_H=75
IMG_W=480
IMG_H=300

export PYTHONPATH="$JP"
export LD_LIBRARY_PATH=/home/gaoweijian/miniforge3/envs/sapiens2/lib/python3.12/site-packages/nvidia/cu13/lib:/home/gaoweijian/miniforge3/envs/sapiens2/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}

mkdir -p "$DATASET/logs" "$DATASET/checkpoints" "$DATASET/eval"

stamp() { date '+%F %T'; }
log() { echo "[$(stamp)] $*" | tee -a "$LOG"; }
set_status() { echo "$1" | tee "$STATUS"; }

exec >>"$LOG" 2>&1
log "===== WEEKEND STAGE23 CONTINUATION (delivery15 480x300/120x75) ====="
set_status "waiting_labels"

wait_for_file() {
  local path=$1
  local label=$2
  while [[ ! -f "$path" ]]; do
    log "waiting for $label: $path"
    sleep 120
  done
  log "ready: $path"
}

for limb in ankle wrist wu; do
  wait_for_file "$DATASET/labels/$limb/$LABEL_NPZ" "heatmap $limb"
done

if [[ ! -f "$PRE_LIMB_MAP" ]]; then
  "$PY" - <<'PY' "$PRE_LIMB_MAP" "$BATCH"
import json, sys
from pathlib import Path
out = Path(sys.argv[1]); batch = Path(sys.argv[2])
mapping = {
    "ankle": str(batch / "ankle/data_root/multiview_3d_results/full/multiview_3d_results_pre_limb.jsonl"),
    "wrist": str(batch / "wrist/data_root/multiview_3d_results/full/multiview_3d_results_pre_limb.jsonl"),
    "wu": str(batch / "wu/data_root/multiview_3d_results/full/multiview_3d_results_pre_limb.jsonl"),
}
out.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
PY
fi

for scheme in v31 v32; do
  wait_for_file "$DATASET/checkpoints/stage1_${scheme}/best.pt" "stage1 $scheme"
done
set_status "stage1_ready"

master_pid=""
while read -r pid _; do [[ -n "$pid" ]] && master_pid="$pid"; done < <(pgrep -f '/run_weekend_training_master.sh' || true)
if [[ -n "$master_pid" ]]; then
  log "master PID $master_pid running — wait for stage2/3 or exit"
  set_status "waiting_master_finish"
  while kill -0 "$master_pid" 2>/dev/null; do
    if [[ -f "$DATASET/checkpoints/stage3_v31/best.pt" && -f "$DATASET/checkpoints/stage3_v32/best.pt" ]]; then
      set_status "done_master_owned"
      exit 0
    fi
    sleep 120
  done
  if [[ -f "$DATASET/checkpoints/stage3_v31/best.pt" && -f "$DATASET/checkpoints/stage3_v32/best.pt" ]]; then
    set_status "done_master_owned"
    exit 0
  fi
fi

for scheme in v31 v32; do
  [[ -f "$DATASET/splits/pack30_${scheme}.npz" ]] || "$PY" "$JP/build_0806_pack_splits.py" --scheme "$scheme"
done

"$PY" "$JP/prepare_0806_pose3d_labels.py" \
  --label-root "$DATASET/labels" \
  --pre-limb-map "$PRE_LIMB_MAP" \
  --output "$POSE3D_LABELS"
set_status "pose3d_labels_ready"

run_stage2() {
  local scheme=$1
  local split_npz="$DATASET/splits/pack30_${scheme}.npz"
  local out="$DATASET/checkpoints/stage2_${scheme}"
  local logd="$DATASET/logs/stage2_${scheme}"
  [[ -f "$out/best.pt" ]] && return 0
  set_status "stage2_${scheme}_running"
  cd "$EGO"
  "$CAMTEST_PY" experiments/stage2_refinement/scripts/train_refinement.py \
    --label-root "$DATASET/labels" \
    --stage1-checkpoint "$DATASET/checkpoints/stage1_${scheme}/best.pt" \
    --output-dir "$out" --log-dir "$logd" --split-manifest "$split_npz" \
    --epochs 9999 --batch-size 16 --workers 8 --lr 0.001 --weight-decay 0.005 \
    --selection-metric refined_pixel_error --min-epochs 1 --early-stop-patience 20 \
    --heatmap-width "$HM_W" --heatmap-height "$HM_H" \
    --image-width "$IMG_W" --image-height "$IMG_H" \
    --base-channels 64 --device cuda --seed 42 --max-hours 72
  set_status "stage2_${scheme}_done"
}

run_stage3() {
  local scheme=$1
  local split_npz="$DATASET/splits/pack30_${scheme}.npz"
  local out="$DATASET/checkpoints/stage3_${scheme}"
  [[ -f "$out/best.pt" ]] && return 0
  set_status "stage3_${scheme}_running"
  cd "$EGO"
  "$CAMTEST_PY" experiments/stage3_pose3d/scripts/train_pose3d.py \
    --label-root "$DATASET/labels" \
    --pose3d-labels "$POSE3D_LABELS" \
    --stage1-checkpoint "$DATASET/checkpoints/stage1_${scheme}/best.pt" \
    --stage2-checkpoint "$DATASET/checkpoints/stage2_${scheme}/best.pt" \
    --output-dir "$out" --split-manifest "$split_npz" \
    --epochs 9999 --batch-size 64 --workers 8 --lr 0.001 --weight-decay 0.0005 \
    --min-epochs 1 --early-stop-patience 20 --device cuda --seed 42
  set_status "stage3_${scheme}_done"
}

for scheme in v31 v32; do run_stage2 "$scheme"; done
for scheme in v31 v32; do run_stage3 "$scheme"; done
for scheme in v31 v32; do
  "$PY" "$JP/eval_0806_test_3d.py" \
    --split-npz "$DATASET/splits/pack30_${scheme}.npz" \
    --pre-limb-map "$PRE_LIMB_MAP" --label-root "$DATASET/labels" \
    --output-dir "$DATASET/eval/${scheme}" --split-name test
done
set_status "done"
log "===== WEEKEND STAGE23 DONE ====="
