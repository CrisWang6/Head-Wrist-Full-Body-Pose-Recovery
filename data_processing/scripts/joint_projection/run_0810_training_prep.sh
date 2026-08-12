#!/usr/bin/env bash
# Finalize triangulation + head 2D CSV (csv-only) for line1/line2, then launch training.
# Skips nose detect / optimize / render (not required for stage1/2/3 labels).
set -euo pipefail

BATCH=/home/gaoweijian/0810_batch
JP="$BATCH/repo/test_code/joint_projection"
PYTHON=/home/gaoweijian/miniforge3/envs/sapiens2/bin/python
export PYTHONPATH="$JP"
export LD_LIBRARY_PATH=/home/gaoweijian/miniforge3/envs/sapiens2/lib/python3.12/site-packages/nvidia/cu13/lib:/home/gaoweijian/miniforge3/envs/sapiens2/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}

finalize_line() {
  local NAME=$1
  case "$NAME" in
    line1) CONFIG="$JP/configs/0810_line1_dual_external_mocap.json" ;;
    line2) CONFIG="$JP/configs/0810_line2_dual_external_mocap.json" ;;
    *) echo "bad line $NAME" >&2; return 1 ;;
  esac
  local ROOT="$BATCH/$NAME"
  local DATA_ROOT="$ROOT/data_root"
  local FULL="$DATA_ROOT/multiview_3d_results/full"
  local HEAD_REPRO="$FULL/head_reprojection"
  local PRE_LIMB="$FULL/multiview_3d_results_pre_limb.jsonl"
  local PLAYBACK="$FULL/skeleton_playback.json"
  local CSV="$HEAD_REPRO/head_reprojection_2d_wo_calibration_proj.csv"
  mkdir -p "$HEAD_REPRO" "$FULL/visualization" "$ROOT/logs"

  echo "[prep] finalize $NAME"
  cp -f "$PRE_LIMB" "$FULL/multiview_3d_results.jsonl"
  "$PYTHON" "$JP/export_playback_from_jsonl.py" \
    --results "$PRE_LIMB" \
    --output "$PLAYBACK" \
    --source "external triangulation (0810 training prep)" \
    --prune \
    >"$ROOT/logs/finalize_training_prep.log" 2>&1

  if [[ ! -s "$CSV" ]]; then
    echo "[prep] head 2D csv-only $NAME"
    "$PYTHON" "$JP/render_multiview_to_head.py" \
      --data-root "$DATA_ROOT" --config "$CONFIG" \
      --skeleton-playback "$PLAYBACK" \
      --output-dir "$HEAD_REPRO" --csv-only \
      >"$ROOT/logs/head_csv_training_prep.log" 2>&1
  fi
  wc -l "$CSV" | tee -a "$ROOT/logs/finalize_training_prep.log"
}

for NAME in line1 line2; do
  finalize_line "$NAME"
done

echo "[prep] launch training master"
bash "$JP/run_0810_training_master.sh"
