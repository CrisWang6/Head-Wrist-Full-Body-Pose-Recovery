#!/usr/bin/env bash
# 0810: re-export heatmap NPZ for line1/line2 after pipeline B→E.
set -euo pipefail

NAME="${1:?usage: $0 line1|line2}"
BATCH=/home/gaoweijian/0810_batch
JP="$BATCH/repo/test_code/joint_projection"
PYTHON=/home/gaoweijian/miniforge3/envs/sapiens2/bin/python
LABEL_ROOT=/home/gaoweijian/0810dataset/labels
LOG_ROOT=/home/gaoweijian/0810dataset/logs/reexport

export PYTHONPATH="$JP"

case "$NAME" in
  line1)
    CONFIG="$JP/configs/0810_line1_dual_external_mocap.json"
    HEAD_DIR_NAME=0712_035226
    ;;
  line2)
    CONFIG="$JP/configs/0810_line2_dual_external_mocap.json"
    HEAD_DIR_NAME=0712_035903
    ;;
  *) echo "NAME must be line1|line2" >&2; exit 1 ;;
esac

ROOT="$BATCH/$NAME"
DATA_ROOT="$ROOT/data_root"
FULL="$DATA_ROOT/multiview_3d_results/full"
HEAD_REPRO="$FULL/head_reprojection"
CSV="$HEAD_REPRO/head_reprojection_2d_wo_calibration_proj.csv"
PRE_LIMB="$FULL/multiview_3d_results_pre_limb.jsonl"
PLAYBACK="$FULL/skeleton_playback_raw.json"
mkdir -p "$LOG_ROOT" "$HEAD_REPRO" "$LABEL_ROOT/$NAME"

if [[ ! -f "$CSV" ]]; then
  echo "[reexport] regenerate head 2D CSV for $NAME"
  if [[ ! -f "$PLAYBACK" ]]; then
    "$PYTHON" "$JP/export_playback_from_jsonl.py" \
      --results "$PRE_LIMB" \
      --output "$PLAYBACK" \
      --source "reexport pre_limb triangulation" --prune
  fi
  "$PYTHON" "$JP/render_multiview_to_head.py" \
    --data-root "$DATA_ROOT" --config "$CONFIG" \
    --skeleton-playback "$PLAYBACK" \
    --output-dir "$HEAD_REPRO" --csv-only
fi

"$PYTHON" "$JP/export_0806_heatmap_labels.py" \
  --csv "$CSV" \
  --limb "$NAME" \
  --head-dir-name "$HEAD_DIR_NAME" \
  --frame-root /home/gaoweijian/0810dataset/frames \
  --output "$LABEL_ROOT/$NAME/heatmap_labels_120x75.npz"

echo "reexport_done_$NAME"
