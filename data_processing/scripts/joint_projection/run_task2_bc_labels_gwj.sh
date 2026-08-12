#!/usr/bin/env bash
# Task2: force B+C for one 0806 dataset, export pre_limb playback + head 2D CSV (no video).
# Usage: bash run_task2_bc_labels_gwj.sh wu|wrist|ankle
set -euo pipefail

NAME="${1:?usage: $0 wu|wrist|ankle}"
BATCH=/home/gaoweijian/0806_batch
JP="$BATCH/repo/test_code/joint_projection"
PYTHON=/home/gaoweijian/miniforge3/envs/sapiens2/bin/python
LABEL_ROOT=/home/gaoweijian/0806dataset/labels
LOG_ROOT=/home/gaoweijian/0806dataset/logs/task2

export PYTHONPATH="$JP"
export LD_LIBRARY_PATH=/home/gaoweijian/miniforge3/envs/sapiens2/lib/python3.12/site-packages/nvidia/cu13/lib:/home/gaoweijian/miniforge3/envs/sapiens2/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

case "$NAME" in
  wu)
    CONFIG="$JP/configs/0806_dual_external_mocap.json"
    HEAD_DIR_NAME=0712_033709
    ;;
  wrist)
    CONFIG="$JP/configs/0806_wrist_dual_external_mocap.json"
    HEAD_DIR_NAME=0712_032704
    ;;
  ankle)
    CONFIG="$JP/configs/0806_ankle_dual_external_mocap.json"
    HEAD_DIR_NAME=0712_033034
    ;;
  *)
    echo "NAME must be wu|wrist|ankle" >&2
    exit 1
    ;;
esac

ROOT="$BATCH/$NAME"
DATA_ROOT="$ROOT/data_root"
MANIFEST="$DATA_ROOT/multiview_3d_results/aligned_manifest.jsonl"
INFER="$ROOT/inference"
FULL="$DATA_ROOT/multiview_3d_results/full"
CHUNKS="$FULL/chunks"
HEAD_REPRO="$FULL/head_reprojection"
LOGS="$LOG_ROOT/$NAME"
N_CHUNKS_TRI=8

mkdir -p "$INFER" "$CHUNKS" "$LOGS" "$HEAD_REPRO" "$LABEL_ROOT/$NAME" \
  "$ROOT/input/module01" "$ROOT/input/module02"

stamp() { date '+%F %T'; }
log() { echo "[$(stamp)] $*" | tee -a "$LOGS/pipeline.log"; }

log "Task2 force B+C start for $NAME"

# --- Force clear B outputs (toes) and C outputs ---
for f in module01_CAM_A module01_CAM_D module02_CAM_A module02_CAM_D; do
  if [[ -f "$INFER/${f}.jsonl" ]]; then
    mv -f "$INFER/${f}.jsonl" "$INFER/${f}.jsonl.task2.bak.$(date +%s)" || true
  fi
done
rm -rf "$CHUNKS"
mkdir -p "$CHUNKS"
rm -f "$FULL/multiview_3d_results.jsonl" \
  "$FULL/multiview_3d_results_pre_limb.jsonl" \
  "$FULL/skeleton_playback_raw.json" \
  "$FULL/skeleton_playback.json" \
  "$FULL/multiview_3d.csv" \
  "$FULL/multiview_3d_report.json"
rm -f "$HEAD_REPRO"/head_reprojection_2d_*.csv "$HEAD_REPRO/report.json"

# --- Stage B ---
log "Stage B RTMW WholeBody (force)"
cams=(
  "module01_CAM_A|$ROOT/input/module01/left_CAM_A_1920x1200_30fps.mjpeg|0"
  "module01_CAM_D|$ROOT/input/module01/right_CAM_D_1920x1200_30fps.mjpeg|0"
  "module02_CAM_A|$ROOT/input/module02/left_CAM_A_1920x1200_30fps.mjpeg|1"
  "module02_CAM_D|$ROOT/input/module02/right_CAM_D_1920x1200_30fps.mjpeg|1"
)
pids=()
for item in "${cams[@]}"; do
  IFS='|' read -r cname video gpu <<<"$item"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$JP/infer_rtmpose_candidates.py" \
    --video "$video" \
    --output "$INFER/${cname}.jsonl" \
    --mode performance --device cuda --backend onnxruntime --rotate-180 \
    >"$LOGS/${cname}.log" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
wc -l "$INFER"/module{01,02}_CAM_{A,D}.jsonl | tee -a "$LOGS/pipeline.log"

# --- Stage C ---
n=$("$PYTHON" - <<'PY' "$MANIFEST"
import json, sys
from pathlib import Path
print(sum(1 for line in Path(sys.argv[1]).open(encoding="utf-8") if line.strip()))
PY
)
log "Stage C triangulate frames=$n"
info=$("$PYTHON" - <<'PY' "$n" "$N_CHUNKS_TRI"
import math, sys
n=int(sys.argv[1]); k=int(sys.argv[2])
size=math.ceil(n/k)
starts,ends=[],[]
for i in range(k):
    s=i*size; e=min(n-1,(i+1)*size-1)
    if s>e: break
    starts.append(s); ends.append(e)
print(",".join(f"{s}:{e}" for s,e in zip(starts,ends)))
print(" ".join(map(str,starts)))
print(" ".join(map(str,ends)))
PY
)
cores=$(echo "$info" | sed -n '1p')
starts=($(echo "$info" | sed -n '2p'))
ends=($(echo "$info" | sed -n '3p'))
pids=()
for i in "${!starts[@]}"; do
  cs=${starts[$i]}; ce=${ends[$i]}
  ctx_s=$((cs>10?cs-10:0)); ctx_e=$((ce+10<n-1?ce+10:n-1))
  chunk=$(printf chunk_%02d "$i")
  "$PYTHON" "$JP/process_external_multiview_3d.py" \
    --manifest "$MANIFEST" --candidates-dir "$INFER" --config "$CONFIG" \
    --output-dir "$CHUNKS/$chunk" --start-seq "$ctx_s" --end-seq "$ctx_e" \
    >"$LOGS/$chunk.log" 2>&1 &
  pids+=($!)
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PYTHON" "$JP/merge_multiview_chunks.py" \
  --chunks-root "$CHUNKS" --cores "$cores" --output-dir "$FULL" \
  --manifest "$MANIFEST" --config "$CONFIG" --candidates-dir "$INFER" \
  --context-frames 10 >"$LOGS/merge.log" 2>&1
cp -f "$FULL/multiview_3d_results.jsonl" "$FULL/multiview_3d_results_pre_limb.jsonl"

# --- Export playback + head 2D CSV (no video) ---
log "export skeleton_playback_raw + head 2D CSV"
"$PYTHON" "$JP/export_playback_from_jsonl.py" \
  --results "$FULL/multiview_3d_results_pre_limb.jsonl" \
  --output "$FULL/skeleton_playback_raw.json" \
  --source "task2 pre_limb triangulation" --prune \
  >"$LOGS/export_playback.log" 2>&1

"$PYTHON" "$JP/render_multiview_to_head.py" \
  --data-root "$DATA_ROOT" --config "$CONFIG" \
  --skeleton-playback "$FULL/skeleton_playback_raw.json" \
  --output-dir "$HEAD_REPRO" --csv-only \
  >"$LOGS/head_reproj_csv.log" 2>&1

CSV="$HEAD_REPRO/head_reprojection_2d_wo_calibration_proj.csv"
"$PYTHON" "$JP/export_0806_heatmap_labels.py" \
  --csv "$CSV" --limb "$NAME" --head-dir-name "$HEAD_DIR_NAME" \
  --output "$LABEL_ROOT/$NAME/heatmap_labels_120x75.npz" \
  >"$LOGS/export_labels.log" 2>&1

log "Task2 DONE $NAME labels=$LABEL_ROOT/$NAME/heatmap_labels_120x75.npz"
echo "task2_done_$NAME" > "$LOGS/STATUS.txt"
