#!/usr/bin/env bash
# Full A→E pipeline for one 0810 session (line1|line2) on gwj.
# Usage: bash run_0810_line_dataset.sh line1|line2
# Stages skip when aligned/detections/triangulation already valid.
set -euo pipefail

NAME="${1:?usage: $0 line1|line2}"
BATCH=/home/gaoweijian/0810_batch
REPO="$BATCH/repo"
JP="$REPO/test_code/joint_projection"
ROOT="$BATCH/$NAME"
PYTHON=/home/gaoweijian/miniforge3/envs/sapiens2/bin/python
export LD_LIBRARY_PATH=/home/gaoweijian/miniforge3/envs/sapiens2/lib/python3.12/site-packages/nvidia/cu13/lib:/home/gaoweijian/miniforge3/envs/sapiens2/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

case "$NAME" in
  line1)
    CONFIG="$JP/configs/0810_line1_dual_external_mocap.json"
    HEAD_DIR_NAME=0712_035226
    ;;
  line2)
    CONFIG="$JP/configs/0810_line2_dual_external_mocap.json"
    HEAD_DIR_NAME=0712_035903
    ;;
  *)
    echo "NAME must be line1 or line2" >&2
    exit 1
    ;;
esac

DATA_ROOT="$ROOT/data_root"
MANIFEST="$DATA_ROOT/multiview_3d_results/aligned_manifest.jsonl"
MANIFEST_REPORT="$DATA_ROOT/multiview_3d_results/aligned_manifest_report.json"
INFER="$ROOT/inference"
FULL="$DATA_ROOT/multiview_3d_results/full"
CHUNKS="$FULL/chunks"
LOGS="$ROOT/logs"
VIZ="$FULL/visualization"
HEAD_OUT="$FULL/head_reprojection"
NOSE_OUT="$HEAD_OUT/nose_offset_opt"
STATUS="$ROOT/STATUS.txt"
N_CHUNKS_TRI=8
N_CHUNKS_RENDER=24

mkdir -p "$INFER" "$CHUNKS" "$LOGS" "$VIZ" "$HEAD_OUT" "$NOSE_OUT/chunks" \
  "$DATA_ROOT/multiview_3d_results" "$ROOT/input/module01" "$ROOT/input/module02"

stamp() { date '+%F %T'; }
log() { echo "[$(stamp)] $*" | tee -a "$LOGS/pipeline.log"; }
set_status() { echo "$1" | tee "$STATUS"; }

n_frames() {
  "$PYTHON" - <<'PY' "$MANIFEST"
import json, sys
from pathlib import Path
n = sum(1 for line in Path(sys.argv[1]).open(encoding="utf-8") if line.strip())
print(n)
PY
}

make_cores() {
  local n=$1 nchunk=$2
  "$PYTHON" - <<'PY' "$n" "$nchunk"
import math, sys
n = int(sys.argv[1]); k = int(sys.argv[2])
size = math.ceil(n / k)
starts, ends = [], []
for i in range(k):
    s = i * size
    e = min(n - 1, (i + 1) * size - 1)
    if s > e:
        break
    starts.append(s); ends.append(e)
print(",".join(f"{s}:{e}" for s, e in zip(starts, ends)))
print(" ".join(map(str, starts)))
print(" ".join(map(str, ends)))
PY
}

############################################
# Stage B — RTMW WholeBody body+foot (4 cams, 2 GPUs)
############################################
inference_has_toes() {
  local path=$1
  [[ -s "$path" ]] || return 1
  "$PYTHON" - <<'PY' "$path"
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
with path.open(encoding="utf-8") as f:
    for _ in range(40):
        line = f.readline()
        if not line:
            break
        row = json.loads(line)
        for cand in row.get("candidates") or []:
            kps = cand.get("keypoints") or {}
            if "left_big_toe" in kps and "right_big_toe" in kps:
                raise SystemExit(0)
raise SystemExit(1)
PY
}

run_rtmpose() {
  set_status "B_rtmpose_running"
  log "Stage B RTMW WholeBody start for $NAME"
  local cams=(
    "module01_CAM_A|$ROOT/input/module01/left_CAM_A_1920x1200_30fps.mjpeg|0"
    "module01_CAM_D|$ROOT/input/module01/right_CAM_D_1920x1200_30fps.mjpeg|0"
    "module02_CAM_A|$ROOT/input/module02/left_CAM_A_1920x1200_30fps.mjpeg|1"
    "module02_CAM_D|$ROOT/input/module02/right_CAM_D_1920x1200_30fps.mjpeg|1"
  )
  local pids=()
  for item in "${cams[@]}"; do
    IFS='|' read -r cname video gpu <<<"$item"
    local out="$INFER/${cname}.jsonl"
    if [[ -s "$out" ]]; then
      local lines
      lines=$(wc -l <"$out")
      if [[ "$lines" -gt 100 ]] && inference_has_toes "$out"; then
        log "skip RTMW $cname (already $lines lines with toes)"
        continue
      fi
      if [[ "$lines" -gt 100 ]]; then
        log "body-only inference detected for $cname; backing up and re-inferring with toes"
        mv -f "$out" "${out}.body_only.bak"
        rm -rf "$CHUNKS"
        mkdir -p "$CHUNKS"
        rm -f "$FULL/multiview_3d_results.jsonl" \
          "$FULL/multiview_3d_results_limb_gt.jsonl" \
          "$FULL/multiview_3d_results_pre_limb.jsonl" \
          "$FULL/skeleton_playback.json" \
          "$FULL/skeleton_playback_raw.json" \
          "$FULL/multiview_3d.csv" \
          "$FULL/multiview_3d_report.json"
        rm -rf "$NOSE_OUT/chunks" "$VIZ"
        mkdir -p "$NOSE_OUT/chunks" "$VIZ"
        rm -f "$NOSE_OUT"/head_*.mp4 "$NOSE_OUT"/render_cache.json \
          "$NOSE_OUT"/report.json 2>/dev/null || true
      fi
    fi
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" "$JP/infer_rtmpose_candidates.py" \
      --video "$video" \
      --output "$out" \
      --mode performance \
      --device cuda \
      --backend onnxruntime \
      --rotate-180 \
      >"$LOGS/${cname}.log" 2>&1 &
    local pid=$!
    pids+=("$pid")
    log "started RTMW $cname gpu=$gpu pid=$pid"
  done
  local failed=0
  if [[ ${#pids[@]} -gt 0 ]]; then
    for pid in "${pids[@]}"; do
      wait "$pid" || failed=1
    done
  fi
  if [[ "$failed" != 0 ]]; then
    set_status "B_rtmpose_failed"
    exit 1
  fi
  for cname in module01_CAM_A module01_CAM_D module02_CAM_A module02_CAM_D; do
    if ! inference_has_toes "$INFER/${cname}.jsonl"; then
      log "ERROR: $cname missing toe keypoints after inference"
      set_status "B_rtmpose_missing_toes"
      exit 1
    fi
  done
  wc -l "$INFER"/module{01,02}_CAM_{A,D}.jsonl | tee -a "$LOGS/pipeline.log"
  set_status "B_rtmpose_done"
}

############################################
# Stage C — Multiview triangulation
############################################
run_triangulate() {
  set_status "C_triangulate_running"
  local n
  n=$(n_frames)
  log "Stage C triangulate $NAME frames=$n"
  # Skip if pre_limb or raw results already present with matching line count.
  if [[ -s "$FULL/multiview_3d_results_pre_limb.jsonl" ]]; then
    local lines
    lines=$(wc -l <"$FULL/multiview_3d_results_pre_limb.jsonl")
    if [[ "$lines" -eq "$n" ]]; then
      log "skip triangulate (pre_limb exists lines=$lines)"
      set_status "C_triangulate_skipped"
      return 0
    fi
  fi
  local info starts ends cores
  info=$(make_cores "$n" "$N_CHUNKS_TRI")
  cores=$(echo "$info" | sed -n '1p')
  starts=($(echo "$info" | sed -n '2p'))
  ends=($(echo "$info" | sed -n '3p'))
  log "cores=$cores"
  local pids=()
  local i
  for i in "${!starts[@]}"; do
    local core_start=${starts[$i]}
    local core_end=${ends[$i]}
    local context_start=$((core_start > 10 ? core_start - 10 : 0))
    local context_end=$((core_end + 10 < n - 1 ? core_end + 10 : n - 1))
    local chunk
    chunk=$(printf "chunk_%02d" "$i")
    if [[ -s "$CHUNKS/$chunk/multiview_3d_results.jsonl" ]]; then
      log "skip $chunk (exists)"
      continue
    fi
    "$PYTHON" "$JP/process_external_multiview_3d.py" \
      --manifest "$MANIFEST" \
      --candidates-dir "$INFER" \
      --config "$CONFIG" \
      --output-dir "$CHUNKS/$chunk" \
      --start-seq "$context_start" \
      --end-seq "$context_end" \
      >"$LOGS/$chunk.log" 2>&1 &
    pids+=("$!")
  done
  local failed=0
  if [[ ${#pids[@]} -gt 0 ]]; then
    for pid in "${pids[@]}"; do
      wait "$pid" || failed=1
    done
  fi
  if [[ "$failed" != 0 ]]; then
    set_status "C_triangulate_failed"
    exit 1
  fi
  # All chunks may already exist; still merge if pre_limb not yet written.
  if [[ -s "$FULL/multiview_3d_results_pre_limb.jsonl" ]]; then
    local plines nlines
    plines=$(wc -l <"$FULL/multiview_3d_results_pre_limb.jsonl")
    nlines=$n
    if [[ "$plines" -eq "$nlines" ]]; then
      log "skip merge (pre_limb already lines=$plines)"
      set_status "C_triangulate_skipped"
      return 0
    fi
  fi
  "$PYTHON" "$JP/merge_multiview_chunks.py" \
    --chunks-root "$CHUNKS" \
    --cores "$cores" \
    --output-dir "$FULL" \
    --manifest "$MANIFEST" \
    --config "$CONFIG" \
    --candidates-dir "$INFER" \
    --context-frames 10 \
    >"$LOGS/merge.log" 2>&1
  cp -f "$FULL/multiview_3d_results.jsonl" "$FULL/multiview_3d_results_pre_limb.jsonl"
  set_status "C_triangulate_done"
  log "triangulate done lines=$(wc -l <"$FULL/multiview_3d_results_pre_limb.jsonl")"
}

ensure_pre_limb() {
  if [[ -s "$FULL/multiview_3d_results_pre_limb.jsonl" ]]; then
    return 0
  fi
  # Rebuild raw merge from existing chunks if limb-GT overwrote the main jsonl.
  local n chunk_ok=0
  n=$(n_frames)
  local i
  for i in $(seq 0 $((N_CHUNKS_TRI - 1))); do
    local chunk
    chunk=$(printf "chunk_%02d" "$i")
    [[ -s "$CHUNKS/$chunk/multiview_3d_results.jsonl" ]] && chunk_ok=$((chunk_ok + 1))
  done
  if [[ "$chunk_ok" -lt "$N_CHUNKS_TRI" ]]; then
    log "pre_limb missing and chunks incomplete ($chunk_ok/$N_CHUNKS_TRI); re-running triangulate"
    run_triangulate
    return 0
  fi
  log "rebuilding pre_limb from $chunk_ok chunks"
  local info cores
  info=$(make_cores "$n" "$N_CHUNKS_TRI")
  cores=$(echo "$info" | sed -n '1p')
  "$PYTHON" "$JP/merge_multiview_chunks.py" \
    --chunks-root "$CHUNKS" \
    --cores "$cores" \
    --output-dir "$FULL" \
    --manifest "$MANIFEST" \
    --config "$CONFIG" \
    --candidates-dir "$INFER" \
    --context-frames 10 \
    >"$LOGS/merge_rebuild_pre_limb.log" 2>&1
  cp -f "$FULL/multiview_3d_results.jsonl" "$FULL/multiview_3d_results_pre_limb.jsonl"
}

############################################
# Stage D — Head RTMW nose detect (.h265 + bytes)
############################################
run_nose_detect() {
  set_status "D_nose_detect_running"
  local head_dir="$DATA_ROOT/$HEAD_DIR_NAME"
  local ts="$head_dir/timestamps.csv"
  local va="$head_dir/module01_D45D2E00_CAM_A.h265"
  local vd="$head_dir/module01_D45D2E00_CAM_D.h265"
  local out_a="$NOSE_OUT/head_CAM_A_rtmw_nose.csv"
  local out_d="$NOSE_OUT/head_CAM_D_rtmw_nose.csv"
  local meta_a="${out_a%.csv}.fixed.json"
  local meta_d="${out_d%.csv}.fixed.json"
  if [[ -s "$out_a" && -s "$out_d" && -s "$meta_a" && -s "$meta_d" ]]; then
    log "skip Stage D nose detect (fixed-nose csv + meta exist)"
    set_status "D_nose_detect_skipped"
    return 0
  fi
  log "Stage D: sample RTMW nose -> fixed UV (production default --sample-count 48)"
  local pid_a="" pid_d=""
  if [[ ! -s "$out_a" || ! -s "$meta_a" ]]; then
    rm -f "$out_a" "$meta_a"
    CUDA_VISIBLE_DEVICES=0 "$PYTHON" "$JP/detect_head_nose_rtmw.py" \
      --video "$va" --timestamps "$ts" --camera CAM_A --output-csv "$out_a" \
      --sample-count 48 \
      >"$LOGS/nose_a.log" 2>&1 &
    pid_a=$!
  fi
  if [[ ! -s "$out_d" || ! -s "$meta_d" ]]; then
    rm -f "$out_d" "$meta_d"
    CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$JP/detect_head_nose_rtmw.py" \
      --video "$vd" --timestamps "$ts" --camera CAM_D --output-csv "$out_d" \
      --sample-count 48 \
      >"$LOGS/nose_d.log" 2>&1 &
    pid_d=$!
  fi
  local failed=0
  [[ -n "$pid_a" ]] && { wait "$pid_a" || failed=1; }
  [[ -n "$pid_d" ]] && { wait "$pid_d" || failed=1; }
  if [[ "$failed" != 0 ]]; then
    set_status "D_nose_detect_failed"
    exit 1
  fi
  set_status "D_nose_detect_done"
  wc -l "$out_a" "$out_d" | tee -a "$LOGS/pipeline.log"
}

############################################
# Viz helpers (required MP4s)
############################################
run_viz_raw() {
  set_status "E_viz_raw_running"
  mkdir -p "$VIZ"
  log "export raw triangulation playback + 4-cam + 3D yaw"
  "$PYTHON" "$JP/export_playback_from_jsonl.py" \
    --results "$FULL/multiview_3d_results_pre_limb.jsonl" \
    --output "$FULL/skeleton_playback_raw.json" \
    --source "raw triangulated methods.filtered.multiview (pre limb-GT)" \
    --prune \
    >"$LOGS/export_raw_playback.log" 2>&1

  "$PYTHON" "$JP/render_skeleton_yaw_video.py" \
    --data "$FULL/skeleton_playback_raw.json" \
    --output "$VIZ/skeleton_3d_raw_yaw.mp4" \
    --yaw-deg 100 --pitch-deg 18 \
    >"$LOGS/viz_raw_yaw.log" 2>&1 &
  local pid_yaw=$!

  if [[ -s "$MANIFEST_REPORT" ]]; then
    "$PYTHON" "$JP/render_external_multiview_results.py" \
      --results "$FULL/multiview_3d_results_pre_limb.jsonl" \
      --manifest-report "$MANIFEST_REPORT" \
      --manifest "$MANIFEST" \
      --video-root "$ROOT/input" \
      --config "$CONFIG" \
      --output-dir "$VIZ" \
      >"$LOGS/viz_four_view.log" 2>&1 &
    local pid_4=$!
  else
    log "WARN: missing $MANIFEST_REPORT; skip four_view_reprojection"
    local pid_4=""
  fi

  wait "$pid_yaw" || { set_status "E_viz_raw_yaw_failed"; exit 1; }
  if [[ -n "$pid_4" ]]; then
    wait "$pid_4" || { set_status "E_viz_four_view_failed"; exit 1; }
  fi
  # Canonical name for external 4-cam 2D skeletons (triangulation observations).
  if [[ -s "$VIZ/four_view_reprojection.mp4" ]]; then
    cp -f "$VIZ/four_view_reprojection.mp4" "$VIZ/external_4cam_2d_skeletons.mp4"
  fi
  ls -lh "$VIZ"/*.mp4 2>/dev/null | tee -a "$LOGS/pipeline.log" || true
  set_status "E_viz_raw_done"
}

############################################
# Stage E — limb GT + joint opt + head render
############################################
run_finalize_triangulation() {
  set_status "E_finalize_triangulation_running"
  ensure_pre_limb
  log "Stage E finalize: use external triangulation only (no wrist/ankle mocap replace)"
  cp -f "$FULL/multiview_3d_results_pre_limb.jsonl" "$FULL/multiview_3d_results.jsonl"
  "$PYTHON" "$JP/export_playback_from_jsonl.py" \
    --results "$FULL/multiview_3d_results_pre_limb.jsonl" \
    --output "$FULL/skeleton_playback.json" \
    --source "external triangulation (0810 no limb mocap GT)" \
    --prune \
    >"$LOGS/finalize_triangulation.log" 2>&1
  "$PYTHON" "$JP/render_skeleton_yaw_video.py" \
    --data "$FULL/skeleton_playback.json" \
    --output "$VIZ/skeleton_3d_triangulated_yaw.mp4" \
    --yaw-deg 100 --pitch-deg 18 \
    >"$LOGS/viz_triangulated_yaw.log" 2>&1 || true
  set_status "E_finalize_triangulation_done"
  tail -n 20 "$LOGS/finalize_triangulation.log" | tee -a "$LOGS/pipeline.log"
}

run_optimize() {
  set_status "E_optimize_running"
  # Clear stale head render chunks to free disk before rebuild.
  rm -rf "$NOSE_OUT/chunks"
  mkdir -p "$NOSE_OUT/chunks"
  rm -f "$NOSE_OUT"/head_*.mp4 "$NOSE_OUT"/render_cache.json 2>/dev/null || true

  "$PYTHON" "$JP/optimize_multiview_head_nose_offset.py" \
    --data-root "$DATA_ROOT" \
    --config "$CONFIG" \
    --skeleton-playback "$FULL/skeleton_playback.json" \
    --head-a-nose-csv "$NOSE_OUT/head_CAM_A_rtmw_nose.csv" \
    --head-d-nose-csv "$NOSE_OUT/head_CAM_D_rtmw_nose.csv" \
    --output-dir "$NOSE_OUT" \
    --mode per_frame \
    --skip-render \
    >"$LOGS/optimize.log" 2>&1
  "$PYTHON" "$JP/render_nose_offset_parallel.py" \
    --prepare \
    --data-root "$DATA_ROOT" \
    --config "$CONFIG" \
    --output-dir "$NOSE_OUT" \
    --head-a-nose-csv "$NOSE_OUT/head_CAM_A_rtmw_nose.csv" \
    --head-d-nose-csv "$NOSE_OUT/head_CAM_D_rtmw_nose.csv" \
    --before-playback "$FULL/skeleton_playback_raw.json" \
    --after-playback "$FULL/skeleton_playback.json" \
    --report "$NOSE_OUT/report.json" \
    >"$LOGS/prepare_render.log" 2>&1
  set_status "E_optimize_done"
  tail -n 30 "$LOGS/optimize.log" | tee -a "$LOGS/pipeline.log"
}

run_parallel_render() {
  set_status "E_render_running"
  local i
  local pids=()
  for i in $(seq 0 $((N_CHUNKS_RENDER - 1))); do
    if [[ -s "$NOSE_OUT/chunks/chunk_$(printf '%02d' "$i")/meta.json" ]]; then
      continue
    fi
    (
      cd "$JP"
      "$PYTHON" render_nose_offset_parallel.py \
        --data-root "$DATA_ROOT" \
        --config "$CONFIG" \
        --output-dir "$NOSE_OUT" \
        --head-a-nose-csv "$NOSE_OUT/head_CAM_A_rtmw_nose.csv" \
        --head-d-nose-csv "$NOSE_OUT/head_CAM_D_rtmw_nose.csv" \
        --chunk "$i" "$N_CHUNKS_RENDER" \
        >"$NOSE_OUT/chunks/chunk_${i}.log" 2>&1
    ) &
    pids+=("$!")
    if (( ${#pids[@]} % 8 == 0 )); then
      sleep 1
    fi
  done
  local failed=0
  if [[ ${#pids[@]} -gt 0 ]]; then
    for pid in "${pids[@]}"; do
      wait "$pid" || failed=1
    done
  fi
  local done_n
  done_n=$(ls "$NOSE_OUT"/chunks/chunk_*/meta.json 2>/dev/null | wc -l)
  if [[ "$done_n" -lt "$N_CHUNKS_RENDER" ]]; then
    log "render incomplete done=$done_n/$N_CHUNKS_RENDER failed=$failed"
    set_status "E_render_failed"
    exit 1
  fi
  (
    cd "$JP"
    "$PYTHON" render_nose_offset_parallel.py \
      --data-root "$DATA_ROOT" \
      --config "$CONFIG" \
      --output-dir "$NOSE_OUT" \
      --head-a-nose-csv "$NOSE_OUT/head_CAM_A_rtmw_nose.csv" \
      --head-d-nose-csv "$NOSE_OUT/head_CAM_D_rtmw_nose.csv" \
      --merge
  ) >"$LOGS/merge_render.log" 2>&1
  set_status "E_render_done"
  ls -lh "$NOSE_OUT"/*.mp4 | tee -a "$LOGS/pipeline.log"
}

############################################
# Main A→E
############################################
log "===== pipeline start $NAME A→E (0810 external triangulation only, no limb mocap GT) ====="
set_status "starting"
[[ -s "$MANIFEST" ]] || { echo "missing manifest $MANIFEST" >&2; set_status "missing_manifest"; exit 1; }
[[ -s "$ROOT/input/module01/left_CAM_A_1920x1200_30fps.mjpeg" ]] || { echo "missing videos" >&2; set_status "missing_videos"; exit 1; }
[[ -s "$DATA_ROOT/$HEAD_DIR_NAME/module01_D45D2E00_CAM_A.h265" ]] || { echo "missing head h265" >&2; set_status "missing_head"; exit 1; }
if [[ -s "$DATA_ROOT/aligned_data/aligned_30hz_strict.csv" ]]; then
  cp -f "$DATA_ROOT/aligned_data/aligned_30hz_strict.csv" "$DATA_ROOT/aligned_data/aligned_30hz.csv"
  log "using aligned_30hz_strict.csv (exact timestamp rows only)"
fi
[[ -s "$DATA_ROOT/aligned_data/aligned_30hz.csv" ]] || { echo "missing aligned" >&2; set_status "A_missing_aligned"; exit 1; }
log "Stage A aligned+manifest present → skip"

run_rtmpose
run_triangulate
ensure_pre_limb
run_nose_detect
run_viz_raw
run_finalize_triangulation
run_optimize
run_parallel_render
set_status "done"
log "===== pipeline DONE $NAME ====="
log "Required videos:"
log "  $VIZ/external_4cam_2d_skeletons.mp4"
log "  $VIZ/skeleton_3d_raw_yaw.mp4"
log "  $NOSE_OUT/head_CAM_A_direct_noseonly.mp4"
log "  $NOSE_OUT/head_CAM_D_direct_noseonly.mp4"
log "  $NOSE_OUT/head_2x2_direct_vs_nose_offset_opt.mp4"
