#!/usr/bin/env bash
set -euo pipefail
base=/home/gaoweijian/0711_214559/realigned_offset0_rawsource_full6207
scripts=/home/gaoweijian/0711_214559/scripts
python=/home/gaoweijian/miniforge3/envs/sapiens2/bin/python
export LD_LIBRARY_PATH=/home/gaoweijian/miniforge3/envs/sapiens2/lib/python3.12/site-packages/nvidia/cu13/lib:/home/gaoweijian/miniforge3/envs/sapiens2/lib/python3.12/site-packages/nvidia/cudnn/lib
mkdir -p "$base/inference/parts"
rm -f "$base/inference/parts"/* "$base/inference/ext_A_rtmpose.jsonl" "$base/inference/ext_D_rtmpose.jsonl"
starts=(0 1552 3104 4656)
counts=(1552 1552 1552 1551)
pids=()
for side in A D; do
  gpu=0; [[ "$side" == D ]] && gpu=1
  for part in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$gpu "$python" "$scripts/infer_rtmpose_candidates.py" \
      --video "$base/external_CAM_${side}.mp4" \
      --output "$base/inference/parts/ext_${side}_part${part}.jsonl" \
      --mode performance --device cuda --backend onnxruntime --rotate-180 \
      --start-frame "${starts[$part]}" --max-frames "${counts[$part]}" \
      >"$base/inference/parts/ext_${side}_part${part}.log" 2>&1 &
    pids+=("$!")
  done
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
[[ "$failed" == 0 ]]
cat "$base/inference/parts"/ext_A_part{0,1,2,3}.jsonl > "$base/inference/ext_A_rtmpose.jsonl"
cat "$base/inference/parts"/ext_D_part{0,1,2,3}.jsonl > "$base/inference/ext_D_rtmpose.jsonl"
wc -l "$base/inference/ext_A_rtmpose.jsonl" "$base/inference/ext_D_rtmpose.jsonl"
