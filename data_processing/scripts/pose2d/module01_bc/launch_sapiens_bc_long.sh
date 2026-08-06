#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the external module01 B/C dataset directory}"
SAPIENS_REPO="${SAPIENS_REPO:?Set SAPIENS_REPO to the upstream sapiens2 checkout}"
SAPIENS_PYTHON="${SAPIENS_PYTHON:-python}"
SAPIENS_CHECKPOINT="${SAPIENS_CHECKPOINT:?Set SAPIENS_CHECKPOINT to a pose checkpoint}"
WORKER="${SCRIPT_DIR}/../camc/sapiens_camc_worker.py"
CONFIG="${SAPIENS_REPO}/sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py"
OUTPUT="${PROJECT_ROOT}/output/sapiens_parts"

mkdir -p "${OUTPUT}"
rm -f "${OUTPUT}"/cam_*.jsonl "${OUTPUT}"/cam_*.jsonl.summary.json "${OUTPUT}"/cam_*.log
cd "${SAPIENS_REPO}/sapiens/pose"
export PYTHONPATH="${SAPIENS_REPO}"

launch() {
  local gpu="$1"
  local camera="$2"
  local part="$3"
  local start="$4"
  local end="$5"
  CUDA_VISIBLE_DEVICES="${gpu}" "${SAPIENS_PYTHON}" "${WORKER}" \
    --video "${PROJECT_ROOT}/input/module01_D45D2E00_CAM_${camera}.h265" \
    --config "${CONFIG}" \
    --checkpoint "${SAPIENS_CHECKPOINT}" \
    --start "${start}" \
    --end "${end}" \
    --batch-size 1 \
    --output "${OUTPUT}/cam_${camera,,}_part_${part}.jsonl" \
    >"${OUTPUT}/cam_${camera,,}_part_${part}.log" 2>&1 &
  LAST_PID="$!"
}

pids=()
launch 0 B 0 0 6642; pids+=("${LAST_PID}")
launch 0 B 1 6643 13285; pids+=("${LAST_PID}")
launch 1 C 0 0 6636; pids+=("${LAST_PID}")
launch 1 C 1 6637 13274; pids+=("${LAST_PID}")
printf "%s\n" "${pids[@]}" >"${OUTPUT}/pids.txt"
echo "Started workers: ${pids[*]}"

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
echo "${failed}" >"${OUTPUT}/exit_status.txt"
exit "${failed}"
