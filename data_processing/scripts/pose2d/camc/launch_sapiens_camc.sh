#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the external pose2d dataset directory}"
SAPIENS_REPO="${SAPIENS_REPO:?Set SAPIENS_REPO to the upstream sapiens2 checkout}"
SAPIENS_PYTHON="${SAPIENS_PYTHON:-python}"
SAPIENS_CHECKPOINT="${SAPIENS_CHECKPOINT:?Set SAPIENS_CHECKPOINT to a pose checkpoint}"
WORKER="${SCRIPT_DIR}/sapiens_camc_worker.py"
VIDEO="${PROJECT_ROOT}/input/module01_D45D2E00_CAM_C.h265"
CONFIG="${SAPIENS_REPO}/sapiens/pose/configs/keypoints308/shutterstock_goliath_3po/sapiens2_0.4b_keypoints308_shutterstock_goliath_3po-1024x768.py"
OUTPUT="${PROJECT_ROOT}/output/sapiens_parts"

mkdir -p "${OUTPUT}"
rm -f "${OUTPUT}"/part_*.jsonl "${OUTPUT}"/part_*.jsonl.summary.json "${OUTPUT}"/part_*.log

cd "${SAPIENS_REPO}/sapiens/pose"
export PYTHONPATH="${SAPIENS_REPO}"

launch() {
  local gpu="$1"
  local part="$2"
  local start="$3"
  local end="$4"
  CUDA_VISIBLE_DEVICES="${gpu}" "${SAPIENS_PYTHON}" "${WORKER}" \
    --video "${VIDEO}" \
    --config "${CONFIG}" \
    --checkpoint "${SAPIENS_CHECKPOINT}" \
    --start "${start}" \
    --end "${end}" \
    --batch-size 1 \
    --output "${OUTPUT}/part_${part}.jsonl" \
    >"${OUTPUT}/part_${part}.log" 2>&1 &
  LAST_PID="$!"
}

pids=()
launch 0 0 0 1292; pids+=("${LAST_PID}")
launch 0 1 1293 2585; pids+=("${LAST_PID}")
launch 1 2 2586 3878; pids+=("${LAST_PID}")
launch 1 3 3879 5171; pids+=("${LAST_PID}")
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
