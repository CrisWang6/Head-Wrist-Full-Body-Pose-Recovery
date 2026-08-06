#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:?Set PROJECT_ROOT to the external module01 B/C dataset directory}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/wrist_tag_full_ch03_01/chunks}"
mkdir -p "${OUTPUT_ROOT}"

run_chunk() {
    local offset="$1"
    local duration="$2"
    local output="${OUTPUT_ROOT}/chunk_${offset}"
    mkdir -p "$output"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/wrist_tag_pose_stereo.py" \
        --input-dir "${PROJECT_ROOT}/input" \
        --intrinsics "${REPO_ROOT}/configs/pose2d/module01_bc/head_intrinsics.json" \
        --camchain "${REPO_ROOT}/configs/pose2d/module01_bc/head_BC-camchain.yaml" \
        --output-dir "$output" \
        --duration "$duration" \
        --start-offset "$offset" \
        --tag-size 0.08 \
        --no-video >"$output/run.log" 2>&1
}
export -f run_chunk
export SCRIPT_DIR REPO_ROOT PROJECT_ROOT PYTHON_BIN OUTPUT_ROOT

printf '%s\n' \
    '0 60' '60 60' '120 60' '180 60' \
    '240 60' '300 60' '360 60' '420 42.51' |
    xargs -P 2 -n 2 bash -c 'run_chunk "$0" "$1"'
