#!/usr/bin/env bash
set -eo pipefail

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
export ROS_HOSTNAME="${ROS_HOSTNAME:-localhost}"
source /catkin_ws/devel/setup.bash

DATASET=/data/handle_dataset
OUT=/data/handle_intrinsics_mixed_results
BAGS=/data/handle_intrinsics_results/bags

mkdir -p "$OUT"

cat > "$DATASET/aprilgrid_handle.yaml" <<'YAML'
target_type: 'aprilgrid'
tagCols: 6
tagRows: 6
tagSize: 0.0352
tagSpacing: 0.3
YAML

run_one() {
  local cam="$1"
  local model="$2"
  rm -rf "$OUT/$cam"
  mkdir -p "$OUT/$cam"
  cd "$OUT/$cam"
  echo "===== ${cam} ${model} ====="
  if [ ! -s "$BAGS/${cam}.bag" ]; then
    echo "Missing or empty bag for ${cam}" | tee "kalibr_${cam}.log"
    echo 99 > exit_code.txt
    return
  fi
  set +e
  /catkin_ws/devel/lib/kalibr/kalibr_calibrate_cameras \
    --target "$DATASET/aprilgrid_handle.yaml" \
    --bag "$BAGS/${cam}.bag" \
    --models "$model" \
    --topics "/${cam}/image_raw" \
    --dont-show-report \
    > "kalibr_${cam}.log" 2>&1
  status=$?
  set -e
  echo "$status" > exit_code.txt
  tail -80 "kalibr_${cam}.log" || true

  for artifact in "$BAGS/${cam}-camchain.yaml" "$BAGS/${cam}-results-cam.txt" "$BAGS/${cam}-report-cam.pdf"; do
    if [ -e "$artifact" ]; then
      cp "$artifact" "$OUT/$cam/"
    fi
  done
}

run_one CAM_A pinhole-radtan
run_one CAM_B omni-radtan
run_one CAM_C omni-radtan
run_one CAM_D omni-radtan
