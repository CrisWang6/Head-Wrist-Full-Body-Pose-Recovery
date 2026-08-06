#!/usr/bin/env bash
set -eo pipefail

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
export ROS_HOSTNAME="${ROS_HOSTNAME:-localhost}"
source /catkin_ws/devel/setup.bash

DATASET=/data/handle_dataset
OUT=/data/handle_intrinsics_omni_results
BAGS=/data/handle_intrinsics_results/bags

mkdir -p "$OUT"

cat > "$DATASET/aprilgrid_handle.yaml" <<'YAML'
target_type: 'aprilgrid'
tagCols: 6
tagRows: 6
tagSize: 0.0352
tagSpacing: 0.3
YAML

for cam in CAM_A CAM_B CAM_C CAM_D; do
  rm -rf "$OUT/$cam"
  mkdir -p "$OUT/$cam"
  cd "$OUT/$cam"
  echo "===== ${cam} omni-radtan ====="
  if [ ! -s "$BAGS/${cam}.bag" ]; then
    echo "Missing or empty bag for ${cam}" | tee "kalibr_${cam}_omni.log"
    echo 99 > exit_code.txt
    continue
  fi
  set +e
  /catkin_ws/devel/lib/kalibr/kalibr_calibrate_cameras \
    --target "$DATASET/aprilgrid_handle.yaml" \
    --bag "$BAGS/${cam}.bag" \
    --models omni-radtan \
    --topics "/${cam}/image_raw" \
    --dont-show-report \
    > "kalibr_${cam}_omni.log" 2>&1
  status=$?
  set -e
  echo "$status" > exit_code.txt
  tail -80 "kalibr_${cam}_omni.log" || true

  for artifact in "$BAGS/${cam}-camchain.yaml" "$BAGS/${cam}-results-cam.txt" "$BAGS/${cam}-report-cam.pdf"; do
    if [ -e "$artifact" ]; then
      cp "$artifact" "$OUT/$cam/"
    fi
  done
done
