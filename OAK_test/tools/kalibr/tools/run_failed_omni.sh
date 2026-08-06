#!/usr/bin/env bash
set -eo pipefail

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
export ROS_HOSTNAME="${ROS_HOSTNAME:-localhost}"
source /catkin_ws/devel/setup.bash

DATASET=/data/head_dataset
OUT=/data/head_intrinsics_results

for cam in CAM_B CAM_C CAM_D; do
  rm -rf "$OUT/${cam}_omni"
  mkdir -p "$OUT/${cam}_omni"
  cd "$OUT/${cam}_omni"
  echo "===== ${cam} omni-radtan ====="
  set +e
  /catkin_ws/devel/lib/kalibr/kalibr_calibrate_cameras \
    --target "$DATASET/aprilgrid_head.yaml" \
    --bag "$OUT/bags/${cam}.bag" \
    --models omni-radtan \
    --topics "/${cam}/image_raw" \
    --dont-show-report \
    > "kalibr_${cam}_omni.log" 2>&1
  status=$?
  set -e
  echo "$status" > exit_code.txt
  tail -90 "kalibr_${cam}_omni.log" || true
done
