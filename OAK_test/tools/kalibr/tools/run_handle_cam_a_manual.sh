#!/usr/bin/env bash
set -eo pipefail

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
export ROS_HOSTNAME="${ROS_HOSTNAME:-localhost}"
source /catkin_ws/devel/setup.bash

OUT=/data/handle_intrinsics_mixed_results/CAM_A_manual
mkdir -p "$OUT"
cd "$OUT"

printf "640\n" | KALIBR_MANUAL_FOCAL_LENGTH_INIT=1 \
  /catkin_ws/devel/lib/kalibr/kalibr_calibrate_cameras \
  --target /data/handle_dataset/aprilgrid_handle.yaml \
  --bag /data/handle_intrinsics_results/bags/CAM_A.bag \
  --models pinhole-radtan \
  --topics /CAM_A/image_raw \
  --dont-show-report \
  > kalibr_CAM_A_manual.log 2>&1
echo "$?" > exit_code.txt

cp /data/handle_intrinsics_results/bags/CAM_A-camchain.yaml "$OUT/" 2>/dev/null || true
cp /data/handle_intrinsics_results/bags/CAM_A-results-cam.txt "$OUT/" 2>/dev/null || true
cp /data/handle_intrinsics_results/bags/CAM_A-report-cam.pdf "$OUT/" 2>/dev/null || true

tail -120 kalibr_CAM_A_manual.log || true
