#!/usr/bin/env bash
set -eo pipefail

export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
export ROS_HOSTNAME="${ROS_HOSTNAME:-localhost}"

source /catkin_ws/devel/setup.bash

DATASET=/data/head_dataset
OUT=/data/head_intrinsics_results
mkdir -p "$OUT"

cat > "$DATASET/aprilgrid_head.yaml" <<'YAML'
target_type: 'aprilgrid'
tagCols: 6
tagRows: 6
tagSize: 0.0352
tagSpacing: 0.3
YAML

python3 /data/tools/make_cam_bags.py --dataset "$DATASET" --out "$OUT/bags" --fps 10

for cam in CAM_A CAM_B CAM_C CAM_D; do
  mkdir -p "$OUT/$cam"
  cd "$OUT/$cam"
  echo "===== ${cam} pinhole-equi ====="
  set +e
  /catkin_ws/devel/lib/kalibr/kalibr_calibrate_cameras \
    --target "$DATASET/aprilgrid_head.yaml" \
    --bag "$OUT/bags/${cam}.bag" \
    --models pinhole-equi \
    --topics "/${cam}/image_raw" \
    > "kalibr_${cam}.log" 2>&1
  status=$?
  set -e
  echo "$status" > "exit_code.txt"
  tail -80 "kalibr_${cam}.log" || true
done
