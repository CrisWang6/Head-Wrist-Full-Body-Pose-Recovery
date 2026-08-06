#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/kalibr_data/head_extrinsics_1920_20260717"
CUSTOM="$HOME/kalibr_custom_head_20260717"
IMAGE="kalibr:ros1_20_04"

docker run --rm \
  -v "$ROOT:/data" \
  -v "$CUSTOM:/custom" \
  --entrypoint bash "$IMAGE" -lc \
  'source /catkin_ws/devel/setup.bash && python3 /custom/build_pair_bag.py --images-root /data/images --cameras CAM_B CAM_C --start 1 --end 20 --output /data/head_BC.bag'

docker run --rm \
  -v "$ROOT:/data" \
  -v "$CUSTOM:/custom" \
  --entrypoint bash "$IMAGE" -lc \
  'source /catkin_ws/devel/setup.bash && python3 /custom/build_pair_bag.py --images-root /data/images --cameras CAM_A CAM_C --start 21 --end 40 --output /data/head_AC.bag'

docker run --rm \
  -v "$ROOT:/data" \
  -v "$CUSTOM:/custom" \
  --entrypoint bash "$IMAGE" -lc \
  'source /catkin_ws/devel/setup.bash && cd /data && python3 /custom/kalibr_calibrate_cameras --models omni-radtan omni-radtan --topics /CAM_B/image_raw /CAM_C/image_raw --bag /data/head_BC.bag --target /data/aprilgrid_head.yaml --fixed-intrinsics /data/CAM_B-camchain.yaml /data/CAM_C-camchain.yaml --approx-sync 0.01 --mi-tol -1 --no-shuffle --dont-show-report' \
  2>&1 | tee "$ROOT/kalibr_head_BC.log"

docker run --rm \
  -v "$ROOT:/data" \
  -v "$CUSTOM:/custom" \
  --entrypoint bash "$IMAGE" -lc \
  'source /catkin_ws/devel/setup.bash && cd /data && python3 /custom/kalibr_calibrate_cameras --models omni-radtan omni-radtan --topics /CAM_A/image_raw /CAM_C/image_raw --bag /data/head_AC.bag --target /data/aprilgrid_head.yaml --fixed-intrinsics /data/CAM_A-camchain.yaml /data/CAM_C-camchain.yaml --approx-sync 0.01 --mi-tol -1 --no-shuffle --dont-show-report' \
  2>&1 | tee "$ROOT/kalibr_head_AC.log"
