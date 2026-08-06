#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/kalibr_data/wrist_extrinsics_1920_20260717"
CUSTOM="$HOME/kalibr_custom_head_20260717"
IMAGE="kalibr:ros1_20_04"

for side in left_wrist right_wrist; do
  docker run --rm \
    -v "$ROOT:/data" -v "$CUSTOM:/custom" --entrypoint bash "$IMAGE" -lc \
    "source /catkin_ws/devel/setup.bash && python3 /custom/build_pair_bag.py --images-root /data/$side/images --cameras CAM_A CAM_C --start 1 --end 20 --output /data/$side/${side}_AC.bag"

  docker run --rm \
    -v "$ROOT:/data" -v "$CUSTOM:/custom" --entrypoint bash "$IMAGE" -lc \
    "source /catkin_ws/devel/setup.bash && python3 /custom/build_pair_bag.py --images-root /data/$side/images --cameras CAM_A CAM_B --start 21 --end 40 --output /data/$side/${side}_AB.bag"

  docker run --rm \
    -v "$ROOT:/data" -v "$CUSTOM:/custom" --entrypoint bash "$IMAGE" -lc \
    "source /catkin_ws/devel/setup.bash && cd /data/$side && python3 /custom/kalibr_calibrate_cameras --models omni-radtan omni-radtan --topics /CAM_A/image_raw /CAM_C/image_raw --bag /data/$side/${side}_AC.bag --target /data/aprilgrid_head.yaml --fixed-intrinsics /data/$side/CAM_A-camchain.yaml /data/$side/CAM_C-camchain.yaml --approx-sync 0.01 --mi-tol -1 --no-shuffle --dont-show-report" \
    2>&1 | tee "$ROOT/$side/kalibr_${side}_AC.log"

  docker run --rm \
    -v "$ROOT:/data" -v "$CUSTOM:/custom" --entrypoint bash "$IMAGE" -lc \
    "source /catkin_ws/devel/setup.bash && cd /data/$side && python3 /custom/kalibr_calibrate_cameras --models omni-radtan omni-radtan --topics /CAM_A/image_raw /CAM_B/image_raw --bag /data/$side/${side}_AB.bag --target /data/aprilgrid_head.yaml --fixed-intrinsics /data/$side/CAM_A-camchain.yaml /data/$side/CAM_B-camchain.yaml --approx-sync 0.01 --mi-tol -1 --no-shuffle --dont-show-report" \
    2>&1 | tee "$ROOT/$side/kalibr_${side}_AB.log"
done
