#!/usr/bin/env bash
set -euo pipefail
DATA_DIR="${1:-$HOME/kalibr_data}"
shift || true
mkdir -p "$DATA_DIR"
DISPLAY_VALUE="${DISPLAY:-}"
XAUTH="${XAUTHORITY:-$HOME/.Xauthority}"
DOCKER=(docker)
if ! docker ps >/dev/null 2>&1; then
  DOCKER=(sudo docker)
fi
DOCKER_ARGS=(--rm -it -e QT_X11_NO_MITSHM=1 -v "${DATA_DIR}:/data")
if [[ -n "$DISPLAY_VALUE" ]]; then
  xhost +local:root >/dev/null 2>&1 || true
  DOCKER_ARGS+=(-e DISPLAY="$DISPLAY_VALUE" -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
  [[ -f "$XAUTH" ]] && DOCKER_ARGS+=(-e XAUTHORITY=/root/.Xauthority -v "${XAUTH}:/root/.Xauthority:ro")
fi
exec "${DOCKER[@]}" run "${DOCKER_ARGS[@]}" --entrypoint bash kalibr:ros1_20_04 -lc 'source /catkin_ws/devel/setup.bash && cd /data && exec bash'
