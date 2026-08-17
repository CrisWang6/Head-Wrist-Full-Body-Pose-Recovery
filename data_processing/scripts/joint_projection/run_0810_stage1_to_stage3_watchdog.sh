#!/usr/bin/env bash
# Wait for stage1 (150ep) to finish, then launch stage3 200ep skipping stage2.
set -euo pipefail

JP="/home/gaoweijian/0810_batch/repo/test_code/joint_projection"
DATASET="/home/gaoweijian/0810dataset"
LOG="$DATASET/logs/0810_stage1_to_stage3_watchdog.log"
STAGE1_BEST="$DATASET/checkpoints/stage1_v31/best.pt"

exec >>"$LOG" 2>&1
echo "[$(date '+%F %T')] watchdog: waiting for stage1 train_heatmap to exit"
while pgrep -f 'train_heatmap.py.*stage1_v31' >/dev/null; do
  sleep 30
done
echo "[$(date '+%F %T')] stage1 process exited"

if [[ ! -f "$STAGE1_BEST" ]]; then
  echo "[$(date '+%F %T')] ERROR: missing $STAGE1_BEST"
  exit 1
fi

echo "[$(date '+%F %T')] stopping stale stage2/stage3 jobs if any"
pkill -f 'train_refinement.py.*stage2_v31' 2>/dev/null || true
pkill -f 'run_0810_stage2_100ep.sh' 2>/dev/null || true
pkill -f 'run_0810_stage3_150ep_watchdog.sh' 2>/dev/null || true
pkill -f 'train_pose3d.py.*stage3_v31_aligned' 2>/dev/null || true
sleep 2

echo "[$(date '+%F %T')] launching stage3 200ep (skip stage2)"
exec bash "$JP/run_0810_stage3_skip_stage2_200ep.sh"
