#!/usr/bin/env bash
# Wait for stage2 to finish, then launch stage3 150ep (override chained script stage3).
set -euo pipefail

JP="/home/gaoweijian/0810_batch/repo/test_code/joint_projection"
LOG="/home/gaoweijian/0810dataset/logs/0810_stage3_150ep_watchdog.log"

exec >>"$LOG" 2>&1
echo "[$(date '+%F %T')] watchdog: waiting for stage2 train_refinement to exit"
while pgrep -f 'train_refinement.py.*stage2_v31' >/dev/null; do
  sleep 20
done
echo "[$(date '+%F %T')] stage2 process exited; stop chained runner if still alive"
pkill -f 'run_0810_stage2_100ep.sh' 2>/dev/null || true
sleep 2
pkill -f 'train_pose3d.py.*stage3_v31_aligned' 2>/dev/null || true
sleep 2
echo "[$(date '+%F %T')] launching stage3 150ep"
exec bash "$JP/run_0810_stage3_150ep.sh"
