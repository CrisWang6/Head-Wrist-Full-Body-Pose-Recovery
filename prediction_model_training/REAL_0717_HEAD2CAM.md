# 0717 real-data head two-camera training

This run trains only the stage-1 head heatmap branch from `module01/CAM_B` and
`module01/CAM_C`. It starts from the best IsaacSim head checkpoint, but creates a
new optimizer and a new run directory.

Labels are projected with the per-camera Kalibr omni+radtan intrinsics. CAM_B uses
the measured head mount `(87, -26, 161) mm`; CAM_C is derived from CAM_B with the
calibrated B-to-C transform. Each camera exposure timestamp is mapped into the
mocap clock independently, then joint positions and the mocap Head quaternion are
interpolated at that exposure time.

The 80/20 split is chronological to avoid leakage between neighboring video
frames. Only joints that project into the front 180-degree image are included in
the loss. The launcher has both a clean 47.75-hour deadline and a 48-hour OS hard
limit. `last.pt`, `best.pt`, and at most three numbered checkpoints are retained.

Commands:

```bash
python scripts/prepare_real_head_heatmap.py
bash scripts/run_real_0717_head2cam_48h.sh
```
