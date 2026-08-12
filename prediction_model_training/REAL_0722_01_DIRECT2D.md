# 0722_01 head CAM_B/C direct-2D stage-1 training

This dataset uses the per-camera 2D coordinates in
`module01_cam_bc_hybrid_skeleton_2d.csv` as heatmap ground truth. It does not
project a 3D human pose through a camera model.

The head branch predicts 12 heatmaps shared by CAM_B and CAM_C, in this order:
`LeftFoot`, `RightFoot`, `LeftUpLeg`, `RightUpLeg`, `LeftArm`, `RightArm`,
`Spine`, `Spine2`, `LeftForeArm`, `RightForeArm`, `LeftHand`, `RightHand`.
In the source convention these correspond to left/right foot, hip, shoulder,
lumbar/thoracic spine, elbow, and wrist.

## Timestamp and frame alignment

Each camera record is first bound to its raw H.265 image by
`decoded_frame_index`. CAM_B and CAM_C are then paired one-to-one in temporal
order using their own `device_ts_ms`. A pair is accepted only when the absolute
timestamp difference is at most 1 ms.

Do not pair these data using the CSV `seq` column. After camera dropouts, the
same `seq` can refer to B/C images separated by multiple frame periods.

## Prepare data

```bash
cd /home/gaoweijian/EgoRear_w_hand

/home/gaoweijian/miniforge3/envs/camtest/bin/python \
  scripts/extract_direct_2d_frames.py \
  --dataset-root /home/gaoweijian/Desktop/0722_01_training

/home/gaoweijian/miniforge3/envs/camtest/bin/python \
  scripts/prepare_direct_2d_heatmap.py \
  --dataset-root /home/gaoweijian/Desktop/0722_01_training
```

The label manifest records the accepted sample count, unpaired frames, and
actual B/C timestamp-delta distribution. The label NPZ also stores both device
timestamps, both decoded frame indices, both original aligned sequence values,
and the per-sample stereo delta.

## Train

The supplied launcher initializes from the simulation stage-1 checkpoint,
trains only the head branch, uses a chronological 80/20 split, and stops cleanly
after 47.75 hours:

```bash
CUDA_VISIBLE_DEVICES=1 \
  bash scripts/run_real_0722_01_head2cam_direct2d_48h.sh
```
