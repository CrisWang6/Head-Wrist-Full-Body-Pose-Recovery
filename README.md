# Head Wrist Full Body Pose Recovery

Codebase for recovering full-body pose from synchronized head-mounted fisheye cameras, wrist cameras, external stereo cameras and motion-capture rigid bodies. This project builds on the EgoRear-style setup with head cameras, wrist cameras and supporting data-collection system design files.

The repository is organized as a source-code and workflow repo. Raw videos, generated CSV/NPZ datasets, model checkpoints, rendered validation videos and temporary reports are intentionally excluded.

## Repository Layout

```text
Head-Wrist-Full-Body-Pose-Recovery/
|-- OAK_test/                    # OAK/DepthAI capture, trigger, IMU and calibration tools
|-- data_processing/             # alignment, pose inference, stereo 3D recovery and projection
|-- prediction_model_training/    # model-training utilities for recovered labels
`-- Issacsim_data_generation/     # synthetic data generation utilities
```

## Current 2026-08 Workflow

The newest workflow focuses on `0711_214559` style synchronized datasets:

1. Capture OAK CAM_A/CAM_D streams with external trigger and device timestamps.
2. Align head stereo, external stereo and motion-capture rigid bodies by trigger sequence.
3. Drop any trigger where a required camera is missing a frame.
4. Run external stereo 2D pose detection with RTMPose/RTMW or YOLO pose.
5. Smooth 2D keypoints first, then triangulate external stereo 3D using the omni/fisheye model.
6. Convert external 3D skeletons through CH01/CH07 rigid frames.
7. Project the skeleton into head CAM_A/CAM_D with the confirmed `xy_swap + z_flip` head-camera basis.
8. Optionally use head-stereo manual/RTMW GT points, especially nose, shoulders, elbows, wrists and toes, to fit and refine the 3D skeleton.
9. Render 3D comparison viewers and head-stereo overlay videos for review.

The detailed audit notes are in:

```text
data_processing/docs/weekly_pose_recovery_20260806/
```

## Important Entry Points

OAK capture:

```bash
python OAK_test/scripts/capture/capture_oak_ad_external_trigger.py --help
```

Alignment:

```bash
python data_processing/scripts/alignment/strict_filter_aligned_50hz.py --help
python data_processing/scripts/alignment/align_0806_multi_stereo_mocap.py --help
```

External stereo pose recovery and head projection:

```bash
python data_processing/scripts/joint_projection/process_external_stereo_to_head.py --help
python data_processing/scripts/joint_projection/stabilize_torso_four_joints_3d.py --help
python data_processing/scripts/joint_projection/fit_head_gt_weighted_external_3d.py --help
python data_processing/scripts/joint_projection/generate_weighted_3d_comparison_visualization.py --help
```

## Confirmed Notes

- Use trigger/timestamp alignment rather than H265 packet order.
- Use the newest `offset0` aligned data for the current `0711_214559` branch.
- Keep strict deleted-frame videos separate from source-time review videos.
- Use the omni/fisheye camera model for head-camera projection.
- Optimize 3D skeleton points and poses; do not modify detected/manual 2D GT points as the optimization target.
- Weak filtering is preferred for full-motion review because strong filtering can visibly lag behind motion.

## Data Policy

Do not commit:

- raw H265/MP4/AVI/MJPEG recordings;
- rendered review videos or image dumps;
- large pose/checkpoint weights such as `.pt`, `.pth` or `.safetensors`;
- generated CSV/NPZ/NPY datasets;
- secrets, SSH keys, `.env` files or machine-local logs.
