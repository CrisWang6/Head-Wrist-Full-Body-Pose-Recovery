# Data Processing

Data-processing code for synchronized head/wrist camera recordings, motion-capture alignment, 2D pose inference, stereo triangulation, full-body 3D recovery, and projection into head-mounted fisheye cameras.

This repository stores source code, lightweight configs, documentation and reproducible processing entry points. It does not store raw H265/MP4 videos, large model weights, generated CSV/NPZ datasets, rendered validation frames, logs, or temporary reports.

## Project Layout

```text
data_processing/
|-- scripts/
|   |-- alignment/          # timestamp, trigger, IMU and mocap alignment
|   |-- joint_projection/   # stereo triangulation, 3D optimization, head projection, viewers
|   `-- pose2d/             # Sapiens/RTMPose/YOLO pose assembly helpers
|-- tools/
|   `-- alignment_viewer/   # browser alignment review utilities
|-- configs/                # lightweight processing presets
`-- docs/                   # workflow notes and code catalog
```

## 2026-08 Head/Wrist Full-Body Pose Recovery Update

This repository now includes the head/external stereo pose-recovery branch used for the 2026-08-03 to 2026-08-05 experiments.

Main additions:

- strict trigger/timestamp alignment for synchronized head and external stereo streams;
- RTMPose/RTMW and YOLO pose candidate inference for external stereo views;
- omni/fisheye stereo triangulation from external CAM_A/CAM_D 2D keypoints;
- CH01/CH07 rigid-frame conversion and head-camera projection with the confirmed `xy_swap + z_flip` basis;
- weak and strong smoothing profiles for torso, limbs, hands and head-related keypoints;
- nose 2D/3D GT anchoring using CH07 and RTMW/manual head-view landmarks;
- manual head-stereo GT labeler and weighted 3D fitting against external stereo triangulation;
- standalone 3D comparison visualization and head-stereo projection rendering.

The current audit notes and confirmed workflow are stored in:

```text
docs/weekly_pose_recovery_20260806/
```

## Important Entry Points

Alignment:

```bash
python scripts/alignment/strict_filter_aligned_50hz.py --help
python scripts/alignment/align_0806_multi_stereo_mocap.py --help
```

External stereo and head projection:

```bash
python scripts/joint_projection/process_external_stereo_to_head.py --help
python scripts/joint_projection/stabilize_torso_four_joints_3d.py --help
python scripts/joint_projection/fit_head_gt_weighted_external_3d.py --help
python scripts/joint_projection/generate_weighted_3d_comparison_visualization.py --help
```

Pose inference:

```bash
python scripts/joint_projection/infer_rtmpose_candidates.py --help
python scripts/joint_projection/detect_head_nose_rtmw.py --help
```

## Confirmed Processing Rules

- Use the newest timestamp-aligned `offset0` strict dataset for `0711_214559`.
- Drop a trigger if any required camera is missing that trigger or frame.
- Filter external stereo 2D keypoints before triangulating to 3D.
- Prefer the weak filter for full-motion review; strong filtering is useful for stability checks but can lag behind motion.
- Use the head projection basis `xy_swap + z_flip`.
- Use the omni/fisheye projection model for head cameras, not a pinhole model.
- Optimize the 3D skeleton, not the detected 2D GT points.
- Keep `strict/event` videos and `source_time` review videos conceptually separate.

## Environment

Core processing uses Python 3.10+ with NumPy, SciPy, OpenCV, Pandas, PyYAML and FFmpeg. Pose inference branches may require separate CUDA/PyTorch environments for RTMPose, RTMW WholeBody, YOLO or Sapiens2.

Large third-party repositories, checkpoints and datasets should be referenced through environment variables or command-line arguments rather than committed here.
