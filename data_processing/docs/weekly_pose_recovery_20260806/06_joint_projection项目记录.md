# joint_projection 项目记录

更新时间：2026-08-06

项目位置：

```text
C:\Users\hand\Desktop\HearWristCam\test_code\joint_projection
```

## 1. 本周主要脚本

### 对齐与视频准备

```text
align_four_triggered_cameras.py
make_strict_synced_outputs.py
make_event_projection_bundle.py
extract_aligned_event_video.py
extract_aligned_video_by_source_frame.py
expand_event_video_by_timestamp.py
make_h265_frame_map.py
inspect_h265_timeline.py
```

用途：处理 timestamp/trigger 对齐、缺帧删除、事件视频和原始时间轴视频。

### 2D pose 与外部双目 3D

```text
infer_pose_candidates.py
infer_rtmpose_candidates.py
process_external_stereo_to_head.py
```

用途：跑 YOLO pose / RTMPose 候选人，做外部双目匹配和三角化。

### 头部鼻子和手部检测

```text
detect_head_nose_rtmw.py
detect_head_nose_mediapipe.py
detect_guided_hands_mediapipe.py
detect_apriltag25h9_hands.py
```

本周结论：

- 鼻尖很难稳定检测，RTMW WholeBody 可用。
- 用户手动画过绿色 nose 点，但后续更希望用 RTMW WholeBody。
- 手部不建议主要依赖 tag，MediaPipe Hand 的 ROI 引导方式更适合 wrist。

### 头部投影与坐标检查

```text
compare_head_projection_axis_modes.py
compare_head_vs_rigid_projection.py
export_head_pose_visualization.py
build_head_fivepoint_3d_visualization.py
build_skeleton_3d_visualization.py
generate_weighted_3d_comparison_visualization.py
```

用途：检查 head/camera/rigid 坐标系、生成 3D 可交互页面、输出投影视频。

### 3D 优化与滤波

```text
optimize_external_skeleton_with_ch07_nose.py
optimize_head_fivepoint_pose.py
stabilize_torso_four_joints_3d.py
fit_head_gt_weighted_external_3d.py
apply_fixed_global_fivepoint_full.py
apply_head_nose_fixed_offset.py
```

本周最新重点脚本：

```text
fit_head_gt_weighted_external_3d.py
```

它用于把头部双目手动 2D GT 反投影为 3D GT，然后和外部相机三角化 3D 骨架做加权拟合。

## 2. 模型使用记录

### Sneaker / 鞋子

早期鞋子检测曾使用 YOLO sneaker 方向做框/分割，目标是取出现时间最多、最像用户本人鞋子的检测结果，避免路人鞋子干扰。

### 全身骨架

早期头部相机试过 Sapiens2 全身骨架，但后续用户要求只保留肩膀和肘关节，其他关节主要来自外部双目投影。

### 外部双目 pose

本周主要在 YOLO pose 和 RTMPose 之间比较。RTMPose/RTMW WholeBody 在 nose 与头部投影约束上更合适，因此当前验证分支偏向 RTMPose。

### 手部

腕部 tag 不稳定。更推荐用手部模型：

- MediaPipe Hand Landmarker
- 通过外部骨架投影得到的肘-腕方向裁 ROI
- 取 hand landmark 0 作为 wrist

## 3. 当前全量结果

全量弱滤波投影：

```text
C:\Users\hand\Desktop\0711_214559\realigned_offset0_rawsource_full6207\final_full_videos\head_stereo_global_fivepoint_projection_weak_filter_full.mp4
```

其它全量输出：

```text
external_stereo_2d_raw_vs_filtered_full.mp4
global_fivepoint_3d_front_side_full.mp4
head_stereo_global_fivepoint_projection_full.mp4
global_fivepoint_ch07_full.csv
head_stereo_projected_2d_full.csv
head_stereo_projected_2d_weak_filter_full.csv
weak_3d_filter_report.json
```

弱滤波报告要点：

- median window：3
- Savitzky-Golay window：9
- ch07_event_offset_frames：0
- phase policy：centered offline filter，无固定时延
- limb policy：肩的 delta 传播到肘/腕，髋的 delta 传播到膝/踝，nose 不动

## 4. 当前手动 GT 拟合结果

输入手动标注：

```text
C:\Users\hand\Desktop\0711_214559\head_stereo_fullbody_2d_gt.csv
```

输出目录：

```text
C:\Users\hand\Desktop\0711_214559\head_manual_gt_fit_5s15s_final_v2
```

关键输出：

```text
head_stereo_external_vs_headGT_weighted_fit_5s15s.mp4
headgt-external-weighted-3d-comparison.html
external_triangulated_ch07_5s15s.csv
external_global_fit_ch07_5s15s.csv
headGT_weighted_bone_optimized_ch07_5s15s.csv
head_stereo_manual_triangulated_ch07.csv
comparison_3d_5s15s.json
report.json
```

验证范围：

- sequence：250-749
- 帧数：500
- 时间：5-15s
- 输出 fps：50

## 5. 重要数值

来自 `report.json`：

- 手动标注 sequence：252、302、1478、1837
- 验证段内标注 sequence：252、302
- head triangulated points：60
- head stereo ray miss median：26.40 mm
- head stereo ray miss p90：58.15 mm
- 拟合前 median error：122.76 mm
- 拟合后 median error：74.47 mm
- 高权重点拟合后 median error：58.76 mm
- 低权重点拟合后 median error：178.44 mm
- 肩宽 GT：353.02 mm
- 上臂 GT：242.92 mm
- 小臂 GT：222.60 mm

## 6. 可视化颜色约定

在手动 GT 拟合验证视频中：

- cyan：外部双目三角化/鼻子锚定 baseline
- yellow：头部 GT 加权优化后的骨架
- green circles：头部手动 2D GT

在 3D HTML 中：

- external：外部原始或 nose-anchored baseline
- optimized：优化后骨架
- headGT：头部双目反投影的手动 GT

## 7. 后续维护建议

1. 新增结果目录时写清楚“基线来源”和“是否使用旧 CH07 列”。
2. 每个视频旁边保留同名 report.json。
3. 如果改了内参或外参，输出名里标明 new_intrinsics / xybasis / offset0。
4. 如果改了滤波强度，输出名里标明 strong_filter 或 weak_filter。
5. 如果做头部投影，必须确认鱼眼/omni 投影模型仍在使用。
