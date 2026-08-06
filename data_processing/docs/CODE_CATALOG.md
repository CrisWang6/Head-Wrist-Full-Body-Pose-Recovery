# 代码功能目录

## 对齐：`scripts/alignment`

| 文件 | 功能 |
|---|---|
| `preprocess_9cam_imu_mocap.py` | 读取 9 相机录制、IMU 和动捕，建立统一 30 Hz 时间线及 aligned CSV。 |
| `global_imu_mocap_alignment.py` | 搜索全局时间偏移和姿态变换，对齐设备 IMU 与 mocap。 |
| `align_wrist_imu_mocap.py` | 针对左右腕 IMU/刚体轨迹细化时间与姿态对齐。 |
| `export_fbx_pose_to_csv.py` | 从 FBX 导出骨架关节 pose CSV。 |
| `export_ascii_fbx_mocap_to_csv.py` | 解析 ASCII FBX 并导出动捕数据。 |
| `export_abx2_mocap_rigid_csv.py` | 从 ABX2/配套 FBX 导出刚体和骨架 CSV。 |
| `export_wrist_mocap_derivatives.py` | 计算腕部 mocap 速度、角速度等派生量。 |
| `play_mocap_skeleton_3d.py` | 交互播放 mocap 3D 骨架。 |
| `run_export_fbx_pose.ps1` | Windows 下批量调用 FBX 导出工具。 |

## Viewer：`tools/alignment_viewer`

| 文件 | 功能 |
|---|---|
| `make_alignment_viewer_data.py` | 从 aligned CSV 生成浏览器 viewer 的 JavaScript 数据。 |
| `make_video_skeleton_viewer_data.py` | 生成视频帧到 3D skeleton 的映射数据。 |
| `make_aligned_video_frames.py` | 按对齐索引抽取 CAM_B 预览帧。 |
| `make_aligned_video_preview.py` | 生成带时间/动捕信息的对齐视频。 |
| `alignment_viewer.html` | 时间轴式相机/IMU/mocap 对齐浏览器。 |
| `video_skeleton_viewer.html` | 视频与 3D skeleton 联动浏览器。 |

## CAM_C pose：`scripts/pose2d/camc`

| 文件 | 功能 |
|---|---|
| `sapiens_camc_worker.py` | 调用 Sapiens2，对指定视频帧区间输出 308 关键点 JSONL。 |
| `launch_sapiens_camc.sh` | 将 CAM_C 序列切成多 GPU 分片并运行 Sapiens worker。 |
| `assemble_camc_pose_csv.py` | 合并分片 JSONL，并按视频时间线生成 CAM_C pose CSV。 |
| `sapiens_batch_benchmark.py` | 测试 Sapiens batch size、速度和显存。 |
| `rtmpose_camc_benchmark.py` | 测试 RTMPose CAM_C 推理速度和关键点输出。 |
| `rtmpose_random_review.py` | 随机抽样 RTMPose 结果并生成质量检查数据。 |

## CAM_B/C pose 与腕部 Tag：`scripts/pose2d/module01_bc`

| 文件 | 功能 |
|---|---|
| `rtmpose_video_worker.py` | 对视频帧区间运行 RTMPose 并写 JSONL/summary。 |
| `launch_rtmpose_bc.py` | 统计 B/C 视频帧数，切分到双 GPU 并启动 worker。 |
| `assemble_rtmpose_bc.py` | 合并 CAM_B/C RTMPose 分片并按时间戳生成 CSV。 |
| `launch_sapiens_bc_long.sh` | 对长序列 CAM_B/C 运行双卡 Sapiens 分片。 |
| `assemble_sapiens_bc.py` | 合并 CAM_B/C Sapiens JSONL 为统一姿态 CSV。 |
| `wrist_tag_pose_stereo.py` | 结合鱼眼内参和 B/C 外参，从 AprilTag 双目观测恢复腕部位姿。 |
| `run_full_wrist_chunks.sh` | 将完整腕部序列分片并并行运行 stereo Tag pose。 |
| `merge_wrist_tag_chunks.py` | 合并分片腕部 pose CSV/report。 |
| `analyze_tag_wrist_calibration.py` | 分析 Tag 位姿与腕部 mocap 之间的固定变换。 |
| `evaluate_wrist_pose_vs_mocap.py` | 计算腕部视觉 pose 与 mocap 的位置/旋转误差。 |
| `filter_wrist_evaluation.py` | 过滤异常帧并重算/绘制腕部误差统计。 |

## 投影与融合：`scripts/joint_projection`

| 文件 | 功能 |
|---|---|
| `project_joints.py` | 通用鱼眼相机投影库和 CLI；加载内外参、骨架、刚体和图片。 |
| `inspect_h265_timeline.py` | 用 ffprobe/FFmpeg 检查 H.265 PTS、帧数和时间线。 |
| `calibrate_0722_h265_fixed_time.py` | 搜索/验证 0722 H.265 固定时间偏移。 |
| `validate_head_imu_mocap_sync.py` | 对比头部 IMU 与 mocap，验证同步质量。 |
| `triangulate_head_bc_upper_body.py` | 用 CAM_B/C 2D 关键点三角化上肢 3D。 |
| `analyze_hybrid_temporal_offset.py` | 分析 Sapiens、刚体、BVH 和视频之间的时序偏移。 |
| `export_hybrid_skeleton_2d_csv.py` | 导出严格时间戳对齐的 CAM_B/C hybrid skeleton 2D CSV。 |
| `build_skeleton_3d_visualization.py` | 构建融合 skeleton 的 3D 可视化数据。 |
| `export_head_pose_visualization.py` | 导出头部 pose/相机轨迹可视化。 |
| `prepare_0722_h265_validation.py` | 准备 H.265 投影验证样本。 |
| `prepare_0722_head_validation.py` | 准备头部相机投影验证样本。 |
| `prepare_0722_2_hybrid_3d_comparison.py` | 准备 0722_2 hybrid 3D 对比数据。 |
| `project_0722_head_final.py` | 0722 头部相机最终投影 preset。 |
| `project_0722_abx2_subject_scaled.py` | 根据受试者比例缩放 ABX2 骨架并投影。 |
| `project_0722_2_ch308_raw_bvh.py` | 投影原始 BVH/308 关键点结果。 |
| `project_0722_2_camc_hybrid_2d_upper_rigid_wrist.py` | CAM_C hybrid 2D，上肢刚体/腕部替换版本。 |
| `compare_0722_2_raw_bvh_semantic_remap.py` | 比较 raw BVH 与语义 remap 骨架。 |
| `compare_0722_rigid_wrist_replacement.py` | 比较原 pose 与刚体腕部替换。 |
| `compare_0722_rigid_wrist_upperbody_ik.py` | 比较刚体腕部驱动的上肢 IK。 |
| `compare_0722_2_rigid_wrist_c7_upperarm_ik.py` | C7/上臂约束的改进 IK 比较。 |
| `compare_head_vs_rigid_projection.py` | 比较头部 pose 投影和刚体投影。 |
| `make_0722_fisheye_comparison.py` | 生成 0722 多种鱼眼投影方法的对比图。 |
| `render_camc_hybrid_skeleton_video.py` | 渲染通用 CAM_C hybrid skeleton overlay 视频。 |
| `render_camc_hybrid_skeleton_video_module01.py` | module01 录制的历史定制渲染版本。 |

配置文件 `projection_config*.json` 和 `subject_001_skeleton_parameters.json` 是上述脚本的实验 preset/骨架参数，不是生成数据。

