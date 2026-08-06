# 代码功能目录

## `scripts/capture`

| 文件 | 功能 |
|---|---|
| `oak_ffc_sync_ar0234.py` | 单 OAK FFC/AR0234 多相机 pipeline，预览同步帧并检查设备时间戳。 |
| `oak_ffc_sync_ar0234_orin_continuous_fsync.py` | 面向 Orin 的连续 FSYNC 版本，支持 H.265 编码、录制进度、统计和封装。 |

## `scripts/recording`

| 文件 | 功能 |
|---|---|
| `record_9cam.py` | 同时连接头部、左腕、右腕三个 OAK，记录 9 路 H.265、设备/主机时间戳和 IMU。 |
| `record_multi_h265_v2.py` | 较早的多设备 H.265 联合录制实现，保留用于兼容和对照。 |
| `start_recording_from_gpio.py` | 树莓派 GPIO 按键监听器；控制 FSYNC/状态灯并启动 `record_9cam.py`。 |
| `sync_imu_to_frames.py` | 将 IMU 样本按时间戳关联到视频帧，生成同步后的结构化记录。 |

## `scripts/calibration`

| 文件 | 功能 |
|---|---|
| `__init__.py` | 标定脚本目录的 Python 包标记。 |
| `common.py` | DepthAI socket、AprilGrid/AprilTag 检测、同步取帧、JSON 和绘图公共函数。 |
| `capture_intrinsics_dataset.py` | 从指定 OAK 模块同步采集内参或外参标定图片。 |
| `inspect_device_calibration.py` | 读取并报告设备 EEPROM 中的相机/IMU 标定参数。 |
| `export_depthai_calibration.py` | 将项目相机参数转换为 DepthAI calibration JSON，并可选备份和写入 EEPROM。 |

## `scripts/diagnostics`

| 文件 | 功能 |
|---|---|
| `debug_oak_imu_euler.py` | 读取 OAK IMU，执行 gyro bias 校准、Mahony 姿态估计并输出 Euler 角/诊断。 |

## `tools/device_ui`

| 文件 | 功能 |
|---|---|
| `cam_test.py` | 相机测试总入口，支持预览、raw、stereo、ToF、YOLO 和 GUI 分支。 |
| `cam_test_gui.py` | PySimpleGUI/PyQt 相机测试界面。 |
| `device_manager.py` | OAK 设备发现、选择和连接管理。 |
| `install_requirements.py` | 设备工具依赖安装辅助。 |
| `oak_ffc_raw_host_ar0234.py` | AR0234 原始流 host 端采集/显示工具。 |
| `stress_test.py` | 多相机长时间运行、温度/资源/丢帧压力测试。 |

## `tools/kalibr/custom`

| 文件 | 功能 |
|---|---|
| `build_pair_bag.py` | 从按 sample/camera 组织的标定图构建双相机 ROS bag。 |
| `CameraCalibrator.py` | 自定义 Kalibr CameraCalibrator，支持项目所需固定内参/执行行为。 |
| `kalibr_calibrate_cameras` | 修改后的 Kalibr 相机标定 CLI。 |
| `kalibr_calibrate_cameras_fixed` | 固定内参标定入口的另一历史版本。 |
| `run_remote.sh` | 构建头部 B/C、A/C bag 并运行双相机外参标定。 |
| `run_wrists_remote.sh` | 批量运行左腕和右腕 A/C、A/B 外参标定。 |

## `tools/kalibr`

| 文件 | 功能 |
|---|---|
| `kalibr_shell.sh` | 启动挂载数据目录的 Kalibr Docker shell。 |
| `kalibr_calibrate_cameras.sh` | 通用 Docker 标定包装脚本。 |
| `tools/make_cam_bags.py` | 为多个单相机图片集生成 ROS bag。 |
| `tools/make_multicam_bag.py` | 从多相机同步图片生成一个多 topic bag。 |
| `tools/run_all_omni_intrinsics.sh` | 批量运行 omni-radtan 内参标定。 |
| `tools/run_failed_omni.sh` | 重跑历史失败的 omni 标定任务。 |
| `tools/run_handle_cam_a_manual.sh` | 手柄 CAM_A 手工参数标定入口。 |
| `tools/run_handle_mixed_intrinsics.sh` | 手柄不同 camera model 的混合内参实验。 |
| `tools/run_handle_omni_intrinsics.sh` | 手柄 omni-radtan 内参标定。 |
| `tools/run_head_intrinsics.sh` | 头部相机内参批处理。 |

