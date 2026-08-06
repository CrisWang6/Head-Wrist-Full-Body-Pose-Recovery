# 第三方依赖

## Python

| 依赖 | 用途 | 版本策略 |
|---|---|---|
| `depthai` | OAK 设备、pipeline、编码器、IMU 和 EEPROM | `>=2.32,<4`；代码同时包含 DepthAI 2/3 兼容分支 |
| `opencv-contrib-python` | 图像预览、AprilTag/ArUco、标定与绘制 | `>=4.8,<5` |
| `numpy` | 矩阵、相机参数和数据转换 | `>=1.26,<3` |
| `scipy` | 旋转和数值工具 | `>=1.11` |
| `pupil-apriltags` | AprilTag fallback 检测 | `>=1.0.4` |
| `matplotlib` | 诊断图 | `>=3.8` |
| `Pillow` | GUI/图像辅助 | `>=10` |
| `psutil` | 压力测试与进程监控 | `>=5.9` |
| `imageio-ffmpeg` | FFmpeg 可执行文件定位 | `>=0.5` |
| `PyQt5`、`PySimpleGUI` | 可选桌面 GUI | 安装 `.[gui]` |

## 系统与上游项目

| 依赖 | 来源/固定版本 | 说明 |
|---|---|---|
| DepthAI | <https://github.com/luxonis/depthai-python> | OAK Python SDK |
| Kalibr | <https://github.com/ethz-asl/kalibr.git> | 历史 commit `1f60227442d25e36365ef5f72cd80b9666d73467` |
| ROS1 Noetic | Kalibr `Dockerfile_ros1_20_04` | 只用于标定容器 |
| FFmpeg | 系统包 | H.265 封装、探测和转换 |
| `pinctrl` | Raspberry Pi 系统工具 | GPIO 按键、FSYNC 和状态灯 |

`tools/kalibr/custom/CameraCalibrator.py` 和两个 `kalibr_calibrate_cameras*` 是针对固定内参、同步容差和无界面运行修改的 Kalibr 入口；它们必须放进与上述 commit 匹配的 Kalibr 容器。

