# 第三方依赖

## 核心环境

| 依赖 | 用途 | 历史/建议版本 |
|---|---|---|
| NumPy | 矩阵、时间序列和骨架 | 历史 2.2.6 |
| Pandas | CSV 与表格对齐 | 历史 2.3.3 |
| SciPy | Rotation、插值和优化 | 历史 1.15.3 |
| OpenCV | H.265 帧、鱼眼投影、PnP 和绘制 | 历史 4.13.0 |
| Matplotlib | 误差曲线和 3D 图 | `>=3.8` |
| PyYAML | Kalibr camchain 配置 | `>=6` |
| pupil-apriltags | 腕部 Tag 检测 | `>=1.0.4` |
| FFmpeg/ffprobe | H.265 时间线和解码 | 系统包 |

## Pose 推理环境

| 依赖 | 用途 | 历史基线 |
|---|---|---|
| Meta Sapiens2 | 308 keypoint pose | <https://github.com/facebookresearch/sapiens2.git>，历史 commit `7e5bae88456ac418ff0e58e74106c9fe192055d4` |
| Sapiens 0.4B pose 权重 | 全身关键点 | 外部 `sapiens2_0.4b_pose.safetensors` |
| RTMLib | RTMPose 推理封装 | 0.0.15 |
| Ultralytics | 人体检测 | 历史 Sapiens/RTMPose 环境依赖 |
| PyTorch/torchvision | GPU 推理 | 与 CUDA 和上游模型匹配 |
| Blender `bpy` | 个别 FBX/骨架工具 | Blender 自带 Python，不通过普通 pip 环境保证 |

Sapiens2 和核心对齐环境可能使用不同 Python/CUDA 版本，推荐分别创建 `.venv` 与 `.venv-pose`，通过 JSONL/CSV 交换结果。

