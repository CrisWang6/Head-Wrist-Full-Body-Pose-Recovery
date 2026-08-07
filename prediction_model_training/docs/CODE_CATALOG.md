# 代码功能目录

## 主包：`src/egorear_sim2d`

| 文件 | 功能 |
|---|---|
| `__init__.py` | `egorear_sim2d` 包初始化。 |
| `labels.py` | 从仿真 metadata/SMPL-X 关节生成 2D 坐标、可见性和 Gaussian heatmap。 |
| `dataset.py` | 发现 label NPZ、读取多视角 RGB、生成 target heatmap 和 DataLoader collate。 |
| `model.py` | 一阶段 ResNet 风格 heatmap 网络；不同相机共享 head/wrist branch 权重，并暴露 stage2 feature。 |
| `refinement.py` | 加载冻结的一阶段模型、双视角 anchor 特征采样、JQA 融合和 residual heatmap。 |
| `pose3d.py` | 三阶段 3D lifting 网络与串联 stage2/stage3 的推理 pipeline。 |
| `splits.py` | 生成、写入和校验 train/val/test split manifest，供各阶段共用同一划分。 |

## 一阶段脚本：`scripts`

| 文件 | 功能 |
|---|---|
| `prepare_heatmap_labels.py` | 从 Isaac/BlenderProc render 与 SMPL-X motion 生成仿真 label NPZ。 |
| `prepare_render_dataset.py` | 批量把仿真视频拆帧并组织训练 labels/frames。 |
| `watch_prepare_frames.py` | 监控渲染目录，增量准备训练帧。 |
| `wait_prepare_train.py` | 等待数据准备完成后启动训练的历史编排器。 |
| `wait_prepare_train_split.py` | 监控 render/prepare 状态并按 head/wrist 分支启动训练。 |
| `prepare_real_head_heatmap.py` | 用真实 aligned 3D mocap 和头部相机标定生成 2D heatmap label。 |
| `extract_direct_2d_frames.py` | 按直接 2D CSV timestamp 从 H.265 严格拆取 CAM_B/C 帧。 |
| `prepare_direct_2d_heatmap.py` | 将直接 2D 坐标和对应 RGB 写成训练 label NPZ。 |
| `prepare_pose3d_labels.py` | 把 stereo-lifted 3D 骨架转到头部坐标系，写成 stage3 的 12 关节 3D label NPZ。 |
| `create_dataset_split.py` | 生成 random / strided-random / chronological 的共享 split manifest。 |
| `train_heatmap.py` | 一阶段训练入口；支持 head/wrist、可见性 mask、resume、双 GPU、限时和 checkpoint 轮转。 |
| `test_heatmap.py` | 加载 checkpoint，在验证集计算 loss/像素误差并导出预测。 |
| `visualize_direct2d_validation.py` | 绘制直接 2D ground truth、预测峰值和 heatmap overlay。 |
| `visualize_random_split_test.py` | 在共享 held-out test set 上评估 stage1/2/3 并导出骨架对比图和 `test_metrics.json`。 |
| `build_split_comparison_report.py` | 汇总多组 split 实验的 stage1/2 像素误差与 stage3 MPJPE，输出 Markdown 和 CSV。 |
| `monitor_head180_start_wrist180.py` | 监控头部分支 epoch，并在条件满足后启动腕部分支训练。 |
| `run_real_0717_head2cam_48h.sh` | 0717 CAM_B/C 真实数据的 48 小时训练 preset。 |
| `run_real_0722_01_head2cam_direct2d_48h.sh` | 0722 直接 2D、12 关节、CAM_B/C 共享模型的双卡 48 小时 preset。 |
| `run_real_0722_01_random_split_stages123.sh` | 全局 random split（seed 42）的 stage1→2→3 串行 preset，支持 `START_STAGE` 续跑。 |
| `run_real_0722_01_strided_stages123.sh` | stride 10/30/90 三组 strided-random split 的 stage1→2→3 批量 preset。 |

## 二阶段：`experiments/stage2_refinement`

| 文件 | 功能 |
|---|---|
| `scripts/train_refinement.py` | 冻结 stage1，基于 stage1/noisy-GT proposal 训练 CAM_B/C residual refiner。 |
| `scripts/test_refinement.py` | 评估 stage2 heatmap loss、像素误差并保存可视化/指标。 |
| `configs/stage2_head_bc_refinement.json` | 历史 16 关节 stage2 结构和数据配置。 |

二阶段核心网络已经移到主包 `src/egorear_sim2d/refinement.py`，不再维护一份冲突的嵌套 `model.py`。

## 三阶段：`experiments/stage3_pose3d`

| 文件 | 功能 |
|---|---|
| `scripts/train_pose3d.py` | 冻结 stage1/stage2，用 stereo-lifted 3D 监督训练 12 关节 3D lifting。 |
| `scripts/visualize_pose3d_video.py` | 导出 train/test 连续 3D 预测视频；`--split-manifest` 复用训练划分。 |
| `configs/stage3_head_bc_pose3d.json` | stage3 的关节表、坐标约定和训练超参记录。 |

## 独立实验：`experiments/multiview_refinement`

| 文件 | 功能 |
|---|---|
| `__init__.py` | 导出 `StereoHeatmapRefiner`。 |
| `dataset.py` | 根据 manifest 读取 CAM_B/C RGB、初始 heatmap、target heatmap 和 visibility。 |
| `model.py` | 独立双视角 heatmap refinement 网络。 |
| `train.py` | 训练独立 refiner，保存配置、best/last checkpoint。 |
| `infer.py` | 加载独立 checkpoint，批量导出 CAM_B/C refined heatmap NPZ。 |
| `smoke_test.py` | 用随机 tensor 验证模型形状、forward 和梯度。 |

