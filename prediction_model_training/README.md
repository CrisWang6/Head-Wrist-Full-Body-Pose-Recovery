# Prediction Model Training

人体 2D heatmap 的一阶段训练、CAM_B/C 二阶段 refinement 和多视角 refinement 仓库。支持仿真投影标签、真实 3D 投影标签，以及 0722 由严格时间戳对齐的直接 2D CSV 生成 ground truth。

## 1. 项目结构

```text
prediction_model_training/
├─ src/egorear_sim2d/
│  ├─ dataset.py       # 多视角数据集与帧读取
│  ├─ labels.py        # 仿真投影标签与 heatmap
│  ├─ model.py         # 一阶段共享权重网络
│  ├─ refinement.py    # CAM_B/C 二阶段 residual refinement
│  ├─ pose3d.py        # 三阶段 3D lifting 网络与推理 pipeline
│  └─ splits.py        # 各阶段共享的 train/val/test split manifest
├─ scripts/            # 一阶段数据准备、训练、测试和可视化
├─ experiments/
│  ├─ stage2_refinement/    # 基于一阶段特征的双视角 refinement
│  ├─ stage3_pose3d/        # 基于二阶段特征的 12 关节 3D lifting
│  └─ multiview_refinement/ # 基于 manifest 的独立双视角实验
├─ configs/            # 相机参数和历史训练配置
└─ docs/
```

逐文件说明见 [代码功能目录](docs/CODE_CATALOG.md)，上游依赖见 [第三方依赖](docs/THIRD_PARTY.md)。

## 2. 环境部署

历史训练基线：

```text
Ubuntu 22.04.5
Python 3.10.20
PyTorch 2.5.1+cu121
CUDA runtime 12.1
OpenCV 4.13.0
TensorBoard 2.20.0
NumPy 2.2.6
2 × RTX 4090
```

推荐先安装匹配 GPU 的 PyTorch，再安装项目：

```bash
cd prediction_model_training
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[multiview]"
```

检查：

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda, torch.cuda.device_count())
PY
```

仿真标签还需要同级 `Issacsim_data_generation`：

```bash
pip install -e ../Issacsim_data_generation
```

## 3. 一阶段 Pipeline

```text
输入 A：Isaac Sim/BlenderProc RGB + metadata + SMPL-X
   → prepare_heatmap_labels.py

输入 B：真实 RGB + aligned 3D mocap + 相机内外参
   → prepare_real_head_heatmap.py

输入 C：真实 H.265 + 严格对齐 2D CSV
   → extract_direct_2d_frames.py
   → prepare_direct_2d_heatmap.py

统一 label NPZ + RGB
   → MultiViewHeatmapDataset
   → 共享 head_branch 的一阶段网络
   → best.pt / last.pt / TensorBoard
   → test_heatmap.py / validation visualization
```

### 3.1 仿真标签

```bash
python scripts/prepare_heatmap_labels.py \
  --simulation-root ../Issacsim_data_generation \
  --render-root /data/simulation/outputs/example \
  --output-dir /data/labels/example
```

### 3.2 真实 3D 投影标签

```bash
python scripts/prepare_real_head_heatmap.py --help
```

默认相机参数位于 `configs/calibration/head/`。

### 3.3 0722 直接 2D ground truth

```bash
python scripts/extract_direct_2d_frames.py --help
python scripts/prepare_direct_2d_heatmap.py --help
```

该链路读取 `module01_cam_bc_hybrid_skeleton_2d.csv` 中每帧 CAM_B/C 关节坐标，并按 timestamp 对应 H.265 拆帧，不再使用人体 3D pose 做二次 projection。

### 3.4 训练

通用入口：

```bash
python scripts/train_heatmap.py --help
```

0722 双卡 48 小时配置：

```bash
export CUDA_VISIBLE_DEVICES=0,1
export DATA_ROOT=/data/egorear
export ARTIFACT_ROOT=/artifacts/egorear
export PYTHON_BIN="$(pwd)/.venv/bin/python"
bash scripts/run_real_0722_01_head2cam_direct2d_48h.sh
```

CAM_B/C 使用同一个共享模型。0722 实验输出 12 个通道：

```text
LeftFoot, RightFoot,
LeftUpLeg, RightUpLeg,
LeftArm, RightArm,
Spine, Spine2,
LeftForeArm, RightForeArm,
LeftHand, RightHand
```

即两脚、两髋、两肩、腰椎/胸椎、两肘、两腕。

### 3.5 测试与 TensorBoard

```bash
python scripts/test_heatmap.py --help
python scripts/visualize_direct2d_validation.py --help
tensorboard --logdir /artifacts/egorear/logs --host 0.0.0.0 --port 6006
```

## 4. 二阶段 refinement

重构后 `refinement.py` 已并入主包 `src/egorear_sim2d`，一阶段和二阶段共用同一个 `dataset.py`、`model.py` 与安装环境，避免原嵌套源码包缺少 `dataset.py`。

训练：

```bash
python experiments/stage2_refinement/scripts/train_refinement.py \
  --label-root /data/labels/real_head \
  --stage1-checkpoint /artifacts/stage1/best.pt \
  --output-dir /artifacts/stage2 \
  --log-dir /artifacts/stage2_logs \
  --proposal-source stage1 \
  --epochs 20 --batch-size 8 --device cuda
```

评估：

```bash
python experiments/stage2_refinement/scripts/test_refinement.py --help
```

stage2 已经跟随 0722 实验统一到 12 关节，配置见 `experiments/stage2_refinement/configs/stage2_head_bc_refinement.json`。复用 16 关节的历史 checkpoint 前，必须同时核对 joint layout、输出头和 checkpoint metadata。

## 5. 三阶段 3D lifting

`experiments/stage3_pose3d` 冻结 stage1/stage2，把 refined heatmap 和特征抬升成头部坐标系下的 12 关节 3D 骨架，监督来自 stereo-lifted 的肩肘关节：

```bash
python scripts/prepare_pose3d_labels.py --help
python experiments/stage3_pose3d/scripts/train_pose3d.py --help
python experiments/stage3_pose3d/scripts/visualize_pose3d_video.py --help
```

导出预测视频时应传入训练用的同一个 `--split-manifest`，否则 test 视频会退回按时间顺序的 80/20 划分，可能混入训练帧。

## 6. 共享数据划分与横向对比

stage1/2/3 通过同一个 split manifest 保证划分一致，避免相邻帧跨子集泄漏：

```bash
python scripts/create_dataset_split.py --help
python scripts/visualize_random_split_test.py --help
python scripts/build_split_comparison_report.py --help
```

整条 stage1→2→3 串行流程有两个 preset：

```bash
bash scripts/run_real_0722_01_random_split_stages123.sh    # 全局 random split，seed 42
bash scripts/run_real_0722_01_strided_stages123.sh         # stride 10/30/90 三组对比
```

两个脚本默认在仓库根目录下用 `python` 运行，可用 `REPO_ROOT`、`PYTHON`、`LABEL_ROOT`、`SPLIT_MANIFEST`、`CUDA_VISIBLE_DEVICES` 覆盖；random split preset 还支持 `START_STAGE` 从中间阶段续跑。

## 7. 独立多视角实验

`experiments/multiview_refinement` 使用 CSV manifest 直接读取 CAM_B/C RGB、初始 heatmap 和 target heatmap：

```bash
python experiments/multiview_refinement/smoke_test.py
python experiments/multiview_refinement/train.py --help
python experiments/multiview_refinement/infer.py --help
```

这套实现与主 `egorear_sim2d.refinement` 是两个实验分支，不应混用 checkpoint。

## 8. 数据与产物边界

以下内容全部在 Git 外：

- H.265、拆帧图片、label NPZ；
- checkpoint、权重和优化器状态；
- TensorBoard event、日志、验证图；
- 仿真输出和 SMPL-X 模型；
- `data/splits/*.npz` split manifest，可用 `create_dataset_split.py` 加相同 seed 复现。

推荐统一挂载：

```text
/data/egorear       # 输入和标签
/artifacts/egorear  # checkpoint、logs、outputs
```

## 9. 未直接纳入的相关代码

上游 EgoRear 参考实现和含明文 SSH 密码的部署脚本没有复制，详见 [未纳入代码](docs/UNINTEGRATED_CODE.md)。

