# Isaac Sim Data Generation

头戴鱼眼相机、左右腕部相机、ArUco/AprilTag 与人体运动的几何仿真和多视角数据生成项目。

> 仓库目录沿用既定名称 `Issacsim_data_generation`；NVIDIA 产品的标准拼写是 **Isaac Sim**。

## 1. 项目结构

```text
Issacsim_data_generation/
├─ src/geosim/       # 可复用的几何、相机、人体、IMU 和渲染模块
├─ scripts/          # CLI：仿真、pipeline、渲染和可视化
├─ configs/          # 默认几何与 realistic camera 配置
├─ tests/            # 几何仿真单元测试
└─ docs/             # 原始说明、Git 来源和第三方依赖
```

逐文件说明见 [代码功能目录](docs/CODE_CATALOG.md)，外部运行时见 [第三方依赖](docs/THIRD_PARTY.md)。

## 2. 环境部署

### 2.1 基础 Python 环境

历史可复现基线：

- Ubuntu 22.04.5
- Python 3.10.20
- NumPy 2.2.6
- OpenCV 4.13.0
- BlenderProc 2.8.0
- NVIDIA 驱动 580.159.03

安装核心几何功能：

```bash
cd Issacsim_data_generation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[vision,visualization]"
```

仅运行无可视化几何测试时：

```bash
pip install -e .
```

### 2.2 BlenderProc

BlenderProc 是外部依赖，不复制进仓库：

```bash
git clone https://github.com/DLR-RM/BlenderProc.git /opt/BlenderProc
pip install -e /opt/BlenderProc
blenderproc --version
```

历史项目使用 BlenderProc 2.8.0 和 Blender 4.2。若新版本渲染行为变化，应优先固定到同一版本。

### 2.3 Isaac Sim

Isaac Sim 必须按新机器 GPU 驱动重新安装。通过参数指定其 Python：

```bash
python scripts/render.py isaacsim --isaacsim-python /opt/isaacsim/python.sh --help
```

不要把完整 Isaac Sim 安装目录、Kit cache 或 50 GB 级 Docker 镜像放入 Git。

### 2.4 外部资产

运行真实人体 motion 需要：

```text
assets/smplx/SMPLX_NEUTRAL_2020.npz
data/motions/HumanEva/..._stageii.npz
```

这些文件受许可或体积限制，不属于仓库。CLI 参数应指向外部路径；仓库 `.gitignore` 已排除模型、motion cache、视频和 mesh。

## 3. Pipeline

```text
AMASS/HumanEva motion
  → SMPL-X 顶点/关节恢复
  → 头部与腕部相机 rig
  → Tag 角点和相机可见性
  → 可选 IMU/成像退化
  → BlenderProc 或 Isaac Sim 多视角渲染
  → metadata + motion cache + 视频/帧
  → prediction_model_training 生成 heatmap 标签
```

### 3.1 几何仿真

```bash
python scripts/sim.py geometry --synthetic --no-visualization
python scripts/sim.py geometry \
  --motion-dir /data/HumanEva \
  --config configs/default_geometry.json
```

### 3.2 Tag/腕部/足部 pipeline

```bash
python scripts/pipeline.py --help
python scripts/pipeline.py wrist-ankle-tags --help
python scripts/pipeline.py foot-landmarks --help
python scripts/pipeline.py foot-tag --help
```

该层组合头部相机 Tag、腕部相机、足部 landmark/Tag 和可选腕部 IMU，输出 truth/estimate pose tracks。

### 3.3 渲染

```bash
python scripts/render.py --help
python scripts/render.py blenderproc --help
python scripts/render.py isaacsim --help
```

`render.py` 负责生成相机轨迹、SMPL-X cache、Tag 贴图并调用渲染后端。所有输出应写到被忽略的 `outputs/` 或外部数据盘。

### 3.4 可视化

```bash
python scripts/vis.py --help
python scripts/vis_human_motion.py --help
```

## 4. 测试

```bash
python -m unittest discover -s tests
```

重构时共运行 9 项测试，通过 8 项，1 项按原测试条件跳过。

## 5. 配置

- `configs/default_geometry.json`：相机 rig、Tag、关节和默认 motion 参数。
- `configs/realistic_camera.json`：模糊、噪声、曝光/畸变等真实成像退化。

历史源码中的默认路径已从 `config/` 统一更新为 `configs/`。

## 6. 与其他仓库的关系

- 输出的 RGB/metadata 由 `prediction_model_training/scripts/prepare_heatmap_labels.py` 转换为一阶段标签。
- 生成结果可由 `data_processing` 做投影、对齐和可视化。
- 相机参数来源于 `OAK_test/configs/calibration`。

