# 第三方依赖

| 依赖 | 用途 | 历史基线 |
|---|---|---|
| PyTorch | 一阶段/二阶段模型和 DataLoader | 2.5.1+cu121 |
| CUDA runtime | PyTorch GPU 后端 | 12.1 wheel runtime |
| NumPy | 标签与指标 | 2.2.6 |
| OpenCV | RGB、heatmap 可视化和视频读取 | 4.13.0 |
| TensorBoard | 训练曲线 | 2.20.0 |
| Pillow | 独立 multiview experiment 图像读取 | 12.2.0 |
| Isaac Sim generation repo | 仿真 metadata/SMPL-X 标签源 | 同级 `Issacsim_data_generation` |
| EgoRear | 模型设计参考 | <https://github.com/hiroyasuakada/EgoRear.git>，历史参考 commit `d9df1e6c26ae98162e4365c4bd109cd1847b8150` |

## 说明

- 当前训练网络是本项目的轻量一阶段实现，不直接依赖完整 EgoRear 上游包。
- 上游 EgoRear 只作为多视角 feature extraction/JQA 设计参考，没有复制进仓库。
- checkpoint、ImageNet 权重、Sapiens 权重和训练数据都不是第三方代码依赖，必须通过外部 artifact/data 目录提供。
- PyTorch wheel 必须根据新主机驱动选择；README 中的 `cu121` 命令用于复现历史环境。

