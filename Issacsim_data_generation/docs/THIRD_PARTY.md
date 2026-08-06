# 第三方依赖

| 依赖 | 用途 | 历史基线/来源 |
|---|---|---|
| NumPy | 几何、SMPL-X、相机与 pose tracks | 2.2.6 |
| OpenCV | Tag、图像和鱼眼工具 | 4.13.0 |
| Open3D | 3D 可视化 | 可选 `visualization` extra |
| BlenderProc | Blender 场景构建和 MP4 渲染 | 2.8.0，<https://github.com/DLR-RM/BlenderProc> |
| Blender | BlenderProc 后端 | 历史 4.2 |
| Isaac Sim | GPU 多视角渲染 | NVIDIA 官方安装，版本需与新驱动匹配 |
| SMPL-X | 人体 mesh/关节 | `SMPLX_NEUTRAL_2020.npz`，受模型许可约束 |
| HumanEva/AMASS | motion 输入 | 外部数据集，受各自许可约束 |

## 不进入 Git 的依赖

- BlenderProc 官方 checkout；
- Isaac Sim/Kit 安装目录和 cache；
- SMPL-X 模型；
- HumanEva/AMASS motion；
- Isaac Lab/Docker 镜像；
- 纹理、mesh、生成视频和 motion cache。

这些依赖使用固定外部目录或 CLI 参数接入。不要把其许可证受限文件提交到公开仓库。

