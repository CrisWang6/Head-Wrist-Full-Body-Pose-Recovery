# 代码功能目录

## CLI

| 文件 | 功能 |
|---|---|
| `scripts/sim.py` | 几何仿真入口；读取 synthetic 或 motion 输入并运行头部/腕部相机与 Tag 几何。 |
| `scripts/pipeline.py` | 组合 wrist/ankle/foot pipeline、IMU 融合和 truth/estimate pose track 输出。 |
| `scripts/render.py` | 生成多相机轨迹和人体 cache，调度 BlenderProc/Isaac Sim 渲染。 |
| `scripts/vis.py` | rig、Tag、pose track 和足部 pipeline 的 3D 可视化入口。 |
| `scripts/vis_human_motion.py` | 查看 AMASS/HumanEva motion、关节和 mesh 动画。 |

## `src/geosim`

| 文件 | 功能 |
|---|---|
| `__init__.py` | `geosim` 包初始化。 |
| `config.py` | 解析 JSON，构建带类型的仿真配置。 |
| `linalg.py` | 旋转、齐次变换和数值线性代数基础函数。 |
| `camera.py` | 相机模型、坐标变换、投影和可见性。 |
| `geometry.py` | 人体关节、相机 rig、Tag 角点和几何仿真主逻辑。 |
| `motion.py` | 加载、采样和标准化 AMASS/HumanEva motion。 |
| `smplx_numpy.py` | 使用 NumPy 读取/计算 SMPL-X 模型数据。 |
| `tag_rig.py` | 头部、腕部和足部 Tag 的刚体定义与位姿。 |
| `imu.py` | 腕部 IMU 测量仿真、噪声和姿态融合。 |
| `realistic.py` | 相机/输入退化模型和 realistic 配置。 |
| `pose_tracks.py` | truth/estimate 位姿轨迹结构、保存和误差数据。 |
| `appearance.py` | 人物/材质 appearance 组合和确定性采样。 |
| `blenderproc_cache.py` | 把人体、相机和 Tag 数据序列化为 BlenderProc motion cache。 |
| `blenderproc_runner.py` | 构建并执行 BlenderProc 渲染命令，整理输出。 |
| `isaacsim_runner.py` | 在 Isaac Sim 中构建场景、相机、人体和批量渲染任务。 |
| `runner.py` | 几何仿真任务编排和公共 runner。 |
| `visualization.py` | Open3D/Matplotlib 可视化辅助。 |

## 测试

| 文件 | 功能 |
|---|---|
| `tests/test_geometry_sim.py` | 验证相机、Tag、motion、pose track 和几何 pipeline 的核心不变量。 |

