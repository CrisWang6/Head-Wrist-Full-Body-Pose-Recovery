# 头戴鱼眼相机与腕部 Tag 几何仿真

这个工作区包含一套分阶段的仿真工具，用于测试头戴四鱼眼相机系统对腕部 tag、腕部相机以及脚部目标的观测与位姿恢复。

项目按仿真真实度分层推进：

- Stage 0 几何基线：理想鱼眼投影、射线三角化，不经过图像检测器。
- Stage 1 渲染 tag 检测：轻量 SMPL-X 人体渲染、ArUco tag 图像、双目/单目 fallback 位姿恢复，以及可复用的轨迹导出。
- Stage 2 真实相机输入：可选的图像域传感器和输入退化，例如运动模糊、读出/散粒噪声、眩光、丢帧和类似遮挡的干扰项。
- Stage 3 BlenderProc 场景仿真：在平坦开阔的 Blender 场景中渲染动态 SMPL-X 人体、头戴四相机、左右腕部四相机和真实 ArUco 腕部 tag。

仿真器会把已知 tag 角点投影到多个 220 度鱼眼相机中，将可见像素反投影成射线，并在多视角之间三角化角点，最后通过刚体对齐估计目标位姿。这给后续加入噪声检测或照片级仿真之前提供了一个清晰的可观测性和误差基线。

## 坐标约定

- 世界/头部坐标系：`x` 向右，`y` 向前，`z` 向上。
- 头部关节是头戴相机 rig 的原点。
- 四个头戴相机在头部 `xy` 平面内组成一个 440 mm x 200 mm 的矩形：
  - 前后长度：沿 `y` 方向 440 mm
  - 左右宽度：沿 `x` 方向 200 mm
- 对原始 AMASS/SMPL-X 输入，仿真器会根据肩线和世界向上方向推导头戴 rig 坐标系，而不是直接使用 SMPL-X 的 `Head` 关节旋转。这样可以保持所有头戴相机都看向世界 `-z`。
- 每个头戴相机都刚性连接在该 rig 坐标系上，并沿 rig `-z` 方向观察。
- 相机坐标系：`x` 向右，`y` 在头部坐标系中向下/向后，`z` 为光轴。
- 左右手腕都可以挂载两个可选虚拟相机，相机分辨率和 FOV 与头戴相机一致。它们的光轴直线穿过腕关节，光线方向远离腕关节。两个轴互相垂直，并位于腕部局部 `yz` 平面内，该平面同时也是腕部 tag 平面。

默认相机模型是理想等距鱼眼，完整 FOV 为 220 度，分辨率为 1920 x 1080，默认 30 Hz。

## 项目结构

- `config/default_geometry.json`：默认相机、tag 和仿真设置。
- `config/realistic_camera.json`：Stage 2 相机/输入退化配置。
- `test_motion/`：放置选定 AMASS 或已预处理 motion 文件。
- `smplx_models/`：SMPL-X 模型文件目录。
- `src/geosim/`：仿真核心包。
  - `camera.py`、`geometry.py`、`runner.py`：Stage 0 几何基线。
  - `tag_rig.py`、`motion.py`、`smplx_numpy.py`：共享 motion、人体和 tag 模型。
  - `pose_tracks.py`：共享位姿序列插值、腕部/tag 融合和 `.npz` 轨迹存储。
  - `realistic.py`：Stage 2 图像退化与物理相机参数容器。
  - `imu.py`：腕部 6 轴 IMU 仿真与视觉/IMU 融合。
  - `blenderproc_cache.py`、`blenderproc_runner.py`：Stage 3 BlenderProc 动画缓存和 Blender 侧渲染 runner。
- `scripts/render.py`：渲染任务脚本，内部按子命令组织 `head-tags`、`wrist-views` 和 `blenderproc`。
- `scripts/pipeline.py`：估计 pipeline 脚本，内部按子命令组织 `foot-landmarks` 和 `foot-tag`。
- `scripts/sim.py`：仿真基线脚本，内部按子命令组织 `geometry`。
- `scripts/vis.py`：交互式可视化脚本，内部按子命令组织 `pose-tracks`、`foot-tracks` 和 `rig-frame`。
- `scripts/vis_human_motion.py`：独立的人体 motion 预览脚本，用 SMPL-X 网格和骨架播放给定 `.npz` 动作轨迹。
- `tests/`：基于 Python `unittest` 的轻量测试。

## Motion 输入

runner 支持两种输入格式。

预处理好的 `.npz` motion 文件可以直接提供这些数组：

- `head_pos`：shape `(F, 3)`，单位米
- `left_wrist_pos`：shape `(F, 3)`，单位米
- `left_elbow_pos`：shape `(F, 3)`，单位米

可选数组：

- `head_rot`：shape `(F, 3, 3)`，从头部坐标系到世界坐标系的旋转
- `left_wrist_rot`：shape `(F, 3, 3)`，从腕部坐标系到世界坐标系的旋转
- `fps`：标量

如果有中性 SMPL-X 模型文件，也支持原始 AMASS SMPL-X `.npz` 文件。默认模型路径为 `smplx_models/SMPLX_NEUTRAL_2020.npz`，也可以通过 `--smplx-model` 指定。当前 adapter 是纯 NumPy 实现，使用 shape blend shapes、SMPL-X 运动学树和近似线性 blend skinning。它用于几何测试和轻量可视化，不用于最终照片级人体渲染。

## 快速开始

运行内置 synthetic motion：

```bash
python3 scripts/sim.py geometry --synthetic
```

运行 `test_motion/HumanEva` 及其子目录下所有可用 AMASS `.npz` 文件。如果该目录存在，并且没有提供 motion 参数，它会作为默认输入目录：

```bash
python3 scripts/sim.py geometry --motion-dir test_motion/HumanEva
```

尝试不同 tag 尺寸：

```bash
python3 scripts/sim.py geometry --synthetic --tag-size-m 0.08
```

跳过可视化，只输出更快的统计结果：

```bash
python3 scripts/sim.py geometry --motion-dir test_motion/HumanEva --no-visualization
```

默认 batch run 会打印每个 motion 的摘要，以及按帧数加权的 `Overall` 结果。计算完成后，它会随机选择一个 AMASS motion，并在 `outputs/visualizations/` 下写出一个 10 Hz 可视化视频。这里的 AMASS 文件通常是 120 Hz，所以可视化帧会按渲染 stride 降采样。

视频中包含：

- 近似 SMPL-X 人体网格
- 四个头戴鱼眼相机的位置和朝向
- 两个共面的腕部 tag 方块，tag 平面垂直于小臂
- 头部和腕部轨迹

使用固定随机种子，让可视化选择可复现：

```bash
python3 scripts/sim.py geometry --motion-dir test_motion/HumanEva --visualization-seed 7
```

运行测试：

```bash
python3 -m unittest discover -s tests
```

预览一段 AMASS/SMPL-X 人体动作。该脚本使用 Open3D 实时 viewer，并默认预计算 SMPL-X 顶点以保证播放时更流畅：

```bash
python3 scripts/vis_human_motion.py --motion test_motion/HumanEva/S1/Walking_3_stageii.npz
```

窗口中空格暂停/播放，左右方向键逐帧，`r` 重置视角；鼠标左键旋转，右键平移，滚轮缩放。

如果希望像 GMR 示例那样快速打开窗口并边计算边播放，可以使用简单模式：

```bash
python3 scripts/vis_human_motion.py S1_Walking_3_stageii.npz --simple
```

也可以叠加相机恢复出的当前位姿。默认会自动加载同名 `outputs/wrist_ankle_recovery` 结果，并只用红绿蓝坐标轴显示恢复出的左右手腕和左右踝关节位姿；原始 AMASS truth 坐标轴不会显示。使用 `--speed 0.25` 可以慢速查看细节：

```bash
python3 scripts/vis_human_motion.py \
  S1_Walking_3_stageii.npz \
  --simple \
  --speed 0.25
```

运行手腕和踝关节 tag 的统一恢复 pipeline。该流程会从输入 motion 出发，先用头戴前两个相机和手腕 tag 恢复左右手腕位姿，再用恢复出的腕部相机位姿和踝/脚背 tag 恢复左右踝关节位姿，最后把 wrist/ankle 的 truth 与 estimate 存进同一个 `.npz` 文件：

```bash
python3 scripts/pipeline.py wrist-ankle-tags \
  --motion-dir test_motion/HumanEva \
  --output-dir outputs/wrist_ankle_recovery
```

## 位姿轨迹导出与可视化

头戴相机 tag 渲染器会把恢复出的腕部/tag 位姿和 truth 写入压缩 NumPy 轨迹文件：

```bash
python3 scripts/render.py head-tags \
  --output-dir outputs/pose_tracks \
  --no-video
```

默认轨迹路径为：

```text
outputs/pose_tracks/S1_walk_head_front_left_right_pose_tracks.npz
```

打开交互式 3D 轨迹查看窗口：

```bash
python3 scripts/vis.py pose-tracks outputs/pose_tracks/S1_walk_head_front_left_right_pose_tracks.npz
```

左键拖拽旋转，右键拖拽平移，滚轮缩放。添加 `--show-tags` 可以额外显示 tag 中心轨迹。

用可自由旋转的 3D 窗口检查某一帧 SMPL-X 人体网格、头戴相机、腕部相机和腕部 tag：

```bash
python3 scripts/vis.py rig-frame --frame 1600
```

从两个腕部相机渲染默认 S1 walking motion，并输出一个左右拼接的 1080p 视频：

```bash
python3 scripts/render.py wrist-views
```

运行四视角脚部 landmark 位姿 pipeline，并打开输出轨迹：

```bash
python3 scripts/pipeline.py foot-landmarks
python3 scripts/vis.py foot-tracks outputs/four_view_foot_pipeline/S1_walk_four_view_foot_tracks_realistic_imu.npz
```

脚部 landmark pipeline 默认会仿真 6 轴腕部 IMU，并在放置腕部相机前把它和头戴相机 tag 估计结果融合。添加 `--no-wrist-imu` 可以跑纯视觉腕部位姿 ablation。

如果只想隔离腕部相机，可以只使用两个腕部视角并只保存左脚轨迹：

```bash
python3 scripts/pipeline.py foot-landmarks \
  --view-set wrist \
  --foot left \
  --output outputs/four_view_foot_pipeline/S1_walk_wrist_only_left_foot_tracks_realistic.npz
```

测试完整的头部到腕部再到脚部传播链，并在左脚脚背上使用一个仿真 tag。头戴前两个相机恢复腕部位姿，仿真腕部 IMU 辅助这些估计，腕部相机再恢复脚部 tag：

```bash
python3 scripts/pipeline.py foot-tag
python3 scripts/vis.py foot-tracks outputs/foot_tag_pipeline/S1_walk_wrist_foot_tag_left_tracks.npz --show-landmarks
```

使用 `--wrist-set left`、`--wrist-set right` 或 `--wrist-set both` 可以选择两个或四个腕部相机。

在不改变 Stage 0/1 clean baseline 的情况下启用 Stage 2 相机/输入退化：

```bash
python3 scripts/render.py head-tags \
  --realistic-config config/realistic_camera.json \
  --output-dir outputs/camera_views_realistic
```

这样可以保留干净的几何和渲染 tag pipeline，同时把更真实的物理相机条件作为独立阶段测试。

## BlenderProc 场景渲染

BlenderProc 需要在 `camtest` 环境中安装命令行入口。当前项目使用本地 `/home/gaoweijian/BlenderProc` checkout：

```bash
/home/gaoweijian/miniforge3/envs/camtest/bin/python -m pip install -e /home/gaoweijian/BlenderProc
```

渲染默认 S1 walking motion。该命令会先按 `--output-fps` 对 AMASS 源数据降采样，只对采样帧计算 SMPL-X 顶点、8 个相机位姿和左右腕部 tag 角点，再通过 `blenderproc run` 在 Blender 4.2 中渲染 8 条 30 Hz MP4：

```bash
/home/gaoweijian/miniforge3/envs/camtest/bin/python scripts/render.py blenderproc \
  --motion test_motion/HumanEva/S1/Walking_3_stageii.npz \
  --output-dir outputs/blenderproc/Walking_3_stageii \
  --width 1920 \
  --height 1080 \
  --output-fps 30
```

默认 `--head-frame smplx_relative` 会用第一帧校准头戴设备坐标系，之后跟随 SMPL-X `Head` 关节的相对旋转。`--width/--height` 表示最终视频尺寸；鱼眼内部会先用 `width x width` 方形 sensor 渲染完整圆形视场，再中心裁剪到 `width x height`。例如 `1920 x 1080` 输出对应 `1920 x 1920` sensor crop。最终视频默认用 `mp4 + libx264` 输出，`--video-crf` 控制质量，数值越低越清晰，默认 `16`。

在 GPU 渲染下，`--parallel-cameras 0` 会自动检测可见 GPU 数并并行渲染相机。8 张 4090 的机器可以显式指定 8 路并行：

```bash
/home/gaoweijian/miniforge3/envs/camtest/bin/python scripts/render.py blenderproc \
  --motion test_motion/HumanEva/S1/Walking_3_stageii.npz \
  --output-dir outputs/blenderproc/Walking_3_stageii \
  --width 1920 \
  --height 1080 \
  --output-fps 30 \
  --parallel-cameras 8 \
  --gpu-ids 0,1,2,3,4,5,6,7
```

每路相机会启动一个独立 BlenderProc/Blender 进程，日志写入 `logs/<camera_name>.log`。

输出目录中包含：

- `Walking_3_stageii_head_front_left.mp4`
- `Walking_3_stageii_head_front_right.mp4`
- `Walking_3_stageii_head_back_left.mp4`
- `Walking_3_stageii_head_back_right.mp4`
- `Walking_3_stageii_left_wrist_palm_normal.mp4`
- `Walking_3_stageii_left_wrist_forward.mp4`
- `Walking_3_stageii_right_wrist_palm_normal.mp4`
- `Walking_3_stageii_right_wrist_forward.mp4`
- `blenderproc_motion_cache.npz`、`metadata.json` 和两个 ArUco tag 贴图

调试时建议先限制帧数和分辨率，例如：

```bash
/home/gaoweijian/miniforge3/envs/camtest/bin/python scripts/render.py blenderproc \
  --max-output-frames 30 \
  --width 640 \
  --height 360 \
  --samples 8 \
  --parallel-cameras 8 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --output-dir outputs/blenderproc/Walking_3_stageii_30f
```
