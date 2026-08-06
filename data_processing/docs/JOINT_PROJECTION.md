# 关节投影说明

`scripts/joint_projection` 将 mocap world joints/rigid bodies 变换到头部或腕部相机坐标系，再用 Kalibr `omni-radtan` 模型投影到鱼眼图像。

## 输入

- 对齐后的 30 Hz CSV；
- 相机帧或 H.265 视频；
- 相机内参 JSON；
- CAM_B/C camchain YAML；
- 头部/腕部刚体姿态；
- 可选 Sapiens/RTMPose 2D 关键点与 BVH 骨架。

## 运行

```bash
python scripts/joint_projection/project_joints.py --help
```

历史实验配置与脚本放在同一目录，便于保留 `Path(__file__).parent` 的默认查找行为。建议复制配置到 Git 外，修改输入和输出路径后执行。

## 坐标约定

- mocap 位置按原 CSV 单位读取，并在配置中明确转换到米或毫米；
- 相机坐标通常使用 `+z` 向前、`+x` 向右、`+y` 向下；
- CAM_B/C 相对变换来自 Kalibr camchain；
- 鱼眼投影只接受相机前方且落入有效图像范围的点；
- skeleton edge 会在 3D 线段上加密采样后逐点投影，因此鱼眼图像中的骨架线可以呈曲线。

## 0722 Hybrid 流程

```text
Sapiens/RTMPose 2D
  + CAM_B/C 三角化上肢
  + mocap/BVH 躯干和下肢
  + 头部/腕部刚体约束
  + 时间偏移校准
  → hybrid 3D skeleton
  → CAM_B/C fish-eye projection
  → module01_cam_bc_hybrid_skeleton_2d.csv
```

该 CSV 是 `prediction_model_training` 直接 2D ground-truth pipeline 的输入。

