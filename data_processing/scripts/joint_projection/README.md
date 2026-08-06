# 动捕关节到头部/腕部相机投影

输入固定为：

- `C:\Users\hand\Desktop\Dataset\0717_training_data\aligned_30hz.csv`
- `C:\Users\hand\Desktop\Dataset\0717_training_data\images`

脚本读取 1920×1200 Kalibr `omni + radtan` 内参，把 mocap world joint 变换到相机坐标系，再投影到图像。只显示相机前方 180° 半球且落在图像内的主关节，并用红点和红色骨架连接。

## 运行

```powershell
python C:\Users\hand\Desktop\HearWristCam\test_code\joint_projection\project_joints.py
```

默认每台相机使用固定随机种子抽取 30 张，共 8 台相机、240 张结果。输出目录为 `results`，包含：

- 每台相机的 `seq_XXXXXX_joints.jpg`
- 每台相机的 JSON 和 Markdown 误差报告
- `visible_joint_projections.csv`
- `run_manifest.json`
- `summary.json`

## 坐标与姿态

- mocap 位置按 cm 读取，乘 10 转换为 mm。
- 相机坐标采用 `+z` 向前、`+x` 向右、`+y` 向下。
- `camera_axes_anchor` 完整定义相机三轴在 Head、LeftHand 或 RightHand 坐标系中的方向。
- 头部 CAM_B/C 的位置分别为 `(87, -26, 161)` mm 和 `(-87, -26, 161)` mm；两者均为 `(相机+x, 相机+y, 相机+z)=(Head+x, Head+z, Head-y)`。
- 右腕 CAM_C 已更正为 `(70±10, -102, -43)` mm，三轴为 `(腕+x, 腕+z, 腕-y)`。

## 头部姿态

- 头部相机的位置逐帧使用 mocap `Head` world 位置。
- 头部相机的姿态逐帧使用 mocap `Head` world 四元数。
- module01 IMU 不参与当前版本的头部相机姿态计算。

保留的骨架为左右踝、膝、髋、脊柱、颈、肩、肘和腕。鱼眼图像中的骨骼边由 3D 骨骼加密采样后逐点投影，因此可以呈曲线。

当前没有人工标注的 2D joint 真值，所以报告不能给出绝对 joint 像素误差；报告包含内参重投影误差、时间同步误差和位置外参不确定性传播。
