# EgoRear Sim2D Heatmap Training

这个仓库独立于 `Simulation`，用于第一阶段 2D heatmap 训练。

目标：

- 输入：8 路相机 RGB 视频帧。
- 输出：每路相机的 2D joint heatmap，默认 `114x64`。
- 头部相机监督 15 个关节：踝、膝、髋、腰椎、肩、肘、腕。
- 腕部相机监督 7 个关节：踝、膝、髋、Spine1。
- loss：MSE。腕部相机未使用的关节通道通过 mask 排除。

## 生成 Label Cache

默认使用刚刚生成的 Isaac 数据：

```bash
/home/gaoweijian/miniforge3/envs/camtest/bin/python scripts/prepare_heatmap_labels.py \
  --simulation-root /home/gaoweijian/Simulation \
  --render-root /home/gaoweijian/Simulation/outputs/isaacsim_thuman/S2_Walking_3_stageii \
  --output-dir data/labels/S2_Walking_3_stageii
```

每个 appearance 会生成一个 `heatmap_labels_114x64.npz`，里面只存 2D 坐标、可见性和视频路径；heatmap 在训练时即时生成，避免缓存膨胀。

## 训练

`camtest` 当前如果还没有 PyTorch，需要先安装：

```bash
/home/gaoweijian/miniforge3/envs/camtest/bin/python -m pip install torch opencv-python
```

然后跑一个小训练：

```bash
/home/gaoweijian/miniforge3/envs/camtest/bin/python scripts/train_heatmap.py \
  --label-root data/labels/S2_Walking_3_stageii \
  --epochs 5 \
  --batch-size 4 \
  --device cuda
```

输出 checkpoint 默认保存到 `checkpoints/`。
