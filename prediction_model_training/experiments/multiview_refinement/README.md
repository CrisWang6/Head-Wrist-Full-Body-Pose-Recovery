# Manifest-based Multi-view Refinement

独立的 CAM_B/C heatmap refinement 实验。输入 CSV manifest，每行关联两视角 RGB、初始 heatmap、target heatmap 和 visibility。

冒烟测试：

```bash
python experiments/multiview_refinement/smoke_test.py
```

训练：

```bash
python experiments/multiview_refinement/train.py \
  --manifest /data/train_manifest.csv \
  --output-dir /artifacts/multiview \
  --num-joints 12
```

推理：

```bash
python experiments/multiview_refinement/infer.py \
  --manifest /data/val_manifest.csv \
  --checkpoint /artifacts/multiview/best.pt \
  --output-dir /artifacts/multiview/predictions \
  --num-joints 12
```

本实验的 checkpoint 与 `egorear_sim2d.refinement` 不兼容。

