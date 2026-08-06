# CAM_B/C Stage-2 Heatmap Refinement

该实验冻结一阶段网络，使用每个 view 的初始 heatmap peak 作为 joint anchor，在 CAM_B/C 特征图中采样可学习 offset，经 joint-query adaptation 融合后预测两视角 residual heatmap。

核心网络位于仓库主包：

```text
src/egorear_sim2d/refinement.py
```

训练：

```bash
python experiments/stage2_refinement/scripts/train_refinement.py \
  --label-root /data/labels \
  --stage1-checkpoint /artifacts/stage1/best.pt \
  --output-dir /artifacts/stage2 \
  --log-dir /artifacts/stage2_logs \
  --proposal-source stage1 \
  --epochs 20 --batch-size 8 --device cuda
```

一阶段尚未准备好时，可以用 `--proposal-source noisy_gt` 冒烟测试数据与网络。

评估：

```bash
python experiments/stage2_refinement/scripts/test_refinement.py --help
```

`configs/stage2_head_bc_refinement.json` 是历史 16 关节配置；接入 0722 的 12 通道模型前必须统一 joint layout 与 checkpoint metadata。

