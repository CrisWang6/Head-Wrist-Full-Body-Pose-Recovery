# Source map

来源：

- `gwj_reinstall_backup_20260724/restored/EgoRear_w_hand` 的源码、脚本和配置
- `gwj_reinstall_backup_20260724/local_sources/stage2_heatmap_refinement`（移至 `experiments/stage2_refinement`，核心 `refinement.py` 并入主包）
- `HearWristCam/test_code/multiview_refinement`（移至 `experiments/multiview_refinement`）

明确排除：

- `data/`
- `checkpoints/`
- `logs/`
- `outputs/`
- label previews
- TensorBoard events
- 远程部署/清理临时脚本及其中的明文凭据
