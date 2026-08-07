# 关键点预测训练代码复核记录

验证日期：2026-08-07
验证范围：`prediction_model_training`
验证环境：Linux / Python 3.12 / PyTorch 2.13.0+cpu（仅用于结构与流程验证，非训练基线环境）

## 1. 结论

代码本身是完整可跑的：stage1 → stage2 → stage3 三阶段以及配套的划分、评估、可视化脚本全部齐全，没有缺失模块或断裂的引用，在合成数据上可以端到端跑通。

需要注意的是，本仓库是 4090 主机 `/home/gaoweijian/EgoRear_w_hand` 的一份代码导出，两边没有共同的 Git remote，因此"是否与远程主机逐字节一致"无法在仓库内自证，需要按第 5 节的方式核对。

## 2. 静态检查

- Python 语法：26 个脚本 + 7 个包模块，`compileall` 错误 0。
- Shell 语法：4 个 preset，`bash -n` 错误 0。
- 内部 import：`egorear_sim2d` 的 `dataset` / `labels` / `model` / `pose3d` / `refinement` / `splits` 全部存在并可导入。
- 跨仓库依赖：`labels.py` 需要的 `geosim.smplx_numpy` 在同级 `Issacsim_data_generation/src/geosim/` 中存在。
- 入口自检：全部 25 个可执行脚本 `--help` 通过。
- preset 与 argparse 的一致性：4 个 `.sh` 中调用的全部命令行参数都能在对应脚本的 argparse 中找到，未知参数 0。
- 配置默认值：`prepare_real_head_heatmap.py` 默认引用的两个标定文件都在 `configs/calibration/head/` 中存在。

## 3. 端到端流程验证

用 40 帧、2 相机（CAM_B/CAM_C）、12 关节的合成 label NPZ 与合成 3D label，在 CPU 上完整跑通：

| 步骤 | 结果 |
|---|---|
| `create_dataset_split.py`（random，seed 42） | 32/4/4 划分，manifest 写出成功 |
| `train_heatmap.py`（2 epoch） | `best.pt` / `last.pt` / TensorBoard / `training_status.json` 正常 |
| `train_refinement.py`（1 epoch） | `best.pt` / `history.jsonl` / `stage1_baseline.json` 正常 |
| `train_pose3d.py`（1 epoch） | `best.pt` / `best_metrics.json` / per-joint MPJPE 正常 |
| `test_heatmap.py` | 见第 4 节，修复后通过 |
| `test_refinement.py` | summary.json 正常 |
| `visualize_direct2d_validation.py` | 单帧图与 contact sheet 正常 |
| `visualize_random_split_test.py` | stage1/2/3 对比图与 `test_metrics.json` 正常 |
| `visualize_pose3d_video.py` | mp4 与 json 正常 |
| `build_split_comparison_report.py` | Markdown + 4 个 CSV 正常 |
| `experiments/multiview_refinement/smoke_test.py` | 通过 |

合成数据只用于验证形状、流程和产物，指标数值没有意义。

## 4. 本次发现并修复的问题

1. `test_heatmap.py` 的样例可视化按八相机仿真机位写死了 4×2 网格，在当前两相机的 CAM_B/CAM_C 数据上 `np.hstack(panels[4:])` 会抛 `need at least one array to concatenate`，导致整个评估在写图时中断。已改为按实际视角数平铺。
2. `run_real_0722_01_random_split_stages123.sh` 和 `run_real_0722_01_strided_stages123.sh` 写死了 `cd /home/gaoweijian/EgoRear_w_hand` 和 miniforge 解释器路径，只能在 4090 主机上运行。已改为从脚本位置推导仓库根目录，并允许用 `REPO_ROOT`、`PYTHON`、`LABEL_ROOT`、`SPLIT_MANIFEST`、`CUDA_VISIBLE_DEVICES` 覆盖，原值保留为默认。
3. `visualize_pose3d_video.py` 只能按 `--train-ratio` 重新做时间顺序 80/20 划分，与 stage1/2/3 实际使用的 split manifest 不一致，导出的 test 视频可能包含训练帧。已增加 `--split-manifest`，复用训练时的同一划分。
4. `docs/CODE_CATALOG.md` 和 `README.md` 停留在 stage2：`pose3d.py`、`splits.py`、整个 `experiments/stage3_pose3d`、`prepare_pose3d_labels.py`、`create_dataset_split.py`、`visualize_random_split_test.py`、`build_split_comparison_report.py` 和两个 stage1-3 preset 都没有记录，README 还把 stage2 描述成 16 关节。已补齐。

根 `VERIFICATION.md` 是 2026-07-24 的快照，其中"`/home/gaoweijian` 可执行代码路径：0"这一条在 2026-08-06 那次提交后已经不成立；本次修复后重新成立。

## 5. 与 4090 主机的同步状态

远程训练主机记录见 `data_processing/docs/weekly_pose_recovery_20260806/02_4090_SSH与远程处理记录.md`：

```text
gaoweijian@192.168.20.221 (gpu222)
远程仓库目录：/home/gaoweijian/EgoRear_w_hand
```

该地址是内网地址，本仓库也没有指向该主机的 Git remote，因此仓库内无法验证两边是否一致。在能访问该主机的机器上按下面的方式核对：

```bash
# 只比较源码，忽略数据、产物和虚拟环境
rsync -avn --delete \
  --include='*/' \
  --include='*.py' --include='*.sh' --include='*.json' --include='*.yaml' --include='*.toml' --include='*.md' \
  --exclude='*' \
  --exclude='.venv/' --exclude='__pycache__/' \
  --exclude='data/' --exclude='checkpoints/' --exclude='logs/' --exclude='outputs/' \
  gaoweijian@192.168.20.221:/home/gaoweijian/EgoRear_w_hand/ \
  ./prediction_model_training/
```

`-n` 是 dry-run，只列差异不改文件。逐文件确认可以用哈希：

```bash
ssh gaoweijian@192.168.20.221 \
  "cd /home/gaoweijian/EgoRear_w_hand && find . -name '*.py' -not -path './.venv/*' -not -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum" \
  > /tmp/remote_hashes.txt
```

核对时注意两点：远程目录名是 `EgoRear_w_hand` 而不是 `prediction_model_training`；`data/`、`checkpoints/`、`logs/`、`*.npz`、`*.pt`、`*.csv` 按 `.gitignore` 本来就不入库，出现差异属于预期。

## 6. 仍需在 4090 环境确认的事项

本次验证用的是 CPU PyTorch 2.13，与 README 记录的训练基线（PyTorch 2.5.1+cu121、双 RTX 4090）不同，因此下面几项没有覆盖：

- 双卡 `DataParallel` 路径与显存占用；
- CUDA 下的实际训练收敛与吞吐；
- 真实 H.265 拆帧链路（`extract_direct_2d_frames.py`）依赖的解码器行为；
- 用真实 checkpoint 复跑 `visualize_pose3d_video.py --split-manifest` 后的 test 集指标，需与训练时记录的 `test_metrics.json` 对齐。
