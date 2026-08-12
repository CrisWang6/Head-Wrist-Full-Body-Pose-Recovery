# 生产 Pipeline：双外部双目 + 头显鼻尖对齐（nose_offset_opt）

本文件描述**唯一固定的生产配方**，对应已验收产出例如：

`0806/无/multiview_3d_results/full/head_reprojection/nose_offset_opt/head_CAM_A_nose_offset_opt.mp4`

新数据时用户在对话中粘贴路径；agent 运行时填入路径，**不要**为每个数据集改本文件。

**弃用**：`ablation/` 下的多 cell 消融矩阵（S0_B* + S1_* + S2_W* 组合搜索）。消融结论为「多方法组合 refinement 效果不佳」；生产环境**只跑本文固定配方**，不再切换 filter profile、骨长软约束权重或 GT 开关组合。

执行顺序：

```mermaid
flowchart LR
  A[A 时间戳对齐] --> B[B 四相机 2D 检测]
  B --> C[C 三角化→3D骨架]
  C --> D[D 头显鼻尖固定 UV]
  D --> E[E 肢端 GT + 头显优化 + 渲染]
```



各昂贵阶段：若产出已存在且上游（尤其 aligned / manifest）有效，则跳过该阶段。

背景与坐标约定见 `data_processing/docs/PIPELINE_PLAN_0806_踝腕.md`（Stage A 同步硬约束、相机↔刚体映射、交付关键点规则）。本文只写**生产默认参数**、**gwj 执行**与**交付物**。

---



## 生产固定配方（参数一览）


| 环节               | 生产值                          | 说明                                                                                                                         |
| ---------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **2D 滤波**        | `default` profile            | `process_external_multiview_3d.py` 内置 `FILTER_2D`（median 半径 + zero-phase EMA α）；**不**传 ablation `--filter-2d-profile`      |
| **3D 时域平滑**      | Gaussian **σ = 1.0** 帧       | `replace_limb_mocap_gt.py --temporal-smooth-sigma` 默认值（脚本未显式传参即 1.0）；0 = 关闭                                                |
| **鼻尖对齐**         | **开启**（逐帧）                   | `replace_limb_mocap_gt.py --mode per_frame`，将外部骨架鼻点对齐到 CH3-08 tip `[0,-15,-125] mm`                                        |
| **肢端 GT**        | **关闭**（不论哪个数据集都直接用三角化3d骨架结果） | 硬替换腕或踝 tip；z 偏移见下表                                                                                                         |
| **头显 2D refine** | **Layer1 + Layer2**          | Layer1：3D 刚性 tip 残余平移；Layer2：对 **固定** RTMW 鼻尖 UV（Stage D 采样+robust median）做单次刚性平移 GN refine；**不**加 `--skip-head-2d-refine` |
| **骨长软约束**        | **关闭**                       | 生产脚本**不传** `--bone-soft-config`；`soft_shoulder_weight` 默认 0                                                                |
| **踝刚体→脚尖**       | **禁止**                       | `ankle_rigid_to_toe_constraint: false`（永不从踝刚体朝向推脚尖/膝）                                                                      |




### 2D 滤波 `default` 表（`process_external_multiview_3d.FILTER_2D`）

每关节 `(median_radius_frames, zero_phase_ema_alpha)`。未列出的关节（含 **鼻尖**）回退为 **(2, 0.27)**。


| 关节组            | radius | α    |
| -------------- | ------ | ---- |
| 肩、髋            | 5      | 0.12 |
| 肘              | 3      | 0.20 |
| 膝              | 2      | 0.26 |
| 腕              | 2      | 0.28 |
| 踝              | 2      | 0.40 |
| 大趾 / toe alias | 2      | 0.45 |


三角化使用 `filtered_uv` 观测；头显侧鼻尖在 Stage D 单独 RTMW 检测（**固定 UV**，不经此表、不逐帧抖动）。

### 头显优化 Layer1 + Layer2（大道至简）

头显相机与鼻子在物理上相对固定 → Stage D 仅对少量均匀采样帧做 RTMW 检测，robust median 得到 **每路相机一个固定 2D 鼻尖 UV**，全序列复用；Layer2 在该固定观测上 refine **同一**刚性平移，而非逐帧检测 + inlier 筛选。

1. **Layer1**（`optimize_multiview_head_nose_offset.py --mode per_frame`）：外部 multiview 鼻尖 vs 头显刚体 CH3-08 固定 tip `[0,-15,-125] mm` 的 3D 刚性平移基线。per_frame 模式下上游 `replace` 已做逐帧 world 鼻对齐，Layer1 拟合的是头显刚体坐标系内的**残余平移**（多帧 robust median → 单一 offset）。
2. **Layer2**（默认开启，`refine_fixed_2d_offset`）：以 Layer1 offset 为初值，对头显 CAM_A/D 上 **固定** RTMW 鼻尖 UV（来自 `.fixed.json` / 全 timeline 恒定 csv）做小幅 Gauss-Newton refine（最多 400 姿态对、6 步、每步 delta clip ±20 mm）。**不使用**逐帧 RTMW 检测或 per-frame inlier mask。成功时 `chosen_offset_source` 为 `layer1_3d_gt_rigid_tip+fixed_rtmw_2d_refine`；Layer2 未应用时回退 `layer1_3d_gt_rigid_tip`。

腕 / 踝 / 无 三套 full 产物的 `report.json` 生产路径均为 `layer1_3d_gt_rigid_tip+fixed_rtmw_2d_refine`（见文末交付表）。

---



## 数据集差异：腕 vs 踝 vs 无


| 项                                 | **腕**                                         | **踝**                                         | **无**（标定/参考）                            |
| --------------------------------- | --------------------------------------------- | --------------------------------------------- | --------------------------------------- |
| 本地目录                              | `0806/腕/`                                     | `0806/踝/`                                     | `0806/无/`                               |
| Config                            | `configs/0806_wrist_dual_external_mocap.json` | `configs/0806_ankle_dual_external_mocap.json` | `configs/0806_dual_external_mocap.json` |
| 头显子目录                             | `0712_032704`                                 | `0712_033034`                                 | `0712_033709`                           |
| `replace_limb_mocap_gt --replace` | `wrist`                                       | `ankle`                                       | **跳过**（无肢端 GT 替换）                       |
| 肢端 z（刚体局部 `[0,0,z]`）              | **−60 mm**（CH3-06/07）                         | **−80 mm**（CH3-06/07）                         | —                                       |
| 硬替换关节                             | `left/right_wrist`                            | `left/right_ankle`                            | 无                                       |
| 保留三角化                             | 踝 + 脚尖                                        | 腕 + 脚尖                                        | 全部肢端三角化                                 |
| Stage E 输入骨架                      | `pre_limb` → `limb_gt`                        | 同左                                            | `pre_limb` 直接进 optimize                 |
| 远端一键脚本                            | `run_0806_limb_dataset.sh wrist`              | `run_0806_limb_dataset.sh ankle`              | 见下方「无数据集手动命令」                           |


鼻尖 GT、头显 `xy_swap`、禁止 remux mp4 同步等规则三数据集相同。

---



## Stage A — 时间戳对齐

与 `data_processing/docs/PIPELINE_PLAN_0806_踝腕.md` Stage A 相同：曝光结束时间主轴、头显 `.h265`+`bytes`、manifest 回查真实 `frame_index`。

产出（相对 `data_root`）：

- `aligned_data/aligned_30hz.csv`
- `multiview_3d_results/aligned_manifest.jsonl`
- `multiview_3d_results/aligned_manifest_report.json`（四路 2D 可视化需要）

---



## Stage B — 四相机 2D 检测


| 项   | 内容                                                                             |
| --- | ------------------------------------------------------------------------------ |
| 脚本  | `infer_rtmpose_candidates.py`                                                  |
| 参数  | `--mode performance --device cuda --backend onnxruntime --rotate-180`          |
| 并行  | module01→GPU0，module02→GPU1；每模组 CAM_A/D 可并行                                    |
| 输出  | `inference/module{01,02}_CAM_{A,D}.jsonl`（须含 `left_big_toe` / `right_big_toe`） |


---



## Stage C — 多相机三角化 → 3D 骨架


| 项     | 内容                                                                                 |
| ----- | ---------------------------------------------------------------------------------- |
| 脚本    | `process_external_multiview_3d.py` + `merge_multiview_chunks.py`                   |
| 2D 滤波 | 内置 `default`（见上表），**无需额外 CLI**                                                     |
| 并行    | **8** chunk；`--context-frames 10`                                                  |
| 输出    | `full/multiview_3d_results.jsonl` → 复制为 `full/multiview_3d_results_pre_limb.jsonl` |


质量门控跟 config `quality.*`（置信度 0.16、最小射线角 0.5°、最大重投影 35 px 等）。

---



## Stage D — 头显鼻尖 2D 检测（固定 UV）


| 项    | 内容                                                                                                                         |
| ---- | -------------------------------------------------------------------------------------------------------------------------- |
| 输入   | `{data_root}/{head_dir}/module01_*_CAM_{A,D}.h265` + `timestamps.csv`                                                      |
| 脚本   | `detect_head_nose_rtmw.py --camera CAM_A` / `CAM_D`                                                                        |
| 默认参数 | `--sample-count 48`（均匀采样 48 帧检测；`0` = 逐帧检测，仅消融/调试）                                                                         |
| 算法   | 对采样帧 RTMW WholeBody 68 点脸鼻尖（face index 30）检测 → radial MAD 去离群 → robust median 得 **一个** `fixed_uv_px` → 展开为全 timeline 恒定 UV |
| 输出   | `full/head_reprojection/nose_offset_opt/head_CAM_{A,D}_rtmw_nose.csv`（每帧相同 `face_nose_u/v_px`，`fixed_nose=1`）              |
| 元数据  | 同名 `head_CAM_{A,D}_rtmw_nose.fixed.json`（`schema: joint_projection.head_rtmw_fixed_nose.v1`，含 `fixed_uv_px`、采样统计）          |
| 跳过条件 | csv **且** `.fixed.json` 均存在且非空（`run_0806_limb_dataset.sh`）                                                                 |


双 GPU 可对 CAM_A / CAM_D 并行。生产脚本显式传 `--sample-count 48`。

**大道至简**：头显模组与佩戴者鼻尖相对位置在整个采集过程中近似恒定；逐帧 RTMW 鼻尖会抖动并污染 Layer2/渲染。故只检测少量代表帧、取稳健中心，全序列使用同一 2D 鼻尖；渲染中绿色圆圈为该固定 UV，优化后的青色鼻尖投影应与绿圈 **帧间重合、无抖动**。

---



## Stage E — 肢端 GT + 头显优化 + 渲染



### E.1 原始三角化可视化（所有数据集）

```bash
python export_playback_from_jsonl.py \
  --results full/multiview_3d_results_pre_limb.jsonl \
  --output full/skeleton_playback_raw.json \
  --source "raw triangulated methods.filtered.multiview (pre limb-GT)" \
  --prune

python render_skeleton_yaw_video.py \
  --data full/skeleton_playback_raw.json \
  --output full/visualization/skeleton_3d_raw_yaw.mp4 \
  --yaw-deg 100 --pitch-deg 18

python render_external_multiview_results.py \
  --results full/multiview_3d_results_pre_limb.jsonl \
  --manifest-report multiview_3d_results/aligned_manifest_report.json \
  --manifest multiview_3d_results/aligned_manifest.jsonl \
  --video-root <input_root> \
  --config <CONFIG> \
  --output-dir full/visualization
# 复制 four_view_reprojection.mp4 → external_4cam_2d_skeletons.mp4
```



### E.2 肢端 GT 替换（仅腕 / 踝）

`run_0806_limb_dataset.sh` 调用等价于：

```bash
python replace_limb_mocap_gt.py \
  --results full/multiview_3d_results_pre_limb.jsonl \
  --aligned aligned_data/aligned_30hz.csv \
  --output full/multiview_3d_results_limb_gt.jsonl \
  --replace wrist|ankle \
  --z-offset-mm -60|-80 \
  --mode per_frame \
  --playback-output full/skeleton_playback.json
# 默认 --temporal-smooth-sigma 1.0；不传 --bone-soft-config / --skip-nose-align / --skip-limb-replace

cp full/multiview_3d_results_limb_gt.jsonl full/multiview_3d_results.jsonl
```

逐帧逻辑（`status==1` 且 `raw_tick_valid==1` 时）：

1. 平移整骨架使外部 `nose` → CH3-08 tip `[0,-15,-125] mm`
2. 硬写入左/右腕（腕集）或踝（踝集）tip：`p = R @ [0,0,z] + t`
3. Gaussian σ=1.0 时域平滑（**跳过**已标记 `rigid_local` 的硬 GT 关节）
4. `prune_joints_inplace` → delivery 关键点（鼻-only 面部、每脚一个大趾）

**无**数据集：跳过本节；optimize 读 `skeleton_playback_raw.json` 或从 `pre_limb` 导出的 playback。

### E.3 头显鼻尖优化 + 并行渲染

```bash
python optimize_multiview_head_nose_offset.py \
  --data-root "$DATA_ROOT" \
  --config "$CONFIG" \
  --skeleton-playback full/skeleton_playback.json \
  --head-a-nose-csv full/head_reprojection/nose_offset_opt/head_CAM_A_rtmw_nose.csv \
  --head-d-nose-csv full/head_reprojection/nose_offset_opt/head_CAM_D_rtmw_nose.csv \
  --output-dir full/head_reprojection/nose_offset_opt \
  --mode per_frame \
  --skip-render

python render_nose_offset_parallel.py \
  --prepare \
  --data-root "$DATA_ROOT" \
  --config "$CONFIG" \
  --output-dir full/head_reprojection/nose_offset_opt \
  --head-a-nose-csv full/head_reprojection/nose_offset_opt/head_CAM_A_rtmw_nose.csv \
  --head-d-nose-csv full/head_reprojection/nose_offset_opt/head_CAM_D_rtmw_nose.csv \
  --before-playback full/skeleton_playback_raw.json \
  --after-playback full/skeleton_playback.json \
  --report full/head_reprojection/nose_offset_opt/report.json

# 24 路并行 chunk，再 merge
for i in $(seq 0 23); do
  python render_nose_offset_parallel.py \
    --data-root "$DATA_ROOT" --config "$CONFIG" \
    --output-dir full/head_reprojection/nose_offset_opt \
    --head-a-nose-csv full/head_reprojection/nose_offset_opt/head_CAM_A_rtmw_nose.csv \
    --head-d-nose-csv full/head_reprojection/nose_offset_opt/head_CAM_D_rtmw_nose.csv \
    --chunk "$i" 24 &
done
wait
python render_nose_offset_parallel.py --merge \
  --data-root "$DATA_ROOT" --config "$CONFIG" \
  --output-dir full/head_reprojection/nose_offset_opt \
  --head-a-nose-csv full/head_reprojection/nose_offset_opt/head_CAM_A_rtmw_nose.csv \
  --head-d-nose-csv full/head_reprojection/nose_offset_opt/head_CAM_D_rtmw_nose.csv
```

渲染读头显帧走 `.h265`+`bytes`（`H265CaptureReader`）；默认 **24** render chunk。面部绘制策略：只画鼻尖，隐藏眼/耳。优化后画面（AFTER）：青色为投影鼻尖；**绿色圆圈**为 Stage D 固定 RTMW UV（`fixed_nose_uv_from_csv`），全片恒定；验收时青点应与绿圈帧间对齐、无 jitter。

---



## 远端执行（gwj）


| 项        | 值                                                                  |
| -------- | ------------------------------------------------------------------ |
| 主机       | `gaoweijian@192.168.20.221`                                        |
| Conda 环境 | `sapiens2`（`/home/gaoweijian/miniforge3/envs/sapiens2/bin/python`） |
| 批处理根     | `/home/gaoweijian/0806_batch`                                      |
| 代码       | `/home/gaoweijian/0806_batch/repo/test_code/joint_projection`      |
| 腕数据根     | `/home/gaoweijian/0806_batch/wrist/data_root`                      |
| 踝数据根     | `/home/gaoweijian/0806_batch/ankle/data_root`                      |




### 腕 / 踝一键（A→E）

```bash
ssh gaoweijian@192.168.20.221
source ~/miniforge3/etc/profile.d/conda.sh && conda activate sapiens2
export LD_LIBRARY_PATH=/home/gaoweijian/miniforge3/envs/sapiens2/lib/python3.12/site-packages/nvidia/cu13/lib:/home/gaoweijian/miniforge3/envs/sapiens2/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
cd /home/gaoweijian/0806_batch/repo/test_code/joint_projection
bash run_0806_limb_dataset.sh wrist   # 或 ankle
```

`run_0806_limb_dataset.sh` 行为摘要：

- Stage A：要求 `aligned_30hz.csv` + `aligned_manifest.jsonl` 已存在则跳过
- Stage D：要求 `head_CAM_{A,D}_rtmw_nose.csv` **与** `head_CAM_{A,D}_rtmw_nose.fixed.json` 均存在且非空则跳过；否则 `--sample-count 48` 重跑
- 腕 `Z_OFFSET_MM=-60`，踝 `Z_OFFSET_MM=-80`；`N_CHUNKS_TRI=8`，`N_CHUNKS_RENDER=24`
- 顺序：B RTMW → C 三角化 → D 固定鼻尖采样检测 → E.1 raw 可视化 → E.2 replace → E.3 optimize + 并行渲染
- 状态：`/home/gaoweijian/0806_batch/{wrist,ankle}/STATUS.txt`
- 日志：`/home/gaoweijian/0806_batch/{wrist,ankle}/logs/pipeline.log`

本地编排上传、监控、拉回可参考 `data_processing/scripts/joint_projection/overnight_0806_limb_batch.py`（不自动启动远端，仅作参考）。

### 无数据集手动命令

无 `run_0806_*` 壳。在数据根已对齐、四路 inference 与三角化（Stage A–C）完成后，从 Stage D 起与腕踝相同；**跳过 E.2** `replace_limb_mocap_gt`。`optimize` 的 `--skeleton-playback` 指向 `skeleton_playback_raw.json`；`--config` 用 `0806_dual_external_mocap.json`。

示例本地数据根：`C:\Users\hand\Desktop\双外部双目\0806\无`

---



## 交付物清单

路径均相对 `{data_root}/multiview_3d_results/full/`。

### JSON / JSONL


| 文件                                                                      | 腕/踝                   | 无             |
| ----------------------------------------------------------------------- | --------------------- | ------------- |
| `multiview_3d_results_pre_limb.jsonl`                                   | 有                     | 有             |
| `multiview_3d_results_limb_gt.jsonl`                                    | 有                     | —             |
| `multiview_3d_results.jsonl`                                            | 有（= limb_gt）          | 有（= pre_limb） |
| `skeleton_playback_raw.json`                                            | 有                     | 有             |
| `skeleton_playback.json`                                                | 有（delivery v2，pruned） | 有             |
| `replace_{wrist,ankle}_report.json`                                     | 有                     | —             |
| `head_reprojection/nose_offset_opt/report.json`                         | 有                     | 有             |
| `head_reprojection/nose_offset_opt/head_CAM_{A,D}_rtmw_nose.csv`        | 有（全序列恒定 UV）           | 有             |
| `head_reprojection/nose_offset_opt/head_CAM_{A,D}_rtmw_nose.fixed.json` | 有                     | 有             |




### 必出 MP4


| #   | 路径                                                                         | 说明                  |
| --- | -------------------------------------------------------------------------- | ------------------- |
| 1   | `visualization/external_4cam_2d_skeletons.mp4`                             | 四路外部 2D 骨架（三角化观测）   |
| 2   | `visualization/skeleton_3d_raw_yaw.mp4`                                    | 原始三角化 3D（yaw 固定视角）  |
| 3a  | `head_reprojection/nose_offset_opt/head_CAM_A_direct_noseonly.mp4`         | 优化前 → 头显 A          |
| 3b  | `head_reprojection/nose_offset_opt/head_CAM_D_direct_noseonly.mp4`         | 优化前 → 头显 D          |
| 4   | `head_reprojection/nose_offset_opt/head_CAM_A_nose_offset_opt.mp4`         | **生产验收主视频**（优化后 A）  |
| 5   | `head_reprojection/nose_offset_opt/head_CAM_D_nose_offset_opt.mp4`         | 优化后 D               |
| 6   | `head_reprojection/nose_offset_opt/head_2x2_direct_vs_nose_offset_opt.mp4` | 2×2 before/after 对比 |


可选：`visualization/skeleton_3d_optimized_yaw.mp4`（limb_gt 后 3D 回放，腕/踝集）。

### 本地已验证路径（示例）


| 数据集 | `head_CAM_A_nose_offset_opt.mp4`                                      |
| --- | --------------------------------------------------------------------- |
| 无   | `0806/无/multiview_3d_results/full/head_reprojection/nose_offset_opt/` |
| 腕   | `0806/腕/multiview_3d_results/full/head_reprojection/nose_offset_opt/` |
| 踝   | `0806/踝/multiview_3d_results/full/head_reprojection/nose_offset_opt/` |


三处 `report.json` 的 `chosen_offset_source` 均为 `layer1_3d_gt_rigid_tip+fixed_rtmw_2d_refine`，`fixed_rtmw_nose_uv` 记录 CAM_A/D 固定 UV，确认同一 nose_offset_opt 生产路径。

---



## 与消融方案的区别


|           | **生产（本文）**                 | **消融（已弃用）**                                               |
| --------- | -------------------------- | --------------------------------------------------------- |
| 目的        | 固定配方交付                     | 多 cell 对比选参                                               |
| 滤波        | 单一 `default` + σ=1.0       | S0_B0…B5 切换                                               |
| GT        | 鼻对齐 + 肢端硬替换（按角色）           | S1_1.1…1.4 开关组合                                           |
| 骨长软约束     | **关**                      | S2_W0…W5 权重扫描                                             |
| 头显 refine | Layer1 + 固定 UV Layer2 固定开  | 可 `--skip-head-2d-refine`；Stage D `--sample-count 0` 逐帧检测 |
| 入口        | `run_0806_limb_dataset.sh` | `ablation/run_ablation_batch.sh`                          |


---



## 脚本索引


| 角色                   | 路径（相对 HearWristCam repo）                                                                      |
| -------------------- | --------------------------------------------------------------------------------------------- |
| 远端一键                 | `data_processing/scripts/joint_projection/run_0806_limb_dataset.sh`                                         |
| 肢端 GT                | `data_processing/scripts/joint_projection/replace_limb_mocap_gt.py`                                         |
| 头显鼻尖固定 UV 检测         | `data_processing/scripts/joint_projection/detect_head_nose_rtmw.py`                                         |
| 头显优化                 | `data_processing/scripts/joint_projection/optimize_multiview_head_nose_offset.py`                           |
| 并行渲染                 | `data_processing/scripts/joint_projection/render_nose_offset_parallel.py`                                   |
| Delivery 关键点         | `data_processing/scripts/joint_projection/delivery_keypoints.py`                                            |
| 腕 / 踝 / 无 config     | `configs/0806_{wrist,ankle}_dual_external_mocap.json`，`configs/0806_dual_external_mocap.json` |
| 完整背景计划（含 Stage A 细节） | `data_processing/docs/PIPELINE_PLAN_0806_踝腕.md`                                                                    |
| 消融（勿用于生产）            | `data_processing/scripts/joint_projection/ablation/`                                                        |


