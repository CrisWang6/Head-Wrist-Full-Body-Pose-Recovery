# 工程化重构验证记录

验证日期：2026-07-24

本文是 2026-07-24 的快照。`prediction_model_training` 在 2026-08-07 做过一次单独复核，包含 stage1-3 端到端跑通结果和与 4090 主机的核对方式，见 [训练代码复核记录](prediction_model_training/docs/VERIFICATION_20260807.md)。

## Git 仓库

| 仓库 | 分支 | 首次提交候选文件 | 代码文件 |
|---|---|---:|---:|
| `OAK_test` | `main` | 75 | 34 |
| `Issacsim_data_generation` | `main` | 36 | 23 |
| `data_processing` | `main` | 74 | 56 |
| `prediction_model_training` | `main` | 43 | 27 |

四个目录均为独立 Git 仓库，没有自动 commit，也没有配置 remote。

## 静态检查

- Python：120 个文件，语法错误 0。
- Shell：15 个文件，通过 Git Bash `bash -n`，错误 0。
- JSON：20 个文件，解析错误 0。
- TOML：4 个文件，解析错误 0。
- 逐文件文档：140 个 Python/Shell/PowerShell/HTML/Kalibr 入口全部能在对应 `docs/CODE_CATALOG.md` 找到，缺失 0。

## 功能检查

- Isaac Sim：9 项单元测试执行成功，其中 1 项按原测试条件跳过。
- `prepare_direct_2d_heatmap.py --help`：通过。
- `extract_direct_2d_frames.py --help`：通过。
- `project_joints.py --help`：通过。
- `preprocess_9cam_imu_mocap.py --help`：通过。

当前 Windows base Python 没有安装 DepthAI，因此 OAK CLI 没有做硬件 runtime 测试。该 Python 的 PyTorch `c10.dll` 也无法加载，因此 stage1/stage2 GPU smoke test没有在本机执行；相关源码已经通过语法、导入路径人工检查和主机历史环境版本核对。应在 README 指定的 Linux/DepthAI/CUDA 环境中做最终硬件测试。

## 安全与仓库边界

- 明文密码/已知凭据模式：0。
- 旧 `/home/gaoweijian`、`/home/whr`、`test_code/...` 可执行代码路径：0。
- H.265、MP4、bag、JPG、CSV、NPZ、checkpoint、权重、日志、Python cache：0。
- `.gitignore` 已验证会忽略 `data/`、`outputs/`、`logs/`、`checkpoints/`、`weights/` 和媒体/模型格式。

历史实验 JSON 中仍可保留采集来源路径作为 provenance，但运行脚本的默认路径已经改为仓库相对路径、命令行参数或环境变量。

## 来源保护

整个过程只在 `C:\Users\hand\Desktop\Code` 中移动或修改复制件。以下三个来源目录未被移动、删除或覆写：

- `C:\Users\hand\Desktop\gwj_reinstall_backup_20260724`
- `C:\Users\hand\Desktop\HearWristCam`
- `C:\Users\hand\Desktop\Dataset`

