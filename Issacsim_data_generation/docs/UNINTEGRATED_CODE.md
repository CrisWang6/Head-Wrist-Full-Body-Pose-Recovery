# 未直接纳入的相关代码

- `gwj_reinstall_backup_20260724/restored/BlenderProc`：第三方官方源码，未确认有本项目修改；应固定上游版本而不是复制。
- 原机 `/home/gaoweijian/isaacsim`：约 21 GB 安装目录，应按新 GPU 驱动重装。
- `Simulation/smplx_models`：模型文件，不属于源码。
- `Simulation/test_motion`：HumanEva motion 数据，不属于源码。
- `Simulation/outputs`：渲染产物，不属于源码。

重装前工作树的三个未提交修改已经体现在当前源码中，并额外保存在 `docs/SOURCE_UNCOMMITTED.patch`。

