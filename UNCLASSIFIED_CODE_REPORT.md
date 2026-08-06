# 未直接进入仓库的代码总表

未归入代码已经分别记录到各仓库：

- [OAK/树莓派运维与 Luxonis 示例](OAK_test/docs/UNINTEGRATED_CODE.md)
- [BlenderProc、Isaac Sim 与仿真资产](Issacsim_data_generation/docs/UNINTEGRATED_CODE.md)
- [远程数据导出/清理和只有产物的 Pose2D 目录](data_processing/docs/UNINTEGRATED_CODE.md)
- [EgoRear 上游参考与远程训练部署脚本](prediction_model_training/docs/UNINTEGRATED_CODE.md)

共同原则：

1. 含明文密码、SSH 主机信息或破坏性清理命令的一次性脚本不进入公开 Git。
2. 上游工作树干净的第三方源码用 URL + commit 固定，不复制到自研仓库。
3. 只有数据、checkpoint、日志、可视化和 report 的目录不视为项目源码。
4. 重复代码保留最终版本；完全重复的副本不再次提交。

如果决定保留远程运维脚本，建议新建私有 `ops` 仓库，并先完成凭据环境变量化、SSH key、dry-run 和目标目录白名单。

