# 未直接纳入的相关代码

## 数据上传与远程训练编排

原 `.codex_tmp/training_export` 和备份 `local_sources/training_export` 中包含：

- H.265 上传/远端解码；
- 0717 训练数据导出；
- 远端训练状态检查；
- 远端清理脚本。

这些脚本横跨数据处理和服务器运维，包含明文 SSH 密码及破坏性清理命令，因此没有放进本仓库。若要恢复，应先去除凭据、增加 dry-run 和目标目录校验，再放入独立 `ops` 仓库。

## 只有产物的目录

- `restored/pose2d_0722_module01`：没有发现源码，只有输出结果。
- Dataset 的 `alignment_data.js`、`video_skeleton_data.js`：viewer 生成数据。
- 所有 `validation_*`、CSV、NPZ、图片、视频和 report：中间或验证产物。

