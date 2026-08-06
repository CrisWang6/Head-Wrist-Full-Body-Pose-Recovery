# 未直接纳入的相关代码

## EgoRear 上游参考

原目录：

```text
HearWristCam/.codex_tmp/EgoRear_ref
```

上游：

```text
https://github.com/hiroyasuakada/EgoRear.git
commit d9df1e6c26ae98162e4365c4bd109cd1847b8150
```

工作树干净，属于第三方参考实现，因此未复制。需要复现实验时建议以 Git submodule 或固定 commit 拉取。

## 远程部署脚本

原 `.codex_tmp/real_head_training_0717` 中的 `deploy.py`、`fetch_previews.py`、`remote_exec.py` 以及 `training_export` 脚本包含硬编码服务器地址和明文密码。核心训练、标签和配置已经由远端最终版覆盖，这些一次性脚本未纳入。

如需保留，应建立不公开的 `ops` 仓库并改为 SSH key、环境变量和明确的远程目标白名单。

