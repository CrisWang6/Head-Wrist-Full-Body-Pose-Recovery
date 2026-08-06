# 未直接纳入的相关代码

## 远程运维脚本

原 `HearWristCam/.codex_tmp` 中存在多批：

- `pi_*.py`
- `run_pi_*.py`
- `read_pi_*.py`
- `stop_pi_9cam.py`

它们用于 SSH 登录树莓派、检查设备、拉取录制和远程启动 9 相机，但包含硬编码 IP、用户名和明文密码，而且多为一次性诊断。为避免未来推送 Git 泄密，本仓库没有复制。

若需要恢复，建议建立独立 `ops` 仓库并统一改成：

```python
host = os.environ["OAK_HOST"]
user = os.environ["OAK_USER"]
# 使用 SSH key，不在代码中保存 password
```

## Luxonis 示例

原 `HearWristCam/examples/downloader` 和 `examples/models` 是 Luxonis 示例下载器与模型描述，没有发现本项目修改，因此未合并。需要时应从官方 DepthAI examples 获取。

