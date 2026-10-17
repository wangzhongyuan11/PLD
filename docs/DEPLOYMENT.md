# 部署、升级与回退

唯一版本为 `pld_dual_20261017`。先按[交付说明](../delivery/pld_dual_20261017/README.md)校验附件，再部署。以下升级入口针对现有已安装的 PYNQ-Z2 双模型服务；它不会安装操作系统、驱动、PYNQ 或修改系统 Python。

## 本次板端布局

```text
/home/xilinx/jupyter_notebooks/pld/
  releases/pld_dual_20261017/{hub,web,skin,camus,units,config}/
  rollback/pre_pld_dual_20261017/       # 一份完整、私有的清理前可运行回退
  camus_deps_20261015/                 # 原隔离 nibabel 依赖，保留
  camus_web_data_20261015/             # CAMUS 历史，保留
/home/xilinx/jupyter_notebooks/PLD_Web/data/  # Skin 当前历史及 .access_token
```

另外一份 Skin 候选目录下独有的 34 个历史任务也保留，不能当重复数据删。原始开发数据和 symlink 目标保留。版本清单中没有访问码；`hub/config.json` 仅保存私有 token 文件路径。

## 升级现有板卡

先确认没有浏览器任务或独立板测正在执行。通过现有 SSH 登录方式上传三个文件：资产 ZIP、`delivery/pld_dual_20261017/MANIFEST.json`、`board/deploy_release.py`。不要将主机完整备份上传为发布包。

```bash
sudo -E /usr/local/share/pynq-venv/bin/python3 -B deploy_release.py install \
  --archive pld_dual_20261017_assets.zip --manifest MANIFEST.json
```

该工具先校验整体和每个成员 SHA256、拒绝路径穿越/重复/符号链接，再暂存新目录；经 hub 正常停止旧后端，核对三个服务的 inactive、PID=0、Result 和 cgroup、取得共享 lease，才备份有效 unit/drop-in 和旧应用/运行包。新 unit 沿用停止信号、禁止 SIGKILL、有界超时、loopback 端口与隔离依赖。旧 stacked drop-in 收敛为三个正式 unit，固定入口仍是8765，仅 hub 开机启用。

脚本返回 `DEPLOY_STARTED` 仅代表已启动，必须等待 `/hub/status` 的 `phase=ready`，同时确认当前模型与另一个服务的状态，不能视为完整验收。若已有该 release 目录，工具拒绝覆盖；不要直接解压覆盖运行目录。发生启动错误保留新目录和 rollback receipt，检查 `journalctl -u pld-hub -u pld-web -u pld-camus`，按下节恢复。时钟不准时以主机时间和单调耗时为准。

## 回退与重新应用

```bash
sudo -E /usr/local/share/pynq-venv/bin/python3 -B deploy_release.py rollback
# 等待 ready、核对功能后，需要恢复正式版本：
sudo -E /usr/local/share/pynq-venv/bin/python3 -B deploy_release.py reapply
```

回退使用完整私有副本，路径重定位后不依赖已清理旧目录；原历史/访问码路径不变。unit 发生未知改动则拒绝自动覆盖。不要在活动 DMA 或未确认关闭的 HDMI 上用 kill -9/直接换 overlay。回退只是软件版本恢复，不回滚用户数据。

## 新板复现

新 PYNQ 镜像需要先具备板端已有环境：PYNQ 3.1.1 / Python3.10.4 / NumPy1.21.5 / Pillow9.0.1 / SciPy1.8.0，以及 XRT、base HDMI overlay。附件已附 nibabel5.2.1 wheel，按其 LICENSE 保留权属，使用隔离 `--target` 目录安装，不升级系统 NumPy/Pillow。ARM Skin 三套库与冻结源码、Pillow core 哈希绑定，不匹配时必须在目标板重新构建并独立验收，不能跳过验证。

将附件中 `hub/web/skin/camus` 放到上述正式根；把 [三个 unit 模板](../delivery/pld_dual_20261017/systemd/) 放入 `/etc/systemd/system/`，按实际环境调整 data/deps 路径。把附件 `config/installation.json` 放入 `hub/installation.json`，保持 backend/source/runtime/data/extra 与 unit 一致。创建仅 root 可读的共享访问码文件（至少24字符），`hub/config.json` 写 `{"token_file":"实际绝对路径"}`；这个值与 Skin data 中 `.access_token` 必须指同一份文件。创建 `hub/state/last_model` 内容 `skin`。这些是本机私有配置，不提交或上传。

运行 `systemd-analyze verify` 检查三个 unit，`systemctl daemon-reload`，只 enable/start `pld-hub.service`；另两个后端 disable，由 hub 调度。新板的模型/位流与源哈希一致只代表输入就绪，仍需真实黄金输出验证与硬件接口测试。

## 短测

在板端使用 PYNQ Python 运行仓库的 `scripts/verify_release_board.py`，参数包括 `--hub-root`、`--model skin|camus`、`--frames 1..10`、全新 `--output`，可加 `--hdmi`。该入口通过认证的8765网关提交原始开发图，验证 frozen golden、ZIP CRC、PNG像素哈希/尺寸及 HDMI 后推理恢复。默认数据路径沿用当前板；其他板应显式准备对应开发 fixture 并调整路径，不能用任意图像代替 golden。交互另用 `scripts/verify_interactive_board.py --model ... --output ...`。

独立驱动/故障注入板测之前，先正常停 hub 并确认三个服务及子进程都已结束；测试结束恢复 hub。网页正在运行时不可另起 DMA 测试进程。物理 HDMI 未连接时仅报告初始化、frame submit 和释放/恢复，不报告画面正确。
