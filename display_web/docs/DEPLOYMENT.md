# 部署与开机自启

## 环境

- 电脑：Python 3.9+、NumPy、Pillow；Windows 入口为 start_windows.cmd。
- 板子：PYNQ-Z2，使用 `/usr/local/share/pynq-venv/bin/python3`。
- B 计算包：`/home/xilinx/jupyter_notebooks/pld/final_mode6_fastq_20260929_v3`，必须包含 deployment_manifest.json、模型、配套 bit/hwh 和已编译的三个原生库。
- 应用目录：`/home/xilinx/jupyter_notebooks/PLD_Web`。

## 已安装自启的板子

本整理版已合并此前的全部更新，不需要为了文件整理重新安装板端程序。日常上电后等待约 1–2 分钟，使用原浏览器打开 `http://192.168.2.99:8765/`。

## 首次部署或迁移

将整个 PLD_Web 目录上传到 Jupyter 首页。网页服务和旧显示程序不能同时运行。先用手动入口验证：

```bash
cd /home/xilinx/jupyter_notebooks/PLD_Web
bash start_board.sh /home/xilinx/jupyter_notebooks/pld/final_mode6_fastq_20260929_v3
```

首次复制打印的完整链接，只把 127.0.0.1 换为板子 IP，保留 # 后内容。验证成功后浏览器记住访问码，以后可收藏不带 # 的地址。清除网站数据、无痕模式或更换浏览器后需重新使用完整链接。

服务访问码保存在 `data/.access_token`；保持 data 目录不变，不要复制到公开仓库。此服务面向可信局域网，不是公网网站。本工程不修改 IP 或路由器设置。

## 开机自启

先等待任务完成、停止 HDMI，再正常退出手动启动的服务。执行：

```bash
cd /home/xilinx/jupyter_notebooks/PLD_Web
sudo -E /usr/local/share/pynq-venv/bin/python3 install_autostart.py
```

看到 AUTOSTART_PASS 后，在无任务运行时执行 `sudo reboot`。启动完成后直接访问固定网址，测试 JPG 分割和 HDMI。安装器检查旧进程、计算包和 systemd 配置，原同名服务文件会备份，不会自动重启板子。

自启配置使用 root 登录 shell 加载 PYNQ/XRT 环境；主进程首先接收 SIGINT，等待任务及显示退出。服务不自动重试硬件故障。

## 操作流程

选择“板端 FPGA 分割”→ 上传 JPG → 开始处理 → 选择切割图或轮廓 → 点击显示到 HDMI。

单文件最多 8 MB；板端网页最多 300 万像素，电脑后处理最多 1200 万像素。电脑“图片 + 配套掩膜”模式支持 JPG/PNG 原图和同尺寸单通道二值 PNG。真实结果回放没有原始 JPG，只显示历史掩膜，历史推理时间不是本次测量。

实体按键：有原图时 B0 原图、B1 切割图、B2 原图与轮廓对比、B3 黑屏。回放模式 B0 掩膜、B1 轮廓、B2 描边掩膜。B3 保持显示链路，网页“停止 HDMI”才释放链路。

## 检查与停机

```bash
systemctl status pld-web.service --no-pager
sudo journalctl -u pld-web.service -n 60 --no-pager
sudo systemctl stop pld-web.service
sudo systemctl start pld-web.service
```

日志可能含初始访问链接，不应公开。取消自启：

```bash
sudo systemctl disable --now pld-web.service
```

出现硬件故障时先保留当前任务下 inference.log 或 hdmi.log，确认推理与显示进程已退出，再处理硬件状态，不要重复启动多个程序。任务超时或 HDMI 未释放时，服务会禁止继续硬件操作。停止服务最多等待 240 秒；超时后强制清理不代表硬件已经正常释放。

## 后续更新代码

无任务时停止服务，备份旧源码，更新源码但保留 `data/`，再启动服务。不要将桌面旧增量 ZIP 反复覆盖到整理版上。

## 组合候选正式切换（版本化 drop-in，2026-10-14 已上板）

`scripts/deploy_pld_web_candidate.py` 把已通过板端隔离验证的组合候选（C 网页双线程 PNG + 可选常驻 B worker）
切换到正式 `pld-web.service`。候选目录与固定校验值：

| 项 | 值 |
|---|---|
| 候选 C | `/home/xilinx/jupyter_notebooks/PLD_Web_png_worker_candidate_v3_20261014` |
| 候选 C `server.py` SHA256 | `4facbcc5d5e385415ead0529e1db21ce704a3f3d0f2fec8f2c58b343383c8835` |
| 候选 B | `/home/xilinx/jupyter_notebooks/pld/final_fastq_worker_candidate_v2_20261013` |
| 候选 B `deployment_manifest.json` SHA256 | `b80c8813dc56f32d1bfbb35161f91a5092cdfb6f27044be9ef998cff3dd1ab55` |

13 个核心源码的 pin 取**板端候选目录的实际文件字节**（GPT 已用只读 SFTP 核对本机工作树与板端逐字节一致），
不是 Git 换行归一化后的 blob 哈希。其中 `static/style.css` 实际含 1 个 CRLF，SHA256 为
`ca38cc0154d41942a17c7d3119f728e03fef1092f5aba7185691e0f2e3931a0a`；不要把它“修”成归一化哈希。

脚本分两步：`preflight` 只读检查，`apply` 才修改系统；`rollback` 用 apply 写下的记录恢复。在板上执行：

```bash
sudo -E /usr/local/share/pynq-venv/bin/python3 deploy_pld_web_candidate.py preflight
sudo -E /usr/local/share/pynq-venv/bin/python3 deploy_pld_web_candidate.py apply
sudo -E /usr/local/share/pynq-venv/bin/python3 deploy_pld_web_candidate.py rollback
```

`preflight` 要求服务 active、HTTP 200、经**原** `PLD_Web/data/.access_token` 内部鉴权确认空闲
（无活动任务、HDMI off、fault 为空），逐文件核对候选 C 的 13 个核心源码 SHA256、候选 B 的
`deployment_manifest.json` 本身哈希和清单内全部文件，并确认原 unit/生效 ExecStart/WorkingDirectory 仍指向
原 `PLD_Web` 与原 data、候选 drop-in 尚不存在。访问令牌只在内存读取，不打印、不写记录。
数据目录被固定为 `ORIGINAL_APP/data`：即使显式传 `--data`，任何其它目录都会在任何 systemctl 调用、
HTTP 请求或文件修改之前被拒绝，原访问码/历史绝不会被切到候选 data。

`apply` 只新增一个版本化 drop-in（`/etc/systemd/system/pld-web.service.d/50-pld-web-png-worker-v3-20261014.conf`）
覆盖 `WorkingDirectory` 与 `ExecStart`：同一 PYNQ venv Python、root 登录 shell 环境、原端口 8765、
显式 `--persistent-b-worker`，**继续使用原 `PLD_Web/data`**（访问码、历史、用户输入与结果不迁移、不重建）。
原 unit、旧冻结运行包、既有 drop-in 均不改写；切换前把可回退配置（不含令牌）与既有 drop-in 备份写入
`/var/lib/pld-web-switch/apply-*.json`。写入使用同目录临时文件 + `os.replace` 原子替换，随后
`daemon-reload`、`restart`，并在有界时间内核验 MainPID 的真实 cmdline/cwd、HTTP 200、原令牌可用、
原历史条目仍在。任一步失败会自动删除该 drop-in、重启原服务并再次验证；若恢复也失败，脚本以退出码 2
报告硬件/DMA 所有权未知，禁止伪成功。`rollback` 显式恢复原配置并同样验证，不删除候选目录或用户数据。
所有等待均有界：普通 systemctl 操作（is-active/show/daemon-reload）默认 30 秒；`restart` 默认 300 秒，
以覆盖 unit 的 `TimeoutStopSec=240` 加重启开销；`--startup-timeout`（默认 120 秒）只界定 restart 返回
之后的健康验证窗口，与 restart 超时互相独立。systemctl 超时会转为明确失败，不会挂死脚本。

**rollback 边界**：自动 rollback 要求当前（候选）服务仍能通过 HTTP 提供 `/api/status` 且处于空闲，
用于确认没有任务/HDMI/硬件操作正在进行。如果服务已经完全停止或无响应，脚本会拒绝回退而不是盲目
重启旧服务——此时硬件/DMA 所有权不明。人工恢复路径：

```bash
# 1. 确认没有推理/显示/网页进程仍持有硬件（三者都为空才继续）
ps -ef | grep -E 'server.py|infer_worker.py|infer_images.py|hdmi_worker.py' | grep -v grep
sudo journalctl -u pld-web.service -n 100 --no-pager
# 2. 确认无持有者后，手工移除候选 drop-in 并恢复原服务
sudo rm /etc/systemd/system/pld-web.service.d/50-pld-web-png-worker-v3-20261014.conf
sudo systemctl daemon-reload
sudo systemctl restart pld-web.service
```

若第 1 步仍能找到持有硬件的进程，先等其有界退出或按故障流程处理，不要在 DMA 状态未知时启动任何
推理、HDMI 或另一套网页程序。

主机回归 `sim/test_deploy_pld_web_candidate.py`（16 项）用假 systemctl/假 /proc/本地鉴权 HTTP 驱动同一入口，
覆盖 preflight 通过、C/B 哈希漂移拒绝、忙碌/HDMI/fault 拒绝、异常 ExecStart 拒绝、重复 drop-in 拒绝、
外来 data 目录在任何副作用前拒绝、is-active 卡住与 restart 卡住的有界失败（含恢复失败退出码 2）、
apply 生成与原子回退、rollback 与忙碌拒绝，并断言输出不含访问令牌。该测试不接触板卡。
正式切换前的 24 帧交错小/大图压力审计已通过。板上 `preflight` 核对候选 C 13 文件及 B manifest 的 162 文件后，`apply` 切换成功；同图小/大图真实 HTTP、六视图像素、原始 head、ZIP 成员与 CRC、HDMI 软件切换和原访问码均通过。随后实际执行一次 `rollback`，旧服务及原有 4 条本次开机历史恢复，再次 `apply` 后一张真实图通过，历史增至 5 条；当前正式服务仍运行组合候选。最终回退记录为 `/var/lib/pld-web-switch/apply-1746420088611781333.json`（板卡时钟不准，文件名时间戳不用于外部计时）；执行入口在 `/home/xilinx/jupyter_notebooks/deploy_pld_web_candidate_20261014.py`。机器记录见 `out/board_sync_20261012/formal_web_v3_audit_20261014.json` 与 `formal_web_reapply_v3_20261014.json`，耗时及验收边界见 [C_WEB_PERFORMANCE](../../docs/validation/C_WEB_PERFORMANCE.md)。

## 更新本次开机历史功能

电脑最终源码已包含此功能。已有板端自启配置无需重新安装；将更新包 `PLD_history.zip` 上传到 Jupyter 根目录后，在没有推理任务运行时执行：

```bash
cd /home/xilinx/jupyter_notebooks
sudo systemctl stop pld-web.service
unzip -o PLD_history.zip
sudo systemctl start pld-web.service
sudo systemctl status pld-web.service --no-pager
```

更新包只覆盖 server.py 和三个网页文件，不包含 data、访问码或模型。建议更新前备份这四个旧文件。网页按 Ctrl+F5 刷新，下方出现“本次开机的处理记录”。新完成的任务会自动加入；更新前的旧任务没有索引，不会导入。

同一次板子开机中，刷新浏览器或重启网页服务仍可恢复历史。板子重启后历史列表从空开始；以前的结果文件仍保留在 data 中，不自动清理。
