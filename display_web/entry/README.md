# 三模型统一入口

板端入口：**http://192.168.2.99:8088/**。首次默认心脏 2.5D；后续服务启动恢复上次成功选择的模型。顶部按钮可切换皮肤镜、CAMUS，并切回心脏 2.5D。所有浏览器共享当前板卡模型，切换前先下载需要保留的结果。

心脏使用原 8092 页面；皮肤镜/CAMUS 经原 hub 的带代次地址转发并嵌入，原访问码仅在板内用于建立原 hub 会话，不传到浏览器。统一入口与当前心脏界面一样供局域网直接访问。只有选择皮肤镜后才显示“ROI / 交互分割”；再次点击返回普通分割。页面保留选图、四种提示工具、运行和 HDMI 控制，增加“清除所有标记”（网页与 HDMI 右下角 Clear all），移除撤销、恢复、类别/阈值调整、回放和下载控件。交互请求校验当前模型代次，CAMUS 和 ACDC 不开放 ROI。

ROI 的网页和 HDMI 共用一个工作区。HDMI 左侧原图可点选、拖框，右侧显示当前修正结果；顶部 FG / BG / ROI / Box 选工具，Run 运行，Web / Stop 正常退出显示。点/框变化复用概率并保持显示；ROI 改变时原后端先关闭 HDMI 完成推理，交互控制随后自动恢复显示。普通模型页面仍显示原始推理结果，不覆盖其历史或 ZIP。

心脏 engine 使用 `KillMode=mixed`：systemd 只先通知主进程，由主进程按顺序停止并等待 HDMI 子进程，避免 control-group 同时发送退出信号。仍要求正常退出及 cgroup 为空，失败时禁止加载另一模型。安装器通过独立 drop-in 设置，回退恢复原配置。更新现有安装时也需应用 `96-entry-hdmi-shutdown.conf`，并保存原文件。

板端交互模块从 hub 的 `installation.json` 中 `backends.<model>.data` 读取原始 head；须与实际服务的数据目录一致。当前 Skin8 两模型分别位于 `/home/xilinx/skin8_web_data/skin` 和 `/home/xilinx/skin8_web_data/camus`，不能沿用旧双模型的数据目录。模型、位流和推理核心不因此改变。

入口等待当前任务完成后切换。支持暂停协议且 unit 无互斥冲突时，旧服务正常关闭 HDMI、释放计算会话及硬件锁后暂停，保留 CPU 软件状态。软件进程可以同时存在，FPGA 始终只有一个所有者。ACDC 的应用适配还要求 hub 与两个后台的 PID、暂停状态及释放状态一致，再取得原有三把文件锁；不支持该协议的旧部署继续正常停止并检查 cgroup。释放失败时拒绝继续加载。不要另行手动启动模型服务。入口本身停止时保留当前模型；仅打开或刷新网页不会切走其他浏览器使用的模型。

顶部“自动显示 HDMI”和“选图后自动分割”保存在板端。普通皮肤镜选图、CAMUS 输入文件完整、心脏 ED/ES 都更新后可自动提交一次；ROI 新图自动初始分割，提示编辑仍点 Run。任务成功由板端触发上屏；显示失败不抹掉分割结果。手动 Stop 同时关闭自动显示并取消旧回调。切回模型仅恢复模型/运行时身份相符且仍存在的上次结果，页面标明“上次结果”；ROI 工作区跨模型代次失效，不恢复旧提示。

2026-10-29 重构候选：打开自动显示开关仅影响后续任务，不立即恢复旧图；状态/图片 GET 不等待显示控制锁；显示软件预热改为异步确认，首帧超时保留故障与硬件所有权保护。任务通知支持有界幂等重试，具体改动、主机回归和待板测范围见 [皮损自动显示重构](../../docs/validation/SKIN_HDMI_REFACTOR_20261029.md)。该记录不表示已更新板端服务。

`PLD_WARM_HDMI=1` 启用显示子进程的软件预热和复用。预热不下载位流或创建视频硬件。显示关闭仍正常回收视频，下一次显示重新取得硬件；同一结果重复显示不重启，CAMUS 纯视图切换更新已有会话。计算与 HDMI 使用不同位流，推理期间不保证 HDMI 信号持续。

ACDC 候选通过 `PLD_RESIDENT_MODELS=1` 启用上述常驻交接。`python -m acdc.appliance.startup_cache RELEASE_ROOT` 在生成应用证书前准备源码绑定的固定调度计划。`ACDC_NATIVE_CACHE` 保存经过原身份检查的原生库；运行时文件校验仍执行，仅在同一进程内以文件设备号、inode、长度及纳秒修改/变更时间复用哈希，硬件恢复失败时清空。核心计算源码、模型和位流不变。

将本目录放入板端独立目录，使用原有 PYNQ Python 安装（root）。安装器从现有 hub unit 识别路径，并保存原开机启用状态。只启用入口自启动，由入口加载默认心脏服务。

```bash
sudo /usr/local/share/pynq-venv/bin/python3 /home/xilinx/pld_entry_20261023/install.py
```

回退时先等任务完成：

```bash
sudo /usr/local/share/pynq-venv/bin/python3 /home/xilinx/pld_entry_20261023/install.py rollback
```

回退恢复原开机启用状态和原心脏页面，不删除模型或数据。保留板端 `installation.json`；不要覆盖正在运行的入口目录。

2026-10-25 性能候选位于 `/home/xilinx/latency_opt_20261024`，ACDC 应用位于 `/home/xilinx/acdc_board_v2/releases/latency_opt_20261024`。`98-latency-opt.conf` 选择候选；原目录保留。该候选的 `rollback.json` 保存替换前后身份，`backup/` 保存文件；专用回退入口为 `sudo /usr/local/share/pynq-venv/bin/python3 /home/xilinx/latency_opt_20261024/rollback.py`。使用此回退入口，不运行下面旧增量版本的回退方式。板端评估 hub 使用逐处适配，该定制 hub 与评测入口已同步到仓库；安装清单只批准已核对的精确版本，未知版本仍拒绝覆盖。

最终候选已通过用户的三模型实屏与触控确认。常驻热切换约 2.4～7.3 秒；新进程首次准备仍约 27～33 秒。软件预热不保证开机即秒切；1022×767 的 ROI 初始到首帧仍观察到 16～28 秒。完整计时和限制见 [优化记录](../../docs/design/MODEL_HDMI_OPTIMIZATION_PLAN_20261024.md#13-2026-10-25-候选实施记录) 与 [测量证据](../../docs/validation/MODEL_HDMI_OPTIMIZATION_20261025.json)。

2026-10-25 的清除按钮增量仅更新 ROI 网页和触屏界面；保留原 HDMI 驱动。该增量的备份与回退入口为 `/home/xilinx/display_stability_20261025/rollback.py`，回退清除按钮更新并保留此前性能优化。

本地检查：

```text
python -m unittest discover -s display_web/entry -p "test_*.py" -v
node display_web/entry/test_ui.cjs
```

2026-10-24 板端增量部署目录为 `/home/xilinx/pld_display_repair_20261024`，入口通过独立 `96-display-repair.conf` 指向该目录的 `entry/`，原入口目录保留。`rollback.json` 记录本轮替换文件及前后哈希，`backup/` 保存原文件；`verification.json` 记录心脏 HDMI 显示中切到皮肤镜、两模型各三项交互请求及切回心脏的通过结果。测试采用缩小至长边128像素的已有开发图，不代表完整尺寸性能或物理屏幕目视验收。恢复前先停止入口与 hub、确认任务完成，再按记录恢复原文件和两个 drop-in，执行 daemon-reload 并重启入口；不要对本增量部署直接运行旧入口安装器的 rollback。
