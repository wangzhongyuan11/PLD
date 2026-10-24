# 三模型统一入口

板端入口：**http://192.168.2.99:8088/**。入口服务启动默认加载心脏 2.5D；顶部按钮可切换皮肤镜、CAMUS，并切回心脏 2.5D。所有浏览器共享当前板卡模型，切换前先下载需要保留的结果。

心脏使用原 8092 页面；皮肤镜/CAMUS 经原 hub 的带代次地址转发并嵌入，原访问码仅在板内用于建立原 hub 会话，不传到浏览器。统一入口与当前心脏界面一样供局域网直接访问。选择皮肤镜或 CAMUS 后，可点顶部“ROI / 交互分割”，复用原工作台的 ROI、前景/背景点、框提示、撤销和回放；再次点击返回普通分割。交互请求校验当前模型代次，切换模型后旧页面不能继续操作。

心脏 engine 使用 `KillMode=mixed`：systemd 只先通知主进程，由主进程按顺序停止并等待 HDMI 子进程，避免 control-group 同时发送退出信号。仍要求正常退出及 cgroup 为空，失败时禁止加载另一模型。安装器通过独立 drop-in 设置，回退恢复原配置。更新现有安装时也需应用 `96-entry-hdmi-shutdown.conf`，并保存原文件。

板端交互模块从 hub 的 `installation.json` 中 `backends.<model>.data` 读取原始 head；须与实际服务的数据目录一致。当前 Skin8 两模型分别位于 `/home/xilinx/skin8_web_data/skin` 和 `/home/xilinx/skin8_web_data/camus`，不能沿用旧双模型的数据目录。模型、位流和推理核心不因此改变。

入口等待当前任务完成后切换，沿用原 hub 的 HDMI/DMA 退出检查；停止失败时拒绝继续加载。两套拥有 FPGA 的服务不会由入口同时启动。不要另行手动启动模型服务。入口本身停止时保留当前模型；重新启动入口默认回到心脏 2.5D。仅打开或刷新网页不会强制切走其他浏览器正在使用的模型。

将本目录放入板端独立目录，使用原有 PYNQ Python 安装（root）。安装器从现有 hub unit 识别路径，并保存原开机启用状态。只启用入口自启动，由入口加载默认心脏服务。

```bash
sudo /usr/local/share/pynq-venv/bin/python3 /home/xilinx/pld_entry_20261023/install.py
```

回退时先等任务完成：

```bash
sudo /usr/local/share/pynq-venv/bin/python3 /home/xilinx/pld_entry_20261023/install.py rollback
```

回退恢复原开机启用状态和原心脏页面，不删除模型或数据。保留板端 `installation.json`；不要覆盖正在运行的入口目录。

本地检查：

```text
python -m unittest discover -s display_web/entry -p "test_*.py" -v
node display_web/entry/test_ui.cjs
```

2026-10-24 板端增量部署目录为 `/home/xilinx/pld_display_repair_20261024`，入口通过独立 `96-display-repair.conf` 指向该目录的 `entry/`，原入口目录保留。`rollback.json` 记录本轮替换文件及前后哈希，`backup/` 保存原文件；`verification.json` 记录心脏 HDMI 显示中切到皮肤镜、两模型各三项交互请求及切回心脏的通过结果。测试采用缩小至长边128像素的已有开发图，不代表完整尺寸性能或物理屏幕目视验收。恢复前先停止入口与 hub、确认任务完成，再按记录恢复原文件和两个 drop-in，执行 daemon-reload 并重启入口；不要对本增量部署直接运行旧入口安装器的 rollback。
