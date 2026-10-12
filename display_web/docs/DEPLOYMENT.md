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
