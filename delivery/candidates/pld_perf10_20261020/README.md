# Perf10 候选部署与原字节源码归档

2026-10-20 已通过网线部署到 PYNQ-Z2，独立目录为 `/home/xilinx/jupyter_notebooks/pld/releases/pld_perf10_20261020`。
部署验收时入口 http://192.168.2.99:8765/ ，沿用原访问码；`pld-perf10.service` active/enabled，skin、busy=false、fault=null。
验收后按用户要求安全关机：服务正常停止、DMA 锁释放、存储同步和关机请求均退出 0，SSH/网页已断开。
部署和自启动配置保留；未做断电重启验收。本次发布不再次操作板卡，见 [安全关机记录](validation/safe_shutdown.json)。
原 `pld_dual_20261017` 保留；仓库 `delivery/current.json` 未变，本候选没有提升为唯一正式交付。

## 资产与原样边界

下载 [候选 Release](https://github.com/wangzhongyuan11/PLD/releases/tag/pld-perf10-candidate-20261020)：

- `PLD_Dev_Source_OriginalBytes_20261020.zip`：用户提供的 PLD_Dev 工作树中 521 个源码快照文件，逐字节保留，包括换行。没有使用 Git 导出的换行规范化副本。逐文件身份见 [source_identity.json](source_identity.json)。
- `PLD_Perf10_Deploy_20261020.zip`：已实测的 800 文件部署包，包括固定计算源码、原样 FastDevice、冻结 INT8 模型、配对 bit/HWH/XSA 和本目录的独立启动适配。
- `PLD_Perf10_Board_Handoff_20261020_Published.zip`：上述两包、完整说明、原始验证 JSON、预览和六次结果 ZIP。

整包 SHA256 和字节数见 [assets.json](assets.json)。模型、位流和完整运行时保存在 Release；Git 记录本入口、部署适配和小型证据。
源码归档不含 `.git`、生成缓存或私有凭据；收到的原 ZIP 指纹也保存在索引中。

## 本轮验证

[summary.json](validation/summary.json) 与 [HTTP 原始记录](validation/http_verification.json) 表明：
两模型各通过 8×8 首层输入的完整 15 层真实板端逐层比较；完整 256×256 head 与独立 CPU INT8 参考均为零差异。
六次网线上传、预览和 ZIP 下载均通过。排除 CPU 对照的两次复测中位数为 skin 3.6969 秒、CAMUS 7.6429 秒，
口径是电脑上传开始至四张预览下载并解码，不包括模型加载或浏览器绘制。

已实际回退原版并对 ISIC_0000003.jpg 完成 frozen-golden head、六图及 ZIP 检查，再切回 Perf10。
[关机前部署身份](validation/deployment_identity.json) 记录运行 bit/HWH 与服务状态。
这是 100 MHz、batch 64、卷积 FPGA/非卷积 CPU 的混合执行候选；未做 1000 帧长稳、完整故障注入、物理 HDMI 或断电重启测试。
8 FPS 与 2 秒出图目标未达到。独立 CPU 对照约需 67/239 秒，日常演示关闭该选项。

## 板端切换

等待当前任务结束后切回原版：

```bash
sudo /usr/local/share/pynq-venv/bin/python3 /home/xilinx/jupyter_notebooks/pld/releases/pld_perf10_20261020/manage.py rollback
```

再次使用候选：

```bash
sudo /usr/local/share/pynq-venv/bin/python3 /home/xilinx/jupyter_notebooks/pld/releases/pld_perf10_20261020/manage.py resume
sudo /usr/local/share/pynq-venv/bin/python3 /home/xilinx/jupyter_notebooks/pld/releases/pld_perf10_20261020/manage.py activate
```

首次安装步骤在交付 ZIP 的 README 中；只适用于已经具备原 PYNQ 环境和原版访问码配置的板卡。
新安装使用独立目录并拒绝覆盖；启动前先检查旧服务已退出及共享 FPGA 锁可获取，停止时禁止强杀活动 DMA。
电脑本次直连地址 `192.168.2.1/24` 仅配置到 Windows ActiveStore；重启后可能需要重新配置。Wi-Fi 未修改。

板卡系统时钟未校准，[服务日志](validation/candidate_service.log) 中的 May 07 时间原样保留；本轮日期以电脑 Asia/Shanghai 时间为准。
