# HDMI 触控与 Skin8 交接包（2026-10-24）

固定开发目录：C:\Users\60398\Desktop\PLD_Dev。本包是当前板卡的增量开发/部署交接包，不是 SD 卡镜像；需要已有 PLD 冻结运行环境与 ACDC hdmi_touch_20261024 基础版本。没有收录患者上传、历史结果、登录凭据或缓存。

## 已完成和正在验收

- ACDC 第四幅分割表面支持触控旋转；前三幅保留 MPR。每幅图可放大，左上角橙色 < BACK 返回四宫格。用户已实屏确认返回正常，旋转效果可以接受。
- 拖动时使用独立的简化显示网格，松手恢复完整网格；模型输出、原网格及测量数据不变。ED 完整 95436 / 预览 6410 三角形，ES 完整 93152 / 预览 6768。
- 四宫格预览画面生成板测 27–29 ms；完整网格稳态约 183–186 ms。此值不含 HDMI 传输、触控采样和排队，不能当作屏幕实际帧率。最初完整网格约 1.015 s。
- 通过缓存切片/背景、预计算法线、C 深度缓冲光栅化、直接生成目标分辨率、消除重复渲染来降低延迟。仅改变显示，不修改模型/RTL。
- 皮肤镜、CAMUS 取消之前的 HDMI 全面阻断，使用原控制接口释放推理硬件后启动显示子进程。CAMUS HTTP 层已有非重入 busy 信号量，包装层不得重复 acquire。
- 发现触控 drop-in 覆盖 CAMUS PYTHONPATH，造成 NIfTI 缺 nibabel。95-camus-dependencies.conf 恢复原有依赖目录，没有重新安装包。
- 最终运行结果以 evidence/hdmi_lifecycle_20261024.json 为准，保留首次 CAMUS 依赖错误及修复后的结果，不把历史失败隐去。

## 文件

- board_source/：板上 ACDC 应用、网页资源、触控公共模块的源码快照，保留同学原有实现。
- integration/：本次 HDMI 四宫格、触控、C 渲染器、验证脚本和 CAMUS 依赖配置。
- skin8_source/：Skin8 工具、仿真及与已部署 bitstream 的 build_receipt 哈希完全一致的 RTL。两份旧 RTL 从本地 Git 历史按 SHA256 找回，不能用开发目录里后来时序失败的版本替代。
- qualified_rtl/：同一套已部署 RTL 的明确快照。
- skin8_runtime/：已通过板测的 bit/hwh、双模型编译参数、驱动和本次网页入口补丁。旧 package.json 是原始构建身份，不覆盖新加入的网页文件；本包 SHA256SUMS.json 覆盖交接文件。
- evidence/：原 Skin8 板测、构建身份及本次显示/生命周期实测。

## 板上位置和恢复

网页入口 http://192.168.2.99:8088/；ACDC 页面 8092。
ACDC 新版 /home/xilinx/acdc_board_v2/releases/hdmi_surface_20261024。
应用 SHA256：c7131026b1a1f2b6668e34ddc628af0d2c536827ebf8e7380359b27cd985186c。
Skin8 运行目录 /home/xilinx/skin8_candidate_20261023。
旧 ACDC release /home/xilinx/acdc_board_v2/releases/hdmi_touch_20261024。
本轮原 unit 和原 skin8 web_entry.py/hdmi_policy.py 备份 /home/xilinx/hdmi_surface_backup_20261024。

所有修改先在桌面源代码完成。重新部署时先确认无推理任务，正常停止入口、hub、ACDC web/engine、skin/camus 服务，检查 MainPID=0、cgroup 无子进程且 FPGA 锁可获取。不要强杀持有 VDMA/DMA 的进程，也不要在显示运行时加载推理位流。

ACDC 在旧 release 的独立副本中替换 integration 的 hdmi.py/surface_hdmi.py/touch_drag.py/surface_raster.c 至 acdc/appliance/。板上该目录编译：

    gcc -O3 -shared -fPIC surface_raster.c -lm -o surface_raster.so

从 release 根目录运行 pynq-venv Python 的 `-m acdc.appliance.identity <release绝对路径>` 生成新的 v2_application.json。必须同步修改两个 systemd unit 的 WorkingDirectory、PYTHONPATH、config 路径和 --app-sha，然后 daemon-reload。不要复用本说明的旧 SHA 来部署发生变化的源码。启动后检查 health 的 application_sha256、ready、pl_selftest.head_equal。

Skin8 替换 web_entry.py 和 hdmi_policy.py 时须成对更新。CAMUS 依赖配置放入 /etc/systemd/system/pld-camus.service.d/95-camus-dependencies.conf，然后 daemon-reload，在服务停止时更新。

回退：先正常停止所有硬件拥有者；恢复备份中的两个 ACDC unit、skin8 两个 Python 文件。95-camus-dependencies.conf 是修正丢失依赖的独立补丁，可保留；如确需回到原配置再移除此确切文件。daemon-reload 后启动 pld-entry.service，检查入口和模型 health。原始 release 未删除。

## 验证边界

用户已经确认 ACDC 旋转和返回的实际体验。HTTP 生命周期只做受影响路径的必要短测，不是长期稳定性或所有异常条件的保证。Skin8 先前固定输入 1000/1000 一致、10.64 FPS，CAMUS 100/100 一致、约3.66 FPS；这是模型核心已有证据，不是本次 HDMI 刷新率，也不是完整数据集准确率。

GitHub PR、压缩包链接和最终状态见 RELEASE_STATUS.md。交付前先保存、校验包，再按用户指令关闭 Windows；板卡服务独立运行。

CAMUS 附加修复：同一依赖 drop-in 还包含原 camus/board 目录，避免显示子进程导入 HardwareLease 时找不到 tensor_accel。先以相同 PYTHONPATH 验证完整导入，再启动显示。
