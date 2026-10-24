# 当前板端运行版本交接（2026-10-24）

本包只保存板卡断开前已经部署、使用或实测的当前版本。没有旧 reuse9/perf10/Sol 实验核、旧网页重构稿、尚未部署的预览补丁、历史模型参数、患者上传和登录凭据。

## 内容与板端对应位置

- acdc_release_source：从 /home/xilinx/acdc_board_v2/releases/hdmi_surface_20261024 取回的 ACDC 应用、网页、HDMI/触控公共模块与应用资源。该版本已部署到两个 ACDC 服务；应用 SHA256 为 c7131026b1a1f2b6668e34ddc628af0d2c536827ebf8e7380359b27cd985186c。
- skin8_runtime：/home/xilinx/skin8_candidate_20261023 对应的当前 Skin8 位流、HWH、双模型编译参数、驱动、验证工具和已更新 web_entry.py/hdmi_policy.py。只含目前使用的皮肤镜和 CAMUS 参数，不带旧实验模型。
- unified_entry：当前 /home/xilinx/pld_entry_20261023 的入口源码（端口8088）。
- systemd：当前 ACDC 两个服务 unit 及 CAMUS 依赖路径修复 drop-in。
- evidence：当前版本既有核心板测和本次 HDMI 生命周期/触控渲染实测。失败和修复后的记录都保留。

这是增量运行版本交接，不是 SD 卡镜像。ACDC 冻结模型、原始 PLD release、系统依赖和既有安装环境未重复打包。ACDC 模型身份以 v2_config.json 为准，不将其他/历史模型替换进来。GitHub PR 只含当前源码、配置和小型证据；当前位流、编译参数、资源二进制位于 Release 压缩包。

## 已验证的行为

ACDC 第四幅分割表面触控拖动旋转，松手恢复完整网格；四幅图均可放大并用左上角橙色 < BACK 返回。用户已确认返回正常、旋转效果可以接受。四宫格拖动预览生成 27–29 ms，不含 HDMI 提交/触控等待，不能当作屏幕帧率。简化只用于拖动显示，不改变模型输出和定量数据。

三模型 HTTP HDMI 生命周期已通过：ACDC 关闭显示后新推理完成；皮肤镜/CAMUS 关闭显示后重新推理 head 与切换前逐字节一致。皮肤镜最后一轮实屏再次确认正常；随后用户断开板卡，CAMUS/ACDC 没有继续该轮目视检查。未声称所有异常/长时间运行无故障。

CAMUS 修复了被触控配置覆盖的 nibabel 依赖目录，以及显示子进程 HardwareLease 所需的 tensor_accel 模块路径。不要删除 95-camus-dependencies.conf 中这些路径。其 HTTP 控制层已持有 busy 信号量，不得在 start_hdmi 包装层重复加锁。

## 继续开发与部署

固定开发目录 C:\Users\60398\Desktop\PLD_Dev。当前本包是可供同学继续开发的快照；不要用旧 handoff 候选包覆盖它。用户已取消 Windows 关机要求。

任何覆盖/重新部署前先确认无推理任务，正常停止入口、hub、相关模型与 HDMI 服务，确认 MainPID=0、cgroup 无子进程、共享 FPGA 锁释放。不要强杀持有 DMA/VDMA 的进程，或在显示运行时加载推理位流。

ACDC 依赖完整的现有 hdmi_surface_20261024 或 hdmi_touch_20261024 release。源码中的原生渲染器在 acdc/appliance 目录编译：

    gcc -O3 -shared -fPIC surface_raster.c -lm -o surface_raster.so

任何文件变化后，从 release 根目录使用 PYNQ Python 运行 `-m acdc.appliance.identity <release绝对路径>` 重新生成清单，再更新两个 unit 的目录、PYTHONPATH、--config 和 --app-sha。包中的 v2_application.json 记录原部署身份，不可用于认证被修改或未补齐依赖的目录。

Skin8 的 web_entry.py 与 hdmi_policy.py 成对部署。skin8_runtime/package.json 是原始构建记录；本次网页文件的最终身份以本交接包 SHA256SUMS.json 为准。

本轮板上备份位置 /home/xilinx/hdmi_surface_backup_20261024；原 ACDC release 保留。需要回退时先正常释放硬件，再恢复备份 unit/入口文件、daemon-reload 并检查健康状态。这里只说明位置，不把旧代码和模型混入交付。
