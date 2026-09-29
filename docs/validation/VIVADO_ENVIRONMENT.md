# Vivado 2026.1 本机环境检查

运行检查：2026-09-25 22:31–22:38；结果整理：2026-09-26。输入源码基线：`b49c705`。本项检查未修改RTL、PS7参数、Vivado安装文件或系统级环境变量；新构建产物在 `out/vivado_env_check/`，未覆盖原有两版overlay。后续TKEEP修复与板测结果见提交 `307da19`；本次独立占位RTL综合结论不覆盖该新版本。

## 启动异常的原因与复现结论

当前工具启动的进程没有 `PROCESSOR_ARCHITECTURE`，但 `.NET RuntimeInformation.OSArchitecture` 实测为 `X64`。安装目录 `Vivado/bin/loader.bat:31-40` 会在该变量缺失时把提示重定向到NUL并 `exit /b 1`，对应此前“返回1且无输出”。

在当前进程设置 `PROCESSOR_ARCHITECTURE=AMD64` 后，同一 `vivado.bat -version` 正常输出：

```text
vivado v2026.1 (64-bit)
Tool Version Limit: 2026.06
SW Build 6511674
IP Build 6504888
SharedData Build 6501428
```

此时退出码仍为1，因为 `loader.bat:578-580` 的版本查询分支调用prodversion后固定 `exit /b 1`。**版本查询的退出码不是健康检查判据**；下面真正的仿真与批处理综合均以退出码0通过。

直接调用 `bin/unwrapped/win64.o/prodversion.exe` 未初始化依赖路径，曾返回 `0xC0000135`；正确的bat入口初始化后版本查询可用，因此没有据此重装工具或运行库。

## 已通过的运行检查

| 检查 | 结果与证据 |
|---|---|
| 安装入口 | `D:\vivado\2026.1\Vivado\bin\vivado.bat`；Git Bash为 `D:\Git\bin\bash.exe` |
| 许可证 | 工具识别有效Vivado Design Suite BASIC，日志显示有效期至2027-09-19；实际取得并释放 `Vivado_Synthesis`、`Vivado_Implementation` / `xc7z020` 许可。未输出许可证内容 |
| 器件数据库 | batch Tcl `get_parts xc7z020clg400-1` 返回唯一目标；建in-memory工程和综合均成功 |
| IP目录 | `processing_system7:5.5`、`axi_dma:7.1`、`axis_data_fifo:2.0` 均存在 |
| RTL仿真 | 实际运行 `scripts/run_sim.sh`：xvlog→xelab→xsim；reg_slice 272 beats，incr 4953 beats/93包；`ALL_SIMS_PASS`，退出码0 |
| Zynq目标综合 | `axis_incr`及其reg_slice在 `xc7z020clg400-1` 上out-of-context综合完成；51 LUT、66 FF，生成DCP；`ENV_CHECK_SYNTH_PASS`，退出码0 |
| 完整FIFO overlay | 重新生成BD/IP→综合→布局布线→bitstream→XSA；`BUILD_OK pl_core=fifo`，退出码0；bitstream前置DRC为0错误 |
| 布线后时序与资源 | 100 MHz，WNS +2.205 ns、TNS 0、WHS +0.024 ns、THS 0，脉宽无违例；2303 LUT、3267 FF、4.5 BRAM36、0 DSP |
| 产物提取 | 从本次XSA成功提取同名 `dma_top.bit`（4,045,770 B）与 `dma_top.hwh`（342,743 B），未部署到板卡 |

仿真输出目录 `out/sim_logs/` 为共享回归目录，会被后续运行更新；表中4953 beats为本次旧基线的运行结果，后续TKEEP回归的5349 beats见当前项目记录。独立综合探测Tcl与完整日志：`out/vivado_env_check/check.tcl`、`check.log`；报告为 `synth_utilization.rpt`、`synth_timing.rpt`，检查点为 `axis_incr_synth.dcp`。独立模块的综合时序不代表最终BD布线后的时序。

完整FIFO构建日志为 `out/vivado_env_check/fifo_build.log`；新产物及 `timing_summary.rpt`、`utilization.rpt` 位于 `out/vivado_env_check/fifo/`。检查结论：本机Vivado的仿真、Zynq综合/实现、bitstream及硬件平台导出链路可用，DDR配置告警仍需独立处理，不能以构建成功消除。

## 运行方式

以下设置只补齐已实测的X64主机在本工具会话缺少的变量，不改系统注册表或用户永久环境：

```powershell
Set-Location D:\PLD
if (-not $env:PROCESSOR_ARCHITECTURE) { $env:PROCESSOR_ARCHITECTURE = 'AMD64' }
& D:\vivado\2026.1\Vivado\bin\vivado.bat -version
# 上一条的退出码1是该安装版本脚本的行为；应确认版本文本。
& D:\Git\bin\bash.exe scripts/run_sim.sh
# 仿真必须退出码0且打印ALL_SIMS_PASS。
```

本次FIFO构建使用现有 `vivado/build_dma_bd.tcl`，另设输出目录，不覆盖原有 `out/vivado_fifo/`：

```powershell
& D:\vivado\2026.1\Vivado\bin\vivado.bat -mode batch -notrace -nojournal -source D:/PLD/vivado/build_dma_bd.tcl -log D:/PLD/out/vivado_env_check/fifo_build.log -tclargs fifo D:/PLD/out/vivado_env_check/fifo
& D:\conda\python.exe scripts/extract_xsa.py out/vivado_env_check/fifo/dma_top.xsa out/vivado_env_check/fifo/dma_top
```

## 告警与验证边界

- `Synth 8-7137`：独立综合对 `axis_reg_slice.sv` 的 `data_q_reg` 给出set/reset优先级告警。RTL仿真已通过，但本轮未做综合网表功能等价验证；保留告警，未为消除告警擅改RTL。
- `Timing 38-242`：独立模块OOC时钟未设 `HD.CLK_SRC`，不能估计完整时钟偏斜；最终以BD布线后报告为准。
- `PSU-1`～`PSU-4`：PS7 DDR DQS-to-clock参数为 `-0.051/-0.006/-0.009/-0.033`。已核对 `vivado/ps7_config.tcl` 与原始 `PYNQ_tutorials/dma/dma_tutorial.tcl` 数值一致；这只能证明来源，不能证明板上DDR稳定。不得为让告警消失直接改成0。
- 旧 `vivado/smoke_synth.tcl` 仍使用Artix-7、宣称缺少Zynq器件，并对 `synth_1 impl_1` 同时使用 `-to_step write_bitstream`。本轮不使用该过时脚本，实际验证使用Zynq目标与现有BD构建脚本；源码保留未改。
- 本项环境检查未连接JTAG、未下载新bitstream、未测试HDMI，也未检查Vitis/HLS工具链。其他工作已记录的板测结果见 `board/README.md`，不与本项环境检查混称。
