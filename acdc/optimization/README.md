# ACDC 下一代计算与存储组件

状态：实现中的实验架构。当前完成精确打包乘法、P256/P384连续乘加与四路Q24、输入空间bank双缓冲，以及可逆布局和物理容量planner。**尚未接通完整tile卷积、DMA和整网控制器，没有新完整位流，也没有新架构板测FPS。** 后续范围以[优化计划](../docs/OPTIMIZATION_4FPS_8FPS.md)为准。

## 当前可复现内容

| 组件 |实现与验证|
|---|---|
| `rtl/int8_pair_mul.sv` |共享权重的两个signed INT8乘积，精确负数修正；343定向、20000随机与复位/背压仿真通过，初始OOC映射1 DSP。后续运行`synth_pair.tcl`同时绑定源码|
| `rtl/conv_array.sv` |P=4/6，Ci8×Co8，完整归约后INT32输出，bias一次、溢出检测、tag对齐；157组初始事务和复位恢复通过|
| `rtl/conv_compute.sv` |4组有界结果FIFO、四路精确Q24、ReLU、lane mask/tag/首末拍；P4/P6各96组密集算术、5976输入拍，两次完整运行并含在途复位通过|
| `rtl/tile_banks.sv` |输入双slot、64-bit字节使能写、P个空间bank、活动slot保护、读响应背压、越界状态；每配置520次读，跨512-word页及不完整末页测试通过|
| `layout.py` |CHW↔C8HW8、OIHW↔W8x8可逆，pad为0；空间bank的唯一坐标映射|
| `planner.py` |逐层tile容量与覆盖、输入/权重/输出有效字节、带明确遗漏项的服务时间模型；不是实测调度周期|
| `board_inventory.py` |只读身份/版本/内存/服务/FCLK，不加载overlay或修改服务|

P256计算组件OOC：144 DSP、6877 LUT、14634 FF；P384计算组件：208 DSP、10252 LUT、19793 FF。输入双缓冲OOC分别确认72/84 BRAM36等效。这些是**分别综合的组件**，不代表整机资源或完整时序已经合格。OOC接口没有完整系统的输入输出约束，不能用其timing.rpt宣称上板时序通过。

第一版深RAM声明导致P256输入缓冲实际映射128 BRAM36，门禁拒绝。改成512×64显式分页后映射为72，P384为84。失败报告保留在`out/acdc/optimization/banks_p4_before_paging/`；新的综合清单绑定源码及报告/检查点hash。

## 命令

仓库根目录PowerShell，Vivado2026.1。运行工具时保持其读取源码不变；脚本和TB只有固定种子和有界watchdog。

```powershell
$env:PROCESSOR_ARCHITECTURE='AMD64'
& D:/Git/bin/bash.exe acdc/optimization/run_pair_sim.sh
& D:/Git/bin/bash.exe acdc/optimization/run_array_sim.sh 4
& D:/Git/bin/bash.exe acdc/optimization/run_array_sim.sh 6
& D:/Git/bin/bash.exe acdc/optimization/run_compute_sim.sh 4
& D:/Git/bin/bash.exe acdc/optimization/run_compute_sim.sh 6
& D:/Git/bin/bash.exe acdc/optimization/run_banks_sim.sh 4
& D:/Git/bin/bash.exe acdc/optimization/run_banks_sim.sh 6
& acdc/.venv/Scripts/python.exe -m pytest acdc/tests -q
& acdc/.venv/Scripts/python.exe -m acdc.optimization.planner --profile P256 --output out/acdc/optimization/plan_p256.json
& acdc/.venv/Scripts/python.exe -m acdc.optimization.planner --profile P384 --output out/acdc/optimization/plan_p384.json
```

独立OOC综合（只在前一进程结束后修改该次源文件；`-log/-journal`须在`-tclargs`之前）：

```powershell
& D:/vivado/2026.1/Vivado/bin/vivado.bat -mode batch -source acdc/optimization/vivado/synth_pair.tcl -log out/acdc/optimization/pair_synth.log
& D:/vivado/2026.1/Vivado/bin/vivado.bat -mode batch -source acdc/optimization/vivado/synth_compute.tcl -log out/acdc/optimization/compute_p4_synth.log -tclargs 4
& D:/vivado/2026.1/Vivado/bin/vivado.bat -mode batch -source acdc/optimization/vivado/synth_compute.tcl -log out/acdc/optimization/compute_p6_synth.log -tclargs 6
& D:/vivado/2026.1/Vivado/bin/vivado.bat -mode batch -source acdc/optimization/vivado/synth_banks.tcl -log out/acdc/optimization/banks_p4_synth.log -tclargs 4
& D:/vivado/2026.1/Vivado/bin/vivado.bat -mode batch -source acdc/optimization/vivado/synth_banks.tcl -log out/acdc/optimization/banks_p6_synth.log -tclargs 6
```

所有成功标志同时要求退出码0。`synth_guard.py`在综合前后验证源码未漂移，并写组件证据；不签发生产运行包。完整IP/BD构建、AXI故障恢复、15层整网和实测4/8 FPS仍是后续任务。

## 板卡当前检查

2026-10-18只读SSH检查成功：ARMv7/Python3.10.4/PYNQ3.1.1/NumPy1.21.5；系统MemTotal503464 kB，CMA总量131072 kB，当时空闲65676 kB；FCLK为125/100/100/100 MHz。hub和web active、camus inactive，FPGA operating。该快照不证明ACDC已加载。未停止正式服务、未更换overlay，也未存储板卡凭据。

## 紧接着的集成工作

连接卷积窗口地址生成、输入bank读取与权重bank读取，在tag/ready对齐下驱动计算组件；完成真实tile层级数值与周期验证。随后实现CAT/Pool流式路径、新DMA执行表及整网调度，完成完整BD布线，再按计划隔离切换板测。A参数尚缺，密集fixture的数值/性能始终标记synthetic。
