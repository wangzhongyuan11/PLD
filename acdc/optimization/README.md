# ACDC 下一代计算与存储组件

状态：实现中的实验架构。当前完成精确打包乘法、P256/P384连续乘加与四路Q24、输入空间bank双缓冲、Co8 tile卷积、CAT输入DMA与Pool分流写DMA，以及可逆布局和容量planner。**输入/参数/计算/写回的整任务控制、整网执行表和完整BD尚未完成，没有新完整位流，也没有新架构板测FPS。** 后续范围以[优化计划](../docs/OPTIMIZATION_4FPS_8FPS.md)为准。

## 当前可复现内容

| 组件 |实现与验证|
|---|---|
| `rtl/int8_pair_mul.sv` |共享权重的两个signed INT8乘积，精确负数修正；343定向、20000随机与复位/背压仿真通过，初始OOC映射1 DSP。后续运行`synth_pair.tcl`同时绑定源码|
| `rtl/conv_array.sv` |P=4/6，Ci8×Co8，完整归约后INT32输出，bias一次、溢出检测、tag对齐；157组初始事务和复位恢复通过|
| `rtl/conv_compute.sv` |4组有界结果FIFO、四路精确Q24、ReLU、lane mask/tag/首末拍；P4/P6各96组密集算术、5976输入拍，两次完整运行并含在途复位通过|
| `rtl/tile_banks.sv` |输入双slot、64-bit字节使能写、P个空间bank、活动slot保护、读响应背压、越界状态；每配置520次读，跨512-word页及不完整末页测试通过|
| `rtl/tile_conv.sv` |窗口/tap地址生成、输入/权重读取对齐、Ci尾掩码、Co8计算；P4/P6各23个密集tile，覆盖15层Ci/kernel、24×24×Ci192和32×256 head、2个预期错误，分别比较22032/23052输出拍|
| `layout.py` |CHW↔C8HW8、OIHW↔W8x8可逆，pad为0；空间bank的唯一坐标映射|
| `planner.py` |逐层tile容量与覆盖、输入/权重/输出有效字节、带明确遗漏项的服务时间模型；不是实测调度周期|
| `board_inventory.py` |只读身份/版本/内存/服务/FCLK，不加载overlay或修改服务|
| `rtl/input_transform.sv`、`pool_stream.sv`、`stream_fork.sv` |精确CAT分支查表/掩码/补零，signed MaxPool和独立握手分流；2048查表字、3656原始像素/912 Pool像素通过|
| `rtl/dma_transfer.sv` |72-bit Full命令/64-bit对齐数据、BTT/TLAST/TKEEP/tag/status、独立数据/状态完成和超时；34正常、12预期故障通过，错误锁存至受控复位|
| `rtl/tile_writer.sv` |32-bit计算输出拼成零填充C8像素、full/pool各双行缓存、公平写调度；P4/P6各13374次指定DDR地址写入与独立Conv/Pool参考相等|
| `rtl/parameter_loader.sv` |Co8 DMA包的bias/M头与W8x8权重装载，支持非活动slot；P4/P6均经参数DMA→真实tile算术→Pool/写回完整比对|
| `rtl/input_loader.sv` |C8HW8图像按halo装入bank；CAT分支LUT、全局nearest地址、deep行缓存；P4/P6各8配置/12845次bank写入、实际源读取字数精确核对|
| `rtl/lut_loader.sv`、`dma_read_arbiter.sv` |512字节CAT表通过DMA装入并逐项核验；原子保留读包直到TLAST与status均完成，三客户端30包公平路由和阻塞命令稳定性通过|

P256计算组件OOC：144 DSP、6877 LUT、14634 FF；P384计算组件：208 DSP、10252 LUT、19793 FF。输入双缓冲OOC分别确认72/84 BRAM36等效。这些是**分别综合的组件**，不代表整机资源或完整时序已经合格。OOC接口没有完整系统的输入输出约束，不能用其timing.rpt宣称上板时序通过。

已集成tile组件OOC：P256为148 DSP/88 BRAM36/10263 LUT/12430 FF；P384为212 DSP/100 BRAM36/15641 LUT/17474 FF。包含输入/权重双slot和计算，不含完整DDR搬运与网络控制器。综合裁去打包乘积43位寄存器的高11位属预期：INT8配对乘积有符号结果可由32位表示，拆分只读取低32位；仍以全系统数值与实际布线作后续门禁。

第一版深RAM声明导致P256输入缓冲实际映射128 BRAM36，门禁拒绝。改成512×64显式分页后映射为72，P384为84。失败报告保留在`out/acdc/optimization/banks_p4_before_paging/`；新的综合清单绑定源码及报告/检查点hash。

新writer的P4 OOC为0 DSP/3 BRAM36/784 LUT/899 FF；其中包含Pool。P4/P6的输入/参数loader OOC均为0 DSP。输入loader使用固定网络幂次图像网格的移位地址生成和独立容量检查流水，避免消耗P384剩余DSP。组件资源不能直接当作完整SoC布线结果。

目前DMA集成仿真使用明确的行为级`datamover_source/sink`施加随机背压并核对每次DDR读写；不是已运行厂商IP模型、真实DDR或板测。厂商DataMover完整IP/BD、故障恢复和时序仍需下一阶段完成。当前Python回归31项通过。

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
& D:/Git/bin/bash.exe acdc/optimization/run_tile_sim.sh 4
& D:/Git/bin/bash.exe acdc/optimization/run_tile_sim.sh 6
& D:/Git/bin/bash.exe acdc/optimization/run_stream_sim.sh
& D:/Git/bin/bash.exe acdc/optimization/run_dma_sim.sh
& D:/Git/bin/bash.exe acdc/optimization/run_writer_sim.sh 4
& D:/Git/bin/bash.exe acdc/optimization/run_writer_sim.sh 6
& D:/Git/bin/bash.exe acdc/optimization/run_parameter_sim.sh 4
& D:/Git/bin/bash.exe acdc/optimization/run_parameter_sim.sh 6
& D:/Git/bin/bash.exe acdc/optimization/run_input_sim.sh 4
& D:/Git/bin/bash.exe acdc/optimization/run_input_sim.sh 6
& D:/Git/bin/bash.exe acdc/optimization/run_arbiter_sim.sh
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
& D:/vivado/2026.1/Vivado/bin/vivado.bat -mode batch -source acdc/optimization/vivado/synth_tile.tcl -log out/acdc/optimization/tile_p4_synth.log -tclargs 4
& D:/vivado/2026.1/Vivado/bin/vivado.bat -mode batch -source acdc/optimization/vivado/synth_tile.tcl -log out/acdc/optimization/tile_p6_synth.log -tclargs 6
& D:/vivado/2026.1/Vivado/bin/vivado.bat -mode batch -source acdc/optimization/vivado/synth_stream.tcl -log out/acdc/optimization/stream_synth.log
& D:/vivado/2026.1/Vivado/bin/vivado.bat -mode batch -source acdc/optimization/vivado/synth_writer.tcl -log out/acdc/optimization/writer_p4_synth.log -tclargs 4
& D:/vivado/2026.1/Vivado/bin/vivado.bat -mode batch -source acdc/optimization/vivado/synth_loaders.tcl -log out/acdc/optimization/loaders_p4_synth.log -tclargs 4
& D:/vivado/2026.1/Vivado/bin/vivado.bat -mode batch -source acdc/optimization/vivado/synth_loaders.tcl -log out/acdc/optimization/loaders_p6_synth.log -tclargs 6
```

所有成功标志同时要求退出码0。`synth_guard.py`在综合前后验证源码未漂移，并写组件证据；不签发生产运行包。完整IP/BD构建、AXI故障恢复、15层整网和实测4/8 FPS仍是后续任务。

## 板卡当前检查

2026-10-18只读SSH检查成功：ARMv7/Python3.10.4/PYNQ3.1.1/NumPy1.21.5；系统MemTotal503464 kB，CMA总量131072 kB，当时空闲65676 kB；FCLK为125/100/100/100 MHz。hub和web active、camus inactive，FPGA operating。该快照不证明ACDC已加载。未停止正式服务、未更换overlay，也未存储板卡凭据。

## 紧接着的集成工作

继续完成各路径的整任务集成、新DMA执行表及整网调度，完成完整BD布线，再按计划隔离切换板测。当前tile模块只处理一个Co8块，Co组切换/跨层调度仍需控制器；调试/状态恢复也不能继承旧整网证据。A参数尚缺，密集fixture的数值/性能始终标记synthetic。

当前组件之后的直接任务是：组合input/parameter/LUT读客户端与计算/写回，形成有界双任务预取与缓存所有权控制；再以层间barrier保证上一层写入完成后才预取下一层输入。缓存键须含输入生产者/帧generation与LUT来源，不能只按地址命中；参数缓存按模型/地址/形状绑定。随后连接DDR描述符执行表和AXI-Lite frame接口，完成全网络及完整BD。上述控制尚未落地，不能把单tile正确性扩大为整网已通过。

`vivado/probe_datamover.tcl`已在Vivado2026.1实际创建并验证AXI DataMover5.1：Full MM2S/S2MM、64-bit数据、32-bit地址、23-bit BTT、16拍burst、DRE关闭。GUI配置枚举使用`Full`，不是数字1。该结果仅说明IP配置/实例化模板有效，尚未生成完整搬运系统或测量DDR带宽。

DMA格式按AMD PG022的[Command Interface](https://docs.amd.com/r/en-US/pg022_axi_datamover/Command-Interface)与[Status Interface](https://docs.amd.com/r/en-US/pg022_axi_datamover/Status-Interface)：INCR/EOF、BTT、tag和状态分别核验；不能仅收到TLAST就宣布写入DDR完成。出错时锁存故障，必须确认DataMover停止或受控复位后才复用缓冲。
