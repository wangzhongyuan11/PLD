# ACDC 下一代计算与存储组件

2026-10-21当前更新：P6完整100MHz布线、1000帧8.151268 FPS/P95 106.257678ms、18物理节点与3类恢复、40体1616片PL、15层实测和原系统回归均通过。当前报告见[验证记录](../docs/VALIDATION_P6_20261021.md)。xsim BASIC不能启动59147实例P6网表；新增有界DDR模型先在xsim RTL与原模型得到相同结果，Icarus完整综合网表补充检查仍在运行。下面2026-10-19文字是历史快照，不覆盖此更新。

状态（2026-10-19）：已接入冻结A模型，形成双任务预取、缓存复用、层屏障、DDR整网执行表、frame调度及ACD8 AXI-Lite/PS后端。厂商DataMover完整BD已实现；前两次100MHz候选因容量检查、地址和BRAM输出路径的时序违例被拒绝。当前增加地址/计算输入两处弹性寄存器，P4/P6各23 tile、P4真实参数小网格AXI顶层、P6真实参数小网格整帧及3故障回归通过；完整尺寸新流水仿真和完整布线仍在运行。**真实PL数值、恢复、4/8 FPS尚未验收。** [ABI](ABI.md)记录整网接口，[实现状态](../docs/IMPLEMENTATION_STATUS.md)区分实际证据，[最终计划](../docs/FINAL_ALIGNMENT_OPTIMIZATION_PLAN_20261019.md)列出剩余门禁。下文初始组件结果只保留其历史范围。

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
| `rtl/tile_job_queue.sv`、`tile_executor.sv` |双任务顺序、输入/参数双缓存、epoch/生产者身份、层写回屏障，串联Conv/CAT/Pool及DDR写回|
| `rtl/frame_dispatch.sv`、`frame_engine.sv`、`frame_control.sv`、`spatial_accel.sv` |一次START读取DDR执行表，独立AXI-Lite AW/W、完成/故障锁存及计数；实际模型小网格回归逐写比较|
| `program.py`、`image.py`、`device.py` |真实物理地址执行表、派生参数、长期CMA、范围同步、独立head、HALT/共同复位与unsafe隔离|
| `vivado/build_spatial.tcl`、`package.py`、`board_verify.py` |厂商完整BD、构建前后源哈希/配对/时序门禁、实际板测入口；具备入口不代表板测已通过|

P256计算组件OOC：144 DSP、6877 LUT、14634 FF；P384计算组件：208 DSP、10252 LUT、19793 FF。输入双缓冲OOC分别确认72/84 BRAM36等效。这些是**分别综合的组件**，不代表整机资源或完整时序已经合格。OOC接口没有完整系统的输入输出约束，不能用其timing.rpt宣称上板时序通过。

已集成tile组件OOC：P256为148 DSP/88 BRAM36/10263 LUT/12430 FF；P384为212 DSP/100 BRAM36/15641 LUT/17474 FF。包含输入/权重双slot和计算，不含完整DDR搬运与网络控制器。综合裁去打包乘积43位寄存器的高11位属预期：INT8配对乘积有符号结果可由32位表示，拆分只读取低32位；仍以全系统数值与实际布线作后续门禁。

第一版深RAM声明导致P256输入缓冲实际映射128 BRAM36，门禁拒绝。改成512×64显式分页后映射为72，P384为84。失败报告保留在`out/acdc/optimization/banks_p4_before_paging/`；新的综合清单绑定源码及报告/检查点hash。

新writer的P4 OOC为0 DSP/3 BRAM36/784 LUT/899 FF；其中包含Pool。P4/P6的输入/参数loader OOC均为0 DSP。输入loader使用固定网络幂次图像网格的移位地址生成和独立容量检查流水，避免消耗P384剩余DSP。组件资源不能直接当作完整SoC布线结果。

DMA数值集成仿真使用行为级DDR/DataMover模型施加随机背压并核对每次DDR读写；不是已运行厂商IP模型、真实DDR或板测。完整厂商BD已写出，时序、板上恢复与性能仍待验收。当前Python回归46项通过，日志`out/acdc/integrated_regression_v4.log`。

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

所有成功标志同时要求退出码0。`synth_guard.py`在综合前后验证源码未漂移，并写组件证据；不签发生产运行包。完整IP/BD构建、真实AXI故障恢复、实际A模型的RTL/板端整网验收和实测4/8 FPS仍是后续任务。

## 板卡当前检查

2026-10-18只读SSH检查成功：ARMv7/Python3.10.4/PYNQ3.1.1/NumPy1.21.5；系统MemTotal503464 kB，CMA总量131072 kB，当时空闲65676 kB；FCLK为125/100/100/100 MHz。hub和web active、camus inactive，FPGA operating。该快照不证明ACDC已加载。未停止正式服务、未更换overlay，也未存储板卡凭据。

## 紧接着的集成工作

packed-tag修改的P4 OOC综合网表小空间全网现已通过：272任务、16768次写回、312848周期，独立DDR oracle零差异。该网表早于容量/弹性流水修改，不继承为当前完整BD的时序或板测证据。A模型格式、真实几何策略与20患者40体软件复现已完成；密集合成fixture仍保持synthetic身份。

2026-10-20更新：P4弹性流水完整真实模型RTL通过（11780473周期），完整SoC 100MHz布线通过（WNS +0.023ns、WHS +0.016ns），146 DSP、94.5 BRAM36等效，配对bit/HWH/XSA已认证。DRC仍有264项Warning和3项Advisory，详见`../docs/IMPLEMENTATION_STATUS.md`；尚无实际ACDC板测。P6弹性流水完整仿真/构建继续运行。下一步隔离上板验证18个物理节点及head/恢复/性能、40体真实PL结果与原系统回归。缓存键保留输入生产者/帧epoch/LUT来源；跨帧参数缓存优化需独立model generation设计。具体依赖、容量与退出条件见最终计划第8～14节。

`vivado/probe_datamover.tcl`用于复现IP配置探测；`vivado/build_spatial.tcl`已集成PS7、HP2读/HP0写、Full MM2S/S2MM、64-bit数据、32-bit地址、23-bit BTT、16拍burst、DRE关闭及GPIO HALT/reset。GUI配置枚举使用`Full`，不是数字1。源码或构建启动均不等于DDR带宽/板卡验收。

DMA格式按AMD PG022的[Command Interface](https://docs.amd.com/r/en-US/pg022_axi_datamover/Command-Interface)与[Status Interface](https://docs.amd.com/r/en-US/pg022_axi_datamover/Status-Interface)：INCR/EOF、BTT、tag和状态分别核验；不能仅收到TLAST就宣布写入DDR完成。出错时锁存故障，必须确认DataMover停止或受控复位后才复用缓冲。
