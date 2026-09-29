# 阶段3/4：B 计算子系统验证记录

2026-09-27补充：用户已确认C接受当前ABI及资源预算，阶段3/4全部条款闭合。以下2026-09-26记录中的“待C确认”为当时状态，已由本次确认关闭；显示工程集成仍未执行。后续阶段5按最新要求先完成B独立优化与测试评估。

2026-09-26，基于当前 A 冻结交付，已完成正式 RTL、PS 驱动、整网调度、100 MHz 布线和 PYNQ-Z2 整网板测。阶段0/1/2保留。B 的实现与硬件数值闭环已通过；C 对接口/资源预算的确认尚未收到，不能声称双方联调完成。阶段5的实时性能、HDMI、后处理、测试集评估和长期稳定性仍未验收。

## 验证结果

| 层次 | 结果 | 证据与边界 |
|---|---|---|
| A 基线 | 124文件hash冻结；15/15小块acc、requant、参数切片一致；NumPy整网/分块参考逐层input/output零差异 | `reference/baseline_manifest.json`；默认拒绝参数漂移，换版生成独立manifest |
| 原始JPEG | 本机Pillow12.3/JPEG8、PS Pillow9.0.1/JPEG8均生成完全一致的196608字节首层输入 | `out/stage34/board_input_int8.json`；Conda Pillow12/JPEG9的6783字节差异来自解码，不能放宽容差掩盖 |
| 算术RTL | MAC 7143拍/166结果；Q24 14515结果，均通过 | 正负ties-even、饱和、溢出、背压、复位 |
| 卷积tile | 小块/容量/尾通道/复位19次、89334字节一致，8种配置拒绝；完整15层80 tile、3866624字节一致 | 最终配置流水版本重跑，`tb_conv_tile_engine_full.log`，退出0；单层分别使用该层golden input |
| 非卷积tile | 21次、127223字节一致，10种配置拒绝 | 14处A层间转换小块及MAP/POOL/UPSAMPLE/CONCAT_UP，真实共享RAM |
| 正式顶层协议 | 33次计算、67次加载、8种错误、49533输出字节一致 | 全15层参数包、小块计算、14处层间转换、TKEEP/TLAST、忙时写/START、末拍背压、计算中复位；`tb_axis_tensor_accel.log` |
| 主机回归 | 26/26通过 | DMA缓存/恢复、独立长度、图调度/跳连/元数据及独立数值oracle；软件oracle不是板测 |
| 正式构建 | Vivado2026.1 batch退出0，bit/hwh与同次XSA一致；源码与打包RTL哈希一致 | `out/vivado_cnn/build_manifest.json`，`BUILD_DONE mode=cnn` |
| 布线时序 | 100 MHz；WNS +0.139 ns、TNS 0；WHS +0.014 ns、THS 0；WPWS +3.750 ns | 布线后报告；不是OOC估计 |
| 整网板测 | 两次连续帧和故障恢复后第三帧，15层input/output及最终65536字节head均mismatch=0 | PS原始JPEG预处理后，只向网络提供一次首层输入；中间golden仅比较，不注入 |
| 板上恢复 | 短INPUT包error2、缺包50ms超时、计算中RESET均恢复 | 计算中复位前COMPUTE_CYCLES=48685、BUSY=1、SENT=0且RX已启动；之后完整下一帧正确 |
| 独立下载核对 | 下载板上head再次与A完整head精确比较通过 | SHA256 `2b497527f2157b1fe31f74a5f5bc06816debf455adbb1788358320defefbf245` |

板上部署目录：`/home/xilinx/jupyter_notebooks/pld/stage34_cnn_20260926_v1/`。137个部署文件在执行前逐项hash通过。主机板测JSON为`out/stage34/cnn_board.json`；结果为`out/stage34/head_int8.bin`。当前PL已加载CNN overlay，验证程序正常结束并复位核心/DMA后释放缓冲区，不存在后台持续推理。

## 实现、存储与预算

正式合同：[寄存器与命令](../design/ACCELERATOR_REGISTERS.md)、[数据合同](../design/ACCELERATOR_ABI.md)、[内部bank布局](../design/TILE_ENGINES.md)。AXI-Lite核心地址`0x43C00000/4KiB`，DMA`0x40400000`；64-bit AXIS、simple DMA、100 MHz。PS使用有界MMIO驱动，不调用PYNQ DMA的无界等待。

每帧15次参数加载、14次LUT加载、80个卷积tile、100个非卷积tile，共389条命令。权重在同层各tile之间复用；跳连保留编码器1/3/5层输出，分别在decoder12/10/8消费后释放。全部卷积、Q24、ReLU/尺度映射、Pool、Upsample、Concat在PL执行；PS负责预处理、LUT参数生成、halo/字节拷贝、调度和结果拼接。

| 资源或流量 | 当前实现 |
|---|---|
| Routed Slice LUT / FF | 7047 / 7204；构建门禁保守计数LUT原语7865，不能和合并后的Slice LUT混用 |
| BRAM36等效 / DSP | 67 / 51，其中50个BRAM36+34个BRAM18；计算引擎特征/权重存储64 BRAM36等效 |
| B侧上限 | 90 BRAM36、96 DSP、32000 LUT；为C至少预留50 BRAM36、124 DSP、21200 LUT，尚待C确认 |
| DMA连续DDR | TX 131072字节；RX 65536+64 guard字节；通过pynq.allocate，地址按64bit对齐 |
| 跳连DDR | 三支同时保留917504字节；普通NumPy特征另有输入/输出与临时拷贝，不假定全部位于连续DMA内存 |
| 最大单张特征 | decoder2输入24×256×256=1572864字节；按tile送PL，不要求整图放进BRAM |
| 每帧参数/LUT | 124584 / 4352字节；参数包括121760字节weights、1412字节bias、1412字节Q24 |
| 每帧MM2S / S2MM | 11020424 / 9273344字节；总有效载荷20293768字节，包含halo重复搬运，不含AXI开销与PS拷贝 |

## 实测性能口径

三帧逐层核对用时3.473 / 3.438 / 3.406秒，包含文件golden读取/比较。另一次关闭golden比较与逐层打印的预热后运行：**2.874秒/帧，B子系统0.348 FPS**，结果仍在计时后精确核对。仅1帧基准，不能推导P95或长期吞吐。

计时包含PS分块、配置、参数/LUT/特征DMA、PL计算及拼接；不含模型加载、overlay编程、JPEG解码/预处理、C后处理或显示。每帧硬件计算计数71385808周期，100MHz约0.714秒；卷积单层仿真共30097648周期约0.301秒。DMA等待约1.280秒包含等待PL产生输出，不能与PL计算时间相加当独立阶段。实际每帧约20.3MB搬运，加上389条Python/MMIO调度和串行字节读回，尚未达到阶段5的≥1 FPS目标。当前未测同口径PS CPU加速比。

## 时序修复与保留告警

第一版正式BD布线WNS=-9.190ns，已拒绝部署。根因是配置尺寸连乘、非卷积地址经外部BRAM的组合长路径及顶层包长度连乘。分别加入配置面积/容量流水、READ_PREP/ADDR_MUL/ADDR_SUM地址寄存、各分支字节长度流水，并提前寄存输入末字节标志。未降低100MHz、未增加掩盖路径的时序例外。最终相关仿真和完整布线均重跑通过。

独立算子OOC正裕量未覆盖跨模块RAM路径，不能替代正式顶层实现。共享RAM须先仲裁地址/写数据，再使用单一RAM访问语句，避免推断成大量LUT RAM。

DRC无Error/Critical Warning，但保留127项报告：DSP流水建议、AMD DMA/interconnect的异步BRAM控制与LUT方程告警、无可布线负载及WRITE_FIRST建议等；`CHECK-3`提示同类报告数量限制，不能称“零告警”。PS7 DDR参数告警保留，长期板级稳定性在阶段5继续验证。

## 复现入口与验收余项

对照`PLAN.md`逐条复核：

| 条款 | 当前判定 | 对应证据 |
|---|---|---|
| 3① 冻结模型、小块acc/requant/切片 | 通过 | 124文件manifest、15层参考核验 |
| 3② 数值规则、算子及完整单层 | 通过 | 可执行预处理/整网参考、MAC/Q24/非卷积TB、最终80个完整卷积tile |
| 3③ B/C冻结数据与控制合同 | B实现通过，双方确认待完成 | ABI 1文档、正式顶层和驱动；未收到C确认 |
| 3④ 计算、存储、带宽与预留 | B侧预算与实测已交付 | 本文资源/流量表；C接受预留额度仍待确认 |
| 4① 所有算子、逐层和head一致 | 通过 | 板上三帧各15层输入/输出、下载head再次精确比较 |
| 4② 分块/跳连/帧/背压/恢复 | 通过 | 主机图调度与halo测试、正式顶层TB、连续帧和三种板上恢复 |
| 4③ 可重建IP、时序、约定资源 | 构建/时序/B预算通过，C约定待确认 | bit/hwh/XSA与源码hash绑定、布线报告；不是双方资源签收 |
| 4④ 板上推理及可复现控制/加载/恢复 | 通过 | 部署清单137文件hash、`cnn_board.json`、README复现命令 |

提交`a14042e`后的完整性复核再次确认：当前源码与部署包内运行时相同；当前bit/hwh与板测报告所记hash相同；三帧每帧15层、389命令、71385808计算周期，三种故障恢复均有证据。未重跑无关硬件测试，也未把C尚未确认的条款改写为已完成。

本机完整命令和SSH部署见[board/README.md](../../board/README.md)。`scripts/package_cnn_board.py`强制检查构建清单、源码/产物哈希和XSA配对；不要部署早期负时序产物。A换版使用独立manifest和部署目录，保留本基线。

阶段3的数值/算子/数据合同和阶段4的B实现/构建/整网硬件验证已具备。仍需C确认接口与资源预算；没有C显示工程联调证据。阶段5另行完成≥1 FPS、端到端P95≤1秒、CPU基线、379张测试集、后处理/HDMI/UI及≥10分钟且≥1000帧稳定性，不以当前三帧通过替代。
