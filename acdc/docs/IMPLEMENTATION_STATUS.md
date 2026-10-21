# ACDC Axial 实施状态

更新：2026-10-21。目标仍为[最终计划](FINAL_ALIGNMENT_OPTIMIZATION_PLAN_20261019.md)的完整链路；当前结果见[验证与使用记录](VALIDATION_P6_20261021.md)。交付包和补充网表验证尚待终态，不将目标标记完成。

## 2026-10-21 完整PL链路与回退回归

- 新增`validate_performance_comparison.py`按结果manifest认证并逐片配对20患者/1616片CPU参考与远程PL输入/head SHA。历史新鲜窗口为2.141482与5.164175 FPS，均为各患者窗口合计；不是控制负载/冷热状态的孤立硬件速度实验。3项输入身份/缓存缺片/backend回归通过。[软件基线对照](evidence/cpu_PL_comparison_20261021.json)。
- 用户选择暂时停`pld-perf10.service`并切回ACDC。该外部服务停止时SIGBUS，保留异常，未按干净交接放行；在无残留进程/独占锁条件下，受控全量重配置P6并确认新DMA共同复位，再恢复原P6 worker。新增一片A固定整数向量head全等，网页远程通路可用。[恢复审计](evidence/manual_handoff_20261021.json)。未改perf10源码、unit或开机设置。
- `validate_preprocessing.py`直接调用固定SHA的A几何实现，覆盖20患者/40体/1616片：采用的原生/工作affine、float32工作体、相邻三层量化INT8均全等，并与实际PL输入checkpoint SHA逐片一致。此项不执行新推理，不增加板测样本数。[预处理审计](evidence/preprocessing_A_20261021.json)。
- `run_vendor_datamover_sim.ps1`将B的`dma_transfer`接入当前P6构建生成的AMD DataMover功能网表和带背压的AXI RAM：5组读写回环逐字节正确，包含跨4KiB，74读burst/73写burst；读/写SLVERR、超时、活动复位后的新事务及HALT完成检查通过。[厂商IP证据](evidence/vendor_datamover_20261021.json)。此项不等于PS7/物理DDR时序仿真，也不替代尚在运行的P6计算网表检查。
- 固定20患者/40体/1616片全部P6新鲜推理通过，逐片输入/head SHA对照独立CPU；原生标签、affine、容积/EF全等，20个ZIP SHA/CRC通过。Axial宏Dice 0.8823042897748437，与A逐体指标/容积差异0。[40体审计](evidence/development_PL_20261021.json)。
- PC远程1616片中位174.748400ms、P95 219.439400ms、最大247.678500ms；逐例新鲜推理窗口合计5.164175 FPS。板内1000帧8.151268 FPS/P95 106.257678ms仍为PS-DMA-PL性能证据，两种口径分列。
- `optimization/profile_layers.py`完成15层×3次真实孤立测量：输入由PL前缀产生，每层及融合Pool输出与独立oracle全等，没有中间golden注入。C09/C11/C13的PL中位分别13.774/14.028/16.258ms，完整分解见[板端回归证据](evidence/board_regression_20261021.json)。
- 正常停ACDC→恢复原hub→Skin两张、CAMUS两张真实PL黄金head/PNG/ZIP回归通过→正常释放原系统→恢复P6 worker。原正式源文件、unit和交付指针未修改。恢复hub后必须等认证状态ready，再切换；首次启动期的两个交接失败记录保留。
- 当前网页已恢复8092，使用P6 runtime。patient039网页82片新鲜推理、原生结果、实际浏览器ZIP/PNG导出、MPR/3D及ED/ES切换通过；服务器上传至ZIP就绪20.306547s，远程单片P95 227.7ms。[网页审计](evidence/web_P6_20261021.json)。
- P6 xsim功能网表受50000实例许可限制。补充Icarus运行使用原厂公开UNISIM模型及同一综合网表，没有修改网表；替代DDR存储先在xsim上以相同272 job/16768写回/313888周期通过，[存储模型证据](evidence/portable_DDR_20261021.json)。Icarus综合网表在首次DDR读写前因未知控制状态失败，LUT/FDRE/GSR基础探针通过，原因待查，未记通过。

## 2026-10-20 板测更新

- P6 `bankstage_v1`完整256×256真实参数RTL通过（9465241周期），完整SoC 100MHz布线通过：WNS +0.189ns、WHS +0.013ns、WPWS +3.750ns；21143 LUT、28274 FF、106.5 BRAM36、210 DSP。构建及bit/HWH/XSA配对已认证。
- P6隔离运行包`acdc_spatial_20261020_p6_rpc_v1`完成20预热+1000新鲜帧，18个物理节点、21输入head和3类故障恢复全等。独立复核全部1000帧head SHA、P95和墙钟：**8.151268 FPS**、P95 **106.257678ms**、最大116.584966ms，墙钟122.680298秒；中位PL 95.540735ms、完整PS 105.822606ms。RSS采样87,326,720～87,719,936字节，CMA空闲33,390,592～34,205,696字节。见[独立板测审计](evidence/p6_board_20261020.json)。性能门槛达到G8；完整40体PL、原系统回归和最终交付仍待完成。
- P6 OOC功能网表已生成（210 DSP/104 BRAM36），仿真未能启动：59147实例超过xsim BASIC的50000限制。明确记为工具受限，不记为数值通过；此前P4网表的通过不覆盖此版本。
- 新增`validate_restore.py`，对固定40体/1616片的已认证整数head，独立调用A冻结`geometry.py`和Torch softmax，与B NumPy/Collector比较。最大原生概率误差 **1.7881393432617188e-7**、平面误差 **2.384185791015625e-7**，原生标签差异 **0**，均满足1e-6/零标签差异门槛；此为后处理审计，不计新鲜PL推理。[证据](evidence/restoration_A_20261020.json)。本地新增边界测试后51项通过。
- P6常驻服务交接尚在处理：恢复hub后不能立即停止；第一次在启动期收到SIGTERM退出15，第二次HTTP已监听但后台模型尚未就绪，hub关闭时连接后台失败退出1。安全门禁均拒绝交接，未放宽退出要求；改为等待认证`/hub/status`达到`phase=ready`后再切换。原正式unit和源码均未修改。

以下保留同日较早的候选证据，不代表上述P6当前交接已完成。

- 晚间当前隔离部署为`acdc_spatial_20261020_p4_native_v3`，使用已布线通过的P4位流和NEON布局复制。21输入及3类恢复短测通过，持续6.746495 FPS、P95 137.355077ms；该版本尚未继承旧运行包1000帧验收。运行包SHA为`642181a2877c63628620dcaa76491e4a80f31d4a093fcfc5c50233d0db291671`。
- 修复PC远程后端持有空闲健康检查连接、断连后HTTP状态未清理的问题；握手后关闭连接、超过20秒空闲重连，传输异常清理连接并明确失败，不自动重放不确定的推理。旧网页任务失败记录保留。历史任务不在当前结果目录时清理浏览器旧指针并显示可操作提示。
- 真实网页上传patient039 ED/ES，经远程worker完成82个新鲜PL head，逐片输入/head SHA及最终统计与独立CPU参考全部一致。上传体到ZIP就绪24.412727秒；下载ZIP字节SHA/CRC通过，浏览器实际显示MPR、三维、ED/ES及定量结果。任务`08cffded91014daf8c01143ad8243e98`，证据见[网页板测审计](evidence/web_pl_pair_20261020.json)。切片持续4.155660 FPS，远程P95 252.029245ms，后者尚超250ms；板内PS推理与PC远程请求计时分别报告。
- P6地址修复版`spatial_p6_100_groupindex_v2`完整真实模型RTL通过9465150周期，但布线WNS仍为−1.062ns、WHS +0.006ns，拒绝上板。资源21455 LUT、106.5 BRAM36、210 DSP。最差路径转为BRAM分页/slot选择、空间旋转至DSP输入。新增一级弹性数据寄存器，权重/tag/掩码同步传递；P4/P6各23个tile回归通过，P6真实参数缩小网格全网通过313888周期。`bankstage_v1`完整尺寸仿真和布线仍在运行，不宣称通过。

- 网络恢复后，冻结P4运行包`runtime_spatial_p4_20261020_v1`已真实上板。21个输入head全等；独立整数见证的15个Conv和3个Pool物理输出全等。3个虚拟CAT通过下游Conv及loader仿真覆盖，不冒充物化节点。
- 1000帧、20次预热、3类故障恢复通过。真实墙钟195.783150s，持续 **5.107692 FPS**；P95 **181.087701ms**，最大188.549169ms。每帧head SHA已按对应case独立审计，运行包/位流/报告哈希一致。证明该候选达到G4的性能门槛；40体PL、网页导出、原系统板测和最终交付仍未完成，因此不标记最终G4完成。证据见[板测审计](evidence/spatial_board_20261020.json)。
- 长测RSS采样88,805,376～89,067,520字节；程序保留帧记录会有有界增长。CMA空闲46,530,560～48,054,272字节，不将其波动解释成单一进程泄漏或“零增长”。该次成功长测结束后已重启正式hub。第一次长测因hub已active而在加载位流前被Lease拒绝，零帧，失败记录保留。
- 当前弹性P4综合网表使用A真实参数、缩小网格通过：272 job、16768次写回、313976周期。源码范围为`executor_p4_synth_elastic_v1`，早于下面P6地址修改。完整尺寸仍以RTL和真实板测证明。
- P6弹性版完整真实模型RTL通过（9465150周期），但完整布线WNS **−12.370ns**，拒绝上板。最差路径是宽位数除6/取模到地址寄存器。已以`x_group`递增索引及小范围银行选择替换，并收窄启动列数除法；P4/P6各23个tile、各8个loader用例通过，`groupindex_v2`完整仿真/构建仍待结果。
- 原系统本地回归250项，3项环境跳过，其余通过。网页套件首次从错误目录调用导致导入失败，改从`display_web`目录调用后78项通过；未修改正式源码。日志在`out/acdc/original_system_regression_20261020`。
- 新增PS原生布局复制。主机48项集成测试通过，ARM独立布局验证通过：输入NumPy约20.5ms/NEON约1.44ms，输出约34.0ms/2.26ms。这是普通内存复制测试，不能直接换算成PL FPS。首次整网候选在申请16MiB CMA时失败，未启动DMA；后续按执行表页对齐申请P4 **6,008,832字节**、P6 **6,004,736字节**，候选整网验证进行中。

## 已完成的软件闭环

- `release.py` 按固定 MANIFEST SHA 加载 A 的真实 Axial 模型；原 spec、参数和 checkpoint SHA 保留。`import-release` 导出只含必要材料的65文件子集，保留原清单和许可，不复制数据集。错方向、缺失哈希、参数篡改均拒绝。
- `model.py` 保留原内部/合成格式，新增发布包分派。真实研究模型的资格与物理板验收状态分别记录。
- `witness.py` 独立 Torch FP64 整数见证与 `integer.py` INT64 参考、tile emulator 的21节点全部精确一致，最终262144元素与 A 原始黄金head相等。
- `volume.py` 通过固定发布清单中的影像SHA识别 ACDC unknown-unit 数据；普通未知单位仍拒绝。新增单位解释及原始来源 sidecar，Torch构建后缀与数值版本分开记录。PC依赖已对齐SciPy1.18.1。
- `TorchReferenceBackend` 是显式CPU参考，不是PL回退；利用已证明INT32上界下的精确FP64卷积完成批量验证。
- `validate_development.py` 已实际重新处理固定20名开发患者、40体、1616个Axial堆栈。宏Dice **0.8823042897748437**；逐体逐类Dice及预测容积相对A报告的最大差异均为 **0**。这是开发集合的软件复现，不是独立测试或板测。
- 真实 patient039 ED/ES 已完成NIfTI、三平面MPR、无平滑3D、容积/EF和ZIP。其六项容积与A报告一致。原82片INT64参考运行保留，后续可从身份绑定checkpoint重新恢复/导出，不计新推理。
- 新增PLY、EF CSV；持续吞吐改为完整新鲜任务时间跨度，保留单片服务率为另一个字段。
- 浏览器已实际检查ED/ES切换、MPR、三维；按分割表面范围自适配三维显示。真实PNG/ZIP下载事件成功，ZIP下载字节SHA与服务器产物一致且CRC通过；浏览器未见warn/error。
- `prepare_board_cases.py` 从已验证CPU结果准备21个有界输入/head及独立节点见证：1个A固定整数向量、10患者各ED/ES一个正式float32堆栈。

本轮40体复现使用Python3.12.5、NumPy2.3.5、SciPy1.18.1、Nibabel5.4.2、Torch2.11.0的CPU路径。A声明Python3.12.14/Torch2.11.0+cu128；解释器patch和构建变体不同，已在此B环境实际复现40体零差异，不宣称环境字节相同。已有Python3.13工程环境的SciPy也已升级并重新冻结；其回归单独记录。

## 硬件演进记录（早间快照，当前状态以上节为准）

- packed-tag修改后，tile_executor P4/P6 OOC重新综合分别为148/212 DSP、92/104 BRAM36；异常量化裁剪没有再次出现。该版本P4功能网表回归已通过，适用范围见下文，不能单凭DSP数量宣布网表数值通过。
- A真实参数的P4完整256×256 RTL已通过：1040 job、1073152次写回、11777085周期。该结果在后续容量控制流水修改之前产生，保留其源码范围。
- 新增完整PS7/DataMover Full/HP0+HP2/ACD8 BD；GPIO提供共同复位、双向HALT及停止/错误状态。命令时钟端口已按厂商IP定义修正。
- 首次完整100MHz布线的setup WNS为 **−4.359ns**。关键路径来自tile容量检查中两级DSP乘积加比较；候选已拒绝，报告保存在`out/acdc/optimization/rejected_capacity_p4_100/`，没有上板。
- `tile_conv.sv` 将平面容量乘积、Ci容量乘积和比较拆成寄存器阶段。P4/P6的23 tile回归与P4真实参数小空间全网通过。该内核综合为146 DSP/92 BRAM36；此后完整布线暴露的路径与修复如下。
- 容量流水修改后的P6真实模型完整256×256 RTL通过：1024 job、1073152次写回、9465241周期。该周期只属于随机延迟DDR行为模型，不构成真实FPS。
- 中间P4布局报告仍有约−2.45ns违例，关键路径转移为坐标地址生成到BRAM使能、分页BRAM输出到DSP。其布线WNS−2.826ns，已终止并归档，未生成可验收位流。
- 两处弹性寄存器已实现并保持每周期吞吐；P4/P6各23 tile回归通过，P4真实参数小网格AXI顶层315256周期通过，P6真实参数小网格整帧及3种执行表故障通过。
- **当前P4完整256×256真实模型RTL通过**：1040 job、15 Conv、21节点、1073152次写回、11780473周期；输入缓存命中633次、参数缓存命中567次。日志`out/acdc/optimization/real_elastic_p4_d1_sim/simulation_resumed.log`。行为DDR模型下约117.8ms的周期预算不能当作PS-DMA-PL实测延迟。
- **当前P4完整SoC 100MHz布线通过**：`out/acdc/optimization/spatial_p4_100_elastic_v3`，WNS **+0.023ns**、WHS **+0.016ns**、WPWS **+3.750ns**；17173 LUT、23154 FF、94.5 BRAM36等效、146 DSP。中断后从同次优化检查点继续完成，源码快照保持一致，bit/HWH/XSA配对及构建认证通过。
- 当前P4 DRC报告267项：264 Warning、3 Advisory、0 Error、0 Critical Warning。248项为DSP输入/输出流水建议；其余为厂商AXI转换/FIFO/DataMover内的LUT项、异步RAM控制、无负载网络和WRITE_FIRST提示。未降低规则级别或屏蔽警告。正常HALT后共同复位是运行策略，异步RAM警告与真实故障恢复仍须板测，不能以静态时序通过替代。
- P6弹性流水候选`spatial_p6_100_elastic_v1`的完整构建及完整尺寸真实模型RTL于2026-10-20启动，结果待定。P4时序裕量仅23ps，不在该产物上直接提频。
- 实际PL数值、吞吐与稳定性尚未验收；4/8 FPS均未达成证明。机器可读证据见`evidence/spatial_implementation_20261019.json`，其中日期表示文件首次建立日期，内容记录当前状态。
- packed-tag修复后的P4独立OOC网表小空间全网功能仿真已通过：272 job、16768次写回、312848周期；原keep/tag常量化问题在该版本关闭。网表早于容量/弹性流水变更，其通过不覆盖当前整机版本。

前次板卡登录确认原`pld-hub`、`pld-web` active，`pld-camus` inactive。2026-10-20 09:50（主机时间）复查192.168.2.99:22连接超时，尚未进入认证；本轮没有上传正式候选、停止服务或加载ACDC overlay。待上传独立运行包为`out/acdc/runtime_spatial_p4_20261020_v1.zip`。板卡时钟仍不准确，以主机时间和monotonic计时记录证据。

## 新PS运行路径

- `optimization/device.py`：独立ACD8设备，长期CMA、一次START整网、参数/LUT常驻DDR、输入/执行表范围同步、独立head交付、有界等待和HALT后共同复位。HALT不能确认时保留unsafe状态，不能宣称安全恢复。
- `optimization/package.py`：完整布线与bit/HWH/XSA配对门禁、资源/源代码/打包RTL核对；构建前快照入口保持纯标准库，避免Vivado环境导入数值库。
- `optimization/board_verify.py`：真实板测入口，输入/head先校验；可重放prefix比较15Conv+3Pool的物理DDR输出。虚拟CAT不伪造为板上物化张量，通过其下游卷积及loader RTL覆盖。
- 同一入口提供错误descriptor、非法Ci和活动帧取消注入；每次恢复后必须从首层重新执行并比对head。当前仅主机控制流测试通过，实际GPIO/HALT恢复仍待板测。
- 新鲜持续FPS包含比较/任务间空隙，记录P50/P95/P99/max、周期、PS分项和内存样本。单脚本不自授最终G4/G8验收。
- `validate_development.py --board-url ... --oracle-root ...` 可以在真实worker到位后逐片核对先前CPU head，完成40体真实PL闭环；不会从oracle填补缺失结果。

## 可复现命令

原模型SHA固定如下。模型/MRI/bit在`out/`或Release中，源码不携带大工件。

```powershell
$AxialSha = 'fef60178abaf8f81cd3206b2c37e3ca1aef56299ee01c2f8cb3dac3c28bf33be'
& acdc/.venv/Scripts/python.exe -m acdc import-release <A解压根> <新模型目录> --model-spec-sha $AxialSha
& acdc/.venv/Scripts/python.exe -m acdc verify-model <模型目录> --model-spec-sha $AxialSha --full
& acdc/.venv/Scripts/python.exe -m acdc run --backend reference_torch --model <模型目录> --model-spec-sha $AxialSha --ed <ED.nii.gz> --es <ES.nii.gz> --geometry-policy acdc-direction-only-sform-v1 --output <新结果目录>
```

40体参考重建：

```powershell
& out/acdc/reference_env/Scripts/python.exe -m acdc.validate_development --model out/acdc/axial_release_v2 --model-spec-sha $AxialSha --dataset <training目录> --output <新开发验证目录>
```

构建必须等同次打包完成后再启动；不能修改运行中的脚本或其实际读取的打包源。带tag输出保留失败候选，不覆盖旧证据。

```powershell
$env:PROCESSOR_ARCHITECTURE='AMD64'
& D:/vivado/2026.1/Vivado/bin/vivado.bat -mode batch -source acdc/optimization/vivado/package_spatial.tcl -tclargs 4
& D:/vivado/2026.1/Vivado/bin/vivado.bat -mode batch -source acdc/optimization/vivado/build_spatial.tcl -tclargs 4 <新候选标签>
& acdc/.venv/Scripts/python.exe scripts/extract_xsa.py <构建目录>/dma_top.xsa <构建目录>/dma_top
& acdc/.venv/Scripts/python.exe -m acdc.optimization.package <构建目录> --p 4
```

后续必须先完整布线合格，再打包、隔离上板短测，随后优化与长稳，最后40体PL结果和原Skin/CAMUS回归。不得把软件复现或RTL周期换算为真实板上FPS。
