# ACDC Axial 实施状态

更新：2026-10-20。后续目标仍为[最终计划](FINAL_ALIGNMENT_OPTIMIZATION_PLAN_20261019.md)的完整链路、真实 G4 验收及 G8 冲刺；本记录不表示目标完成。

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

## 硬件进度与当前阻断项

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
