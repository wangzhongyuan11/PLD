# 阶段5：独立优化与评估

> 本文件记录旧模型阶段5及当时的候选状态。最终模型迁移、新增ARM控制板测、dev638和后续加速进度见 [FINAL_MODEL_STATUS.md](../FINAL_MODEL_STATUS.md)。

2026-09-27，用户确认C接受既有ABI与资源预算，阶段3/4验收关闭。本轮按用户要求暂不接入C，继续B子系统完整优化、测试评估；原计划中的C后处理/HDMI/UI及整机显示计时保留为后续集成要求。不得将未接C的B性能称为已验收整机性能。

## 最终模型接收后的当前状态

2026-09-27收到 `role_a_final/soup_v5_a050`，新版方案见[最终交付审查与迁移计划](ROLE_A_FINAL_REVIEW.md)。本文件后续优化/板测记录均属于旧模型 `759a4967…fdab`，不能直接转为最终模型 `fe137ca5…6aef` 的证据。

本轮141条包文件hash/字节数检查通过，15层小块、整网、halo分块及PS调度软件替身通过，36项既有Python回归通过；已有CNN构建manifest/时序/配对复核通过。旧 `stage5_ps_range` 的1000帧报告按实际部署包重新审查通过。本轮没有新模型xsim/构建/板测，没有从638张原图独立重算指标，也没有切换默认模型。

最终模型结构和容量兼容现有计算引擎。新增工作是模型身份贯通、参数/LUT换版、0.45概率阈值（q≥−1）和dev638协议；旧379图及q≥0入口仍存在。先建立最终模型的板上基线，再继续性能优化；native-control/native-resize仍为单独候选。旧稳定组合在线B吞吐0.636627 FPS、P95 1.886086秒，新版性能待测，项目性能目标保持未达成。

## 已验证的PS优化

使用同一100 MHz bit/hwh、同一量化输入和冻结模型；每组1帧预热、5帧计时，比较在计时外。另跑单帧cProfile诊断，其耗时不混入性能统计。各帧head与A逐字节一致。

| 版本 | 中位秒/帧 | 5样本P95秒 | 改动 |
|---|---:|---:|---|
| 原始基线 | 2.804156 | 2.816506 | `a14042e`运行时及原始bit |
| PS opt1 | 2.294331 | 2.316847 | 直接张量拷入DMA、普通ndarray视图、尺寸分块结果有界缓存 |
| PS opt2 | 2.132056 | 2.147534 | 在opt1上启用XRT cacheable分配，保留flush/invalidate |

累计耗时下降约24.0%，当前约0.469 FPS，尚未达到1 FPS。原始cProfile定位到重复`tobytes`、PynqBuffer切片元数据、重复分块计算及设备缓冲区访问开销；未移除数值范围、长度、tail guard、帧/命令完成或异常恢复检查。缓存仅保存由尺寸决定的分块方案，不缓存推理输出。

opt1和opt2均另外重跑两次连续整帧、短包/缺包/计算中复位及恢复后的第三帧，全部15层输入/输出与head一致。优化后的输入拷贝还覆盖独立的非连续/反向切片分支测试。

板上目录分别为`stage5_20260927_baseline`、`stage5_ps_opt1`、`stage5_ps_opt2`，均位于`/home/xilinx/jupyter_notebooks/pld/`；各自使用独立文件，保留阶段3/4部署。主机证据在`out/stage5/`，原bit/hwh/xsa、构建清单及报告另存`out/stage5/baseline/`。

复现入口：`board/profile_cnn.py --bit overlay/dma_top.bit --repeat 5 --profile --report <版本>.json`。报告包含源码/产物hash、各命令模式的周期/字节/时间以及逐层耗时；DMA等待与PL计算重叠，不能将两者相加。5帧用于迭代比较，最终稳定性/P95仍须较长运行。

## 379图软件参考与板载CPU基线

`scripts/prepare_evaluation.py`已在全部379个官方test ID上执行，JPEG/GT与A清单一一对应。FP32由冻结checkpoint重建；INT8复用既有整网整数参考。统一为模型分辨率letterbox有效区、最近邻GT、logit/q≥0阈值、逐图宏平均。

| 软件路径 | Dice | IoU |
|---|---:|---:|
| FP32 | 0.8756085579695327 | 0.7985204359068742 |
| INT8 | 0.8723558712663897 | 0.7943660653612900 |

与A的`int8/test_compare.json`一致，最大浮点统计差异1.11e-16。A另一个`metrics/test_per_image.csv`来自略不同的FP32记录，不混作本次量化对照基线。参考源码hash、逐图交集/面积计数、原图/GT/输入/head/NPZ哈希及运行库版本保存在`out/stage5/evaluation/manifest.json`。本机Python3.13.9、NumPy2.5.2、Pillow12.3/JPEG8、Torch2.14.0+cpu，依赖固定在`requirements-evaluation.txt`。拆分解码/预处理后，冻结首层196608字节重新比较通过；主机回归29/29。

保留基线bit的完整379图板测已通过：每张原始JPEG的量化输入、有效区、PL head及Dice/IoU计数均与主机一致，三个故障注入及恢复后完整帧通过。报告`out/stage5/baseline_379.json`及JSONL已下载，运行时代码hash匹配；宏平均Dice/IoU为0.8723558712663897/0.7943660653612900。量化输入到head中位2.123413秒、P95 2.152743秒；含在线预处理为2.244540/2.827683秒，均不含JPEG解码/比较/文件I/O/C显示。总墙钟1050.887秒，但仅379帧，不满足1000帧稳定性要求。后100帧RSS中位较早期下降约19MiB，全部原始内存样本保留，不能用它替代长期测试。

完整数据包276643514字节已上传并校验SHA256，解到`/home/xilinx/jupyter_notebooks/pld/stage5_dataset_20260927/dataset/`。流程、计时与稳定性口径见[评估说明](../validation/EVALUATION.md)。`out/stage5/report/index.html`已附入完整基线板测曲线，保留软件最低分样例；模型误分不当作PL数值错误。

板载CPU执行同一冻结INT8参考，1帧预热55.479秒，3帧计时57.846/54.625/55.106秒，head全部精确一致；中位55.105524秒。对已板测PS opt2的2.132056秒，约25.846倍。此CPU实现为NumPy INT64精确参考，不是优化ARM推理库；双方均从量化输入至head计时，比较、JPEG预处理与C显示不计入。证据`out/stage5/cpu_baseline.json`已下载，板上脚本及参考源码hash与本机匹配。

## PL优化进度

非卷积连续读流水和顶层输出预取队列已通过算子21次及顶层33次仿真。首轮正式100MHz布线WNS=-0.022ns，路径为输入BRAM→池化比较→尺度LUT→输出寄存器，已拒绝部署；报告及对应RTL保留在`out/stage5/pl_opt1_failed/`。

在比较与LUT之间增加一级流水后，相关仿真再次通过。第二轮原路径已修复，但原有卷积BRAM→DSP路径WNS=-0.001ns，仍被门禁拒绝；报告保留在`out/stage5/pl_opt2_failed/`。Vivado布线前估计WNS=+0.057ns，因而默认phys_opt跳过优化。

第三次构建为CNN启用`POST_ROUTE_PHYS_OPT_DESIGN`，不改100MHz或约束，已通过：WNS+0.209ns、WHS+0.018ns、WPWS3.750ns，setup/hold均无违例，bit/hwh与XSA一致、源码/打包RTL/产物hash绑定。资源6924 Slice LUT、7299 FF、67 BRAM36等效、50 DSP。构建退出0，包`out/stage5/pl_opt3_bundle.tar.gz`含141文件。

待基线379图进程正常结束后，已部署到`/home/xilinx/jupyter_notebooks/pld/stage5_pl_opt3/`并核验141文件hash。三帧全部15层输入/输出及head零差异，短包/缺包/计算中复位后恢复通过；5帧同条件性能中位1.566322秒、P95 1.592653秒，约0.638 FPS。相对PS opt2配原bit的2.132056秒降低26.5%，相对受控原始2.804156秒降低44.1%。硬件计算周期从71385808降至36193276；命令数和数据流量未改。证据`out/stage5/pl_verification.json`、`pl_pl_opt3.json`及profile，已核对下载报告中的bit和运行时代码hash。新bit SHA256为`a1daac7d16b636d0545c0cf95a8b18da06af015169af60e3d4fa552fbc4e7cb7`。新版379图及1000帧验证仍待完成，未达到≥1 FPS目标。

## 未保留的PS实验

依据cProfile对MMIO调用开销的定位，实验了基于同一PYNQ映射数组的受检查32位访问，30项主机测试、三帧逐层与故障恢复均通过。实验版5帧中位2.084391秒；紧接着在原版上重测5帧为2.093455秒，差异约0.43%，不足以证明稳定收益，已撤回。原PS opt2驱动按已验证部署包逐字节恢复；不保留无证据的复杂度，不将旧2.132秒与新实验直接相减宣称收益。原始JSON为`out/stage5/ps_opt3.json`、`ps_opt3_verification.json`、`ps_opt2_recheck.json`，实验源码单独保留在`out/stage5/ps_opt3_tensor_accel.py`。

## 已通过整网验证的NEON输入拷贝

最新PL版cProfile仍显示`load_input`自身约0.169秒，原路径分开执行INT8范围检查和拷贝。`board/native/tensor_copy.c`提供单遍检查+NCHW gather，ARM NEON按16字节处理并保留标量尾部；不访问DMA/MMIO，不执行网络算子。Windows标量及PYNQ ARM版本均通过六类布局、反向/广播切片、向量主体与尾部-128拒绝、容量、重叠和尾保护测试。

板上GCC11.2.0使用`-O3 -std=c11 -shared -fPIC -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard`。真实cacheable TX缓冲区上交替运行、去掉两轮预热后的六轮测试：NumPy检查+拷贝中位0.161647秒，NEON 0.062429秒，约2.59倍；无NEON的首版仅约5.7%收益。证据`out/stage5/native_neon_benchmark.json`与`native_neon_hashes.txt`。

已通过显式`--native-copy`接入`TileAccelerator`，默认NumPy路径保留。所有分支检查和拷贝在启动命令之前完成，第二分支非法也不能触发MMIO、DMA或硬件恢复。`build_native_copy.py`在板上编译并自测，生成C源码、Python包装、动态库哈希及编译器/参数清单；正式驱动先核对清单再加载overlay。

部署目录`/home/xilinx/jupyter_notebooks/pld/stage5_ps_neon/`，145文件部署包。三帧各15层input/output和head完全一致，短包、缺包、计算中复位及下一帧恢复通过。1帧预热、5帧同条件测量中位1.437212秒、P95 1.456613秒，约0.696 FPS，相比1.566322秒下降8.2%；仍不含JPEG预处理和显示，尚未达到1 FPS。证据`out/stage5/neon_verification.json`、`neon_native.json`、`neon_native.json.profile.txt`及`neon_build.json`。本机30项回归通过，报告的驱动/调度器/原生源码/bit/hwh哈希已与当前文件核对。最终379图和1000帧仍待完成。

## 并行预处理与TX范围同步

`ParallelBilinear`将源图水平缩放按行分给两个Pillow线程，再执行原垂直滤波，保留8位中间结果和原采样位置。依据[Pillow 9.0.1两遍缩放实现](https://github.com/python-pillow/Pillow/blob/9.0.1/src/libImaging/Resample.c)。本机及板上全部379图的缩放RGB字节完全一致；板上Pillow9.0.1/JPEG8缩放中位由60.057ms降至50.115ms，P95由664.693ms降至434.256ms，总缩放时间51.895→39.048秒。仅测解码后缩放，不包含量化或推理。报告`out/stage5/parallel_resize_host.json`、`parallel_resize_board.json`；已核对候选源码哈希。`evaluate_dataset.py --parallel-resize --decode-images`显式启用，默认路径保留，量化输入/有效区/几何仍须逐张匹配冻结参考。

板上PYNQ3.1.1的`xrt_device.flush`丢弃size/offset并同步整个BO。已核对安装的pyxrt三参数重载及[XRT接口顺序](https://github.com/Xilinx/XRT/blob/master/src/runtime_src/core/include/xrt/xrt_bo.h)，使用`bo.sync(direction,size,offset)`只同步TX前缀，范围向外对齐64字节且不得越过原分配。包装对象保留原缓冲区至DMA完成或复位，不拥有释放权；RX保留整块flush/invalidate和全部尾保护。未修改系统PYNQ/XRT库。独立7种长度的缓存测量见`out/stage5/cache_sync.json`。

147文件包部署到`/home/xilinx/jupyter_notebooks/pld/stage5_ps_range/`，逐项hash通过。`--tx-range-sync`已通过三帧各15层输入/输出/head及短包、缺包、计算中复位恢复。顺序测量原路径10帧→范围同步10帧→原路径10帧，中位1.429481→1.392920→1.449766秒，中位下降2.6%～3.9%；P95分别1.448510/1.464706/1.473197秒，候选含两个较慢样本，不能宣称P95改善。报告`range_verify.json`、`full_before.json`、`range_profile.json`、`full_after.json`，源码/bit/hwh及同步包装哈希已核对。本机32项回归通过。

NEON、TX范围同步和并行预处理组合已完成1000帧板测，循环全部379张原始JPEG，墙钟2054.243秒，退出0。下载的`out/stage5/range_stability.json`与JSONL一致；独立审查脚本逐项核对147文件部署包、运行时及bit/hwh哈希、1000帧head/指标、379图覆盖和统计重算，通过。三类故障及恢复后整帧通过，无清理错误。该版本满足帧数、时长、数值和恢复要求。

量化输入至head中位1.397499秒、P95 1.435489秒、0.711790 FPS；包含在线预处理中位1.524842秒、P95 1.886086秒、0.636627 FPS。均排除JPEG解码、文件I/O、比较与C显示，性能目标未达成。Dice/IoU仍为0.872355871/0.794366065。

RSS前/后100帧中位80040/51070 KiB，同一图片下一轮差值中位-6116 KiB；后600帧各百帧中位58480→57530→52036→52386→51708→51070 KiB，未见持续增长。CMA余量范围26600～98732 KiB，无分配失败。本次内存记录支持稳定运行，不代替未来版本或无限时长验证。审查证据`range_stability_audit.json`，图表`out/stage5/range_report/index.html`；后续修改仍须按最终版本重新验收。

## 尚待完成

### 等待板测的候选（不计入性能成果）

- `board/native/tensor_control.c`与`board/native_control.py`将ABI-1配置、单向DMA命令和有界等待合并到C；使用现有PYNQ映射，缓存同步、缓冲区所有权及异常恢复仍由Python管理。失败保留`SimpleDMA._pending`和活动标志，只有完成或复位后才能释放。默认仍为Python；显式`--native-control out/native/tensor_control.so`启用。本机GCC严格编译通过，40个独立寄存器模型场景覆盖W1C旧IOC、错误优先级、短接收、过期帧/命令、字节计数和超时；生产库另用普通RAM验证两个真实20ms期限。36项Python回归通过。151文件包`out/stage5/ps_control_bundle.tar.gz`已准备；原1000帧测试现已结束。后续仍须在独立目录编译、自测、板测及同条件测速，尚无ARM板测/收益结论。
- `board/native/rgb_resize.c`与`reference/native_resize.py`是独立RGB预处理候选，按Pillow数值规则使用Q22滤波系数和8位水平中间结果，带ARM NEON实现；没有接入推理运行时。本机标量库通过60个随机/全白/退化尺寸/边界/容量用例和379图RGB精确比较。主机中位3.016ms、P95 43.540ms，对照Pillow为2.046/21.284ms，主机上更慢。报告`out/stage5/native_resize_host.json`；尚未验证ARM NEON正确性和收益，不以主机数值通过宣称优化有效。

- 优先完成最终模型入口、独立manifest、阈值与评估/审查/渲染工具迁移；完整材料和验收顺序见迁移方案M2–M5。
- 补齐冻结dev638的源图/GT及来源hash、划分和评估规则，独立复现FP32/INT8，再完成新版PL逐图原始head一致性。旧379图全量与 `stage5_ps_range` 的1000帧已完成，不重复列为该旧版本缺口。
- 最终模型及最终优化运行时重新测板载CPU基线、分项耗时、吞吐/P95；继续降低PS和预处理开销，非卷积遍历与结果预取优化保留已有证据。
- 最终模型/最终运行时覆盖dev638且至少10分钟、1000帧，复核数值、内存、恢复和实际部署身份；旧模型的长期运行不自动覆盖新版或未板测候选。
- C集成按用户要求暂缓；后处理/HDMI/UI与包含显示的端到端验收不冒充完成。
