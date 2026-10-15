# AGENTS.md — PYNQ-Z2 皮损分割（B：PL 加速器与 PS 交互）

## 目标与边界

在 PYNQ-Z2 上将 ISIC Task 1 皮肤镜图像经 PS 输入、DDR/DMA 搬运、PL INT8 TinyUNet 推理，交给后处理与 HDMI/UI 展示；旧基线使用2016数据，最终模型使用2016/2017训练与开发口径。目标与验收条件见 `docs/PLAN.md`；题目依据与分工分析见 `docs/PROJECT_SCOPE.md`。

**最终模型状态**：用户指定 `role_a_final` 为A最终交付（`soup_v5_a050`，不采用A21 QAT），授权B实现、加速、真实板测和交付。默认已切换最终模型，`reference/final_manifest.json` 锚定124文件，旧manifest及部署保留。ABI 1.2最终采用版已完成dev638全部638图原始JPEG的PL head逐图一致、1000帧/1483.827秒、三类故障恢复和部署审查；核心1.422 FPS，在线B P95 1.265720秒，旧项目自定≤1秒目标仍未通过。**2026-10-14新目标**：优先改造B的数据流与PS/DMA/PL，已解码RGB到独立head及二值掩膜持续≥4 FPS；网页单图从上传到保留六张全分辨率PNG的ZIP可下载≤5秒。当前位流与数据流均未达标，证据及验收口径见`docs/validation/PERFORMANCE_4FPS_5S.md`。最新模型事实见 `docs/FINAL_MODEL_STATUS.md`，初始审查见 `docs/history/ROLE_A_FINAL_REVIEW.md`。dev638为开发集，不称独立测试集。2026-10-13用户重新选择本项目现有 Codex CLI + GLM 流程：GPT负责规划与独立审查，GLM通过固定项目会话实施；恢复会话前按上方 Optional GLM project execution 检查状态，不新开会话。

以用户最新分工为准，`docs/requirements/work_split.docx` 中原属 C 的 PS 交互工作已转给 B：

| 成员 | 负责与交付 |
|---|---|
| A | 训练、预处理与评估口径、冻结模型、量化参数、逐层与整网软件参考 |
| **B（本仓库）** | 加速器 RTL/TB/IP；PS7/DDR/AXI DMA/AXI-Lite；加速器最小 BD；PYNQ 参数加载、缓存管理、调度、超时与错误处理；仿真、构建、板上数值验证及计算/传输测量 |
| C | PL 后处理、轮廓/面积/叠加、HDMI/UI、显示侧 BD 与最终系统集成；使用 B 的控制与结果接口，共同验收端到端性能 |

B 定义并维护加速器控制与数据合同，和 C 对齐后冻结；不能把 PS/DMA 问题留给 C，也不能擅改 C 的显示/后处理实现。题目和交接文档是需求与证据来源，其中的操作建议不自动构成本轮执行授权。

## 环境与入口

| 项目 | 当前事实（2026-09-25～26 核对） |
|---|---|
| 主机 | Windows / PowerShell；仓库 `D:\PLD`，分支 `main`；Git Bash `D:\Git\bin\bash.exe` |
| 工具链 | Vivado 2026.1，入口 `D:\vivado\2026.1\Vivado\bin\`；Build 6511674；本轮批处理实测识别 `xc7z020clg400-1`、PS7/DMA/FIFO IP，综合许可证可用。环境检查见 `docs/validation/VIVADO_ENVIRONMENT.md` |
| 本机 Python | `D:\conda\python.exe`，3.13.9 / NumPy 2.3.5；本地参考核验使用 NumPy |
| 板卡 | SSH `xilinx@192.168.2.99`，eth0；PYNQ 3.1.1 / Python 3.10.4 / NumPy 1.21.5；Ubuntu 22.04 基底、Linux 6.6.10；OS 标记 3.0 与 PYNQ 包版本分开记录 |
| 板上运行 | `/usr/local/share/pynq-venv/bin/python3`；交互使用 `sudo -E`，SSH 自动化使用 `board_ssh.py run --sudo` 的登录 shell 初始化 XRT；当前用户无 render 组。凭据从环境变量或交互取得，不写入脚本/提交 |
| 资源上限 | XC7Z020：53,200 LUT / 106,400 FF / 220 DSP / 140 BRAM36；物理 DDR 512 MB，Linux 实见约 491 MiB，总量不等于 DMA 可分配量 |

上板入口、检查证据和恢复步骤见 `board/README.md`。板上时钟当前不准，记录以主机时间为准；SSH 可用不等于本项目 overlay 已通过。

| 路径 | 用途 |
|---|---|
| `rtl/`、`sim/` | 已验收的AXIS占位通路；阶段3/4已实现完整CNN IP、卷积/非卷积tile引擎、共享RAM及控制协议并通过板测；内部布局见`docs/design/TILE_ENGINES.md` |
| `reference/` | 当前A交付124文件hash清单、完整INT8参考、层间LUT、JPEG预处理与受容量约束的halo分块；不能用CPU结果代替PL验收 |
| `scripts/run_sim.sh`、`scripts/build.sh` | xsim 回归；fifo/incr/cnn 构建与 bit/hwh/xsa 提取；`check_build.py` 检查时序和 XSA 配对 |
| `vivado/` | 参数化 BD、IP 打包、PS7 配置；PS7 配置由 `scripts/gen_ps7_config.py` 从种子机械提取 |
| `board/` | `simple_dma.py` 有界 MMIO 控制；`verify_overlay.py` 精确比较、重复包、超时恢复与 JSON 证据；操作见 README |
| `role_a_handoff/role_a/int8/model_spec.json` | 旧模型；仅保留历史回归 |
| `role_a_final/int8/model_spec.json` | 最终目标模型的结构、尺度、偏移与参数哈希；迁移状态见 `docs/history/ROLE_A_FINAL_REVIEW.md` |
| `PYNQ_tutorials/dma/` | PS7+DMA 最小工程基座；`PYNQ_tutorials/dma_hls_axis/` 参考自定义 IP 接入 |
| `local/third_party/finn/finn-rtllib/`、`local/third_party/Pynq资料（含实验指导书等）/` | 按需查阅 RTL 模块、board files、XDC、原理图；复用时保留许可 |
| `out/`、`ip_repo/` | 可重建的日志、报告、板卡产物与打包 IP，不入 Git；数据集/权重/第三方工程同样不入 Git |

## 数值与接口约束

- 当前旧基线：`best.pt` SHA256 `5d2c6b07b85b681190017aa831cf5214856dc9978973b866f647883098b25b77`；`model_spec.json` SHA256 `759a496708ae3bbde1dd50cfab7343a380fe7204cfd96031dc0855f35250fdab`。最终目标：checkpoint `f488719fe6d4d09c9b86be84d06b19d34241d6c14060b87da7bab25d1367eb77`、spec `fe137ca5b128fc1983fabad9bee39b3208cda4256e0d04c4e27cef5c07666aef`。换版用独立manifest，更新参数/LUT/golden与哈希并重跑核验，不使用 A 的试验权重。初审manifest保留在 `out/role_a_final_review/`；当前正式默认使用 `reference/final_manifest.json`。
- TinyUNet：输入 `[1,3,256,256]`，15 个 Conv，通道 8/16/32/64；3×3 stride1 pad1，head 为 1×1 pad0；ReLU、MaxPool2×2、nearest upsample×2、Concat（编码器特征在前）。输出 `[1,1,256,256]` 有符号 INT8 logit；不输出病种诊断。
- 激活 NCHW、权重 OIHW；INT8 范围 `[-127,127]`、零点 0；bias/acc 为小端 INT32。`sat[-127,127](round_ties_even(acc * multiplier_q24 / 2^24))`，正负数均最近偶数舍入。乘积位宽按实际范围证明不溢出，NumPy 参考用 INT64，不能直接截成 INT32。
- **层间必须转换尺度**：Conv 输出尺度与下一层输入尺度可能不同；ReLU/Pool/Upsample/Concat 后按下层尺度重量化，跳连两支分别对齐。现有 Q24 卷积核验不代表层间转换已经验证。
- 首层输入按 A 的 EXIF→RGB→256 letterbox→`/255`→首层 input_scale 量化得到，不是 `RGB8-128`。旧模型评估默认 `q>=0`（概率0.50）；最终包要求概率0.45，在head尺度0.1995087721216397下等价于有符号 `q>=-1`。当前驱动元数据/评估已使用最终模型q≥−1，旧q≥0只属于旧模型协议。补边强制背景；面积分母是非补边有效区域。
- 小块 golden 是独立 8×8 卷积测试，不能直接和整图同坐标比较；`output_golden` 和 `output_full_golden` 均为 requant 后、ReLU 前。整网与小块必须分开验收。
- 阶段2保留`user:pld:axis_incr:1.0`，逐字节+1、无AXI-Lite。正式`user:pld:axis_tensor_accel:1.0`为64-bit AXIS、100 MHz、AXI-Lite `0x43C00000`；命令、寄存器、包长度、TKEEP/TLAST、ready标签和恢复以`docs/design/ACCELERATOR_REGISTERS.md`为准。B实现已板测；用户于2026-09-27确认C已认可现有接口与资源预算。

## 实现、仿真、构建、上板闭环

1. **定位与复现**：先 `git status --short --branch`，读受影响源文件和当前合同；明确本轮输入、期望输出与失败判据。分析任务不扩大成 RTL 修改。
2. **参考对齐**：核对模型哈希与布局，用 NumPy 生成/复算数据；先小块累加、requant，再单层整图、层间转换和整网。错误报告包含层、坐标、期望值、实际值。
3. **实现与自校验仿真**：SystemVerilog 一文件一模块，文件名与模块名一致。TB 覆盖正负值、舍入平局、饱和、边缘补零、背压、源间隙、TLAST、复位与重复帧；随机测试固定种子。用断言、超时和退出码判断，波形用于定位。
4. **综合与实现**：Tcl 重建工程/IP，检查 DRC、资源和布线后 setup/hold/脉宽时序；不能用综合成功或 `BUILD_DONE` 代替时序合格。为 C 的后处理与显示保留已协商的资源预算。
5. **板上验证**：获准上板后，使用同次构建的 bit/hwh、独立部署目录和 SHA256；先 fifo、再 incr、再计算 IP。先单帧逐字节/逐元素比较，再重复运行与测速；所有等待有上限，失败保留 DMA 状态和首个差异。
6. **调试回归**：按数值→协议→BD/时钟/复位→驱动/缓存→整机时序分层定位。修复后重跑失败用例及受影响回归；板上失败要回灌成本地可复现测试。仿真可重复，不代表硬件可以无界重试。
7. **记录与提交**：记录源码提交、模型/产物哈希、命令、退出码与验证层次；更新事实变化。按仓库约定，每个已验证结果做聚焦 commit，保留无关修改，不提交密码和生成物；已提交变更用 `git revert <commit>` 回退。

在 PowerShell 中明确调用 Git Bash，避免命中 `C:\Windows\System32\bash.exe`（WSL）：

```powershell
Set-Location D:\PLD
# 当前工具进程若缺少架构变量，在已确认本机为X64的前提下补齐；仅影响当前进程。
if (-not $env:PROCESSOR_ARCHITECTURE) { $env:PROCESSOR_ARCHITECTURE = 'AMD64' }
git status --short --branch
& D:\conda\python.exe scripts/verify_a_handoff.py
& D:\conda\python.exe scripts/verify_full_reference.py
& D:\conda\python.exe -m unittest discover -s sim -p 'test_*.py' -v
& D:\Git\bin\bash.exe scripts/run_sim.sh
& D:\Git\bin\bash.exe scripts/build.sh fifo
& D:\Git\bin\bash.exe scripts/build.sh incr
& D:\Git\bin\bash.exe scripts/run_sim.sh tile-full
& D:\Git\bin\bash.exe scripts/build.sh cnn
& D:\conda\python.exe scripts/package_cnn_board.py
git diff --check
```

对应成功标志：`ALL PASS` / `ALL_SIMS_PASS` / `BUILD_DONE mode=...`，均还须退出码 0。日志在 `out/sim_logs/`；构建产物和时序/资源报告在 `out/vivado_<mode>/`。文档修改只需核对事实、链接与 diff，不要求无关重建。

Vivado 2026.1 的 `-version` 启动分支固定返回 1，不能用它的退出码判断安装失败；实际 batch/仿真仍要求退出码 0。不要直接运行 `bin/unwrapped/` 中的程序绕过环境初始化。旧 `vivado/smoke_synth.tcl` 含过时器件说明和构建调用，不作为当前环境验收入口。

## 当前验证状态与缺口

| 层次 | 已有证据 | 尚未证明 |
|---|---|---|
| A 数值基线 | 15/15小块通过；完整NumPy网络和分块网络从单一首层输入运行，15层input/output逐字节一致；124文件hash锚定 | 完整算子与板测见下文；其他模型/图像仍需换版/评估 |
| 占位 RTL | xsim 全绿：reg_slice 272 beats；incr 5,349 beats / 116 包（含 TKEEP 尾拍/随机 lanes 定向、wrap/背压/源间隙/TLAST/中途复位/8×随机） | 此行为阶段2历史证据；正式CNN见下文 |
| 占位构建 | 2026.1 fifo/incr bit+hwh+xsa；WNS +2.205/+2.226 ns（incr 为 TKEEP 修复后重建）；hold 亦通过 | PS7 DDR 参数告警的长期板级表现；后续版本重建 |
| 板卡基础 | SSH、PYNQ 3.1.1、`sudo -E` 设备识别、4 KiB PYNQ 缓冲区校验通过；详见 `board/README.md` | HDMI |

阶段 0/1/2 已于 2026-09-26 验收，证据见 `docs/history/STAGE012_VALIDATION.md`。复用 2026.1 fifo/incr 产物（XSA 配对、时序复核通过），本轮重跑 xsim 全绿、PS 异常回归 10/10。tutorial/fifo/incr 每套板测 1035 个正确包，含四种模式、1000 随机包和超时恢复后 7 包；1 MiB 吞吐中位数约 434/426/434 MB/s。超时与错误期望负例均退出 1，随后 incr 7/7 再通过。此为阶段0/1/2历史证据；当前板上已加载CNN，HDMI仍未验收。

已关闭：TKEEP 尾拍（2026-09-25）和 PS 无界 DMA 等待/常规异常恢复（2026-09-26）。不要重新引入 PYNQ 3.1.1 DMA 构造/start/stop/wait 内的无界循环；`SimpleDMA` 只支持当前 simple/non-micro 配置，单一拥有者，不能与 `ol.axi_dma` 并用。异常先确认 DMA 复位再释放活动 DDR，重载 overlay 清空外接 PL 状态；硬件无法复位/重配置时不能宣称安全恢复。

阶段3/4的B实现已于2026-09-26通过，详细证据见`docs/history/STAGE34_STATUS.md`。正式CNN布线100 MHz：WNS+0.139ns、WHS+0.014ns，bit/hwh与XSA配对；7047 Slice LUT、7204 FF、67 BRAM36等效、51 DSP。板上三帧15层输入/输出和最终head零差异，短包、缺包超时、计算中复位后整帧恢复通过。当前CNN overlay验证程序已结束并清理DMA，不在后台推理。

当前入口：`board/tensor_accel.py`负责有界控制/缓存/恢复；`board/pl_graph.py`负责15层调度、halo和跳连生命周期；`board/verify_cnn.py`板测。只允许首层图像输入；golden用于比较，禁止逐层注入代替整网执行。所有卷积及ReLU/Pool/Upsample/Concat/尺度映射均在PL，PS生成LUT参数。阶段3/4原100MHz版每帧80卷积tile、100非卷积tile、389命令；最终采用版以ABI 1.2融合decoder并增大tile后为175命令/帧，参数在同层tile间复用。

`SimpleDMA`支持`send`/`receive`/独立收发长度/`submit`+`wait`；旧`receive_only=True`仍是超时注入。加载命令只用MM2S，RUN只用S2MM，结果复制为独立数组后交付C；不能异步保留staging buffer视图。共享RAM先仲裁地址/数据再写单一RAM访问语句；独立OOC正裕量不等于跨模块路径或正式BD时序通过。

运行仿真/构建时不要修改其脚本或已编译源码。`write_build_manifest.py`绑定源码、打包RTL与产物；`package_cnn_board.py`拒绝hash漂移或配对/时序失败的构建。日志`out/sim_logs/`、正式产物`out/vivado_cnn/`、板测`out/stage34/cnn_board.json`，部署目录`/home/xilinx/jupyter_notebooks/pld/stage34_cnn_20260926_v1/`。

JPEG环境已核验：本机uv Pillow12.3/JPEG8、板上Pillow9.0.1/JPEG8，原始JPEG生成的196608字节与A一致；Conda Pillow12/JPEG9有解码差异，不能忽略输入不匹配。`scripts/prepare_cnn_input.py --compare-golden`同时保存有效区域和像素/模型hash。A后续换版显式使用新manifest，保留当前基线。

2026-09-27用户确认C已认可ABI与资源预算，阶段3/4验收关闭。本轮继续阶段5的B独立优化、测试评估，暂不接入C。PS opt2已在原100MHz bit上通过逐层与故障恢复，5帧中位2.132秒（复测2.093秒）。完整379图软件及基线板测已复现A：FP32 Dice/IoU 0.875608558/0.798520436，INT8及PL 0.872355871/0.794366065。板载NumPy整数参考中位55.106秒，不能代表优化ARM库。

PL流水与结果预取优化已完成第三次构建和三帧逐层/故障恢复板测：100MHz WNS+0.209ns、WHS+0.018ns，6924 LUT/7299 FF/67 BRAM36等效/50 DSP；五帧中位1.566秒。NEON输入检查/拷贝已接入并完成同样数值与恢复验证，五帧中位1.437秒、约0.696 FPS，当时尚未达标；报告`out/stage5/neon_native.json`。该阶段验证部署为`stage5_ps_neon`，显式`--native-copy out/native/tensor_copy.so`启用；库和编译清单通过`scripts/build_native_copy.py`板上生成，加载前核对源码/库哈希。全部CNN算术仍在PL。保留基线产物在`out/stage5/baseline/`。基线原bit的完整379图JPEG预处理及PL head逐字节一致、指标复现A，总运行1050.9秒但只有379帧；该阶段后续优化版本须重新覆盖379图及至少1000帧。完整测试数据在板上`/home/xilinx/jupyter_notebooks/pld/stage5_dataset_20260927/dataset/`，执行与口径见`docs/validation/EVALUATION.md`，进度见`docs/history/STAGE5_STATUS.md`。未关闭的DRC与PS7 DDR告警仍需记录，不把DMA吞吐或核心周期换算成系统帧率；C后处理/HDMI按用户要求暂不接入。

后续PS优化在`stage5_ps_range`：可选`--tx-range-sync`同步TX前缀，RX整块同步和尾保护保留；可选`--parallel-resize --decode-images`使用两个Pillow线程进行精确缩放。该组合1000帧/2054秒测试已退出0，覆盖379图，输入/head/指标一致，三类故障及下一帧恢复通过，部署哈希和统计经独立审查；未见持续RSS增长。量化输入到head中位1.3975秒/P95 1.4355秒；含在线预处理1.5248/1.8861秒、0.6366 FPS，仍未达标。证据`out/stage5/range_stability_audit.json`，图表`out/stage5/range_report/index.html`。此为该独立版本的稳定性证据，后续候选须重新验收；详细版本范围见`docs/history/STAGE5_STATUS.md`。

最终采用版为`out/final_model/mode6_tile32_fastq_v3.tar.gz`，板端隔离目录`/home/xilinx/jupyter_notebooks/pld/final_mode6_fastq_20260929_v3/`；部署包161文件hash、bit/hwh/XSA及125MHz布线时序核验通过（WNS+0.013ns、WHS+0.049ns、WPWS+2.75ns，7108 LUT/67 BRAM36等效/53 DSP）。运行显式选项`--native-copy`、`--native-control`、`--native-resize`、`--tx-range-sync`、`--fuse-maps`、`--fuse-decoder`、`--fast-quantize`。`out/final_model/fastq_stability1000_audit.json`独立核对1000/1000 head和指标、638唯一图、三类故障恢复，核心中位0.700375秒/P95 0.726514秒，在线B中位0.758002秒/P95 1.265720秒。后续优化若改变源码/部署须按版本重验；短测不能继承该1000帧证据。125MHz 13ps setup裕量很窄，任何RTL/时钟变更都须重新布线。C后处理/HDMI仍未联调。用户要求减少重复测量：优先做能决定候选去留的短测，再只对最终采用版做必要验收。

C的网页/HDMI应用源码已于2026-10-12合并入`display_web/`（板上原数据路径`/home/xilinx/jupyter_notebooks/PLD_Web/data`）。`pld_result.py`锚定spec `fe137c…`、manifest `e47eb18b…`（即`reference/final_manifest.json`）、head scale 0.1995087721216397与q≥−1；样例`display_web/samples/frame_000001.bin`为ISIC_0000003真实板测head。2026-10-14正式`pld-web.service`已通过版本化drop-in切到`PLD_Web_png_worker_candidate_v3_20261014`与`final_fastq_worker_candidate_v2_20261013`，保留原`data/`、访问码和历史；常驻B worker加双线程level1 PNG在隔离目录完成24帧逐图head/六视图核对，正式服务又完成四图与HDMI软件切换，实际rollback→reapply后真实图仍正确。原unit/旧运行包未覆盖，最终回退记录`/var/lib/pld-web-switch/apply-1746420088611781333.json`。B在线P95≤1秒、PL后处理和物理HDMI目视仍待验；板卡尚未接显示器。2026-10-15本轮仅本地同步预览优先、`InferenceSession`硬件复用和`fast_sub` PNG候选，未连接板卡；用户提供`PLD_timing_2ce70135.json`记录`preview-first-v2`的一次真实运行，单次报告不构成性能验收，本轮不判断当前板端目录/服务版本。版本、口径和证据见`docs/validation/C_WEB_PERFORMANCE.md`。运行B板测脚本前先停止`pld-web.service`，不得让网页与板测同时拥有DMA。

团队共享入口为 `README.md`、`docs/IMPLEMENTATION.md`、`delivery/c_v3/README.md`和`display_web/README.md`。原始题目/分工在`docs/requirements/`，设计细节在`docs/design/`，方法在`docs/validation/`，旧阶段证据在`docs/history/`。用户授权将约10MB冻结交付材料作为`delivery/`例外随Git共享；展开模型、数据集、系统镜像和实验产物仍不进Git。闲置第三方仓库、镜像及缓存已移到被忽略的`local/`，没有删除。初次GitHub发布使用干净快照，旧提交完整保留在本地归档分支；不得推送该旧分支或使用`--all/--mirror`带出误收缓存。
