# A 最终交付审查与 B 迁移方案

> 本文件保留初始审查快照。用户随后授权全面实施和上板，当前默认、板测、数据及性能状态见 [最终模型实施进度](../FINAL_MODEL_STATUS.md)。下文“本轮”指最初文档审查，不限制后续授权。

2026-09-27；审查起点 `55d6e31`。目标模型为 `role_a_final` 的 `soup_v5_a050`，A21 QAT 候选不采用。本轮只检查源码、产物及软件参考并修订文档；没有替换运行时默认模型、修改 RTL/驱动、重新综合或上板。旧包与旧板测证据保留。

## 1. 结论与版本边界

现有实现已经是完整的 PL TinyUNet：PS 负责预处理、参数/LUT生成、halo 搬运、调度与缓存；卷积、requant、ReLU、Pool、Upsample、Concat 和尺度映射均走 PL。无需因为最终权重换版而从头设计加速器。

最终包与旧包的 15 层名称、次序、张量形状、卷积核、步幅、padding、参数偏移和容量一致。权重、bias、所有层 Q24 乘数和输出尺度已变，除首层外的输入尺度也变。迁移的主要工作是**显式绑定新模型、重新生成参数/LUT/golden、贯通阈值和数据集口径，再做新版板上验收**。只把文件路径换成 `role_a_final` 不能完成迁移。

| 范围 | 已有证据 | 本轮结论 |
|---|---|---|
| 旧模型基础实现 | 阶段0–4和 `stage5_ps_range` 板测 | 保留验收，不清零 |
| 旧模型稳定组合 | 379图、1000帧、故障恢复、部署哈希审查 | 本轮重新审查保存的报告通过；不是新上板测试 |
| 最终模型 | 包完整性、15层小块、整网和分块 NumPy、PS 调度软件替身 | 本轮通过；不等于 RTL 仿真或板测 |
| 最终模型 dev638 精度 | A 的逐图软件记录和汇总 | 本轮重算汇总一致；未从638张原图独立重跑推理 |
| 最终模型 PL、稳定性、速度 | 尚无本轮证据 | 必须单独验收 |
| C 后处理/HDMI/UI | 暂不接入 | 原ABI/资源预算认可不代表新版阈值已联调 |

## 2. 包身份与数值差异

包中共142个文件；`MANIFEST.json`列出其余141个文件，SHA256和字节数全部一致，无未列出的额外文件（manifest自身除外）。B运行所需的模型子集仍为124个文件；两种清单覆盖范围和格式不同，不能把A的包manifest直接传给 `Model`。

| 文件 | 最终包 SHA256 |
|---|---|
| `MANIFEST.json` | `ee017f85e6991590f70a25cb22cd6d1a78944cb433c1e605a82f28003131c44b` |
| `best.pt` | `f488719fe6d4d09c9b86be84d06b19d34241d6c14060b87da7bab25d1367eb77` |
| `int8/model_spec.json` | `fe137ca5b128fc1983fabad9bee39b3208cda4256e0d04c4e27cef5c07666aef` |
| `int8/weights_int8.bin` | `6d1eb288acc14c0dbdcec70ba9b1bab43711b10b5934b93b5a514d16adc969cb` |
| `int8/bias.bin` | `fb93cb922debc16d12153df56091858f7c0454cc713e571720d265588f87d216` |

旧模型仍由 `reference/baseline_manifest.json` 锚定：checkpoint `5d2c6b07…b77`、spec `759a4967…fdab`。该manifest和默认常量本轮未改。审查用的新manifest写在 `out/role_a_final_review/model_manifest.json`，不是正式部署切换。

- 结构仍为8基通道、3次下采样、15个Conv，122,113个参数；权重121,760 B、INT32 bias 1,412 B，Q24乘数另随参数包发送。
- 每帧仍为675,282,944 MAC，最大单个INT8张量1,572,864 B。基于相同形状的现有分块调度仍为80个卷积tile、100个非卷积tile，389条加载/运行命令。换权重本身不能消除原有性能缺口。
- 新模型按逐通道 `127*sum(abs(weight))+abs(bias)` 保守界检查，最大acc界4,091,808，最大 `acc_bound*Q24` 为46,896,770,595；Q24最大175,034。INT32累加和现有64位requant乘积足够，不能把乘积压成INT32。
- 首层input_scale未变，导出的 `ISIC_0000000` 首层196,608字节也与旧包一致；这仅验证该样例，不能替代全部新数据的JPEG解码/几何核对。
- 新版15层整网输入/输出均与A一致，说明B现有float32顺序生成的层间LUT在该完整样例上兼容；仍须重新生成所有LUT，不能沿用旧模型的表。

## 3. 实现检查与需改动的位置

| 路径/符号 | 实际行为 | 后续迁移要求 |
|---|---|---|
| `rtl/conv_mac_array.sv`、`rtl/requant_q24.sv` | 8输出×4输入乘加；INT32累加/溢出标记；64位乘积、ties-even和饱和 | 新参数范围适配；优先复用RTL，不因模型名称变化重写 |
| `board/tensor_accel.py:shape/load_parameters` | bank容量检查；参数为运行时weights+bias+Q24 | 保留容量、范围、包长和缓存保护 |
| `reference/tinyunet.py:Model` | 默认绑定旧spec和124文件manifest，规格/参数/黄金文件hash及溢出检查 | 所有消费入口显式传新spec和新manifest；不得删除hash门禁 |
| `Model.prepare_input`、`board/pl_graph.py:transition` | 按新规格可生成LUT；编码器跳连在前；Pool/Concat调度依赖既有15层拓扑 | 本轮新模型软件调度180次run通过；仍需新模型PL逐层验证 |
| `PLGraph.run`结果元数据 | head尺度动态读取，但 `threshold_q` 写死0 | 输出明确的模型/概率阈值/整数阈值身份；与C结果合同同时更新 |
| `scripts/freeze_model_manifest.py`、`verify_full_reference.py` | 已支持显式spec及新manifest（冻结脚本输出不可覆盖） | 可直接复用进行主机核验 |
| `board/verify_cnn.py`、`profile_cnn.py`、`evaluate_dataset.py`、`cpu_baseline.py` | 虽有 `--model`，内部仍 `Model(args.model)`，沿用旧hash默认值 | 仅加 `--model role_a_final/int8` 会拒绝；统一传递模型身份后才能部署 |
| `scripts/package_cnn_board.py` | 锁定旧根目录、旧manifest、旧包内路径 | 参数化模型选择，部署清单绑定新模型和实际运行时；同时保留旧版重现入口 |
| `scripts/prepare_cnn_input.py`、`render_cnn_result.py` | 锁定旧Model；渲染标题仍为q≥0 | 与模型身份、有效区和阈值同步迁移 |
| `scripts/prepare_evaluation.py` | 依赖旧 `manifests/split_with_test.json`、`int8/test_compare.json`，固定379张、FP32/INT8阈值0 | final包没有这两个旧文件；需要独立数据协议，不能复制旧文件补齐 |
| `board/evaluate_dataset.py` | 完整集判断为379个ID；输出按q≥0评估 | 由冻结数据清单确定集合及数量，执行新阈值；完整集和冒烟子集分开 |
| `scripts/audit_board_run.py`、`render_evaluation_report.py` | 379循环周期、汇总、覆盖检查及标题固定 | 随数据协议更新；不能仅改推理入口遗漏审查和图表 |
| `scripts/package_evaluation.py` | 可遍历manifest记录，但仍依赖旧 `complete_test_set` 语义及单一图像根目录 | 审核dev638的集合身份、多来源JPEG和完整性语义，不把标志改名视为完成 |

本轮36项现有Python回归全部通过；这主要保护既有实现。另使用 `sim/test_pl_graph.py:OracleTile` 对新模型运行完整 `PLGraph`，15层输入/输出及head一致、参数加载15次、算子运行180次；软件替身不作为PL证据。

现有 `out/vivado_cnn` 的源码/产物manifest检查通过，bit/hwh与XSA配对；布线WNS +0.209 ns、WHS +0.018 ns、WPWS 3.750 ns。现有构建记录为6924 LUT、7299 FF、67 BRAM36等效、50 DSP，仍在B已协商的32,000 LUT/90 BRAM36/96 DSP预算内。可将该bit作为迁移候选，前提是其硬件源码/构建合同未变且新模型板测通过；如修改RTL/BD/时钟，则重新仿真、构建和时序验收。未关闭DRC与PS7 DDR告警继续保留。

## 4. 阈值是本次接口变化的重点

最终head输出尺度为 **0.1995087721216397**，主概率阈值 **0.45**。对正尺度和包含等号的比较：

`sigmoid(q * head_scale) >= p` 等价于 `q >= ceil(log(p/(1-p)) / head_scale)`。

| 概率阈值p | 最终模型整数阈值q |
|---|---:|
| 0.45（默认） | -1 |
| 0.50 | 0 |
| 0.53 | 1 |
| 0.56 | 2 |
| 0.60 | 3 |

本轮枚举255个合法INT8值，五个阈值均与float32反量化后sigmoid比较一致。FP32主阈值为logit≥−0.20067069546215124。不能把0.45直接当作logit阈值，也不能按普通四舍五入得到整数阈值。

计划在B返回元数据及评估manifest中共同记录概率阈值、整数阈值、head_scale和模型hash。C按有符号INT8比较，掩膜与valid区域相与，补边强制背景，面积分母为valid像素数。最低回归包括q=-2/-1/0、阈值等号、负数符号解释和补边屏蔽。换阈值不改变原始head数值，不能只比较掩膜代替head逐字节比较。

AXI-Lite地址、64位AXIS、寄存器ABI 1无需因这次参数迁移改变；阈值是结果/评估语义变化，需要更新B/C交接说明。A文档中对B/C职责的简称不覆盖本仓库“B承担PS/DMA，C承担后处理/HDMI”的分工。

## 5. 精度数据集不能沿用旧379图结论

| 协议 | 图数与用途 | FP32 Dice / IoU | INT8 Dice / IoU |
|---|---|---|---|
| 旧模型、p=0.50 | ISIC2016官方379图；旧软件与PL历史证据 | 0.875608558 / 0.798520436 | 0.872355871 / 0.794366065 |
| 最终模型、p=0.45 | dev638；A用于开发/选模，当前主复现口径 | 0.844054368 / 0.761204057 | 0.842050807 / 0.758341084 |
| 最终模型、p=0.45 | dev中的2016子集376图；A软件结果 | 0.885989629 / 0.809982271 | 0.883390449 / 0.806452557 |

dev638 = 2016 test 376 + 2017 validation 145 + 2017 exposed validation 117。与旧379图相比，2016子集少 `ISIC_0000279`、`ISIC_0001484`、`ISIC_0001685`。这些变化来自A冻结划分，不能随意补回或将376图成绩称为379图成绩。新旧模型、阈值、样本集合同时变化，不能直接相减宣称精度提升或退化；若要比较，需在同一冻结交集、同一阈值下分别重跑。

本轮从六份逐图JSON重算FP32/INT8五阈值宏平均，与 `metrics/comparison.json` 最大差2.22e−16；FP32与INT8各有638个唯一ID。该检查确认报告内部一致性，不是独立复现全部原始图像推理。A声明的开发集ID摘要为 `4ead5256747a87003cb341ce647e2863ec60e5d2d82a737a9572d2963805133d`；摘要序列化算法仍需随A源码核对，不能另造算法并宣称匹配。

**数据角色保持诚实**：dev638不是未见过的独立测试集；2017 exposed部分已参与开发，2018最终测试掩膜据A记录未打开/未评分。本次不启动2018评估，不将dev成绩写成holdout成绩。旧379图保留历史回归；未来若用最终模型重跑全部379图，应单列“旧集合兼容性回归”，不冒充A的主指标。

### 完整复现尚需补齐的材料

- 包不含原始JPEG/GT、A的 `software/role_a` 源码、冻结split/排除清单原件和逐图数据来源hash。`metrics/*.json`可恢复638个ID及分组，但不能代替图像/GT身份和几何处理证据。
- 需取得与包内 `evaluator_sha256` 对应的评估器，以及 `data.py`、`quant_math.py`、`quant.py`、模型/划分构造等依赖或等价的明确合同；核对EXIF、双线性取整、GT最近邻、二值化、面积分组边界和宏平均规则。
- 复用本机已有2016原始数据时逐图核对hash；补齐145+117张2017图像/GT及来源清单。不得仅凭文件名认为内容一致。
- 包内A命令依赖其原仓库，不能在 `D:\PLD` 直接运行 `software.role_a.*`。B现有NumPy可以验证交付的完整样例，但不因此声称独立复现dev638。

## 6. 迁移顺序与退出条件

在现有阶段3/4/5下增加模型换版检查，不重新编号原阶段，不清零旧版本验收。

| 步骤 | 工作与交付 | 成功标准/依赖 |
|---|---|---|
| M1 接收与主机兼容性（本轮完成） | 记录包身份、小块/整网/分块/调度、位宽容量、差异清单 | 141文件完整；15层零差异；明确仍为主机证据 |
| M2 模型入口与阈值贯通（待实施） | 独立正式manifest；统一主机、板端、打包、元数据和渲染入口 | 新旧模型显式选择；错误spec/参数/manifest拒绝；q=-1与有效区一致；默认切换可追溯 |
| M3 开发集协议与参考（待材料/实施） | 冻结dev638源图/GT/ID/来源/阈值；生成新版FP32/INT8参考和报告 | 全集与子集不可混淆；主指标及五阈值、来源/面积组与A核对；保留首个差异 |
| M4 最终模型板上正确性（待授权执行） | 独立目录部署同次bit/hwh、模型及运行时；从首层贯通 | 新版逐层input/output与head零差异；重复帧及短包/缺包/处理中复位恢复通过 |
| M5 最终版本评估与性能（待M3/M4） | dev638逐图PL head比较；≥1000帧且≥600秒；CPU及分项性能；审查部署包 | 全集覆盖、数值、恢复、内存分别通过；最终版本吞吐/P95另判定，不能用旧模型1000帧代替 |
| M6 C交接与整机（继续暂缓接入） | 同步模型身份、尺度、阈值、有效区及结果所有权 | C确认新结果语义；后续才验收后处理/HDMI/UI与含显示的端到端性能 |

建议先以已完成长期验证的 `stage5_ps_range` 功能组合建立最终模型基线，再单独评估native-control/native-resize候选。当前HEAD含这两类尚未完成ARM整网验收的候选，默认不启用；不能将其本机测试记为长期板测成果。M2可先于完整数据补齐，M4的单样例验证也不依赖dev638全部到齐。

旧组合含在线预处理中位1.524842秒、P95 1.886086秒、吞吐0.636627 FPS；新版尚未测量。项目自定≥1 FPS、P95≤1秒的目标保留，B单独计时与最终C集成计时分开。拓扑未变，因此换权重没有来源可支持“自动达到性能目标”的判断。

## 7. 本轮验证与复现入口

以下为本机实际使用的命令；全部退出0。冻结命令只接受不存在的输出路径，重跑时选新的目录，不覆盖旧证据。

```powershell
D:\conda\python.exe scripts/verify_a_handoff.py role_a_final
D:\conda\python.exe scripts/freeze_model_manifest.py --model role_a_final/int8 --spec-sha256 fe137ca5b128fc1983fabad9bee39b3208cda4256e0d04c4e27cef5c07666aef --output out/role_a_final_review/model_manifest.json
D:\conda\python.exe scripts/verify_full_reference.py --model role_a_final/int8 --spec-sha256 fe137ca5b128fc1983fabad9bee39b3208cda4256e0d04c4e27cef5c07666aef --manifest out/role_a_final_review/model_manifest.json --report out/role_a_final_review/full_reference.json --tiled
D:\conda\python.exe -m unittest discover -s sim -p 'test_*.py' -v
D:\conda\python.exe scripts/write_build_manifest.py out/vivado_cnn --verify
D:\conda\python.exe scripts/audit_board_run.py out/stage5/range_stability.json --dataset out/stage5/evaluation --deployment out/stage5/ps_range_bundle.tar.gz --output out/role_a_final_review/old_range_audit.json
```

`freeze_model_manifest.py`自身先执行非分块整网逐层比较，再冻结124文件。临时主机审查还枚举阈值、校验141条包清单、重算指标汇总和运行OracleTile，结果在 `out/role_a_final_review/audit.json`；`full_reference.json`保存分块逐层证据。`out/`不入Git，本文保留身份、结论和命令。

本轮未运行新模型xsim、Vivado构建、SSH/overlay加载、638图原始推理或C联调。后续RTL向量生成入口也需审查其模型默认值，再执行新版算子回归；不能将旧向量全绿记作新参数已仿真。正式部署前保留旧模型、旧manifest及 `stage5_ps_range` 独立目录用于回退；文档提交可用 `git revert <commit>` 回退。
