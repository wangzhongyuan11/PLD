# ACDC Axial 离线实现验证

> 历史记录：本页为10月18～19日离线阶段。当前P6真实模型、板测、网页及交付状态以[2026-10-21验证与使用记录](VALIDATION_P6_20261021.md)为准。

> 2026-10-19新增证据见[真实Release对齐审计](evidence/release_alignment_20261019.json)和[最终计划](FINAL_ALIGNMENT_OPTIMIZATION_PLAN_20261019.md)：完整包校验、45层参数审计、一个真实Axial黄金head的B独立参考零差异、真实MRI几何/预处理定位已完成。本次未运行RTL/综合/板测。下文为10月18日历史验证，其“未取得模型/病例”限制不代表当前状态，也不被新证据改写成真实模型硬件验收。

日期：2026-10-18。执行环境：Windows、Vivado 2026.1、XC7Z020-1目标；PC依赖见`../requirements-pc.lock`。本轮没有连接板卡，没有A合格生产模型，没有下载真实病例或训练模型。下列测试、仿真和实现不能替代真实板测。

## 已实现的链路

两个显式ED/ES 3D NIfTI → 独立整卷归一化与几何核验 → RAS 2mm重采样 → 相邻三层Axial输入 → 15Conv/21节点INT8网络 → 四类softmax → 去letterbox与概率插值 → 原生网格标签 → 无平滑三维表面 → 三平面MPR与定量 → 异步网页、NIfTI/VTK/JSON/CSV/ZIP导出。

- 只有一个Axial模型；网页的三个MPR来自同一个三维体，不是三模型推理或融合。
- INT64独立参考与分块设备分离；PL路径只输入首层图像，全部Conv/Pool/CAT在PL执行，无CPU自动回退。
- 生产模型按外部SHA、参数和尺度、21节点golden、A的FP32/INT8验收记录与实现绑定校验。合成包默认拒绝，必须显式`--allow-synthetic`。
- PS驱动复用既有有界DMA/native控制与搬运；ACDC使用独立ID/IP/位流/运行包。只读引用的正式工程源没有改动。
- PC预取和前一片恢复通过有界线程池重叠；相邻工作层只resize/量化一次；同层tile复用参数；跳连用后释放；worker常驻且只有一个硬件所有者。
- 断点与RPC缓存绑定输入/模型/几何/runtime和字节hash。命中缓存不计新增推理样本或FPS。
- worker/网页支持优雅停止和取消；部署管理器只管理自己的单元，并检查hub/旧后端互斥。已提供三类板端故障注入入口，但本轮没有执行真实systemd切换或板端故障恢复。

## 本地证据

日志和大产物在仓库忽略目录`out/acdc/`。轻量证据清单随源码保存，位流、模型、病例和虚拟环境不进入Git。

| 验证 | 结果 | 证据或入口 |
|---|---|---|
| Python功能/边界/恢复流程测试 | 24项通过 | `python -m pytest acdc/tests -q`；`out/acdc/acdc_tests_final.log` |
| 完整256×256合成网络 | 21/21节点逐元素一致，最大误差0 | `verify-model --allow-synthetic --full`；`out/acdc/full_numerics_final.json` |
| 完整整数网络接入ED/ES链路 | 六个工作片分别真实执行21节点后恢复/导出完成，明确合成模型 | `out/acdc/neural_pair/`，backend=emulator；不报告为PL FPS |
| 兼容RTL系统仿真 | 30次RUN、60次LOAD、5个预期错误，225080个输出字节比较通过 | `acdc/run_sim.sh`；`out/acdc/simulation.log` |
| 128MAC独立阵列 | 41个事务通过 | `acdc/performance_ip/run_sim.sh`；`out/acdc/mac128_sim/simulation.log` |
| 128MAC完整tile/AXI系统 | 与兼容RTL相同的30/60/5覆盖通过 | `acdc/performance_ip/run_tile_sim.sh`；`out/acdc/fast/simulation.log` |
| 周期模型与RTL标定 | 两套各30组，逐组完全一致 | `planning.calibrate`；`out/acdc/performance_projection.json` |
| 前端语法及乱序加载 | 通过；旧ED响应不能覆盖新ES或混合GPU网格 | `node --check acdc/web_assets/app.js`、`node acdc/tests/test_viewer.cjs` |
| 实际浏览器完整任务 | 合成ED/ES共58片完成，切换/MPR/3D旋转/ZIP下载通过 | `out/acdc/browser_jobs/adc24e31f3c44e2dbc661626f8f991ce`；浏览器截图 |
| 原系统Python回归 | 250项：247通过、3项按环境跳过 | Skin 79、CAMUS 44、网页78、hub 49；对应`*_regression.log` |
| 原系统Node回归 | 6个脚本通过 | `out/acdc/legacy_node_regression.log` |

三维PNG导出已通过浏览器实际下载并用Pillow校验，图像600×560。使用Blob下载，避免内置浏览器对data URL下载的限制。源码归档解压到新目录后的CLI/测试复现、runtime完整性和篡改拒绝结果记入`out/acdc/DELIVERY.json`；该复现复用已锁定的PC虚拟环境，不冒充新机器的安装测试。构建证书、静态预算及关键日志SHA见本目录`evidence/`。

RTL向量包含15层全部通道形状、C7/C8/C9的Co分块、三种CAT通道拆分、Pool、带符号head、舍入/饱和、随机背压/源间隙与协议错误；额外覆盖C7/C8/C9的大空间tile和32×256 head tile。它不是用生产权重执行整卷的证据。合成网络稀疏参数的CPU耗时也不能作为真实模型性能。

网页演示为明确标记的解析强度分类器，**不执行15层神经网络**。合成输入为64×64×12、spacing=2/2/5mm，每期RAS工作层29，总58片。一次最新完整任务上传读取至ZIP就绪为6.293秒；该时间只证明本机软件/展示链路，不能转成PL FPS或真实ACDC精度。下载文件SHA与服务器completion记录一致：`559f4d6c6fa071ae796650ab79e00dd63d742354aa7685d0acb15da89ce72927`。

## 构建与性能结论

兼容ACD3（32MAC）100MHz完整构建通过：WNS +0.265ns、WHS +0.018ns、WPWS +3.750ns；Slice LUT 6867、FF 7718、DSP 52、BRAM36等效67。bit/HWH/XSA配对及源码/IP hash检查通过。门禁使用布线时序，不能只看Vivado的BUILD_OK。

128MAC ACD4候选125MHz生成了bit/XSA，但**未通过时序**：WNS −0.861ns、WHS +0.025ns、WPWS +2.750ns；Slice LUT 10298、FF 14664、DSP 151、BRAM36等效83。`check_build.py`按预期拒绝该产物，没有为它签发运行包证书。相关日志保留在`out/acdc/vivado_fast125/`，此前BRAM超限失败也单独保留。真双口RAM访问模板修复后，BRAM已经回到器件容量内。

128MAC候选100MHz完整布线通过：WNS +0.339ns、WHS +0.014ns、WPWS +3.750ns；Slice LUT 10183、FF 14235、DSP 151、BRAM36等效83。bit/HWH/XSA配对、源码/IP hash和构建证书通过，产物在`out/acdc/vivado_fast100/`。最初脚本的Python `-c`内嵌MSYS路径导致证书步骤报错，已改成独立argv传参，并用Git Bash实际调用同一证书入口复验通过；没有因此重写RTL或继承旧时序结果。该配置尚未真实上板。

每中心片固定2,674,917,376 MAC；参数492304字节。当前容量调度为158次卷积RUN、103次非卷积RUN、547个命令，包载荷32,293,264字节。

| 项目 | 周期/时间 | 解释 |
|---|---|---|
| 兼容核心计算 | 110,396,848 cycles | 含当前Conv及非卷积调度 |
| 128MAC纯发射项 | 21,512,192 cycles | 仅MAC发射，不能换算成完整FPS |
| 当前128MAC实现计算 | 43,760,048 cycles | 含逐组排空、串行Q24、输出写回和非卷积 |
| 当前128MAC串行计算+字节搬运下界，100MHz | 0.76053312秒/中心片 | 未加PS打包/同步/MMIO/DMA间隙和系统争用 |
| 同实现假设125MHz的下界 | 0.608426496秒/中心片 | 125MHz版本时序失败，只用于静态比较 |

因此，**当前实现尚不可能达到完整PS-DMA-PL ≥4FPS**。即使忽略PS开销，100MHz当前下界也约1.31FPS；不是等待板卡后就能直接宣称4FPS。128路MAC、RAM复制及地址流水已实现；64bit宽装载/排空、PL计算与DMA双缓冲重叠、并行Q24/跨像素流水和进一步DDR复用仍是后续性能工作，不能写成已经完成。未来任何更改须重新仿真、布线和板测。

原始256×256×12不能单独确定工作层数。若spacing=1.5/1.5/5mm且轴对齐，每期工作体193×193×29，共58片。当前100MHz实现上述下界约44.11秒/ED+ES，加上实际PS/网络/重采样/恢复/表面/导出会更长。4FPS假设下58片的纯推理也至少14.5秒。斜切affine可能增加工作层数，不允许用只推理24片或跳层美化总时间。

## 尚未验收

1. A的最终Axial权重、FP32源、实际机器合同/布局、尺度、固定患者划分、预处理检查样例与完整golden尚缺；当前内部包格式不冒称A的最终导出格式。
2. 所有真实PL逐层/整网一致性、1000片稳定性、故障恢复、FPS/P95、RSS/CMA/功耗，以及真实ED/ES端到端时间尚未测量。
3. 真实病例的Dice/HD95/ASSD、分割体积和EF误差尚未评价。已提供患者ED/ES配对与hash固定的evaluate入口。
4. 没有修改正式Skin/CAMUS源码、模型/位流、服务或`delivery/current.json`，也没有继承旧版本的板测证据。局部复用与本地回归不代表新ACDC板端已经验收。

本次交付的状态为：可运行的离线完整链路、独立可综合RTL和板端接入/验收工具；生产模型、真实硬件验收与4FPS性能闭环仍未完成。
