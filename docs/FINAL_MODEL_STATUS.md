# A最终模型迁移与B交付进度

2026-09-29用户决定先向C交付已跑通版本、停止继续试验优化。当前选定的B版本与C调用合同见[C交接说明](C_HANDOFF.md)；在线B P95目标仍开放，不影响已完成数值与恢复证据的版本边界。

2026-09-27。用户在初始审查后授权完成B实现、加速和真实板测；C后处理、HDMI/UI及C确认不再是本轮B交付的前置条件。保留≥1 FPS、在线B延迟P95≤1秒、dev638全覆盖、≥1000帧且≥600秒稳定性目标。尚未满足这些条件时不宣称B全部完成。

## 模型及数据身份

- 默认 `role_a_final/int8`，`soup_v5_a050`；spec SHA256 `fe137ca5b128fc1983fabad9bee39b3208cda4256e0d04c4e27cef5c07666aef`，checkpoint SHA256 `f488719fe6d4d09c9b86be84d06b19d34241d6c14060b87da7bab25d1367eb77`。
- 124文件固定于 `reference/final_manifest.json`。旧模型显式路径仍可使用，未知spec或文件漂移拒绝加载。
- head尺度0.1995087721216397；概率0.45对应有符号INT8 `q >= -1`。五概率阈值0.45/0.50/0.53/0.56/0.60对应q=-1/0/1/2/3，补边不参与指标或面积分母。
- `reference/dataset_contracts.json` 固定ID、来源和A面积组：dev638＝376+145+117；376子集及旧379回归单列。B来源/ID摘要有独立序列化定义，不冒充A未提供算法的摘要。
- 262张缺少的2017图像及GT已按A声明的官方S3路径取得，来源/hash见 `out/final_model/data2017/provenance.json`。dev638用于开发选模，不是独立测试集。

## 已完成验证

| 层次 | 结果 | 证据 |
|---|---|---|
| Python与门禁 | 42项通过；错误模型、manifest、集合、重复ID和阈值漂移拒绝 | `sim/test_model_versioning.py`、`sim/test_dataset_contracts.py`及原回归 |
| 新版参考 | 15层整网/halo分块input/output零差异；调度软件替身180次运行一致 | `out/final_model/full_reference.json` |
| 新版RTL仿真 | 全套xsim通过；完整卷积15层/80tiles/3,866,624输出字节零差异 | `out/final_model/sim_logs/migration/` |
| 迁移基线构建 | 当时RTL未变，复用配套100MHz bit/hwh/XSA；WNS +0.209ns、WHS +0.018ns、WPWS 3.750ns | `out/final_model/migration_build/`；6924 LUT、7299 FF、67 BRAM36等效、50 DSP |
| 真实板卡迁移 | 三帧15层input/output及head零差异；短包、缺包、计算中复位及下一帧恢复通过 | `out/final_model/migration_verify.json` |
| ARM原生控制 | 编译、40项寄存器模型、有界期限、逐层及三类恢复通过 | `out/final_model/native_control_verify.json` |
| 376图软件 | INT8 Dice 0.8833904491556513、IoU 0.8064525567678688；全部图像五阈值与A一致 | `out/final_model/evaluation_376/manifest.json`、`out/final_model/report_376/index.html` |
| dev638软件 | 全638图五阈值INT8与A逐图零差异；Dice 0.8420508068333175、IoU 0.7583410835598974，来源及面积组均已统计 | `out/final_model/evaluation_638/manifest.json`、`out/final_model/report_638/index.html` |
| 376图板测 | 在线JPEG预处理、原始head和指标全一致，故障恢复通过，部署审查通过 | `out/final_model/evaluation_376_audit.json` |
| 旧证据兼容 | 旧1000帧报告及参考重新审查通过，不是重跑旧版板测 | `out/final_model/legacy_audit.json` |

FP32独立CPU重构在dev638上为Dice 0.8440510356409867、IoU 0.7612000695925929；与A逐图五阈值最大差0.0025510204081633514。显式sigmoid阈值及关闭MKLDNN的诊断未消除差异，原因尚未确认，不能归因于某个后端或将INT8一致扩展成FP32完全一致。诊断见 `out/final_model/fp32_delta_probe.json`；完整参考的FP32值保留原结果。

部署 `/home/xilinx/jupyter_notebooks/pld/final_model_20260927_v1/` 对应 `out/final_model/migration_v1.tar.gz`，154文件hash核验通过；原生库在板端编译并核对源码/库manifest。bit SHA256 `a1daac7d16b636d0545c0cf95a8b18da06af015169af60e3d4fa552fbc4e7cb7`。迁移包保留未提交源码状态及逐文件hash；正式发布重新打包，不覆盖该部署后沿用证据。

## 性能与剩余工作

原生拷贝+TX前缀同步的新版量化输入到head约1.43秒；加原生控制后10帧中位1.354668秒、P95 1.417304秒，见 `out/final_model/native_control_profile.json`。不含预处理，仍未达标。ARM原生resize候选379图像素一致，但中位0.0753秒慢于Pillow的0.0589秒，不纳入正式路径。

376图板测含在线预处理中位1.486524秒、P95 1.843776秒、吞吐0.655433 FPS，仍未达标；376帧不等于1000帧稳定性验收。

新ABI 1.1模式5在PL装载卷积输入时融合ReLU/尺度LUT，省去8个独立MAP过渡，143次算子运行、315条命令/帧；15层卷积输出仍为原始requant结果。43项Python回归通过，命令级仿真46次运行/102次加载/11个期望错误通过，覆盖全部255个合法INT8查表索引和更换LUT后输入失效。100MHz新构建WNS +0.062ns、WHS +0.020ns，6962 LUT/7299 FF/67 BRAM36等效/50 DSP；三帧全部原始层输出和三类故障恢复通过。原始输入在融合层取上一层PL输出，不能把它称为融合后输入RAM的直接读回比较。

融合版10帧中位1.136330秒、P95 1.184557秒（量化输入到head），较原生控制基线中位约快16%；仍未满足在线性能目标。证据 `out/final_model/fused_verify.json`、`fused_profile.json`、`fused_accel_sim.log`；155文件部署包 `fused_v1.tar.gz` hash核验通过，板端目录 `final_fused_20260927_v1`。原1.0产物保存到 `out/final_model/migration_build/`，本版产物保存到 `out/final_model/fused_build/`，旧部署独立保留。

同模型板载NumPy INT64基线预热1帧后3帧全部正确，中位56.063447秒；此处与融合PL相比约49.3倍，仅代表该精确整数参考，不是优化ARM库对比，见 `out/final_model/final_cpu_baseline.json`。

接下来继续加速，完成最终版本dev638、≥1000帧/≥600秒及部署审查。新增 `board/infer_images.py` 为不依赖GT/golden的JPEG推理入口，输出独立原始head二进制及帧号、模型、阈值、有效区和耗时JSON；三种数据来源各一张真实JPEG已上板，输出hash、输入hash、有效区和阈值独立核验一致。该三图检查不是全集或性能验收。

## Pillow行缓冲区缩放候选

`reference/pillow_resize.py` 直接读取Pillow 9.0.1已解码RGB行，避免再复制整张大图；两线程NEON水平滤波仍保持Pillow Q22系数、横向8bit中间值和纵向舍入。原图只读且保留到两个工作线程全部退出。`decode_rgb` 对已有RGB结果不再重复转换，全部8种EXIF方向及灰度输入回归通过。

板上dev638全部638张缩放结果与标准Pillow逐像素一致，见 `out/final_model/rows_dev638.json`：候选中位0.080043秒、P95 0.476944秒，标准Pillow中位0.206026秒、P95 1.584019秒。这里仅计已解码RGB缩放，不含量化、DMA或推理，不能当成整网FPS。24张跨来源先导集也逐像素一致。

`scripts/build_native_resize.py` 在ARM/Pillow 9.0.1上编译并通过60项像素/边界检查，`board/native_resize.py` 拒绝库、源码集合、Pillow版本及核心二进制hash漂移。运行包装器像素检查与四类manifest篡改负例通过，构建清单为 `out/final_model/rows_runtime_build.json`。板端候选目录 `final_rows_candidate`、运行包装器目录 `final_rows_runtime` 独立保留。`--native-resize` 已接入评估及无golden推理入口，但该组合的整网与稳定性验收尚待执行。

142.857 MHz首个流水候选布线WNS −0.893ns、WHS +0.015ns，被 `check_build.py` 以退出码1拒绝，未部署。补充重量化输入、输入RAM写口流水后的第二次构建仍为WNS −0.768ns、WHS +0.041ns，也被时序门禁拒绝。第二次版本的45项Python回归、命令级xsim和15层/80 tile完整仿真通过；这仅证明逻辑/数值仿真，不是板上可用产物。失败时序及关键路径见 `out/final_model/clock_failed_build/` 和 `clock_pipeline_build.log`，当前最差路径从顶层`co`配置寄存器至`param_word`复位端。不能把候选频率当成已通过硬件事实。100MHz融合版配套产物仍保存在 `out/final_model/fused_build/`。

## 125 MHz ABI 1.1候选与板门验证

142.857 MHz两轮失败后改用PS7 FCLK0=1000/8的125MHz：控制路径改为寄存软复位并分段校验`check_shape`/`check_dims`，tensor BRAM数据口加捕获级，保留候选的MAC/重量化流水与`LOAD_DRAIN`。`check_build.py`通过：WNS +0.135ns、WHS +0.024ns、WPWS +2.750ns，bit/hwh与XSA配对，DRC无违例，6709 LUT/67 BRAM36等效/50 DSP，仍在C预算内。45项Python、全套命令级xsim和15层/80 tile完整仿真先通过；产物与源码manifest见 `out/final_model/clock125_build/`（bit `acc8520db…ed1c7`、hwh `c7830e4a…c944`、xsa `c5878c05…4b88`），源码已随commit `1deb6d9` 提交。

2026-09-28板门验证通过（该门当时仅含24图先导集，未含全集）。部署包 `out/final_model/clock125_v1.tar.gz`（161文件，SHA256 `ddea3f0e…3284`）解压到新目录 `/home/xilinx/jupyter_notebooks/pld/final_clock125_20260928_v1/`，`DEPLOYMENT_HASH_PASS 161`；原生copy/control/resize三库在板端编译并各自通过模型与hash检查。三帧全部15层原始input/output及head零差异，短包、缺包超时、计算中复位及下一整帧恢复通过，报告`clock_mhz=125.0`。24张跨2016/2017train/2017dev真实JPEG经`--decode-images --native-resize`整网推理，全部`output_sha256`与100MHz融合版`fused_smoke24.json`逐一相同（24/24），dice/iou均值完全一致；三类故障再次恢复。量化输入到head 10帧预热profiling中位1.045928秒、P95 1.069705秒，每帧34,849,592计算周期；在线JPEG到head样本中位1.241991秒、P95 1.707140秒。相对融合版100MHz的1.136330秒/在线1.463562秒有提升，但仍未达到≥1FPS或≤1秒目标；dev638全集与≥1000帧稳定性当时尚未执行，已在同日晚完成（见下段）。证据：`out/final_model/clock125_verify.json`、`clock125_smoke24.json`、`clock125_profile.json`；旧100MHz部署`final_fused_20260927_v1`未改动，可按原目录回退。

同日晚在完全相同的125MHz部署/数据上完成dev638全集与稳定性门：数据manifest SHA256 `85ed9c1c…7afc`与主机冻结参考逐字节一致（638 ID，2016 test 376 / 2017 validation 145 / 2017 exposed validation 117），部署bit/hwh SHA与构建产物一致，三块原生库manifest hash核验通过，运行前无竞争推理进程。全集评估638帧全部从原始JPEG在线解码+NEON行缩放，量化输入hash 638/638与冻结参考一致，PL原始head `output_sha256` 638/638逐一相等，指标计数与A逐图INT8参考一致；Dice/IoU宏平均0.8420508068333175/0.7583410835598974，三个来源组与tiny/medium/large组、638×5五阈值重算均与A冻结结果零差异。独立稳定性运行1000帧（覆盖全部638唯一ID，其中362帧为第二循环重复图）、wall 1881.747秒（主机时间22:52:38～23:24:32），1000/1000帧head逐一精确（含362个重复head），短包/缺包/计算中复位三故障恢复及恢复后整帧通过；RSS从首百帧中位78,214KiB降至第二循环的约21MiB区间，随后为有界波动（第639–1000帧中位21,014KiB、范围19,572–22,268KiB、末帧22,268KiB，同图下一循环差中位−49,270KiB），CmaFree最低21,888KiB；1000帧观测内未见失控增长，但不构成无限时长或无泄漏保证。量化输入到head全集中位1.044811秒/P95 1.085599秒（稳定性运行1.046468/1.087104），在线预处理到head中位1.198779秒/P95 1.660312秒（稳定性1.155472/1.654199），核心0.952FPS、在线0.793FPS——**仍未达到≥1FPS和在线≤1秒目标**，该门只关闭数值/覆盖/恢复/稳定性，不关闭性能。证据：`out/final_model/clock125_eval638.json`（SHA `4e99a9a0…57c0`）、`clock125_stability1000.json`（`c3376b05…c8e1`）、对应`.jsonl`、`clock125_eval638_audit.json`、`clock125_stability1000_audit.json`、`clock125_eval638_reference_check.json`；板端报告留在`final_clock125_20260928_v1/`，运行后板卡无残留推理进程。

## 125 MHz ABI 1.2 双分支融合卷积

2026-09-29，新增模式6，把第8/10/12层decoder入口原先独立的Concat/Upsample/尺度转换折入PL卷积。PS仅采集skip完整halo及deep半分辨率紧凑halo；PL分别查两张LUT、按nearest×2读deep并计算原有卷积。保留ABI 1.1模式0–5及显式`--fuse-decoder`开关，精确合同见[寄存器文档](design/ACCELERATOR_REGISTERS.md)。每帧命令数315→203。47项Python回归、引擎随机/边界/复位xsim、命令级58次运行及24个期望错误、三个真实decoder入口的28 tile/31次xsim运行均退出0；新增真实tile累计比较1,015,808输出字节零差异。原先15层加新增tile的混合长仿真只到第100 run后超时中断，不记作通过；新增全部tile已由独立`tile-fused`入口完整覆盖。

当前构建`out/final_model/mode6_build/`通过布线后检查：FCLK0 125 MHz、WNS +0.013ns、WHS +0.049ns、WPWS +2.75ns，bit/hwh与XSA配对；7,108 Slice LUT、67 BRAM36等效、53 DSP，仍在B预算内。13ps的setup裕量很窄，后续改动必须重新布线验收。bit/hwh/XSA SHA256分别为`0df9c718…7f0baf`、`7adfbe2d…81da13`、`834270c2…27e037d`；部署包`out/final_model/mode6_v1.tar.gz` SHA256 `44688973…7bce0e0`，161文件逐项hash通过，板端独立目录`/home/xilinx/jupyter_notebooks/pld/final_mode6_20260929_v1/`。旧125 MHz部署保持可回退。

板上4帧全部15层原始输出/head逐字节一致，短输入、缺输入超时、计算中复位三类故障恢复及下一整帧通过。预热后10帧核心中位0.770894秒/P95 0.791643秒。24张跨来源真实JPEG的head hash和指标计数与上一版逐图相同，在线B中位0.956214秒/P95 1.376893秒。最终模型dev638全集638张原JPEG+NEON行缩放全部输入/head与冻结参考一致，和上一版638个head hash、逐图指标计数独立重算相等；Dice/IoU仍为0.8420508068333175/0.7583410835598974。全集核心中位0.758427秒/P95 0.783818秒，在线B中位0.898447秒/P95 1.371000秒，核心1.315 FPS；三个故障恢复及下一帧再次通过。全集运行墙钟1129.949秒，但只有638帧，**不能代替此版本的≥1000帧稳定性门**。证据`out/final_model/mode6_verify.json`、`mode6_profile.json`、`mode6_smoke24.json`、`mode6_eval638.json`及对应JSONL。核心≥1 FPS已达到；在线P95≤1秒尚未达到，B最终性能/稳定性仍未全部验收。

## ABI 1.2紧凑输入tile与精确快速量化

模式6的deep分支按紧凑半分辨率存入输入bank后，调度器按真实bank容量把decoder入口tile高度从16增至32，28个融合卷积tile减为14个，每帧命令203→175。三个decoder入口共17次tile xsim、1,114,112输出字节与参考零差异；48项Python回归通过。RTL/bit未变，仍使用同一125MHz ABI 1.2构建。独立部署`final_mode6_tile32_20260929_v2`的161文件hash通过；板上三帧15层原始输出/head一致、三类故障及下一帧恢复，10帧核心中位0.704515秒/P95 0.721568秒。24图在线B中位0.899487秒/P95 1.314201秒，head及指标与前版逐图相同。证据`out/final_model/tile32_verify.json`、`tile32_profile.json`、`tile32_smoke24.json`；该短样本不代替全集性能分布。

在相同版图上，首层RGB8量化改为由冻结尺度生成的256项查表；枚举全部RGB8值，与原float32路径逐值一致。旧路径仍可选；显式`--fast-quantize`只在原始JPEG在线解码时启用。独立部署`final_mode6_fastq_20260929_v3`，包SHA256 `7651de35eade4af88d6ea79407e06d6ee4c87e8847a0bd64b732698f51b42873`，161文件及三块原生库构建/核验通过。24图输入/head/指标与v2一致，在线B中位0.854879秒/P95 1.267007秒，故障恢复通过；报告`out/final_model/fastq_smoke24.json`。

v3随后独立完成1000帧、1483.827秒稳定性运行，覆盖dev638全部638张原始JPEG，第二循环重复362张。`scripts/audit_board_run.py`按该部署包和冻结数据审查通过，报告SHA256 `5d057cf35915142ed2a8b28948a79c644a1a903c3fd2402d620836af984b83fd`；与旧125MHz版同顺序1000个ID、PL head SHA256和逐图指标均1000/1000一致。Dice/IoU宏平均仍为0.8420508068333175/0.7583410835598974。短包、缺包、计算中复位及下一帧恢复通过；RSS前100帧中位77,728KiB、末100帧34,768KiB，十个百帧窗口未见持续增长，不能据此保证无限期无泄漏。核心中位0.700375秒/P95 0.726514秒、1.422 FPS；在线B（已解码RGB进入预处理至独立head，排除JPEG解码、文件IO和C显示）中位0.758002秒/P95 **1.265720秒**。所以这一冻结版本的全图数值、1000帧/≥600秒、恢复与核心吞吐通过，项目自定在线P95≤1秒**仍未通过**。证据`out/final_model/fastq_stability1000.json`、`.jsonl`和`fastq_stability1000_audit.json`；板端目录`final_mode6_fastq_20260929_v3`。后续预处理候选不得沿用v3稳定性证据称作自身已验收。

在独立v4候选包中试验NEON水平滤波一次处理两行、复用系数。ARM编译及两种行核各60项像素/边界检查通过；24张真实JPEG的输入/head/指标与v3逐一一致，但在线B中位0.876401秒（v3为0.854879）、P95 1.259074秒（v3为1.267007）。高分辨率图的耗时有快有慢，短样本不足以说明全集性能提高，也未达到1秒目标。该候选的源码改动已撤回，**不作为交付版本**；短测报告留在`out/final_model/rows2_smoke24.json`，板端隔离目录不覆盖v3。后续若继续攻在线P95，需在冻结输入、相同计时口径下先验证有实际收益，再对最终采用的版本做必要验收，不对每个实验重复跑全集。
