# 阶段5评估口径与复现

评估B独立子系统；C后处理、HDMI/UI尚未接入。精度统计不包含形态学或其他后处理。完整测试集成功与速度、稳定性分别判定，不能相互替代。

## 最终模型的当前口径

2026-09-27收到 `role_a_final`。详见[最终交付审查](../history/ROLE_A_FINAL_REVIEW.md)。运行时和评估程序已迁移，当前证据见 [实施进度](../FINAL_MODEL_STATUS.md)。后文旧版命令和旧379图报告保留历史语义。

- 模型：`soup_v5_a050`；spec SHA256 `fe137ca5b128fc1983fabad9bee39b3208cda4256e0d04c4e27cef5c07666aef`，checkpoint及包身份见审查文档。新模型使用独立manifest/输出目录，不覆盖旧参考。
- 主集合：冻结dev638＝2016 test 376、2017 validation 145、2017 exposed validation 117。每个ID绑定来源、JPEG/GT哈希、预处理几何和有效区；原始材料已补齐，2017下载来源与hash保存在 `out/final_model/data2017/provenance.json`。指标JSON可恢复ID，但不能代替数据身份；开发集不能称为独立holdout。
- 主阈值：概率0.45；FP32 logit≥−0.20067069546215124，最终INT8 head_scale=0.1995087721216397，对应q≥−1。五阈值0.45/0.50/0.53/0.56/0.60分别对应q≥−1/0/1/2/3；补边仍强制背景。阈值、模型尺度及比较是否含等号写入manifest，掩膜比较不能代替原始head比较。
- 目标结果：FP32 Dice/IoU 0.8440543675818143/0.7612040567680144；INT8 0.8420508068333175/0.7583410835598973。2026-09-28已在125MHz板上从638张原始JPEG逐图复现：PL原始head全部与冻结INT8参考逐字节相等，宏平均、五阈值、来源组及tiny/medium/large组零差异（见[实施进度](../FINAL_MODEL_STATUS.md)）；FP32差异仍保留A独立口径，不宣称FP32完全复现。
- 保留旧379图历史回归。若最终模型另跑379图，单列模型和阈值；不能与A的376图子集或dev638混算。新旧模型比较需同集合、同阈值重新测量。
- 新版稳定性：覆盖全部638个唯一ID，循环至少1000帧且600秒，重复图不重复计入精度宏平均；数值、覆盖、恢复、内存、性能各自判定。CPU基线使用同一最终模型、量化输入与head比较口径。

`prepare_evaluation.py` 默认最终模型/dev638，支持重复 `--images` 和 `--masks`；`--collection final_isic2016_dev376` 只代表完整376子集，不能通过dev638验收。`package_evaluation.py`、板端和审查程序按冻结ID/来源摘要判断完整性；旧schema只接受旧模型379图。`--allow-subset` 仅限smoke。新版完整软件命令：

```powershell
uv run --index https://download.pytorch.org/whl/cpu --with-requirements requirements-evaluation.txt python scripts/prepare_evaluation.py --collection final_dev638 --images ISBI2016_ISIC_Part1_Test_Data --images out/final_model/data2017/images --masks ISBI2016_ISIC_Part1_Test_GroundTruth --masks out/final_model/data2017/masks --output out/final_model/evaluation_638
```

FP32独立重构与A存在少量边界指标差异，报告分别记录 `int8_matched` 与 `fp32_matched`；不得隐藏或以INT8一致替代FP32复现。最终B验收首先要求原始PL head与冻结INT8参考逐字节相等。

## 旧模型379图：已实现的复现入口

以下各节均针对 `reference/baseline_manifest.json` 锚定的旧模型，阈值0.50。`stage5_ps_range` 的1000帧证据仅属于该旧模型/运行时组合；最终包须重新生成数据并验收。

### 数据和数值

- 使用A清单`manifests/split_with_test.json`中的379个test ID；本机源JPEG及GT必须与这些ID一一对应，无缺失或重复。
- 图像使用EXIF方向、RGB、双线性letterbox，按首层尺度量化；GT使用相同几何位置和最近邻缩放。只统计256×256模型空间中的非补边有效区域。
- FP32由冻结`best.pt`的30个weight/bias张量重建15层图，ReLU、Pool、nearest Upsample和Concat顺序与当前合同一致。A原始预处理/训练源码未提供；如与A报告存在差异，保留原始计数与差异，不调整预测迎合报告。
- INT8软件使用既有NumPy INT64累加、Q24最近偶数舍入与层间LUT的完整网络。每次准备测试集前先通过冻结样例的15层输入/输出比较。
- FP32以logit≥0、INT8以q≥0得到掩膜；每张图计算Dice和IoU后作宏平均。空预测且空GT记1；有效区域不能为空。原始交集、预测面积、GT面积和有效像素数均保留。
- 每个NPZ保存首层输入、完整INT8 head、GT、有效区和FP32掩膜，禁用pickle。manifest绑定源图/GT、输入/head、NPZ、模型、checkpoint、参考源码和运行库hash/版本。

## 主机准备

使用经过首层字节核对的Pillow12.3/JPEG8。`requirements-evaluation.txt`固定本次主机运行库；不向板卡安装Torch。

```powershell
uv run --index https://download.pytorch.org/whl/cpu --with-requirements requirements-evaluation.txt python scripts/prepare_evaluation.py --output out/stage5/evaluation
D:\conda\python.exe scripts/package_evaluation.py out/stage5/evaluation out/stage5/evaluation.tar.gz --images ISBI2016_ISIC_Part1_Test_Data
```

`--limit 3`仅用于预检查，结果显式标记`complete_test_set=false`。完整运行应有379条记录、`passed=true`及`complete_test_set=true`。`progress.json`仅表示中间进度，不能作为最终验收结果。包中只包含上述数据、manifest及可选JPEG，不包含来源未核验的文件。

## 板上评估

在独立版本目录部署通过时序和配对检查的CNN包，并逐项验证`deployment_manifest.json`；将评估包解到同一目录，得到`dataset/`。不要覆盖已有验证版本。使用登录shell初始化的PYNQ Python，并确保没有其他进程拥有DMA或执行CPU基准。

```bash
/usr/local/share/pynq-venv/bin/python3 board/evaluate_dataset.py --bit overlay/dma_top.bit --dataset dataset --decode-images --min-frames 379 --check-recovery --report evaluation.json
```

每张图先验证源文件hash，再从原始JPEG预处理。板上输入、有效区和几何信息必须与主机参考精确一致，才执行PL整网；只注入首层输入，禁止注入中间golden。head逐字节比较后，由板上输出独立计算Dice/IoU并核对原始计数。失败保留已完成JSONL、最终错误JSON和非零退出码；硬件清理失败同样不能通过。

阶段5优化包可显式附加`--native-copy out/native/tensor_copy.so --tx-range-sync --parallel-resize`。先运行`scripts/build_native_copy.py`生成并核验NEON库及清单；并行缩放必须同时指定`--decode-images`。报告记录拷贝、缓存同步和预处理后端以及相应源码哈希。TX范围同步只作用于发送，接收与尾保护维持完整同步；并行缩放的输出仍逐张与原始冻结输入精确比较，不能重做golden迁就候选。

`seconds`为量化输入到head的PS调度/搬运/PL计算/拼接耗时。`online_b_seconds`另加解码后在线预处理。两者均排除比较、文件读取和C显示；JPEG解码另报。逐帧延迟用于中位数、P95及总推理时间计算的吞吐。DMA等待包含PL计算等待，不能与计算周期简单相加。

## 稳定性与CPU对比

```bash
/usr/local/share/pynq-venv/bin/python3 board/evaluate_dataset.py --bit overlay/dma_top.bit --dataset dataset --decode-images --min-frames 1000 --min-seconds 600 --check-recovery --report stability.json
/usr/local/share/pynq-venv/bin/python3 board/cpu_baseline.py --repeat 3 --report cpu_baseline.json
```

稳定性循环379图并逐帧比较，至少1000帧且600秒；默认3000帧上限，未达到要求时明确失败。逐帧保留RSS/HWM、可用内存及CMA余量，检查暖机后的平台和持续增长趋势。JSON中的`stability_duration_and_frames`仅证明时长与帧数，最终还必须审核数值、错误恢复与内存趋势。重复帧不重复计入379图精度宏平均。

CPU基线在同一板卡运行同一冻结整数网络，从相同量化输入到head，1次预热后3次计时，比较在计时外。该实现使用NumPy INT64精确参考，不是NEON/TFLite等优化推理库，报告不得把所得加速比推广到所有CPU实现。基准测量与PL测量不能并发。

## 可视化报告

完整运行下载后，用实际运行的部署包做独立核验；不能以当前正在修改的源码替代旧部署版本：

```powershell
D:\conda\python.exe scripts/audit_board_run.py out/stage5/range_stability.json --dataset out/stage5/evaluation --deployment out/stage5/ps_range_bundle.tar.gz --output out/stage5/range_stability_audit.json
```

审查核对部署/运行时哈希、每帧输出和指标、JSONL一致性及统计重算，分别报告数值、恢复、时长帧数和性能。内存仍需结合图像尺寸及同图循环差值审查；数值通过不代表性能达标。

```powershell
D:\conda\python.exe scripts/render_evaluation_report.py out/stage5/evaluation --output out/stage5/report
# 完整板测下载后追加：--board out/stage5/baseline_379.json
```

本机使用Matplotlib3.10.6导出PNG和HTML；数据来自已核验的NPZ及manifest，不在绘图环境重新解码JPEG。输入图显示反量化后的输入，明确标注Quantized input。报告展示逐图Dice散点、量化差异分布和五张最低分样例，保留模型误分，不只挑选成功样例。加入板测报告前会核对379个ID、head哈希、原始指标计数及参考manifest哈希；未附完整板测时明确标记软件参考。在线报告不包含C显示验收。
