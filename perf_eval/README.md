# PLD 离线性能评测工具（2026-10-17）

本工具复用目前皮肤镜 soup_v5_a050 和 CAMUS hidden_histogram 候选的冻结模型与软件参考。现在没有连接板子；本次只做电脑端开发、测试和离线报告，不改现有网页、模型、RTL或服务。

## 已实现

- 准备一次固定输入，保存配对 INT8 / FP32 数组与有效区域、模型和输入SHA256；两后端使用同一份准备好的数据。
- 模型只初始化一次；区分初始化、首次前向、额外预热与热态测量。
- CPU INT8 数值参考批测；可选CPU FP32 PyTorch入口；显式选择PL时才导入硬件驱动。
- 输入数组到head的推理耗时、读取/标签时间、完整循环吞吐率；均值、中位数、样本P95。
- 逐帧RSS、Linux可用内存/CMA采样；采样最大值不能称为连续监测峰值。Windows当前RSS使用系统接口，不依赖psutil。
- 输出整数head的逐元素差异、标签差异；同板同数据且位精确时生成CPU INT8/PL推理加速比。
- 对用户给出的真实标注计算各结构Dice/IoU、按图平均与像素池化结果；同一GT和区域时计算候选减基线的百分点变化。
- Vivado资源/功耗报告解析，功率计采样积分得到整板能耗/帧；缺资料时不填写功耗数字。
- 现有网页计时、CAMUS DMA事件报告导入，生成HTML、Markdown、JSON和CSV报告。

组织上的优化：复用模型对象、把原图预处理与推理计时分开、结果编码退出推理计时范围，并按帧释放临时数组。此处没有新的模型/RTL提速结论。

## 依赖

基本命令（help、timing、utilization、power、dma-report、report）只需Python标准库。模型推理、prepare、score、compare需要NumPy/Pillow；FP32另需PyTorch。医学格式按原reader安装nibabel等依赖。

```powershell
python -m pip install numpy pillow
# 仅需要FP32时再安装：
python -m pip install torch
```

旧板Pillow使用兼容常量，不要求升级系统。CPU INT8使用NumPy完整整数参考，并非充分优化的ARM推理库，不能据此夸大FPGA相对优化CPU的优势。当前机器没有PyTorch，本轮未执行FP32前向；该入口仍需安装依赖后与A方FP32结果回归。

## 文件结构和运行入口

仓库入口为 `python -m perf_eval`。可选运行 `python scripts/package_perf_eval.py` 导出工具包，包含 `run_eval.py`、`perf_eval/`、`pld_source/`（必要的原有参考/驱动源码）、模板和校验脚本。工具包不包含权重、位流、私有输出或既有成绩，模型来自唯一正式Release。

仓库使用 `python -m perf_eval` 和 `--source .`。独立工具包使用 `python run_eval.py` 和 `--source pld_source`；`--model-root`须明确指向通过正式Release准备的模型目录。

## 无板子现在可以做什么

先按仓库根 README 下载唯一正式 Release 的 `pld_dual_20261017_assets.zip` 到 `out/`，核对并准备模型：

```powershell
python scripts/release_assets.py out/pld_dual_20261017_assets.zip --prepare
python scripts/verify_delivery.py
```

以下命令均在完整仓库根目录运行；不会连接板子或改动服务。

```powershell
python -m perf_eval --help
python -m unittest discover -s perf_eval/tests -q
```

重新准备皮肤镜合成数据并跑CPU（输出目录必须是新的，防止覆盖已有结果）：

```powershell
$skinModel = "role_a_final/int8"
python -m perf_eval prepare --source . --model skin --model-root $skinModel --synthetic --output work/skin_inputs
python -m perf_eval bench --source . --model skin --model-root $skinModel --dataset work/skin_inputs/dataset.json --backend cpu-int8 --warmup 1 --repeat 5 --output work/skin_cpu
```

CAMUS：

```powershell
$camusModel = "camus_candidate_hidden_histogram_int8_20261015/int8_export"
python -m perf_eval prepare --source . --model camus --model-root $camusModel --synthetic --output work/camus_inputs
python -m perf_eval bench --source . --model camus --model-root $camusModel --dataset work/camus_inputs/dataset.json --backend cpu-int8 --warmup 1 --repeat 5 --output work/camus_cpu
```

真实数据把 `--synthetic` 改成 `--input image1.jpg --input image2.jpg`；CAMUS支持原reader允许的灰度/数组/医学格式。准备阶段只提供图像，不使用标注。多数组Role-A NPZ请先转为原图/单数组；本工具未复制网页NPZ缓存适配。

FP32准备了对应的float数组，不会简单将INT8反量化伪装成原始浮点输入。安装PyTorch后，例如皮肤镜：

```powershell
python -m perf_eval bench --source . --model skin --model-root $skinModel --dataset work/skin_inputs/dataset.json --backend cpu-fp32 --checkpoint "role_a_final/best.pt" --repeat 5 --output work/skin_fp32
```

CAMUS使用同样命令形式，checkpoint须显式提供已核验候选FP32权重（SHA256 `3d8f2689…`）；若正式Release中未提供它，则由A方提供对应权重并核验，不能用不同版本代替。两种FP32仅接受当前已知checkpoint SHA256，CPU线程固定2并记入报告。

## 计时口径

- `inference`：已准备好的输入数组到独立head；PL包含PS调度、DMA等待、拼接和输入校验，不能当作纯PL算术周期。
- `processing`：每帧读取数组 + inference + 生成256网格标签；排除PNG、ZIP和HDMI。
- `wall_throughput_fps`：实际完成帧数 / 测量循环墙钟；包含结果哈希、首轮产物写入、采样等工具开销。
- 初始化和首次前向单独报告；首次前向后再执行`warmup`指定次数，均不混进热态统计。
- 合成图只能证明流程可运行，短测P95只是样本分位数；不能用于医学精度、真实业务分布或验收性能保证。
- 目前标签是256×256原始网络网格，补边排除。没有原尺寸恢复、CAMUS固定小岛/孔洞后处理或交互图割。不要与现有原尺寸后处理Dice混比。

## 输出对比和精度

两个bench报告可比较：

```powershell
python -m perf_eval compare work/skin_cpu/benchmark.json work/skin_pl/benchmark.json --output work/head_comparison.json
```

会核对模型、输入manifest、输入哈希、顺序/重复次数、有效区域、输出文件哈希。不同电脑/板子的耗时可以在表中并列，但不自动当成同板FPGA加速比；host fingerprint只用于阻止不同主机误配。FP32和INT8输出不能逐字节比较，仍可比较标签，医学精度需要GT。

按 `templates/score_pairs.json` 建立标注配对清单。prediction/truth为相同尺寸二维类别ID数组或PNG。皮肤标签0/1，CAMUS标签0/1/2/3；可提供bool NPY有效区域。0/255的显示掩膜、0/85/170/255的CAMUS显示图、RGB叠加图不可直接当类别ID。不自动缩放或猜测编码。

```powershell
python -m perf_eval score work/fp32_pairs.json --output work/fp32_scores.json
python -m perf_eval score work/int8_pairs.json --output work/int8_scores.json
python -m perf_eval precision-change work/fp32_scores.json work/int8_scores.json --output work/quantization_change.json
```

两者皆空的类别返回null并从宏平均排除，不凭空记成1分。报告同时保留各类计数；不同空类策略不可直接比较。`precision-change`要求完全相同的GT集合、类别数、有效区域、分组和标签坐标范围；变化单位为百分点，候选减基线。

## 旧记录、资源、DMA与功耗

```powershell
python -m perf_eval timing PLD_timing_xxx.json --output work/web_timing.json
python -m perf_eval utilization utilization.rpt --output work/resources.json
python -m perf_eval dma-report CAMUS_report.json --output work/dma.json
python -m perf_eval power-report power.rpt --output work/fpga_power_estimate.json
python -m perf_eval power-samples work/power_samples.json --output work/board_energy.json
```

网页总时间和子阶段重叠，工具不会累加。DMA导入要求原始`command_events`及`timings_seconds.network`；按加速器命令记录统计收发字节，再除以网络墙钟。它只覆盖DMA包，不包含全部PS内存拷贝、缓存回写或HDMI，因此是流量/带宽估算，不能称DDR控制器实测或峰值带宽。

Vivado资源/功耗输出默认`bitstream_association_verified=false`；必须由对应build manifest确认与当前位流匹配。功耗工具估计和功率计测得的整板输入功耗分别标注。功率采样积分必须涵盖所填写帧数的真实完成区间；模板没有伪造样本，需填真实仪器读数。现在没有功耗数据，报告明确待补测。

## 生成综合报告

```powershell
python -m perf_eval report --run work/skin_cpu/benchmark.json --run work/camus_cpu/benchmark.json --evidence work/resources.json --evidence work/web_timing.json --output work/final_report
```

输出 `report.html`、`report.md`、`report.json`、`comparison.csv`。可把`head_comparison.json`、`quantization_change.json`、功耗和DMA结果继续作为`--evidence`加入。每个output文件/目录必须是新的。

## 板子回来后

1. 在Jupyter新建独立评测目录，上传本工具与支持源码；保留网站、模型、位流与安装状态。不要覆盖现有服务工作目录。
2. 提前下载需要保留的结果、结束当前任务。按原正常释放流程执行 `sudo systemctl stop pld-hub.service`，确认hub/skin/CAMUS三服务均inactive、MainPID=0。为了公平计时，CPU基线也建议在同样服务状态下运行。
3. 使用同一准备好的dataset，在同板先CPU、后PL分别运行。PL入口会检查三服务状态并获取统一FPGA锁，发现占用即拒绝；不会自动停止、强杀或复位服务。
4. 皮肤镜PL额外提供当前配套`--bit 路径.bit`，支持`--native-copy`、`--native-control`、`--fuse`；只有匹配ABI的位流才能启用融合。CAMUS额外提供当前`--runtime`目录。`--model-root`需明确指定对应冻结模型目录。不要拿示例报告或旧位流代替当前版本实测。
5. 保存报告后正常退出，执行 `sudo systemctl start pld-hub.service` 恢复网站。若PL释放报错，报告标为不完整，先核查日志与硬件状态再恢复。

PL命令示例（路径需以板上实际安装记录为准）：

```bash
sudo /usr/local/share/pynq-venv/bin/python3 -m perf_eval bench --source . --model skin --model-root /实际运行包/role_a_final/int8 --dataset /实际评测目录/skin_inputs/dataset.json --backend pl --bit /实际运行包/overlay/dma_top.bit --fuse --repeat 5 --output work/skin_pl
```

本轮没有板子，PL适配、FP32前向、真实数据精度、功耗/DDR实测仍待验证。已有四宫格、网页和模型的历史验收记录不能替代本工具的新板端验收。

仅用于工程教学与算法演示，不作为临床诊断依据。
