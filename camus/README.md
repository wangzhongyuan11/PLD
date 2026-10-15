# CAMUS：独立 INT8 四结构分割

CAMUS 从单帧灰度输入经过精确预处理、15 层整数网络、原生尺寸恢复和固定后处理，输出背景、左心室腔、心肌、左心房四类。CLI、独立网页和性能测试使用同一个 `CamusPipeline`。现有 ISIC 网页、RTL、构建脚本和冻结交付未替换。

本版本已完成真实 PYNQ 数值板测：CAM2 ID/ABI、100 MHz 时钟、首帧 15 层输入/输出、三类故障后的整帧恢复，以及 50 人/200 张开发集原图全部通过。按用户要求在 218 帧后停止长测，218 帧输入/head/原生标签/后处理均零差异；没有完成或宣称 1000 帧验收。CPU reference / emulator 的耗时不能当作板卡性能。

## 使用

本机依赖 NumPy、SciPy、Pillow；NIfTI 输入另外需要 nibabel。支持 PYNQ 的 Python 3.10 / NumPy 1.21 API，不依赖 Torch、OpenCV。当前主机实测为 Python 3.13.9、NumPy 2.3.5。

默认模型：仓库外部交付目录 `camus_candidate_hidden_histogram_int8_20261015/int8_export`。解压运行包后，默认使用包内 `model/int8_export`。模型数据不随源码提交，须保留原交付或另行指定 `--model`。

```powershell
Set-Location D:\PLD
& D:\conda\python.exe -m camus --plan
& D:\conda\python.exe -m camus --backend emulator frame.mhd out/camus/my_frame
& D:\conda\python.exe -m camus --backend reference gray.png out/camus/reference_frame --spacing 0.2 0.3
& D:\conda\python.exe -m camus.web --backend emulator --port 8090
```

网页为 `http://127.0.0.1:8090/`，只绑定本机回环地址。MHD 同时上传对应 RAW；一次执行一个任务，忙时明确拒绝新任务。远程查看使用已有 SSH 隧道，不自动开放网络监听。该网页与原 ISIC 服务独立。

输出包括 `head.bin/head.json`、保留真实 0..3 类别的 `labels_native.npz`、原始/处理后标签 PNG、彩色分割、叠加、报告、静态预览及 `result.zip`。标签 PNG 的 0/85/170/255 是显示编码。报告记录每类像素；有合法 Y/X spacing 时增加 mm²。单帧不推算 EF、EDV、ESV。

输出目录必须新建且位于模型交付目录之外。默认输入上限 800 万原生像素；网页请求上限 32 MiB，NPZ 解压后上限 64 MiB。

## 模型与数值

spec SHA256：`68b4f425cc3612ad72d5e83955c2018981edee912fbc0937e9ac8c1718e7b5ae`。`model_manifest.json` 锚定 173 个导出文件，加载时重新校验，不使用跨调用的隐式验证缓存。完整交付 852 文件的核验在原包 `verify_package.py`。

- 输入 `1×1×256×256`；强度 `clip(0,255)/255`，letterbox 后按 `1/127` ties-even 量化。不做直方图增强、CLAHE 或 uint16 自适应拉伸。`hidden_histogram` 是隐藏层校准方法。
- 15 个 Conv，14 个 hidden Conv 已融合 ReLU；head 不做 ReLU。NCHW/OIHW，INT8 `[-127,127]`，INT32 累加，INT64 Q24 乘积和最近偶数舍入。
- 跳连 skip 在前，两支按导出 Q24 分别转换到共同尺度。整数 LUT 不采用旧 ISIC 的 float32 往返映射。
- head `1×4×256×256`，262144 字节，共享 scale `0.19834566491795338`。裁补边、逐通道双线性恢复原尺寸后 argmax，平局保留较小类别。
- 输入 resize 保持 A 的 double 坐标、float32 权重、纵向后横向；输出保持 float32 反量化/坐标、横向后纵向。两条路径不能合并成任意通用 resize。
- 后处理移植自交付 `sources/software/role_a_camus_opt/postprocess.py`：LV-only island5_hole1，不覆盖 MYO/LA。原始标签同时保留。
- MHD 是 X 最快字节顺序，读取为 H/W；支持 2D 或 Z=1，明确字节序。NIfTI XY→HW 转置、spacing 同步交换，不自动重排方向。真正多帧需要外部选定端点。

A 的 50 人/200 帧 Dice 0.8805233756 是开发证据，不能称独立测试集结果，也未证明该候选综合优于原 CAMUS 交付。当前包的 200 份病例是导出 logits/标签/GT，不含对应原始 MHD/RAW/NIfTI。回放不是 200 次新原图推理。

## CAM2 硬件合同

新 VLNV `user:pld:axis_tensor_accel_camus:1.0`；ID `0x43414d32`，ABI `0x00020000`；AXI-Lite `0x43c00000`，DMA `0x40400000`，64-bit AXIS。命令 1 参数、2 输入、3 LUT、4 RUN；沿用有界标签、字节计数及 TKEEP/TLAST 语义。

模式 0 卷积，1 LUT，2 Pool+LUT，3 Upsample+LUT，4 skip-first Concat+Upsample+双 LUT。拒绝旧模式 5/6。新增 `0x05c[0]` 为 Conv ReLU，仅 idle 可写，支持 byte strobe，Conv 启动时锁存，head 写 0。非 Conv 带 ReLU 标志会拒绝。

Ci≤192、Co≤64/命令。RAM 仍为输入 4×32768、权重 32×2048、输出 8×8192 字节。权重 bank 检查：`ceil(Co/8)*ceil(Ci/4)*K²≤2048`，在装载和运行前均检查。

完整 Ci 累加后一次 Q24；按 Co 分块：64→128 为 64+64；128→128 为 56+56+16；192→64 为 32+32。禁止对 Ci 分块后分别量化再相加。卷积源自冻结 `7d99b861…f25943d`，未使用本地 `3de993e` 重叠实验。共享 MAC/Q24、AXIS、AXI-Lite 文件只读复用。

`CamusTileAccelerator` 复用有界 DMA、缓存及恢复方法，但独立构造并检查新 HWH/ID/ABI，不经过旧 ABI 构造器。首版使用 Python MMIO/NumPy，不使用含旧通道限制的 native control。输出是独立数组；硬件错误不会切换 CPU 后端。

## 验证、构建与打包

```powershell
& D:\conda\python.exe -m unittest discover -s camus/tests -p 'test_*.py' -v
& D:\conda\python.exe -m camus.verify --output out/camus/verify_new.json
if (-not $env:PROCESSOR_ARCHITECTURE) { $env:PROCESSOR_ARCHITECTURE = 'AMD64' } # 本机已确认 X64
& D:\Git\bin\bash.exe camus/run_sim.sh
& D:\Git\bin\bash.exe camus/build.sh
& D:\conda\python.exe -m camus.package
```

所有生成物仅进入 `out/camus/`、`ip_repo/camus/`。源码、打包 RTL、报告、bit/hwh/xsa 由 SHA256 绑定；运行包校验实际执行的 Python 源码与包内冻结版本一致。构建中不得修改编译源/Tcl。资源门为 90 BRAM36 等效、96 DSP、32000 LUT，setup/hold/pulse 均须通过。

2026-10-15 本地验证：完整参考及分块网络从同一首层输入运行，15 层输入/输出零差异；200 原生恢复及后处理回放零差异。集成 xsim 41 个正确输出、7 个预期错误，含真实 15 层小块、128/192 通道、Co56、非整倍通道、舍入/饱和/ReLU、非卷积、完整 bank 容量边界、256 宽实际 tile、底/右 halo、包错误及计算中复位。完整单测及 HTTP/浏览器上传→显示→ZIP 验证见 `out/camus/`。

100 MHz 路由：WNS +0.344 ns、WHS +0.028 ns、WPWS +3.750 ns；6843 Slice LUT、7718 FF、67 BRAM36 等效、52 DSP。DRC 仍列出 DSP 流水、RAM 异步控制等告警及报告数量上限提示，均保留在 `drc.rpt`；时序通过不等于板级验收。

在已安排独占使用的 PYNQ 上，从解压包根运行：

```bash
sudo -E /usr/local/share/pynq-venv/bin/python3 -m camus --backend pl --runtime . frame.mhd results/frame001
```

PL 入口先确认旧 `pld-web.service` inactive，并获取 CAMUS 进程锁；不会自动停止服务。该锁不能阻止不参与锁的手动旧程序。加载新位流会替换当前 PL，不能与 ISIC 同时推理。运行包默认不携带 200 例 dev 回放文件，包内软件验证可用 `camus.verify --skip-native-replay`。

### 板端服务与切换

板端依赖实测为 Python 3.10.4、NumPy 1.21.5、SciPy 1.8.0、Pillow 9.0.1、PYNQ 3.1.1。原始 NIfTI 使用隔离目录中的 nibabel 5.2.1，不替换系统环境。板上时钟不准，测量使用单调时钟，运行日期以主机记录为准。

独立 `pld-camus.service` 绑定板端 `127.0.0.1:8090`。unit 通过 `Conflicts` 和启动顺序互斥原 `pld-web.service`，不改原 unit、drop-in、运行包、访问码或历史数据，也不改变原开机启动设置。从 CAMUS 运行包根执行：

```bash
sudo -E /usr/local/share/pynq-venv/bin/python3 -m camus.board_service status
sudo -E /usr/local/share/pynq-venv/bin/python3 -m camus.board_service camus
sudo -E /usr/local/share/pynq-venv/bin/python3 -m camus.board_service isic
```

首次安装使用 `board_service install --runtime ... --dependencies ... --data ...`；安装前校验冻结运行包，并用 `systemd-analyze verify` 检查新 unit。已有 unit 会拒绝覆盖，安装不会自动启用开机启动。验证进程还在运行时，切换命令拒绝抢占 PL。

主机用 SSH 转发访问，认证信息来自交互或 `PLD_BOARD_PASS`，不得写入源码：

```powershell
& D:\conda\python.exe -m camus.tunnel --local-port 8091
```

打开 `http://127.0.0.1:8091/`，后端必须标为“真实 PL”。主机的隧道模块需要 paramiko，板端推理不依赖它。

### 原图、长稳及网页验收

`prepare_validation` 只按官方 validation 名单选择冻结导出对应的 50 名患者/200 张 ED/ES 端点，先核对原始 GT 与导出 GT，再冻结原 NIfTI、首层量化输入、head、原生及处理后标签的哈希。不会打开封存测试患者影像。

```powershell
& D:\conda\python.exe -m camus.prepare_validation --dataset 数据/CAMUS --candidate camus_candidate_hidden_histogram_int8_20261015 --output out/camus/new_dev200
```

传输并校验数据包后，板端通过登录 shell 初始化 XRT，在独占 PL 时运行：

```bash
sudo -E /usr/local/share/pynq-venv/bin/python3 -m camus.board_verify --runtime . --dataset /path/to/dev200 --output new_board_validation --frames 1000
```

该程序从第一层输入验证全部 30 个黄金张量，注入短包、缺包超时和计算中复位，每次恢复后重跑整帧。随后从原始 NIfTI 执行 200 张开发图循环，逐帧比较输入/head/原生标签/固定后处理；记录命令、字节、engine/core cycles、网络和在线处理耗时、RSS。`frames.jsonl` 持续落盘；`summary.json.complete=false` 不能当作长稳通过。在线处理统计不含 PNG/ZIP，完整结果生成另列。

网页启动后，主机可运行 `camus.board_web_verify --url http://127.0.0.1:8091 --dataset ... --output ... --frames 20 --hdmi`：验证上传→真实 PL→四张全分辨率 PNG→ZIP、忙时 409、head/标签精确比较、ZIP CRC/报告一致性，并区分常驻请求与 HDMI 后的冷初始化。

### HDMI

CAMUS 网页支持叠加、四分类和标签视图。开始 HDMI 前关闭推理 DMA、释放 CAMUS 锁，再由独立子进程加载 PYNQ base overlay；下一次推理先等待 HDMI 关闭，再重建 CAM2 会话。C 的 `pld_hdmi.py` 只读复用，没有修改其二值掩膜或网页接口。超时或未确认释放时停止接受新硬件操作。

独立 CLI：`python -m camus.hdmi --result /path/to/completed/result --view segmentation --duration 10`。HDMI 初始化及帧提交通过不代表真实屏幕目视通过；用户确认本轮未连接显示器。

板测发现并修复两个仅靠主机仿真未暴露的问题：PYNQ 3.1.1 的同名 `.xsa` 旁路判定（XSA 改存 `overlay/evidence/`）；ARM 32 位 NumPy 的 `take` 索引必须为 `intp`（仅索引转换，不改坐标和权重算术）。

## 性能分析

```powershell
& D:\conda\python.exe -m camus.performance --input gray.png --backend emulator --repeat 3 --output out/camus/new_benchmark.json
```

工具将解析 xsim 的 41 组 engine cycles 与状态机公式逐项核对，再投影整帧。当前调度 164 个卷积 tile、19 组参数、605 条命令、33273648 字节/帧；卷积投影 103056016 周期，含 tensor 的 engine 总量 110397056 周期。

100 MHz 下 engine 投影约 1.104 秒，加上不可重叠的单字节装载/读回最低约 0.333 秒，得到约 1.437 秒/帧的流水下限（吞吐上限约 0.696 FPS）。这不是板测，也未包含所有 AXI 控制、PS、缓存、图像处理开销。125 MHz 仅作数学比较，当前通过构建的 CAMUS 位流是 100 MHz。

真实 218 帧板测：网络中位 3.434083 秒、P95 3.611954 秒（中位约 0.291 FPS）；原图读取/预处理、PL 网络和原生恢复/后处理合计中位 4.271206 秒、P95 4.854623 秒，此口径不含 PNG/ZIP。首张原始 NIfTI 全流程生成四张 PNG 与 ZIP 为 6.078725 秒，不构成网页 P95≤5秒验收。每帧实测 engine 都为 110397056 周期，与 RTL 投影完全一致；PS/控制/DMA 等额外时间必须计入系统性能。

性能报告同时保留本机 reference/emulator 的实际端到端生成耗时和样本数。验证演示图由 A 首层量化输入反构造，能逐字节复现输入/head，但不称原始临床图。3 帧短测不继承 ISIC 的 dev638/1000 帧证据。

若要求 CAMUS 4 FPS，必须改变计算并行度和数据流；当前 32 路 MAC 在 125 MHz 下即便满载也至少需 0.664 秒。三个 decoder 入口占 MAC 的 51.16%，应优先剖析；再减少中间特征 DDR 往返、单字节瓶颈和命令依赖。PS 调用优化无法单独达到 4 FPS。
