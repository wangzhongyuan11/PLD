> 团队当前版本从[冻结交付包](../delivery/c_v3/README.md)和[C交接说明](../docs/C_HANDOFF.md)开始。本文保留历史板卡操作命令，各命令仅对应其标明的版本。

# PYNQ-Z2 阶段 0/1/2：构建、部署、验证

阶段0–4已验收，当前为完整PL TinyUNet计算链路。阶段0/1/2的FIFO与逐字节+1通路保留，证据见[STAGE012_VALIDATION.md](../docs/history/STAGE012_VALIDATION.md)。CNN复现见下文；阶段5的优化与评估见[STAGE5_STATUS.md](../docs/history/STAGE5_STATUS.md)。C后处理与HDMI暂未接入。

**模型版本（2026-09-27）**：当前CNN入口默认 `role_a_final/int8`，独立manifest校验，p=0.45（q≥−1）。旧模型需显式 `--model role_a_handoff/role_a/int8`，旧证据仍保留原版本范围。当前ABI 1.2 v3已完成dev638全部638图、1000帧及故障恢复，完整验收状态见[最终模型进度](../docs/FINAL_MODEL_STATUS.md)。后文旧部署命令可用于历史复现；新产物必须另建部署目录，不能覆盖旧部署后沿用其报告。

## 最终模型使用入口

`verify_cnn.py` 做冻结样例逐层检查，`evaluate_dataset.py` 对冻结集合做原图/原始head/指标验证；普通推理使用 `infer_images.py`，不需要GT或中间golden。ABI 1.1通过 `--fuse-maps` 启用PL输入装载查表；ABI 1.2再通过 `--fuse-decoder` 启用双分支融合卷积，两个选项可组合，旧bit会拒绝不支持的选项。新命令与输入包格式见[寄存器合同](../docs/design/ACCELERATOR_REGISTERS.md)。原生库先在同一部署中通过 `scripts/build_native_copy.py` 和 `scripts/build_native_control.py` 构建，加载时核对源码/库hash。

```bash
# 在配套CNN部署目录运行；output必须为新的目录。
/usr/local/share/pynq-venv/bin/python3 board/infer_images.py image1.jpg image2.jpg --bit overlay/dma_top.bit --output results/run01 --native-copy out/native/tensor_copy.so --native-control out/native/tensor_control.so --tx-range-sync --native-resize out/native/rgb_resize.so --fuse-maps --fuse-decoder
```

每张图得到 `frame_000001.bin` 等65536字节NCHW有符号INT8 head及同名JSON。JSON包含帧号、模型/输入/输出hash、阈值、尺度、有效区域、耗时和状态；C按有效区域排除补边，使用q≥−1做默认分割。head已复制到独立数组，内存接口可直接调用 `PLGraph.run`，不能保留底层staging视图。失败非零退出，核心/DMA按既有有界恢复和清理合同处理；已完成的输出可保留，但不能将失败帧标为成功。

## 当前环境（2026-09-26 实测）

- 主机：Windows/PowerShell，`D:\PLD`；Vivado 2026.1，Git Bash `D:\Git\bin\bash.exe`，NumPy 参考使用 `D:\conda\python.exe`。
- 板卡：`xilinx@192.168.2.99`，PYNQ 3.1.1 / NumPy 1.21.5；内核 6.6.10，设备 `Pynq-Z2`。PYNQ Python 为 `/usr/local/share/pynq-venv/bin/python3`。
- 当前普通用户组为 xilinx/adm/sudo，不能依赖历史记录中的 render 组。自动化使用 `sudo bash -lc` 初始化 XRT/PYNQ 环境；密码经 stdin 传入。未改网络、用户组、系统时间或 SD 卡。
- SSH 脚本从 `PLD_BOARD_PASS` 或交互输入取得密码，不保存默认密码。主机默认 Python 缺少 paramiko，使用 `uv run --with paramiko` 的隔离依赖环境。
- 阶段0/1/2保留部署：`/home/xilinx/jupyter_notebooks/pld/stage012_20260926/`。后续CNN部署见下文；不要从此历史目录推断当前PL配置。板上时间不准，以主机日期标记证据。

## 本地检查与构建

```powershell
Set-Location D:\PLD
if (-not $env:PROCESSOR_ARCHITECTURE) { $env:PROCESSOR_ARCHITECTURE = 'AMD64' }
& D:\conda\python.exe -m unittest discover -s sim -p 'test_*.py' -v
& D:\Git\bin\bash.exe scripts/run_sim.sh
& D:\Git\bin\bash.exe scripts/build.sh fifo
& D:\Git\bin\bash.exe scripts/build.sh incr
```

每条命令都必须退出 0。已有对应源码的产物可直接复核，无需重复构建：

```powershell
& D:\conda\python.exe scripts/check_build.py out/vivado_fifo
& D:\conda\python.exe scripts/check_build.py out/vivado_incr
```

`check_build.py` 检查布线后 setup/hold/脉宽和 XSA 中 bit/hwh 与解包文件逐字节一致；`build.sh` 在此门禁通过后才打印 `BUILD_DONE`。DRC 和告警仍需查看实现日志，不能仅凭成功标志忽略告警。资源/时序见 `out/vivado_<mode>/`，仿真日志见 `out/sim_logs/`。

## 部署与复现

一次只允许一个进程拥有 PL/DMA。加载 overlay 会替换当前 PL 配置；部署前检查无其他 DMA/Notebook 内核任务。不同版本用独立目录，同名 bit/hwh 必须成对；保留上一版目录便于恢复。

PowerShell 连接示例（缺少凭据时交互输入）：

```powershell
uv run --with paramiko python scripts/board_ssh.py run 'id'
uv run --with paramiko python scripts/board_ssh.py run 'mkdir -p /home/xilinx/jupyter_notebooks/pld/my_build/fifo /home/xilinx/jupyter_notebooks/pld/my_build/incr'
uv run --with paramiko python scripts/board_ssh.py put board/simple_dma.py /home/xilinx/jupyter_notebooks/pld/my_build/simple_dma.py
uv run --with paramiko python scripts/board_ssh.py put board/verify_overlay.py /home/xilinx/jupyter_notebooks/pld/my_build/verify_overlay.py
uv run --with paramiko python scripts/board_ssh.py put out/vivado_incr/dma_top.bit /home/xilinx/jupyter_notebooks/pld/my_build/incr/dma_top.bit
uv run --with paramiko python scripts/board_ssh.py put out/vivado_incr/dma_top.hwh /home/xilinx/jupyter_notebooks/pld/my_build/incr/dma_top.hwh
uv run --with paramiko python scripts/board_ssh.py run --sudo --timeout 180 'cd /home/xilinx/jupyter_notebooks/pld/my_build && /usr/local/share/pynq-venv/bin/python3 -u verify_overlay.py incr --bit incr/dma_top.bit --repeat 4 --random-packets 1000 --check-recovery --report incr.json'
```

FIFO 使用 `out/vivado_fifo/` 的配对文件，模式改为 `fifo`；tutorial 使用 `PYNQ_tutorials/dma/dma_tutorial.bit/.hwh`，模式同为 `fifo`。脚本打印 bit/hwh/两份 Python 的 SHA256；必须与本地主机 `Get-FileHash` 一致。原始 JSON/日志保留在版本目录，本机下载到 `out/stage012/`，不提交生成物。

成功条件：退出 0、`BOARD_PASS`、JSON `passed=true`；开启恢复测试时还须 `RECOVERY_PASS`。默认 7 种长度为 1/7/8/64/1000/65536/1048576 字节。上述完整命令覆盖四轮随机/全零/全255/递增字节、1000 个固定种子随机长度包以及恢复后 7 包，共 1035 个正确包。每包检查全部有效字节、接收长度和末尾 64 字节保护区。

以下负例必须退出 1，不能当作正常成功：

```bash
# 位于版本目录，以有权限的 PYNQ Python 运行：
python3 verify_overlay.py incr --bit incr/dma_top.bit --inject-timeout --report negative_timeout.json
# 故意将 +1 输出与 FIFO 期望比较，验证首差异与失败返回：
python3 verify_overlay.py fifo --bit incr/dma_top.bit --report negative_mismatch.json
# 负例后正常包必须仍正确：
python3 verify_overlay.py incr --bit incr/dma_top.bit --report final_smoke.json
```

## DMA 控制与恢复

`simple_dma.py` 直接使用 PYNQ MMIO，避开 PYNQ 3.1.1 DMA 驱动构造、start/stop/wait 内的无界轮询。只支持当前 simple/non-micro DMA，单线程单包拥有硬件；不能与 `ol.axi_dma` 驱动同时使用。长度与地址范围按 HWH 校验，缓冲区须按总线宽度对齐。

1. 复位 DMA，等待 reset 清除且两通道 halted；启动并有界等待 running。
2. 分配独立源/目的缓冲区，flush，先启动接收再发送。
3. 同时检查两通道错误位，要求本次 IOC 与 idle 均到达；检查实际接收长度，再 invalidate。
4. 默认每个轮询上限 2 秒。异常保留收发 CR/SR、地址、长度和首个数据差异；先复位两通道再释放活动缓冲区，随后重载同版 overlay 清空 PL FIFO/寄存切片，再启动新包。
5. 复位失败时尝试重配置 PL；若连重配置也失败则报 `RECOVERY_FAIL`/退出 2，硬件状态未确认，需恢复设备后再运行。SSH 外层超时不证明远端进程或 DMA 已停止，不能据此直接释放/复用活动内存。

受控超时只启动接收、不启动发送；预期捕获 `transfer timeout`，完成硬件恢复后再算对 7 种长度。不会通过杀进程伪造恢复通过。DMA reset 只复位 DMA，外接 FIFO 的清理由重载 overlay 完成，依据 [AMD PG021 Reset 说明](https://docs.amd.com/r/en-US/pg021_axi_dma/MM2S_DMACR-MM2S-DMA-Control-Register-Offset-00h)。

吞吐为有效负载字节数/传输时间，包含缓存同步、寄存器调用和完成轮询，排除分配、数据生成、比较与 overlay 加载；不是 CNN FPS，也不把双向读写各算一遍。

### 阶段3/4的单向与非等长API

`SimpleDMA`保留原来的等长`transfer(source,destination,nbytes)`，新增：

```python
dma.send(parameters, parameter_bytes)  # 只启动/等待MM2S，PL必须消费输入
dma.receive(result, result_bytes)     # 只启动/等待S2MM，PL必须产生输出
dma.transfer(source, destination, send_nbytes=input_bytes, receive_nbytes=output_bytes)
dma.submit(source, destination, send_nbytes=input_bytes, receive_nbytes=output_bytes)
dma.wait()                            # 对同一笔submit有界等待
```

`submit`先启动接收、再启动发送；仅等待本次参与的通道，未使用通道的halted/idle不作为完成条件。驱动保留活动buffer引用，成功完成或确认DMA复位后才清除；调用者仍不得在等待期间手动`freebuffer()`。接收长度严格比较，缓存同步及错误后禁止直接复用的规则不变。

旧参数`receive_only=True`仍专门表示“省略发送但等待两通道”的超时注入；它不是新`receive()`的别名。

2026-09-26在独立目录`/home/xilinx/jupyter_notebooks/pld/stage34_dma_20260926/`复测当前驱动：`verify_dma_modes.py`通过55包（48包单独send后receive，7包submit/wait）；保留的`verify_overlay.py`通过1035包及超时恢复。前者单向测试只发送1/7/8字节，利用incr的一拍输出寄存器暂存；大于一拍的包不能照此TX先发完、再启动RX。后续CNN板测已覆盖独立长度的参数/输入加载与输出接收，见下文。原阶段0/1/2目录及配套产物保留。

```bash
python3 verify_dma_modes.py --bit ../stage012_20260926/incr/dma_top.bit --report dma_modes.json
```

本轮JSON下载至`out/stage34/`，已核对板上脚本哈希与主机一致。incr的1 MiB吞吐中位数约423 MB/s；只是驱动回归数据，不是神经网络性能。

## 给 C 的占位 IP 接口

| 项目 | 当前合同 |
|---|---|
| 打包/标识 | `vivado/package_axis_incr.tcl` → `ip_repo/axis_incr/`，VLNV `user:pld:axis_incr:1.0`；`build.sh incr` 自动打包 |
| 时钟/复位 | `aclk` 100 MHz，`aresetn` 低有效异步复位；复位丢弃在途 beat，PS 必须同步中止该包 |
| AXIS | `s_axis` 输入、`m_axis` 输出；TDATA 64、TKEEP 8、TVALID/TREADY/TLAST 1；字节 i 在 `[8*i +: 8]` |
| 计算 | 每个有效字节 `(x+1) mod 256`；TKEEP/TLAST 与数据同步透传，无效 lane 不参与结果比较 |
| 时序 | 空载一拍延迟；无背压时每拍一 beat；阻塞时保持输出 payload，源端只在 valid&&ready 时提交 |
| PS 控制 | IP 无 AXI-Lite/start/done；PS 通过 DMA 地址/字节长度启动、完成状态收尾。正式计算核寄存器合同留在阶段 3 冻结 |
| 现有 BD | PS7 + AXI DMA + FIFO/incr；AXI-Lite `0x40400000`；64 bit、100 MHz、simple、26 bit 长度、DRE 关闭；MM2S→HP2，S2MM→HP0 |

复用已有 `axis_reg_slice`、TKEEP/TLAST TB、PS7 参数、BD 和 PYNQ 分配/恢复框架。C 将 IP repo 加入搜索路径后例化上述 VLNV；本轮已在自建 BD 验证，尚未在 C 的显示工程中实测集成。

## 正式 CNN：阶段3/4复现

结果与限制见[验证记录](../docs/history/STAGE34_STATUS.md)，接口见[ABI 1](../docs/design/ACCELERATOR_REGISTERS.md)。2026-09-26部署在`/home/xilinx/jupyter_notebooks/pld/stage34_cnn_20260926_v1/`：从PS原始JPEG得到输入，三帧逐层一致，短包/缺包/计算中复位恢复通过；另一次纯推理计时2.874秒（0.348 FPS）。这是B计算子系统，不含HDMI/后处理。

本机重建只在源码或工具链变化时执行；使用Git Bash与正确Python PATH：

```powershell
Set-Location D:\PLD
$env:PROCESSOR_ARCHITECTURE='AMD64'
$env:PATH='D:\conda;'+$env:PATH
& D:\conda\python.exe -m unittest discover -s sim -p 'test_*.py' -v
& D:\Git\bin\bash.exe scripts/run_sim.sh
& D:\Git\bin\bash.exe scripts/run_sim.sh tile-full
& D:\Git\bin\bash.exe scripts/build.sh cnn
& D:\conda\python.exe scripts/package_cnn_board.py
```

包为`out/stage34/cnn_bundle.tar.gz`，含同版bit/hwh、冻结模型、驱动、预处理入口与逐文件SHA256部署清单。打包前会检查布线时序、XSA配对与源码/产物hash，不能用早期失败构建凑齐文件。原始JPEG不入包，单独上传用户要处理的图。`board_ssh.py`从交互或环境变量取得凭据，以下不存密码：

```powershell
uv run --with paramiko python scripts/board_ssh.py put out/stage34/cnn_bundle.tar.gz /home/xilinx/jupyter_notebooks/pld/stage34_cnn_20260926_v1.tar.gz
# 新版本目录首次部署时执行；复测已有目录时不要覆盖：
uv run --with paramiko python scripts/board_ssh.py run 'mkdir -p /home/xilinx/jupyter_notebooks/pld/stage34_cnn_20260926_v1 && tar --warning=no-timestamp -xzf /home/xilinx/jupyter_notebooks/pld/stage34_cnn_20260926_v1.tar.gz -C /home/xilinx/jupyter_notebooks/pld/stage34_cnn_20260926_v1'
uv run --with paramiko python scripts/board_ssh.py put ISBI2016_ISIC_Part1_Training_Data/ISBI2016_ISIC_Part1_Training_Data/ISIC_0000000.jpg /home/xilinx/jupyter_notebooks/pld/stage34_cnn_20260926_v1/input.jpg
```

新的部署先建立独立版本目录并解包，再上传JPEG；避免覆盖已验证版本。当前目录已经部署，可直接复测。使用`board_ssh.py run --sudo --timeout 300 '<下列远端命令>'`；登录shell初始化XRT，不直接使用系统Python：

```bash
cd /home/xilinx/jupyter_notebooks/pld/stage34_cnn_20260926_v1
/usr/local/share/pynq-venv/bin/python3 -c "import json,hashlib,pathlib; d=json.load(open('deployment_manifest.json')); assert all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h for p,h in d['files'].items()); print('DEPLOYMENT_HASH_PASS',len(d['files']))"
/usr/local/share/pynq-venv/bin/python3 scripts/prepare_cnn_input.py input.jpg --compare-golden
/usr/local/share/pynq-venv/bin/python3 board/verify_cnn.py --bit overlay/dma_top.bit --repeat 2 --check-recovery --benchmark-frames 1 --input out/stage34/input_int8.bin --input-metadata out/stage34/input_int8.json --report cnn.json --output head_int8.bin
```

`--compare-golden`和`verify_cnn.py`专用于有A完整golden的ISIC_0000000样例，其他图不能强行与它比较。一般输入的调用入口是`PLGraph.run(image, input_metadata=metadata)`，不设置`verify=True`；模型仍须通过manifest验证。

成功必须看到每帧15个`LAYER_PASS`、三个`FRAME_PASS`、`FAULT_RECOVERY_PASS`、`BENCHMARK_PASS`及`CNN_BOARD_PASS`，且SSH退出0。JSON的`passed`为true、没有`close_error`；最后打印成功但清理失败仍退出2，不可忽略。下载`cnn.json`、`head_int8.bin`和`head_int8.json`，后者携带frame_id/模型hash/有效区域/输出hash；head不是PNG，类型为signed INT8。

默认每条DMA/核心等待上限2秒。原始JPEG与A输入完全一致的板上运行库为Pillow9.0.1/JPEG8；不要替换成未经核对的解码器。整网参数每层加载一次，在同层tile间复用；每帧389命令，PS持有三处跳连直到对应decoder消费。

异常按DMA reset确认停止→核心RESET→重新加载数据恢复；计算中复位测试的RX已启动，但不会在确认DMA停止前释放DDR。`close()`正常结束同样复位后释放buffer，overlay留在PL但不持续推理。回退到阶段2时，待当前进程结束并清理硬件后，按本文件原命令加载保留的incr bit/hwh；不要让两套驱动同时拥有DMA。

下载结果后，本机`D:\conda\python.exe scripts/render_cnn_result.py`生成`out/stage34/cnn_result.png`：输入、真实PL logit和`q>=0`叠加图。脚本校验输入/输出/模型hash并强制补边背景，只是离线证据图，不代表C的PL后处理或HDMI已实现。

## 阶段5：NEON优化版复现

已验证目录为`/home/xilinx/jupyter_notebooks/pld/stage5_ps_neon/`，使用优化后的100MHz CNN bit。默认仍支持NumPy拷贝；显式参数启用NEON，所有网络算术仍在PL。新部署包也包含C源码及板上构建入口。进入该独立部署目录后，使用上述sudo登录shell运行：

```bash
/usr/local/share/pynq-venv/bin/python3 scripts/build_native_copy.py
/usr/local/share/pynq-venv/bin/python3 board/verify_cnn.py --bit overlay/dma_top.bit --native-copy out/native/tensor_copy.so --repeat 2 --check-recovery --benchmark-frames 0 --report neon_verification.json
/usr/local/share/pynq-venv/bin/python3 board/profile_cnn.py --bit overlay/dma_top.bit --native-copy out/native/tensor_copy.so --repeat 5 --profile --report neon_native.json
```

编译后须有`NATIVE_BUILD_PASS`、同名`.so`及`.json`；驱动在加载overlay前核对库/C源码/Python包装哈希。数值与恢复报告`passed=true`，三帧各15层一致；五帧中位1.437212秒、P95 1.456613秒，仅为量化输入到head的短基准。379图评估入口`evaluate_dataset.py`同样接受`--native-copy`；最终优化版全量评估与1000帧稳定性尚未完成，详见[评估说明](../docs/validation/EVALUATION.md)。
