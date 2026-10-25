> 当前唯一正式版本为 pld_dual_20261017，部署/回退和验证入口见仓库根 README 及 docs/DEPLOYMENT.md。本页独立应用/补丁步骤保留作模块说明，旧目录名和单模型启动方式不作为正式系统入口。

# PLD 网页与显示系统

本项目负责 PYNQ-Z2 皮损分割系统的网页操作、结果读取、轮廓处理、面积统计和 HDMI 显示。模型训练与 FPGA 加速器由同学的 PLD 工程提供，本项目调用其冻结运行包，不包含训练数据、模型权重或硬件比特流。

当前整合了固定网址、浏览器记住访问码以及开机自启。用户已反馈网页分割、HDMI 和重启后直接访问正常；具体测试范围见 [验证记录](docs/VALIDATION.md)。

## 功能

- 上传 JPG，调用板端 FPGA 完成分割。
- 查看原图、切割图、二值掩膜和轮廓描边，调整轮廓线宽；任务先发布原图与叠加图（`preview_*.png` 为已完成全尺寸图片的访问别名），其余视图和结果包在后台继续生成，完成后才开放 ZIP 下载与 HDMI。
- 显示面积占比、边界像素与后处理耗时，导出结果包。
- 浏览器先在内存中准备上传副本：超过 300 万像素时等比例缩小，掩膜同步最近邻缩放并保持二值；推理模式上传 PNG 或文件过大时转换为 JPG 副本。用户本地原文件不会被修改，结果按上传副本尺寸生成。
- 本次开机历史：显示已完成结果的切割图预览，单独下载切割图或结果包；刷新浏览器和同次开机内重启服务后可恢复。
- 控制 HDMI 显示，支持开发板实体按键。
- 固定网址访问；可安装 systemd 开机自启服务。
- 电脑端提供历史真实掩膜回放，以及图片和配套掩膜的后处理测试。

轮廓处理不会自动修正模型分割错误；目前未实现去噪、填孔、阈值交互或训练。网页服务串行使用计算和显示硬件，不能一边推理一边连续 HDMI 输出。

## 电脑端运行

Windows 双击 `start_windows.cmd`。本机优先使用已有 Python 环境，其他电脑可安装依赖后运行：

```bash
python -m pip install -r requirements.txt
python server.py --open
```

电脑端不执行 FPGA 推理或驱动 HDMI。可以点击“加载真实结果示例”，或上传 `samples/synthetic_original.png` 和 `samples/synthetic_mask.png` 验证操作。合成图片不是模型预测。

## 板端使用

需要同一网址切换皮肤镜和 CAMUS 时，使用可选的 [统一入口](unified/README.md)。它保留两套现有操作界面，等待任务结束后停止旧模型，再启动所选模型；安装与回退均通过独立 systemd 配置完成。当前只完成本地自动化验证，双模型实际 FPGA / HDMI 切换仍待板端验收。

目录保持为 `/home/xilinx/jupyter_notebooks/PLD_Web`。部署步骤、开机自启和故障排查见 [部署说明](docs/DEPLOYMENT.md)。

已配置自启的板子直接打开固定网址：`http://192.168.2.99:8765/`，不要同时手动运行另一份网页或 HDMI 程序。首次使用新浏览器仍需终端/服务日志中的完整访问链接。固定网址的前提是板子 IP 保持不变。

## 文件说明

| 文件或目录 | 用途 |
|---|---|
| `server.py` | HTTP 后端、上传检查、任务管理与硬件互斥 |
| `static/` | 网页 HTML、CSS、JavaScript 和图标，无需 npm 构建 |
| `contour.py` | CPU 轮廓提取，可独立使用 |
| `pld_result.py` | B→C 输出合同验证、掩膜回映射及统计 |
| `infer_then_display.py` | 运行包校验与推理命令构造，也保留独立命令入口 |
| `persistent_worker.py` | 可选常驻 B worker 的主机端监督、协议校验和有界超时 |
| `pld_hdmi.py`、`hdmi_buttons.py`、`hdmi_worker.py` | HDMI 输出、按键和网页控制子进程 |
| `show_result.py` | 独立查看已保存结果；也是现有模块的导入依赖 |
| `start_windows.cmd`、`start_board.sh` | 电脑与板端手动启动 |
| `install_autostart.py` | 板端自启安装与检查 |
| `tests/` | 网页后端、轮廓、访问码和上传副本逻辑测试 |
| `samples/` | 小型真实结果与合成测试样例 |
| `docs/` | 部署、模型交接与验证说明 |

## 本次开机历史

历史只显示当前操作系统启动后，由本版本完成的任务；Linux 使用 boot_id，Windows 使用系统启动 GUID。操作系统重启后列表为空，旧版未写入历史索引的结果不会自动导入。历史不包含失败或尚未完成的任务。

现有 data 中的旧文件不会自动删除，因此“本次开机”指列表和访问范围，而不是自动清空磁盘；需要长期连续使用时应另定磁盘清理策略。访问码文件始终保留。此版本仍保留原流程生成的图片和报告，结果包内容不变。

## 测试

在项目根目录运行：

```bash
python -m unittest discover -s tests -v
```

如有 Node.js，可额外运行浏览器访问码逻辑测试（运行网页本身不需要 Node.js）：

```bash
node tests/test_browser_access.cjs
node tests/test_preview.cjs
node tests/test_timing.cjs
```

如需在本机比较输出管线耗时，可运行固定输入基准（不访问 FPGA/HDMI；结果只代表当前电脑）：

```bash
python tests/benchmark_output.py --runs 5 --output-json benchmark_host.json
python tests/benchmark_png.py
```

`test_fastboot.py` 校验 `fast_sub` PNG 编码与 Pillow 回退的逐像素一致性及运行包校验缓存；
`test_preview.py` 覆盖预览先发布、完整包门控和预览/导出失败语义；`test_reuse.py` 覆盖
`InferenceSession` 模型/硬件复用、预热与释放协议；`test_refine.py` 覆盖可选 OpenCV 轮廓后端的
逐像素等价（无 cv2 环境自动跳过）与符号链接校验；`test_upload_v2.py` 覆盖 EXIF/格式解码与已解码
RGB 复用合同。OpenCV 为可选依赖，缺失时相关测试跳过，不自动安装。

任务结果中的 `processing_seconds` 保留旧口径：任务线程启动到全部 PNG 写出，不含上传/浏览器、report、ZIP 和历史索引提交。下载包内 `report.json` 的 `stage_seconds` 是序列化时点的真实快照，包含到全部 PNG 与 ZIP PNG 载荷为止的阶段；自身写 report、把 report 追加进 ZIP 和最终总耗时不可能预先写在同一份 report 里。网页任务对象/`job.json` 另有 `final_stage_seconds`、`job_total_seconds` 和 `request_to_job_complete_seconds`，口径为 HTTP 请求体开始接收到含 report 的 ZIP 关闭，适合作为用户可见总任务耗时；历史索引提交不包含在该总时内。B 子进程另有启动/等待/总时长分离字段。

网页“量化结果”区显示上述服务端任务总时，并同时给出 CPU 轮廓耗时和 B 耗时；浏览器上传副本预处理和结果渲染不在这个服务端总时内。

六个导出视图 PNG 由每个任务专属的最多 2 个有界工作线程写出：解码像素、视图集合、文件名和 ZIP 成员/顺序不变；`comparison.png` 保持历史 1280×480，其余五个与浏览器提交的上传副本同尺寸（浏览器超过 300 万像素会先缩小原片），网页预览复用同一份全尺寸 PNG，不再重复缩放编码；旧任务仍读取其原有预览文件。当前默认编码器为 `fast_sub`（向量化 PNG Sub 滤波 + zlib RLE level 1，逐字节无损），可用环境变量 `PLD_PNG_BACKEND=pillow` 回退 Pillow `compress_level=1` 路径；两种后端解码像素一致，报告记录 `output_png_backend` 与 `output_png_bytes`。两个线程全部结束并关闭文件后才创建结果包和成功历史；任一保存失败则整个任务报错，不产生结果包或成功历史（若预览已完成则保留预览但永不提供 ZIP）。本仓库版本的主机回归已通过；本轮未做板测，见 [C_WEB_PERFORMANCE](../docs/validation/C_WEB_PERFORMANCE.md)。

上传解码仅在需要时做 EXIF 方向/RGB 转换，避免无方向标记图片的整帧冗余拷贝；推理结果加载可复用已解码 RGB（`load_result(..., decoded_rgb=...)`），但仍校验原始文件 SHA256 与尺寸，原图不匹配即拒绝。`RuntimeVerifier` 在 Linux 上仅当清单 identity 和全文件 stat 签名均未变化时复用上次哈希结果；任何属性变化都会触发全量 SHA256，内容仍全部匹配则校验通过并缓存新签名。`deployment_manifest.json` identity 变化则立即拒绝并要求重启服务，父目录/符号链接变化不会被缓存掩盖。

2026-10-12 只读核对发现板端手动 `start_board.sh` 默认运行包曾指向不存在的 `pld/c_v3_team`，而实际 systemd 服务使用 `final_mode6_fastq_20260929_v3`。本仓库脚本保留后者，后续同步时不要用板端旧脚本覆盖。

### 可选常驻 B worker

B 推理的源码默认仍为每个任务启动一个 `infer_images.py` 子进程；2026-10-14 板上正式服务曾显式追加
`server.py --persistent-b-worker`。该模式要求部署包 `deployment_manifest.json` 已包含
`board/infer_worker.py`（可用 `python scripts/package_cnn_board.py --list-board-files` 核对）。worker 启动到
`READY` 只完成 PYNQ 预导入和部署协议检查，不实例化 Overlay、不下载 bitstream；服务启动后的
`begin_warmup()` 会发出 `--warmup-only` 请求，由该请求初始化冻结模型与 `TileAccelerator`/DMA。当前版本进一步以 `InferenceSession`
在同一 worker 进程内复用冻结 `Model` 与 `TileAccelerator`（模型路径或 bit/native 选项变化即重建），
推理日志仍按任务分文件写出；异常会失效并关闭整个会话，不跨失败复用。服务启动时 `begin_warmup()`
后台执行一次 `--warmup-only` 请求完成预热，预热期间禁止新任务和 HDMI。

持久会话在空闲但存活时持有 DMA；因此 HDMI 启动前必须先向 worker 发送 `release` 请求并得到
`device_released` 确认，释放失败或 worker 已退出会把服务置为 fault 并禁止 HDMI。服务关闭同样先
release 再停止 worker。worker 进程缓存退出（非服务主动关闭）不再被静默重启，而是报告硬件状态未知、
要求重启服务。frame JSON 额外记录 `session_reuse` 字典（包含 `model_reused`/`hardware_reused`），热请求的
`b_timing` 冷启动字段被清空，避免重复报告旧初始化耗时。

`persistent_worker_startup` 只在本次冷启动时为正数，热请求为 0；`persistent_worker_request` 从请求写出
开始计时，不包含本次启动。`persistent_worker_total` 覆盖监督器本次调用全流程，除前两者外还包含
manifest 复查、请求构造/写管道和回复检查等少量剩余时间，不能把三个数字当成互不重叠的独立阶段求和
后解释为总耗时。worker 启动、回复、超时、退出或部署 manifest 漂移都会关闭该进程；推理请求期间的
失败按硬件状态未知处理并设置服务 fault，阻止后续硬件操作直到服务重启。2026-10-14 冻结候选曾在 PYNQ-Z2 上通过真实 JPEG→FPGA→六视图与 HDMI 软件切换；物理显示器尚未连接。

Linux 上 `PR_SET_PDEATHSIG` 跟随创建 worker 的父线程；若由处理单个 HTTP 任务的短生命周期线程创建，
任务线程结束就会杀死健康 worker。因此本版本由一个与 persistent supervisor 同生命周期的长生命周期
launcher 线程创建 worker，任务线程只发送请求。Linux 安装 parent-death signal 失败会让 worker 启动
失败，而不是静默继续。任务统计额外记录 `persistent_worker_pid` 和回复时的
`persistent_worker_alive_after_reply`，便于板端确认连续任务是否复用同一进程。

持久模式下每份 frame JSON 仍来自一次 `infer_images.main` 调用，但其中 `b_timing` 的
`module_entry_to_ready` 和 `best_effort_process_entry_to_ready` 从 worker 首次导入开始计时，热请求会
包含两次任务之间的空闲时间；不能拿它们与旧逐任务子进程的冷启动直接比较。比较worker 内单次初始化
应看 `main_entry_to_ready`，监督开销看上文的 persistent worker 三个字段。

## 上传 GitHub

上传本目录的源码及文档即可；无需上传桌面旧压缩包。`.gitignore` 已排除运行生成的 `data/`、访问码、缓存、虚拟环境、日志与本地环境文件。GitHub 网页手动上传不执行忽略规则，因此也应只选择本整理目录，不混入板端 data 文件夹。

本仓库保留源码在 `display_web/`。板上正式服务通过版本化 systemd drop-in 指向独立候选源码与 B 运行包，继续使用原 `PLD_Web/data/`；旧 unit 与冻结运行包保留。实际回退与重新切换已验证，具体命令见 [部署文档](docs/DEPLOYMENT.md)。
