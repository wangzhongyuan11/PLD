# 验证与整理记录

## 已有证据

- 2026-10-01：本机浏览器通过真实板测掩膜回放、合成原图及掩膜上传、后处理和视图切换；窄屏检查未见横向溢出。
- HTTP 结果下载测试通过并检查 ZIP；内置浏览器自动下载事件未成功确认保存路径。普通浏览器使用与文件导出请以实际结果为准。
- 固定访问码后端重启持久化、浏览器跨页面恢复、同页面更新链接、禁用存储回退均有自动测试。
- 用户后续反馈网页功能、HDMI、固定网址以及自启安装后重启访问均可实现。自启安装截图出现 AUTOSTART_PASS。本次整理没有重新连接开发板复测。

## 本次整理

- 将固定网址及开机自启合并为一个最终源码目录。
- 对比桌面 PLD_Web 与其原始压缩包，未发现桌面源码额外修改。
- 对比桌面旧 HDMI 工程的五个被复用模块，与本项目版本一致；保留它们以满足导入依赖和板端功能。
- 合并轮廓、网页后端和访问码测试；不保留缓存、虚拟环境、运行日志、历史截图及旧补丁包。
- 桌面旧文件保存在 Documents/Codex 下的归档目录，同学 PLD 工程保持原样。
- 当前整理版 25 项 Python 测试全部通过，JavaScript 访问码逻辑检查通过。

本地硬件互斥测试使用模拟子进程，不替代硬件验收。未来模型或硬件版本变化后，需重新验证接口与整机流程。

## 本次开机历史（2026-10-02）

- 完成记录持久化，同一开机标识下重建服务可恢复历史。模拟另一开机标识后列表为空。
- 验证失败/未完成任务、损坏索引、缺失结果包不会出现在恢复列表。
- 验证历史排序、切割图像素与下载接口、未授权请求拒绝访问。
- Windows 开机标识直接读取系统启动 GUID，无需 WMI 或管理员权限。Linux 使用内核 boot_id。
- 25 项 Python 测试和 JavaScript 访问码测试通过。板端重启和 HDMI 未在本次更新中复测。
- 本机浏览器确认历史切割图正常显示，刷新页面和刷新记录按钮后仍可查看；图片与 ZIP 下载内容由 HTTP 自动测试验证。

## 输出耗时与计时（2026-10-12，仅主机）

- 本轮代码修改未连接开发板、未停止 `pld-web.service`、未加载 bitstream 或运行 HDMI；以下主机数字不能当作板端收益。GPT 另做的只读同步检查和隔离 CPU-only 板上对照见 [C_WEB_PERFORMANCE](../../docs/validation/C_WEB_PERFORMANCE.md)，该对照显示 2125x1411 输出外部墙钟中位 43.189→17.013 s，ZIP 约 5.93→6.73 MB（+13.4%），全部解码像素与成员语义一致。
- 固定输入基准使用真实 `frame_000001` 掩膜和确定性噪声 JPEG，通过 `Console.submit(mode="mask")` 走完整网页输出路径；Windows 主机、Pillow 12.0.0、NumPy 2.3.5、22 个逻辑 CPU。最终脚本每档连续 5 次，基线源码从 Git HEAD `a686295` 独立展开，候选为当前工作区。
- 基线（Pillow 默认 PNG level 6 + ZIP deflate）中位 external/processing：1022x767 为 0.61209/0.48271 s，2125x1411 为 1.91776/1.57765 s。候选（PNG level 3 + ZIP stored）为 0.30187/0.21472 s 和 0.88966/0.70179 s，外部墙钟中位分别约 2.03 倍和 2.16 倍；均保留 1280x480 比较图和输入像素尺寸。
- 两档全部 5 次、每次 6 个视图的解码后像素 SHA256 与基线一致，前景/边界像素统计一致；ZIP 成员名和顺序不变。主机 results.zip 变化 +6,807 B（2,281,678→2,288,485 B，+0.30%）和 +5,631 B（7,604,995→7,610,626 B，+0.074%）。
- 新增 wall-time 计时覆盖请求接收到任务启动、停止 HDMI、运行包校验、输入写出、B 子进程启动/等待/总时长、结果加载、掩膜回映射、轮廓、视图合成、每种 PNG、report、ZIP 和任务完成。下载包内 `report.json` 的 `stage_seconds` 明确为序列化前快照；任务 API/`job.json` 的 `final_stage_seconds` 与 `request_to_job_complete_seconds` 覆盖到含 report 的 ZIP 关闭，历史索引提交不在总时内。失败任务保留已发生阶段且不进入可下载成功状态；已写出的中间文件保留用于诊断。
- 已审查并合入板端 `static/app.js`、`static/index.html` 的上传副本缩放 / PNG 转 JPG / 界面提示功能；本地原文件不被修改。Node 测试新增上传尺寸和灰度 PNG 字节流校验。
- 26 项 Python 测试与 JavaScript 访问码/上传逻辑测试通过。完整网页推理、真实浏览器大图上传、HDMI 和板端存储写入仍需下一阶段在 PYNQ-Z2 部署后验证；当前服务尚未切换到本候选。

## 可选常驻 B worker（2026-10-13，仅主机）

- 新增候选保持默认网页仍使用逐任务 `infer_images.py` 子进程；只有显式 `--persistent-b-worker` 才启用。主机监督测试使用临时运行包和假/真实 worker 协议，不访问 FPGA、HDMI、bitstream 或板卡服务。
- 2026-10-13 板上候选审查发现三张真实图均数值/视图一致但每帧重新冷启动。根因是 Linux `PR_SET_PDEATHSIG` 跟随创建 worker 的父线程：worker 由单个 HTTP 任务线程创建，该线程在任务完成后退出，于是健康 worker 被杀死。本地修复为增加与 supervisor 同生命周期的长生命周期 launcher 线程，由它创建 worker；HTTP 任务线程仅请求/等待。Linux parent-death signal 安装失败改为显式启动失败，非 Linux 主机测试保留无 prctl 路径。
- 覆盖同一 PID 连续两次请求、冷/热启动计时、启动失败、请求前退出、超时后终止、过期成功回复、推理 FATAL、超过 64 KiB 协议行、关闭后 fail-closed、部署 manifest 漂移后拒绝复用、服务 fault 阻止后续任务，以及请求持有 DMA 时 HDMI 启动被拒绝。`persistent_worker_startup` 冷请求为本次启动耗时、热请求为 0；`persistent_worker_request` 不含本次启动，`persistent_worker_total` 另含 manifest 复查与管道/监督开销。
- 新增主机回归证明请求线程结束后 worker 仍存活、第二请求同 PID 且 startup 为 0、关闭后 worker 与 launcher 均退出；并模拟 Linux `PR_SET_PDEATHSIG` 安装失败必须抛错。任务阶段新增 `persistent_worker_pid` 与 `persistent_worker_alive_after_reply` 供板端审计。
- 持久模式继承的每份 frame `b_timing` 中，`module_entry_to_ready` 与 `best_effort_process_entry_to_ready` 从 worker 进程首次导入开始计时，热请求包含任务间空闲时间；单次请求初始化应看 `main_entry_to_ready`。
- `python -m unittest discover -s tests -v` 在 Windows 主机通过 41 项；`node tests/test_browser_access.cjs` 通过；`python -m unittest sim.test_infer_timing -v` 通过 9 项；目标 Python 文件 `py_compile` 通过。`scripts/package_cnn_board.py --list-board-files` 显示 `board/infer_worker.py` 已进入打包文件映射；另用现有 `out/vivado_cnn` 检查通过的 bit/hwh/XSA 做主机打包 smoke，生成 `out/final_model/persistent_worker_package_smoke.tar.gz`（162 个文件，2,962,763 B），tar 内含 worker 且 deployment_manifest 哈希 `871f9673…24a7e` 与当前源码一致。该包未部署，不替代任何冻结交付。
- 上述条目为本机代码验证；同日补做的隔离板测见下节。不得把主机协议测试单独解释为板端加速成功。

## 可选常驻 B worker 隔离板测（2026-10-13）

- 初版连续小/小/大三张真实 JPEG 的 head 和六视图虽全等，但每帧 worker 均退出、启动耗时约 9.6 秒，未通过常驻验收。修复版使用长生命周期 launcher 线程，候选部署在 `PLD_Web_worker_candidate_v2_20261013` 和 `final_fastq_worker_candidate_v2_20261013`，正式服务未切换。
- 修复版小/小/大/HDMI显示停止/小共四帧均得到历史相同的原始 head 和六张视图解码像素；四次 worker PID 同为 5468，后三次 `persistent_worker_startup == 0`，回复时 worker 均存活。请求到 ZIP 完成依次 19.003、8.614、20.367、10.842 秒。HDMI 软件就绪 37.736 秒，停止后下一帧正确；物理画面未目视。
- 候选服务正常退出无残留 worker。另一次单图后强制 SIGKILL 候选网页进程，原 worker PID 6399 随后消失；正式 `pld-web.service` 恢复并返回 HTTP 200。强杀发生在推理完成后，没有验证 DMA 正在运行时的硬件恢复。
- 机器证据分别为 `out/board_sync_20261012/worker_candidate_board_audit_v2_20261013.json` 与 `out/board_sync_20261012/worker_parent_kill_audit_v2_20261013.json`；输出性能基线、版本哈希和统计口径见 [C_WEB_PERFORMANCE](../../docs/validation/C_WEB_PERFORMANCE.md)。仍需正式服务同图交错重复、长时内存和错误路径验收后才能采用。
