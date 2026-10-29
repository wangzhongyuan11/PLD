# 皮损 8088 自动显示重构（待板端验收）

日期：2026-10-29。基线：`a18c51af56d8bda0dbfb25b7f7688cfbc2a45b29`。
现象：关闭自动 HDMI 时网页完成很快，开启后观察到约十倍等待。
本轮完成源码重构与主机回归；没有部署、连接板卡或测量真实显示时间。

## 按审查优先级处理

| 优先级 / 原因 | 本轮修改 | 保留的边界 |
|---|---|---|
| 高：每次 HDMI 后重新准备计算会话 | SkinWorker 保留 PreparedArtifacts 普通内存缓存；每个权重、LUT、参数文件首次校验后只读取一次。两次地址重定位及后续会话复用相同字节。文件路径、设备号、inode、长度、修改/变更时间发生变化立即拒绝复用。 | 每次仍校验模型 spec、识别并加载计算位流、重新分配 DDR、重定位描述符；不复用已释放缓冲或旧 MMIO。完整退出丢弃缓存。没有消除计算/显示位流切换。 |
| 高：显示等待阻塞网页代理 | 外层 8088 与内层 hub 都把网络等待移出全局状态锁。GET 状态、预览和文件可并行；hub 对未确认的 POST 保留提交占用标记，阻止另一 POST、模型切换、ROI 和评测抢先取得硬件；关闭服务等待该请求结束。外层代理记录写请求锁等待。 | 保留入口模型检查及 hub 原有 generation 检查；不允许并行硬件写操作。 |
| 中：重复解码全尺寸 PNG | 普通皮损四宫格优先使用已有 preview_original、preview_mask、preview_outlined。旧历史无预览时回退原图。同步独立部署的两份 quad_display。 | 只改变显示采样路径；推理输入、阈值、掩膜、六张原尺寸导出图和 ZIP 合同不变。缩放后屏幕像素可能因两级采样与旧显示有细微不同。 |
| 中：耗时终点不一致 | 增加计算初始化细分、参数缓存复用标记，以及独立的 HDMI 解码合成、位流准备、视频启动、首帧提交记录；页面分开标识 ZIP 与首帧。 | 首帧指驱动调用返回及提交完成，不能代替物理屏幕目视确认；各总计和子项有重叠，不能相加。 |
| 中：10 秒 HTTP 超时与 90 秒预热等待冲突 | 常驻显示 start 立即返回会话，由后台线程等软件就绪并发送 show；取消早于 show 时不会再启动显示。自动请求保留到确认后才移除，超时先核对相同 job/view 的状态，皮损最多两次尝试。 | 软件预热上限 90 秒、发出 show 后首帧上限 30 秒；非预热子进程首帧上限 120 秒。超时发 stop 并报告故障，不强杀驱动、不将超时当作 DDR 已释放；只有真实 idle/退出确认才能结束会话。新任务 admission 会检查异步故障。 |
| 中：打开开关先恢复旧图 | 打开自动显示只保存后续任务偏好，不立即 restore 历史。 | 模型切回时仍按身份校验恢复上次结果，手动显示保持可用。 |
| 中：通知超时被静默丢弃 | begin 使用稳定 request_id，done/failed 使用 token；入口保留有限的去重记录，传输失败最多两次尝试。失败写日志及任务/状态 API，页面提示。 | 去重记录只在本次入口进程内保留最近 64 项；不会在 finish 时补发一个新的 begin，避免旧任务重新取得显示权。手动 stop/show 不自动重放。 |

`display_web/unified/accepted_backends.json` 仅增加这次经主机接口回归的三份皮损文件 LF/CRLF 精确哈希，保留历史版本，不放宽为任意哈希。这是安装接口兼容记录，不是板测通过证明。

## 主机验证

- Python 3.12：185 项测试通过、14 项子测试通过；2 项跳过，分别为未安装的可选 OpenCV 路径及 Windows 不覆盖的 Linux 符号链接语义。
- 8 个 Node 浏览器脚本通过，覆盖原有页面、上传副本、预览/ZIP 门控、计时、入口路由、ROI 与四宫格交互。
- 新回归涵盖：GET 不被显示锁挡住而 POST 仍串行；打开开关不恢复历史；丢失回复后的幂等重试；旧任务不能清除新意图；有限重试与错误可见；预览复用及旧结果回退；预热前取消不启动硬件；首帧超时不伪报释放；没有轮询状态也不能绕过异步故障；软件缓存重用与 DMA 重新分配；参数改变拒绝分配；超时保留活动缓冲；结果数组独立拥有数据。
- 原有 FPGA 释放顺序、过期 overlay 拒绝访问、完整 PNG/ZIP 与历史恢复等回归通过。
- 未执行 RTL 构建，因为未修改 RTL、模型、量化或 ABI；未继承旧版本的板端数值和性能验收。

可复现 Python 选择（环境需有 pytest、NumPy、Pillow、SciPy；OpenCV 可选）：

```text
PYTHONPATH=<repo>;<repo>/display_web;<repo>/display_web/entry;<repo>/display_web/unified;<repo>/experimental/skin8/tools
python -m pytest display_web/tests display_web/entry display_web/unified/tests/test_quad.py display_web/unified/tests/test_repository.py display_web/unified/tests/test_skin_roi_touch.py display_web/unified/tests/test_hub.py display_web/unified/tests/test_interactive.py display_web/unified/tests/test_interactive_install.py display_web/unified/tests/test_proxy_concurrency.py display_web/unified/tests/test_evaluation_bridge.py experimental/skin8/tools/test_prepared_artifacts.py experimental/skin8/tools/test_hdmi_policy.py experimental/skin8/tools/test_web_lifecycle.py -q -rs
```

分号是 Windows PYTHONPATH 分隔符；Linux 使用冒号。主机检查使用真实本机 HTTP/子进程及模拟 PYNQ/MMIO，不是物理板测。

## 部署与测量所需边界

源码应作为独立候选整体部署，不能只替换 server.py：入口的重试协议须与 display_completion.py 配套；server.py 与 warm_hdmi.py 的异步会话接口须配套；Skin8 的 web_entry、frame_driver、board_image 须配套。共享模块若在 web 与 hub 两处部署，需要同步同一版本。部署路径先依据当前服务配置及文件 SHA 核对，不能仅依据 10 月 25 日记录猜测。保留旧目录与 `delivery/current.json`。

板端验收应使用同一原图交错重复以下三组：自动显示关闭、开启后首张、上一张已上屏后的连续上传。另测关闭/开启开关后立即上传、显示启动期间取消、模型切换与回退。

每组记录：原图尺寸与 SHA、运行源码/模型/位流身份、head 与六图像素哈希、上传至预览、上传至 ZIP、上传至 HDMI 首帧，以及物理显示是否正确。关注 `artifacts_reused`、`hardware_reused`、`core_startup_stages`、`ENTRY_PROXY_WAIT`、`hdmi.timing.json`。自动显示状态中的 `last_request.request_seconds` 是入口请求确认时间，不是首帧时间。

`/api/status.hdmi_timing` 给出当前显示任务的首帧记录；普通新任务包含 `request_to_first_frame_submit_seconds` 和 `result_to_first_frame_submit_seconds`。历史任务缺少同次启动的 ZIP 单调时钟时只提供渲染阶段记录。很晚才手动显示会包含等待用户操作的间隔，不能用它比较自动显示速度。ROI 目前继续沿用原有显示计时，不宣称本轮新增了 ROI 端到端指标。

在这些实测完成之前，只能称为通过主机回归的重构候选，不能称为十倍延迟已经消除或已经部署。
