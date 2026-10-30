# P0 开源复现与显示恢复演示候选

本候选落实“开放 GitHub 仓库并完善 README、异常处理与 HDMI HUD”中的软件展示部分，基于 main `cb710efeb2debea9a5c17f938b227eb9beb1b3e0`。2026-10-30 通过 GitHub 元数据核实 `wangzhongyuan11/PLD` 的 visibility 已为 public，无需再切换公开状态。源码候选和主机回归记录不表示板端已经部署。

## 复现材料

根 README 已列出三模型入口、RTL、Vivado Tcl、PS7 配置、冻结 Release 和板端验证入口。正式版本仍由 `delivery/current.json` 指定。当前板端应用与冻结双模型交付是两个身份，不应混用各自性能或回退步骤。

仓库现有工程采用 Tcl 生成 BD 和 IP，未跟踪独立手写 XDC。时钟及接口约束来源需按对应构建脚本、PS7 配置和 IP 生成文件审查；不要为了提供 `.xdc` 而添加未参与实际构建的占位约束。复现某个位流时，应使用其同次源码清单、bit/hwh/XSA、实现时序、资源和 DRC 报告。生成约束如需发布，应随相应构建证据进入 Release。本候选没有重新构建或发布硬件。

Windows 命令中的 Git Bash 和 Vivado 路径须在使用者机器核实；历史 `D:\Git` 不作为所有机器的默认路径。明确调用 Git Bash，避免由 WSL 执行构建。依赖和模型准备沿用根 README 及对应模型合同。

## HDMI 状态栏

状态栏在显示帧提交前绘制，不改写原图、掩膜、head、PNG 导出或 ZIP。皮肤镜和 CAMUS 四宫格放大后仍显示同一任务信息；ACDC 在触控按钮下方绘制，不改变 ED/ES、平面、切片或旋转的坐标。底部常驻工程教学与非临床诊断声明。状态栏覆盖少量显示像素，不改变底层医学数据或统计。

| 字段 | 数据来源与口径 |
|---|---|
| 模式、INT8、阈值、面积 | 皮肤镜成功任务的 job.json 和 inference/frame_000001.json；回放、上传掩膜明确标识没有新 PL 推理。其他模型使用各自 schema，不套用皮肤阈值 |
| 皮肤镜网络耗时 | 成功 infer 帧的 inference_seconds；PLGraph 墙钟时间，包含其 DMA/调度，不称纯 MAC 周期 |
| 皮肤镜任务耗时 | request_to_job_complete_seconds；请求接收开始到 ZIP 完成，不含浏览器绘制 |
| CAMUS 网络与任务耗时 | report.json 的 timings_seconds.network 及 seconds_to_metadata_finalize；后者不含最终元数据追加，屏幕标为 To metadata |
| ACDC 网络耗时 | numerics.json 的真实 PL、非 synthetic、完整成功 records.seconds 的切片中位数，不能当作整卷时间 |
| ACDC 任务与吞吐 | completion.json 的 server_upload_to_assets_seconds；有完整有序时间窗时显示切片数除以首片开始至末片结束的墙钟时间，标为 Slice/s (job window)，不是整机 FPS |
| FPS | 单张保存结果显示 unmeasured；不使用 1/单帧延迟伪装实时帧率，也不使用 HDMI 60 Hz |
| 核心频率、DSP、BRAM | 可选的部署静态构建记录，必须匹配当前任务的模型和运行时 SHA256，并提供利用率报告 SHA256。显示 Build；不称实时资源负载 |

没有证据的字段显示 unmeasured 或 unreported。已保存任务的注记标明 saved task，不表示硬件此刻仍运行该计算核心；计算完成后显示 overlay 已占用 PL。

`display_web/hud_build_info.example.json` 是格式模板，所有占位值默认无效。部署者从对应布线后报告填写真实值，在显示子进程环境设置 `PLD_HUD_BUILD_INFO` 指向私有配置。皮肤镜以 model_spec_sha256 + bit_sha256 绑定；ACDC 以 model_sha256 + runtime_sha256 绑定。CAMUS 由独立显示适配器生成绑定报告 SHA 的 hud_runtime.json，不修改冻结推理源码或原报告；缺少可信身份时资源仍保持未报告。不能使用附件中的示例数值填充模板。配置中的数值是人工整理的构建摘要，哈希匹配不代替对原报告内容的审查。

## 入口异常与恢复演示

入口的切换、显示、连接和模拟演示错误会出现在原错误区域及可关闭弹窗中。同一持续错误不会每次轮询重复打开弹窗。各模型原有输入校验和错误路径继续使用各自接口；本候选没有声称完成所有输入错误的跨页面弹窗。

进入“超时恢复演示（模拟）”可选择成功或失败场景。独立软件设备首先保持无响应；看门狗以单调时钟检测超时，模拟确认 DMA 复位后释放缓冲区、重新映射，然后仅重试一次并比较固定响应的 SHA256。成功流程展示 SIM_TIMEOUT → SIM_RESET → SIM_RELOAD → SIM_VERIFIED → SIM_OK。失败场景返回损坏的模拟响应，校验失败后 SIM_FAILED 停止，不会继续重试。页面、事件代码及保存记录均标注 simulated。

每次流程保存最近最多 24 条事件到入口 state 目录的 recovery_demo.json，含主机 UTC 时间和单调耗时；文件覆盖为最近一次运行，不无限增长。准备或切换未完成时不允许开始演示；跨站 POST、未知参数、并发启动及停止后的新请求会被拒绝。取消和日志写入失败不会标为成功。演示不调用板端接口，不重启真实服务，不清除实际 fault，不修改显示偏好或正式任务状态。

真实故障继续服从既有驱动的有界超时、DMA 复位、活动 DDR 释放及所有权确认。不能把文档建议的 1 秒直接设为所有模型/整卷任务的超时，也不能用网页按钮强行 kill 或热换 overlay。实板验证入口包括 `acdc/board_verify.py --recovery-checks`、对应优化设备的恢复验证及 `camus/board_verify.py`；必须使用各自匹配模型和运行包，在停止冲突服务并确认唯一拥有者后执行。主机模拟不能作为这些检查的通过证据。

## 候选同步与回退

用 `python scripts/package_p0_demo.py --output out/PLD_P0_Demo_Candidate_20261030.zip` 生成白名单源码包和逐文件 SHA256。包只供代码审阅与独立候选部署准备，不含访问码、状态、数据、模型或位流，不会自动安装或上传。包内历史安装器的源码兼容修改也需审阅，不作为当前系统的一键部署指令。验证记录见 [P0 主机验收](validation/P0_DEMO_20261030.md)。

同步必须把新依赖一起带上：入口需要 recovery_demo.py；所有受影响显示路径需要 display_web/telemetry_hud.py。当前板端存在多份 display_web，须按实际启动命令映射，不能只更新仓库根对应的一份。CAMUS 的独立 hub 包也携带该共享模块；旧四宫格补丁打包补充了新模块的“仅缺失或精确相同版本”门禁，旧备份仍按原成员集回退。该旧补丁不构成当前三模型系统的全套部署入口。

不要直接覆盖运行目录。板端实施应先记录当前服务、源码/模型/运行时哈希、原文件与 drop-in；将候选放入独立版本目录，重新生成受影响的源码身份记录，再按现有正常停机流程切换。ACDC 的应用证书绑定源码，不能沿用旧证书冒充新候选。未知安装版本须停止并人工核对，不扩充哈希白名单来绕过检查。回退恢复本次实际保存的原文件、证书和服务路径，保留原数据及偏好；正式 delivery 指针不变。

仍需实板验收：三模型各自固定输入输出逐元素检查；首次与连续任务的计时；HDMI 四宫格、放大、ED/ES/旋转触控目视检查；HDMI 后推理恢复；模拟演示前后真实任务与硬件所有权不变；按模型合同进行真实超时恢复与输出校验。记录实际 source/model/runtime SHA、主机时间、退出码和失败项。新的软件候选不能继承旧版板测结果。
