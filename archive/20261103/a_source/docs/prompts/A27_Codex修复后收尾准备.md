# A27：Codex关键修复后的准备收尾（仍不训练）

Codex独立5个数值/验收反例在初版全部失败，已修复a27_losses.py的逐图非空内外分量平均、几何计数，以及a27_driver.py的completed必要门禁、自有进程kill后wait观测、隐藏子进程窗口。不要改回这些修复，也不要再把双侧均值相加。旧失败日志 CODEX_counterexamples.before.log 保留。

请完成这些稳定的收尾工作，约10工具/10分钟：

1. 将test_a27.py内显式TemporaryDirectory均指定dir=a27/qa_probes（先创建）；不要在用户Temp写本项目手工测试产物。Codex已修正原错误的双侧手算预期，独立test_codex_a27.py保留。跑两套必要测试、保存独立日志/实际退出码，若失败短报真实原因，不自动改变Codex的数学契约。
2. protocol端点phase应为实际v1的evaluated，而不是不存在的completed phase；完成标记是completed.json，明确二者不同。环带aggregation明确每图非空side均值后按图平均，safe_core7x7固定，原48名单包含案例一次。
3. baseline_background_from_a26中远背景确实可从已有计数导出：background_pixels=valid_pixels-(tp+fn)，background_fp=fp；far_background_pixels=background_pixels-ring_pixels，far_background_fp=fp-ring_fp（外环=膨胀GT减GT交valid）。逐图逐阈值验证分母/计数非负，空分母null，保存48图逐图及macro/micro派生基线。**不再新跑48图或638基线**；原A26 Dice/外环诊断原样复用。
4. 只用原最终INT8模型对报告案例 ISIC_0000012 做一张正向诊断冒烟，使用现有a27_diagnostics.evaluate_model实际函数；保存真实记录，与A26同图各阈值dice/tp/fp/fn/ring/中央块核对。修正实际schema问题如有，但不能改原A26或原模型。其余队列不重复前向。
5. 更新PREPARED说明Codex接管范围、5个初次失败反例/修复后的结果、基线可导出远背景，不再声称远背景必须新前向。刷新各审批哈希但approved=false、只列control；不得授予自己训练权限。
6. 真实preflight仍应仅因approved=false失败，余门禁均为True。写简短READY状态和约1KB回传，然后停止。由Codex决定训练放行。

原交付/数据/既有训练器不改，不读最终标签，不下载/安装/连接板卡。所有手工产物仅a27。原额度保护及停止指令有效。
