# A27：补齐对照诊断后执行edge/supervised01

2026-10-09。control/supervised02全部5阶段exit0，outer真实exit0/17:59:41结束；4轮端点5369744a647f0c76ed1140e4654a767fd8b4a57191bb8ed76ed211db27727444，dev638 INT8 0.839783097886465 < soup0.8420508068333175，不采用，保留作匹配普通续训对照。

Codex发现诊断inner公式误用 GT & ~dilate(GT)，恒为0；已改为GT减腐蚀GT，并排除valid边界伪环带，10个独立反例全部通过。外环仍A26原valid口径，主dev638/权重/阈值都不变。旧cohort/case已归档在control/supervised02/diagnostic_history/before_inner_fix。当前审批只edge，真实CODEX_preflight_edge.log全部True。

在完成上一条A28准备修订并保存后执行本任务；不要同时修改任何已审A27代码/协议/审批。无下载、不改原交付/数据、不读2018最终掩膜、不连板卡。

## 有界步骤

1. 仅重跑control的48图诊断（使用现有a27_diagnostics.py --run-dir a27/runs/control/supervised02 --variant control），独立日志/真实exit；不重训、不重校准、不重复638评估。确认48图Dice、外环/背景TPFPFN和案例中央FP768与旧记录相同（1e-6或整数一致），新增inner字段真实非恒0，若不符即停保留证据，不启动edge。
2. 更新简短control负结果说明与state，明确内环旧0无效、原主指标不变；不改静态protocol或已审代码。rootA02及原交付不动，由Codex统一维护。
3. 核验control自有wrapper已结束、edge/supervised01无产物、STOP_REQUEST不存在。只用已审launch_wrapper.ps1 -ProjectRoot E:/嵌赛/pld-medical -Candidate edge -Attempt supervised01，隐藏启动一次。不要按名称杀Python，不自动追加hardbg或A28训练。
4. 启动后短报真实wrapper/driver PID、日志/result及当前阶段，然后停止生成。已审driver自行执行4轮→96校准→15层verify→完整dev638→48诊断；只有outer真实exit和finished才叫完成。失败即停，不自行重试。

少量确定性步骤约6工具/10分钟；实际诊断可脚本等待，勿让模型重复读长日志。Codex监督阶段与额度，≤10%缩批、≤7%保护。收到保护先保存/停止，不因为已排队任务继续启动。
