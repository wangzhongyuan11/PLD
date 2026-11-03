# A27 control：仅执行已审查的一个对照

日期2026-10-09。Codex已修复并核验关键计算；原30项准备测试和独立5反例全部pass，单图冒烟五阈值全部一致；真实 CODEX_preflight_control.log exit0、全部门禁True。CODEX_RUN_APPROVAL.json仅放行control，其他候选未放行。

请使用现有嵌赛工作区，读取本任务和A27当前短state即可，不重做准备/基线、不要运行wrapup_run、apply_rev3、write_ready等修补脚本，不改已审代码/协议/审批。原数据/交付不改、无下载、不连板卡、不读2018最终掩膜。

已批准的唯一命令（工作目录E:/嵌赛/pld-medical）：

```
.venv/Scripts/python.exe -X utf8 runs/role_a_redness_triage/20261009/a27/a27_driver.py run --candidate control --attempt supervised01
```

只启动一次；如果supervised01产物/外层PID已经存在，先核验，禁止重复启动。创建a27内的小型PowerShell外层wrapper，以隐藏窗口后台启动；stdout/stderr及真实退出码、起止时间分别落盘，日志不覆盖。外层记录PID并启动上述已审driver，不重新实现训练/评估。新派生进程仅为此任务，不杀无关进程。所有显式日志/脚本均仅a27。

driver已经按4轮训练→96训练校准→15层verify→dev638 FP32+INT8五阈值→固定48诊断/案例顺序执行，并保存状态与每轮latest；非零/超时/STOP立即停止，不追加其他候选。

启动后给出约1KB的STARTED（真实PID、时间、command、日志和状态入口）并停止生成，由本地driver持续确定性执行，避免模型反复轮询。Codex负责阶段监督与额度保护；候选完成后另派短结果整理。不要把STARTED称为完成，不自行更改参数、绕过门禁、使用重置券。

如启动失败，保留首失败日志，短报并停止；只有具体错误修复经审查后才可新尝试。收到暂停/保护时在A27写STOP_REQUEST并核验对应自有进程退出/有效checkpoint，不去停止其他项目。
