# A27 control/supervised02：修复启动契约后重试

2026-10-09。首次control/supervised01在真正训练前因v1 plan_mode发现非空目录而拒绝：0轮、没有latest；FIRST_FAILURE_control_supervised01.json及旧日志/目录保留，不删除、不冒充已训练。

Codex已修复：a27_train鲜启动不预写variant.json；driver所有阶段日志移到a27/safe_logs/<candidate>/<attempt>/，不污染鲜训练目录。新增启动兼容性反例与原独立反例共7/7通过；真实CODEX_preflight_control_retry.log全部门禁True。当前审批仍只control，允许新attempt supervised02，旧失败不复用。

先仅准备一个可靠launcher，写完停止等审查，约3工具/5分钟：

- a27内写小型outer_wrapper.py，调用唯一已审命令 `.venv/Scripts/python.exe -X utf8 a27_driver.py run --candidate control --attempt supervised02`（driver使用相对项目的完整路径），设置cwd项目、进程级UTF8、隐藏子进程；driver stdout/stderr独立独占日志。
- wrapper必须等待driver实际退出并在a27内原子保存result JSON（真实exit_code、started/finished、wrapper PID、driver PID、命令、日志），再以同一实际退出码退出。不得靠解析stdout猜外层exit。wrapper自身异常也要非零和真实摘要。
- 准备隐藏启动该wrapper的PowerShell脚本，Start-Process -WindowStyle Hidden，-WorkingDirectory项目。记录真实外层PID及路径；所有显式文件都在a27，日志不覆盖。
- 启动前须核验该候选没有活任务、supervised02没有产物；旧supervised01失败记录只能作为历史。不要按名称杀Python，不改已审核心/协议/审批，不重做准备、不追加候选。

这一条消息仅准备可靠launcher，不启动训练。给出READY和两个文件路径即停。Codex审查launcher后另发唯一启动命令，再由DeepSeek执行并短报STARTED停止生成，由本地driver确定性运行。

## 已审查的启动放行（2026-10-09，Codex）

outer_wrapper.py与launch_wrapper.ps1已检查并修正：参数/写路径限制、launcher只阻止本任务有记录的活wrapper（不因其他项目Python存在拒绝启动）、失败不伪报STARTED、wrapper异常清理自有driver。仅启动control/supervised02一次，不改这些文件。

请先核验旧supervised01相关PID已经结束，新supervised02无产物，再执行现有隐藏launcher。传ProjectRoot真实项目路径，Candidate=control，Attempt=supervised02。不要安装/改全局/读凭据/追加候选。

启动后短报真实外层PID、driver PID（result初始记录）、a27_state当前阶段、日志/result路径，然后停止生成。本地driver负责确定性训练与验收，Codex监督/额度保护。首次结果文件仅含PID、不含exit时表示运行中，必须真实wait后有exit和finished才能称完成。失败保存并停止，不自行重试。
