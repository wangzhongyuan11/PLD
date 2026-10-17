# DeepSeek 执行任务：完成现有 v5 的第 2 轮训练

你是本项目的执行助手。先读项目 `AGENTS.md` 和 `docs/A02_当前断点.md`。用户刚把选模目标改为“面积分层验证集上的总体 Dice 优先”；本次仅**完成已经开始的 v5 训练并保存真实结果**，不要沿用旧的“每组不得退步”规则替 Codex 裁定是否发布模型。

## 输入与允许范围

- 工作目录：`E:\嵌赛\pld-medical`。
- 只读核验 `software/role_a/refine_retention_v5.py`、`runs/role_a_retention_v5/checkpoint.pt`、`runs/role_a_retention_v5/protocol.json`、`runs/role_a_augmented_v4/split.json` 与所需本项目数据。
- 允许在 `runs/role_a_retention_v5/` 写入该训练器自己的断点、日志、候选和评估结果；不得改训练器源码、原始数据、旧基线、其他试验目录或全局配置。
- 不读/输出密钥、私有模型或 API 配置；不要读取 ISIC 2018 测试集或掩膜。

## 执行

1. 简短检查 `.venv\Scripts\python.exe` 可用、CUDA 可用、现有断点可恢复；`runs/role_a_retention_v5/checkpoint.pt` 应是第 1 轮已评估状态，现有 `selection.json` 不应表示训练已完成。核对项目总空间预计仍低于 17 GB。
2. 若可恢复，在项目虚拟环境中运行：`& .\.venv\Scripts\python.exe -m software.role_a.refine_retention_v5 --resume`。记录命令、开始/结束时刻和退出码于 `runs/role_a_retention_v5/resume_execution.md`。该脚本自带断点与哈希检查；出错只记录原因，不改代码、不强行恢复或无参数重训。
3. 成功后读 `selection.json`、`history.json`、`checkpoint.pt`，核对第 2 轮完整保存。给出在已公开的三套开发集（2017 validation 145、2016 clean test 376、2017 exposed validation 117）上、按样本数加权的总体 Dice 排名前 5 的 `(epoch, alpha, threshold)` 和各自三个来源的 Dice。**这是开发集排名，不是最终测试结果。**
4. 简短报告文件路径、最终检查点的轮数/阶段、是否完成、异常、空间变化。长日志落盘，不把完整逐图结果发回。

## 停止条件

发现输入/哈希不符、项目总空间接近或超过 17 GB、需要新增权限或下载、CUDA/训练重复失败时停止并报告；不修改全局设置，不接触 2018 留出集。Codex 通知额度保护时完成当前安全步骤并保存后停止。
