你是本项目执行助手。任务只改 `software/role_a/quant.py` 和必要的 `software/role_a/tests/test_quant.py`，为现有 `quant export` 增加 `--overall-train` 开关，以便新模型仅用冻结的 2309 张训练图做跨来源 INT8 校准。Codex 负责审查、实测和最终裁定。

要求：
1. 保留旧 `--split --image-dir --mask-dir` 导出路径和二进制格式。使用 `--overall-train` 时，从 `train_overall_v1.assemble(build_default_spec())` 获取三来源训练项，禁止读取 2018 最终留出集；校准图在各来源内确定性均匀取样，`--calibration-images 96` 应各取 32 张。其他正整数也应合理分配，不能重复 ID。`--overall-train` 与旧数据路径参数组合时明确拒绝，避免静默忽略。
2. `model_spec.json` 记录各来源数量、校准 ID 和冻结训练划分摘要；旧模式字段及读回保持兼容。必须验证校准 ID 属于训练且不在三套开发集。输出位置仍由命令参数指定，项目内。
3. 对新取样逻辑写小而有效的测试。不要修改 `train_overall_v1.py`、`data.py`、`model.py`、当前训练断点或用户全局配置。不下载文件，不显示密钥。
4. 你的内部 shell 曾因项目 ACL 错误无法执行；这次若仍失败，只通过文件读写工具完成代码，不要反复尝试 shell 或声称测试通过。完成后简短列出改动和未运行的检查，停止。不要输出长篇推理；目标是最少无意义 token。

项目 `AGENTS.md` 的既有持续授权适用：允许读取项目数据、断点、模型及必要代码，只能在项目内写产物，不能泄露私人信息。新增空间预算远低于 1 MB；项目总量仍不得超过约 17 GB。
