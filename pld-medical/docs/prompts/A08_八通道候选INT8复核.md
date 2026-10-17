# A08：8 通道 v4 候选的 INT8 复核（待 Codex 放行）

本文件是预备派工；**没有收到 Codex 明确启动消息前，不执行**。你是本项目 DeepSeek 执行助手，先读 `AGENTS.md`。

## 输入与前置门槛

- `runs/role_a_overall_v4_dice2_base8/` 必须有训练器正常写出的 `completed.json`、8 条 `history.json`、`best.pt`；最佳 FP32 开发集总体 Dice 必须大于旧版 `runs/role_a_overall_v1_base8/completed.json` 的 0.8411796459。核对 `best.pt` 对应完成标记的轮次和阈值，记录 SHA256。
- 只在 Codex 对上述证据复核并放行后执行；不得打开 2018 最终留出集掩膜，不更改旧模型包、旧试验或 B/C 交付包。

## 固定执行

沿用旧版 v1 的冻结三来源训练校准方案，96 张（每来源 32 张），不把开发集图像用于校准：

```powershell
.\.venv\Scripts\python.exe -m software.role_a.quant export --checkpoint runs/role_a_overall_v4_dice2_base8/best.pt --overall-train --output-dir runs/role_a_overall_v4_dice2_base8/int8_overall96 --calibration-images 96
.\.venv\Scripts\python.exe -m software.role_a.verify_export --spec runs/role_a_overall_v4_dice2_base8/int8_overall96/model_spec.json --output-json runs/role_a_overall_v4_dice2_base8/int8_overall96/verification.json
.\.venv\Scripts\python.exe -m software.role_a.evaluate_quant_overall --checkpoint runs/role_a_overall_v4_dice2_base8/best.pt --spec runs/role_a_overall_v4_dice2_base8/int8_overall96/model_spec.json --output-dir runs/role_a_overall_v4_dice2_base8/int8_development
```

若已有同名产物，先按校验哈希和任务状态确认，禁止覆盖不同输入。校准/评估的中间记录可续用，但必须确认 checkpoint、量化规格、开发集 ID 和评估器哈希一致。不得为提高分数临时改变校准样本数、阈值集合或量化算法。

## 验收与预算

- 输出模型规格、15 层逐层权重和黄金向量校验、638 张同口径 FP32/INT8 对比及三个来源/三面积组报告。核对导出的 checkpoint SHA 与输入一致、所有校准 ID 属于训练集、各来源 32 张；记录 INT8 相对 v1 INT8 0.8381469742 的增量。
- 本任务预计新增 <0.2 GB，项目总量 <17,000,000,000 字节；开始前测量剩余空间。若工具报错、数据/哈希不一致或占用异常，保留进度并停止。完整日志落项目内，启动、阶段完成、异常和结束时只给摘要与证据路径。
- INT8 指标即使提高，也只能作为软件数值仿真结论。是否采用、何时做一次性 2018 最终留出集评价以及是否更新 B/C 包，均由 Codex 裁定。
