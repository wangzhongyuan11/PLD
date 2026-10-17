# role_a_final 交付包导读（README）

本包是 A 组分割模型的最终交付包。**所有指标均为桌面软件结果，未上板**（无 PYNQ-Z2 的 RTL/综合/时序/帧率/功耗证据）。

## 1. 最终模型

| 项 | 值 |
|---|---|
| 模型 | `best.pt`（FP32 权重），= **已验证的权重 soup 候选 `soup_v5_a050`** |
| SHA256 | `f488719fe6d4d09c9b86be84d06b19d34241d6c14060b87da7bab25d1367eb77` |
| 结构 | TinyUNet，**8 基通道 / 3 次下采样 / 15 个卷积层 / 122,113 参数**，输入 3×256×256 → 1 通道 logits |
| INT8 | `int8/`，15 层逐张量激活 + 逐通道权重对称 INT8、Q24 半偶重定标；`int8/verification.json` = 15/15 层黄金向量通过 |
| 开发集（dev638） | FP32 **0.8440543675818143** @0.45；INT8 **0.8420508068333175** @0.45 |

> 本包**不采用** A21 联合量化感知微调（8 轮为负结果，最佳 epoch 8 INT8 = 0.8411142990 < 0.8420508068）；该实验仅作记录放在 `experiments/`，**不要把它的候选当作最终模型**。

## 2. B/C 组应先读哪些文件

1. **`B_C_接口说明.md`**（最重要）：预处理、张量布局/尺寸、15 层逐层尺度与 Q24 乘子、舍入与饱和规则、参数/内存规模、**可直接复制运行的最小检查脚本**、已知限制。
2. `int8/model_spec.json` + `int8/<layer>/config.json`：机器可读的逐层契约（含 `files_sha256`）。
3. `REPORT.md`：完整结果（总体/三面积组/三来源、FP32+INT8、五阈值）、A21 负结果、复现步骤。
4. `MANIFEST.json`：包内每个文件的 SHA256、字节数、出处。
5. `metrics/comparison.json`：dev638 的 FP32/INT8 汇总与 identity（checkpoint/spec/开发集 ID 摘要）。

## 3. 验证命令（在仓库根目录执行）

```powershell
$py = '.\.venv\Scripts\python.exe'

# 15/15 层黄金向量 + 逐层张量契约
& $py -m software.role_a.verify_export `
        --spec deliverables/role_a_final/int8/model_spec.json `
        --output-json runs/verify_final.json

# dev638 指标（一次评估同时给出 FP32 + INT8 五阈值）
& $py -m software.role_a.evaluate_quant_overall `
        --checkpoint deliverables/role_a_final/best.pt `
        --spec deliverables/role_a_final/int8/model_spec.json `
        --output-dir runs/reproduce_final

# 包完整性（逐条复核 MANIFEST 的 SHA256 与字节）
& $py software/role_a/final_docs/verify_package.py
```

期望：`layers_verified = 15`；dev638 FP32 `0.8440543675818143`、INT8 `0.8420508068333175`（@0.45）；MANIFEST 0 处不符。

## 4. 包依赖：**不是完全独立的自包含包**

- 权重与 INT8 产物本身可直接被 B 组的 RTL 流程消费（逐层 `weight.bin`/`bias.bin`/`config.json`/黄金向量），这部分自包含。
- 但 **Python 侧的验证/评估必须与本仓库的 `software/role_a` 代码一起使用**：`quant.py`、`verify_export.py`、`quant_math.py`、`evaluate_quant_overall.py`、`data.py`、`train_overall_v1.py`、`model.py`。包内不含这些源码，也不含数据集。
- 运行环境：项目虚拟环境 `.venv`（Python 3.12.14、torch 2.11.0+cu128）；仅做校验/评估可用 CPU。
- 因此：**拷走本目录不能单独复现指标**；请连同仓库使用（或由 B/C 自行实现等价算子，并以 `int8/**/acc_golden.bin`、`output_golden.bin` 作为逐层一致性基准）。

## 5. 明确限制

1. **未上板**：无板端实测；不得把本包描述为"已通过板卡验收"。
2. **2018 最终测试掩膜未打开**：未下载、未评分、未用于选模。
3. 数据只用了 ISIC 2016/2017（见 `DATA_PROVENANCE.md`）；48 对 novice 试点与 IMA++ 候选**未入训**。
4. 旧交付包 `deliverables/role_a/` 保留未改；本包为并存新目录。
