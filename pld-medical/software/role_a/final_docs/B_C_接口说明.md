# A 组最终交付包：B/C 接口说明（role_a_final）

本包最终模型 = **已验证的 `soup_v5_a050`**（FP32 权重 SHA256 `f488719fe6d4d09c9b86be84d06b19d34241d6c14060b87da7bab25d1367eb77`）。
A21 联合量化感知微调（8 轮）为**负结果**，其候选不作为最终模型，仅作实验记录放在 `experiments/`。
本说明描述 B（RTL/硬件）与 C（板端集成）对接所需的全部数值与布局约定。**所有数字均为软件侧导出值与桌面测量，不代表 PYNQ-Z2 上板实测。**

## 1. 预处理（软件侧固定，必须与板端一致）

1. 输入为 RGB 三通道图像；缩放到模型分辨率 **256×256**，采用**保持宽高比的 letterbox 补零**（不足处补 0，不拉伸）。
2. 归一化：`float32 = image_uint8 / 255`，像素值落在 `[0, 1]`；补边像素为 0。
3. 允许的像素 padding 值为 0（INT8 域为量化后的 0）。
4. 输出为 **1 通道 logits**（未过 sigmoid）；二值化在评估侧用 `sigmoid(logit) >= 阈值`，阈值见第 5 节。
5. 指标只在 **valid 区域**（非 letterbox 补边）内统计；板端若做可视化，需同样屏蔽补边。

## 2. 张量布局与尺寸

- 特征张量：**NCHW**（batch=1）；权重：**OIHW**；小端字节序。
- 数据类型：激活 `int8`、权重 `int8`、偏置 `int32`、累加器 `int32`、Q24 乘子 `int32`。
- 单层输入/输出形状与逐层尺度见第 3 节表格；与 `int8/model_spec.json`、`int8/*/config.json` 一致。
- 各层后处理（ReLU、MaxPool 2×2、nearest Upsample、Concat）在 FP32/整数域按层定义执行；本规则只替换了 15 个卷积层。

## 3. 逐层量化尺度（导出值，来自 96 张训练集校准）

激活为**逐张量对称 INT8、零点 0**，`scale = max_abs / 127`；权重为**逐输出通道对称 INT8**，`scale = 该通道 max_abs / 127`（`weight.bin` 存 int8，`config.json` 内 `weight_scales` 记录逐通道浮点尺度）。

| # | 层名 | 权重形状 OIHW | 输入 NCHW | 输出 NCHW | padding | input_scale | output_scale | Q24 乘子范围 |
|---|---|---|---|---|---|---|---|---|
| 1 | `encoders.0.0` | 8x3x3x3 | 1x3x256x256 | 1x8x256x256 | 1 | 0.00787402 | 0.01629656 | 13853–21860 |
| 2 | `encoders.0.2` | 8x8x3x3 | 1x8x256x256 | 1x8x256x256 | 1 | 0.01123417 | 0.01276044 | 14423–41601 |
| 3 | `encoders.1.0` | 16x8x3x3 | 1x8x128x128 | 1x16x128x128 | 1 | 0.01276044 | 0.02779864 | 10894–36823 |
| 4 | `encoders.1.2` | 16x16x3x3 | 1x16x128x128 | 1x16x128x128 | 1 | 0.01880545 | 0.05180068 | 4517–19862 |
| 5 | `encoders.2.0` | 32x16x3x3 | 1x16x64x64 | 1x32x64x64 | 1 | 0.03172678 | 0.07777638 | 5201–27883 |
| 6 | `encoders.2.2` | 32x32x3x3 | 1x32x64x64 | 1x32x64x64 | 1 | 0.02897855 | 0.05488971 | 5387–43380 |
| 7 | `encoders.3.0` | 64x32x3x3 | 1x32x32x32 | 1x64x32x32 | 1 | 0.03851143 | 0.10810469 | 4137–39483 |
| 8 | `encoders.3.2` | 64x64x3x3 | 1x64x32x32 | 1x64x32x32 | 1 | 0.04458977 | 0.24929874 | 1559–18123 |
| 9 | `decoder_blocks.0.0` | 32x96x3x3 | 1x96x64x64 | 1x32x64x64 | 1 | 0.07982909 | 0.53191135 | 1448–10424 |
| 10 | `decoder_blocks.0.2` | 32x32x3x3 | 1x32x64x64 | 1x32x64x64 | 1 | 0.17054951 | 0.42664902 | 7551–26509 |
| 11 | `decoder_blocks.1.0` | 16x48x3x3 | 1x48x128x128 | 1x16x128x128 | 1 | 0.15406790 | 0.49517149 | 3730–20240 |
| 12 | `decoder_blocks.1.2` | 16x16x3x3 | 1x16x128x128 | 1x16x128x128 | 1 | 0.22881114 | 0.83205690 | 6737–20314 |
| 13 | `decoder_blocks.2.0` | 8x24x3x3 | 1x24x256x256 | 1x8x256x256 | 1 | 0.31862031 | 0.37142521 | 38518–135373 |
| 14 | `decoder_blocks.2.2` | 8x8x3x3 | 1x8x256x256 | 1x8x256x256 | 1 | 0.37142521 | 0.51691202 | 27306–85187 |
| 15 | `head` | 1x8x1x1 | 1x8x256x256 | 1x1x256x256 | 0 | 0.51691202 | 0.19950877 | 175034–175034 |

## 4. 量化与 Q24 舍入规则（必须逐条一致）

1. **激活量化**：`q = clamp(round(x / input_scale), -127, 127)`，`round` 为**最近偶数**（ties-to-even）。
2. **权重量化**：逐输出通道 `q_w = clamp(round(w / weight_scale_c), -127, 127)`。
3. **偏置**：`bias_int32 = round(bias / (input_scale * weight_scale_c))`，存 int32。
4. **累加**：卷积累加为 **INT32**（`acc = Σ q_x·q_w + bias_int32`）。
5. **重定标**：`q_out = clamp( round_even(acc * multiplier_q24 / 2^24), -127, 127 )`，其中 `multiplier_q24 = round(input_scale * weight_scale_c / output_scale * 2^24)`（INT32）。
6. **反量化**：下一层输入 = `q_out * output_scale`（浮点域），或保持整数并在下一层按其 `input_scale` 重新取整；两种写法在本包的 `golden` 向量中已给出对照。
7. 注意：上一层 `output_scale` 与本层 `input_scale` **数值不一定相同**（本包第 3 节可见，如第 1 层 output 0.0163 vs 第 2 层 input 0.0112），因此跨层必须按**本层** `input_scale` 重新取整（等效于消费端 clamp），否则会引入额外误差。

## 5. 二值化阈值

开发集固定五阈值 `{0.45, 0.50, 0.53, 0.56, 0.60}` 中，主指标最优为 **0.45**（`sigmoid(logit) >= 0.45`）。
A18 曾探索部署阈值 0.35/0.40（开发集候选，未替换交付），板端若要改用需同步改阈值常数与量化数值映射并做一致性测试。

## 6. 参数/内存规模

- 参数量 **122,113**（权重 121,760 字节 + 偏置 1,412 字节，INT8/INT32 打包后）；FP32 权重文件 501,949 字节。
- 单层最大 INT8 特征缓冲 **1,572,864 字节 ≈ 1.5 MB**（256×256×24 通道的 decoder 层输入，NCHW）。
- 最深下采样 3 次（32×32 瓶颈），最大 64 通道。
- **以上为算术估计，未做上板实测**：PYNQ-Z2 的帧率、DDR 带宽、DSP/BRAM 占用、功耗均未测量。

## 7. 最小可运行检查（已实测通过，可直接复制执行）

下面这段是**完整可运行**的检查脚本（不含省略号）：加载 15 层整数模型、取一张真实开发图做 letterbox 预处理、前向、按 0.45 二值化，并在 **valid 区域**统计前景像素；随后调用 `verify_export` 校验 15 层黄金向量。

```python
from pathlib import Path
import numpy as np
import torch

from software.role_a import quant, verify_export
from software.role_a.data import ISICDataset
from software.role_a.train_overall_v1 import assemble, build_default_spec

ROOT = Path.cwd()                                           # 在仓库根目录 E:\嵌赛\pld-medical 执行
PACKAGE = ROOT / "deliverables" / "role_a_final"
spec = PACKAGE / "int8" / "model_spec.json"                 # 注意：必须是 Path，不是 str

# 1) 15 层黄金向量校验
print("verify_export:", verify_export.verify(spec))          # 期望 layers_verified = 15

# 2) 真 INT8 前向（数据来自冻结开发集，仅用于接口自检，不重算主指标）
model, _spec = quant._load_quantized(spec)
model.eval()
assembly = assemble(build_default_spec())
dev = assembly.dev_sets["isic2016_test"]
dataset = ISICDataset(list(dev.items[:1]), root=None, size=256)
item = dataset[0]
image = item.image.unsqueeze(0)                              # [1,3,256,256]，letterbox 补零、值域 [0,1]
with torch.no_grad():
    logits = model(image)                                    # [1,1,256,256]
probability = torch.sigmoid(logits).squeeze(0).squeeze(0).numpy()
valid = item.valid.squeeze(0).numpy() > 0                    # 非补边区域
binary = (probability >= 0.45) & valid
print("image:", dev.items[0].stem, "| valid pixels:", int(valid.sum()),
      "| predicted foreground:", int(binary.sum()), "| total pixels:", int(valid.size))
```

实测结果（2026-09-27，`torch 2.11.0+cu128`、CPU 前向；脚本 `software/role_a/final_docs/check_interface_example.py`，exit 0）：
`verify_export` 返回 `{"layers_verified": 15, "weights_int8_bytes": 121760, "bias_int32_bytes": 1412, "checkpoint_sha256": "f488719fe6d4d09c9b86be84d06b19d34241d6c14060b87da7bab25d1367eb77"}`；
示例图 `ISIC_0000003`：valid 像素 **49,152**、预测前景 **17,433**（模型输入总像素 65,536，其余 16,384 为 letterbox 补边）。该结果仅证明接口可运行，**不代表精度**。

逐层产物（供 B 组对照）：

```python
from pathlib import Path
from software.role_a import verify_export
spec = Path("deliverables/role_a_final/int8/model_spec.json")
print(verify_export.verify(spec))          # 15/15 层 INT32 累加 + INT8 输出黄金向量
# 每层目录含 input.bin / weight.bin / bias.bin / acc_golden.bin / output_golden.bin
#                   input_full.bin / output_full_golden.bin（整张 256×256 的完整向量）
```

## 8. 已知限制（务必随结论引用）

1. **未上板**：本包全部指标为桌面软件结果；没有 PYNQ-Z2 的 RTL/综合/时序/帧率/功耗证据。
2. **激活量化是主要精度损失来源**：只对激活做量化（权重保持 FP32）就会把 dev638 总体 Dice 从 0.8440543676 降到 0.8421502487，几乎等于整段 INT8 差距（−0.0020036）；权重舍入贡献约 1e-4。
3. **校准口径**：激活尺度取自 96 张训练图的 **FP32 直通**统计（`runs/role_a_a14_soup/int8_soup_v5_a050`）；部署环路里上一层的量化输出可能略超该尺度而被 clamp。
4. **A21 QAT 负结果**：从 soup 出发的 8 轮联合假量化微调没有超过基线（最佳 epoch 8 INT8 = 0.8411142990 < 0.8420508068），故**未采用**；实验记录见 `experiments/`。
5. **2018 最终测试掩膜未打开**：未下载、未评分、未用于任何选模。
6. **旧包未改动**：`deliverables/role_a/`（旧 ISIC 2016 基线包）保持原样，本包为并存的新目录。
