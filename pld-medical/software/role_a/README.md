# A 组：ISIC 2016 轻量 U-Net 与 INT8 导出工具

本目录是**新增的独立**软件基线，与 `software/baseline.py`（早期传统图像处理教学基线）互不影响。
它提供 Tiny U-Net、可复现的数据划分、训练与独立测试评估、INT8 导出及逐层黄金向量。

> **用途声明：仅供科研与教学使用，不构成医疗器械，不能用于临床诊断。**
> 本目录是源码；训练权重、精度报告和 INT8 文件见 `deliverables/role_a/`。
> 当前仍无 FPGA 资源/时序或板级运行数据。
> 任何在本目录下产出的 Dice/IoU 都必须连同数据版本、划分文件与评估空间一起报告。

---

## 1. 目录布局

```
software/role_a/
├─ model.py                  TinyUNet（base_channels=8，可选 1..3 次 MaxPool）
├─ data.py                   配对、letterbox、固定种子划分、Dataset
├─ metrics.py                只统计有效区域的 Dice/IoU 与 BCE+soft Dice 损失
├─ train.py                  CLI：prepare-split / train / evaluate-test / describe-model
├─ verify_reference.py       纯 Python 参考计算器（参数量/逐层 shape/MAC，不依赖 torch）
├─ quant_math.py             精确 INT32 卷积与 Q24 重新量化参考
├─ quant.py                  训练集校准、INT8 导出、黄金向量、FP32/INT8 对比
├─ extract_official.py       官方四个 ZIP 安全解包、配对检查与来源清单
├─ verify_export.py          SHA256、打包偏移与逐层整数黄金向量独立复核
├─ README.md                 本文件
└─ tests/
   ├─ _support.py            依赖检查 + 合成 ISIC 文件生成
   ├─ test_data.py           配对错误、letterbox 对齐、划分一致性
   ├─ test_model.py          shape、参数量、算子白名单、logits 语义
   ├─ test_metrics.py        Dice/IoU 边界、valid 掩膜、损失对补边不敏感
   ├─ test_reference.py      参考计算器 vs torch 模型参数量/shape 交叉验证
   ├─ test_train_pipeline.py 三个子命令的真实端到端运行（CPU、小图、极少样本）
   └─ test_quant.py          INT8 数值规则与二进制交付自检
```

代码不自动下载数据。本机已从官方地址取得四个 ZIP 并解包；`data/raw/` 在 `.gitignore`
中，不属于可移植的代码交付。另一台机器需自行从官网取得数据并运行 `extract_official.py`。

```
data/raw/isic2016/
├─ train/
│  ├─ images/ISIC_0000001.jpg ...           官方训练集 900 张
│  └─ masks/ISIC_0000001_Segmentation.png  与图像一一对应
└─ test/
   ├─ images/ISIC_0000016.jpg ...           官方测试集 379 张（只用于最终评估）
   └─ masks/ISIC_0000016_Segmentation.png  若标注不可得，evaluate-test 无法计算 Dice/IoU
```

## 2. 数据来源

- 官方入口（ISIC 2016 Task 1 "Lesion Segmentation"）：
  <https://challenge.isic-archive.com/data/>
- 该挑战的官方训练集为 900 张皮肤镜 JPEG + 对应 PNG 掩膜，官方测试集为 379 张。
- 官方页面标注该数据为 CC0。本机的 `data/raw/isic2016/provenance.json` 记录四个原始
  ZIP 的 URL、SHA256 和文件数量；共有 900 对训练、379 对测试图像/掩膜，标识无重叠。
  项目文档 `docs/06_数据与实验规范.md` 还要求记录版本、日期和引用信息。

从上面的官方页面下载四个 Task 1 ZIP，并分别保存为
`data/raw/isic2016/downloads/train_images.zip`、`train_masks.zip`、`test_images.zip`、
`test_masks.zip`；随后运行：

```powershell
py software/role_a/extract_official.py
```

脚本核对文件名、数量、图像/掩膜配对和训练/测试标识，再写来源清单。

## 3. 依赖

| 依赖 | 用途 | 备注 |
| --- | --- | --- |
| Python 3.10+ | 类型注解 `X | None`、`dataclass` | 3.12 亦可 |
| `torch` | 模型与训练 | CPU 即可跑通；有 CUDA 会自动使用 |
| `numpy` | 数组处理与 letterbox | |
| `Pillow` | JPEG/PNG 读取与缩放 | 双线性/最近邻 |

安装（Windows PowerShell，项目根目录）：

```powershell
py -m pip install torch numpy Pillow
```

训练和完整测试需要上述依赖。`verify_reference.py` 可单独用标准库运行。

## 4. Windows PowerShell 运行命令

以下命令均假设当前目录是项目根目录 `E:\嵌赛\pld-medical`。

### 4.1 查看网络结构（不读数据、不训练）

```powershell
py software/role_a/train.py describe-model --size 256 --base-channels 8 --num-pools 3
py software/role_a/train.py describe-model --size 256 --output-json runs/role_a_model_graph.json
```

输出包含逐层算子、每层参数、总参数量、输出 shape，以及 `allowed_only`（是否只用
Conv 3x3 / Conv 1x1 / ReLU / MaxPool / nearest Upsample / Concat）。

如果本机没有 torch（或只想核对硬件规模），用**纯 Python** 的参考计算器复算同一批数字：

```powershell
py software/role_a/verify_reference.py --base-channels 8 --num-pools 3 --size 256
py software/role_a/verify_reference.py --base-channels 8 --size 256 --export runs/role_a_reference.json
```

它给出逐层 shape、每层与总参数量、bias/权重个数、FP32 与 INT8 的**权重字节数**、
以及卷积 MAC 总数（不含 ReLU/MaxPool/Upsample/Concat）。这些仅供规模估计，
**不是**实测资源占用、时序或功耗。

### 4.2 生成固定种子的 train/val 划分（**只扫描官方训练集**）

```powershell
py software/role_a/train.py prepare-split `
  --image-dir data/raw/isic2016/train/images `
  --mask-dir  data/raw/isic2016/train/masks `
  --output-dir runs/role_a_split `
  --seed 42 --val-fraction 0.2
```

可选：额外登记官方测试集文件名（**只记录，不训练**）：

```powershell
py software/role_a/train.py prepare-split `
  --image-dir data/raw/isic2016/train/images `
  --mask-dir  data/raw/isic2016/train/masks `
  --output-dir runs/role_a_split_with_test `
  --seed 42 --val-fraction 0.2 `
  --include-test `
  --test-image-dir data/raw/isic2016/test/images `
  --test-mask-dir  data/raw/isic2016/test/masks
```

产物：对应输出目录中的 `split.json`（含 seed、排序后的文件名、配对表）、
`split.csv`（`role,stem,image,mask`）、`split_report.json`（非二值掩膜统计）。
同一 `--seed` 与同一数据目录会得到**逐字节相同**的 `split.json`。
训练只用不含 `test` 角色的 `runs/role_a_split/split.json`；第二次登记测试集时必须用
**另一输出目录**，不能覆盖训练划分。

### 4.3 训练（最佳模型只由验证集选择）

```powershell
py software/role_a/train.py train `
  --image-dir data/raw/isic2016/train/images `
  --mask-dir  data/raw/isic2016/train/masks `
  --split runs/role_a_split/split.json `
  --output-dir runs/role_a_tinyunet `
  --epochs 40 --batch-size 4 --lr 0.001 --size 256 `
  --base-channels 8 --num-pools 3 --seed 42 --device auto --num-workers 2 --augment
```

CPU 上可先做流程自检（不是性能测试）：

```powershell
py software/role_a/train.py train `
  --image-dir data/raw/isic2016/train/images --mask-dir data/raw/isic2016/train/masks `
  --split runs/role_a_split/split.json --output-dir runs/role_a_smoke `
  --epochs 1 --batch-size 2 --size 64 --base-channels 4 --num-pools 3 --device cpu
```

### 4.4 官方测试集评估（显式单独命令，不参与选模型）

```powershell
py software/role_a/train.py evaluate-test `
  --image-dir data/raw/isic2016/test/images `
  --mask-dir  data/raw/isic2016/test/masks `
  --split runs/role_a_split_with_test/split.json `
  --checkpoint runs/role_a_tinyunet/best.pt `
  --size 256 --device auto `
  --output-json runs/role_a_tinyunet/test_metrics.json `
  --per-image-csv runs/role_a_tinyunet/test_per_image.csv
```

### 4.5 运行测试

```powershell
py -m unittest discover -s software/role_a/tests -v
py -m unittest discover -s software/role_a/tests -p "test_data.py" -v
```

完整测试需先安装依赖。当前已在合成小图上执行 92 项测试，全部通过；这不代表
ISIC 测试集精度、FPGA 时序或板级运行结果。

## 5. 实现要点

### 5.1 网络（`model.py`）

```
enc0 Conv3x3(3→8)+ReLU ×2        256×256 ─skip─────────────────┐
MaxPool2x2 → 128                                              │
enc1 Conv3x3(8→16)+ReLU ×2       128×128 ─skip────────┐       │
MaxPool2x2 → 64                                       │       │
enc2 Conv3x3(16→32)+ReLU ×2       64×64 ─skip─┐       │       │
MaxPool2x2 → 32                               │       │       │
enc3 Conv3x3(32→64)+ReLU ×2       32×32       │       │       │
Upsample(nearest)+Concat+Conv×2 → 64×64       │       │       │
Upsample(nearest)+Concat+Conv×2 → 128×128     │       │       │
Upsample(nearest)+Concat+Conv×2 → 256×256     │       │       │
Conv1x1(8→1) → logits 256×256                 └───────┴───────┘
```

- 基通道 8，最多 3 次 2×2 MaxPool；没有 BatchNorm / LayerNorm / Dropout / 注意力 /
  转置卷积 / Sigmoid / Softmax。
- 输出是 **logits**，sigmoid 只出现在 `metrics.py` 的损失与评估里。
- `TinyUNet.num_parameters()`、`parameter_breakdown()`、`assert_shapes()`、`describe_graph()`
  便于 B 组核对权重规模、逐层 shape 与算子可映射性。
- `forward_with_features()` 额外返回最深编码器特征（32×32×64），方便后续板卡端比对。

### 5.2 预处理（`data.py`）

- 配对规则严格按官方命名：`ISIC_*.jpg` ↔ `ISIC_*_Segmentation.png`（stem 全等匹配）。
- 缺失掩膜、孤立掩膜、重复标识、图像/掩膜尺寸不一致 ⇒ 立即抛 `DataLayoutError`，
  错误信息里给出具体文件名，绝不静默跳过。
- letterbox：等比缩放 → 居中补零（`pad_value=0`，可用 `--size` 改输出边长）；
  **图像双线性、掩膜最近邻**；`valid` 掩膜由 letterbox 几何直接生成，
  而不是从像素值反推（因为补边像素本身就是 0）。
- 掩膜二值化：原始掩膜若严格只含 0/255 就直接用；否则按 `>= 127` 二值化，
  并在 `split_report.json` 里记录非二值掩膜的数量与文件名。
- 翻转增强默认关闭；训练命令加 `--augment` 时图像/掩膜/valid 同步翻转。

### 5.3 划分（`data.py` / `train.py`）

- 只对**官方训练集目录**做 train/val 划分，默认 80%/20%。
- 默认按掩膜前景占比分成 5 档后再交替分配（`--no-stratify` 可关闭），
  目的是让两集的前景比例分布接近，而**不是**患者级划分。
- 划分文件对同一输入与 seed 可逐字节重现；训练固定 Python/NumPy/PyTorch 随机源，
  CUDA 算子未强制确定性，因此重新训练不承诺权重逐位一致。
- 官方测试集只能通过 `--include-test` 登记文件名，且 `train` 命令遇到含 `test`
  角色的划分文件会直接报错退出（防泄漏护栏）。
- ⚠️ **限制**：ISIC 2016 公开文件名不含患者/病灶分组信息，本划分**无法排除患者级
  泄漏**，报告中必须如实说明这一点。

### 5.4 损失与指标（`metrics.py`）

- 训练损失 = 掩膜区域内 `BCEWithLogits` + soft Dice：`L = w_bce·BCE + w_dice·(1 - (2I+ε)/(P+T+ε))`，
  只在 `valid > 0` 的像素上统计，补边不产生任何梯度。
- 验证/测试：逐图算 Dice 与 IoU 再取均值（同时给中位数、非空真值子集均值、
  空真值/空预测计数）。
- 边界约定（已在测试中固定）：
  - 有效区域内预测与真值**都为空** ⇒ Dice = 1.0，IoU = 1.0；
  - 只有一方为空 ⇒ Dice = 0.0，IoU = 0.0；
  - 有效区域本身为空 ⇒ 直接抛 `ValueError`，不猜。

## 6. 输出说明

`prepare-split` 产物：

| 文件 | 内容 |
| --- | --- |
| `split.json` | seed、角色计数、排序后的 train/val(/test) 文件名、每个 stem 的图像/掩膜路径、配对的非二值掩膜统计入口 |
| `split.csv` | 逐行 `role,stem,image,mask`，便于用 Excel 复核 |
| `split_report.json` | 分层开关、非二值掩膜数量与文件名（仅分层模式生成） |

`train` 产物：

| 文件 | 内容 |
| --- | --- |
| `best.pt` | 最佳验证 Dice 的模型权重 + epoch + 验证指标 + 完整配置；**只含张量与基本类型**，可用 `torch.load(..., weights_only=True)` 加载；不存优化器状态、不支持断点续训 |
| `config.json` | 与 checkpoint 内一致的运行配置（数据目录、split、epoch、batch、lr、seed、device、size、网络通道） |
| `split_used.json` | 本次训练实际使用的 train/val 文件名快照 |
| `metrics.csv` | 每轮：train/val 的 loss、BCE、soft Dice，验证 Dice/IoU 的均值与中位数、空真值/空预测计数、是否最佳 |
| `metrics.json` | 最佳 epoch、最佳验证 Dice、最后一轮、选择规则说明 |
| `model_graph.json` | 算子序列与参数量 |

`evaluate-test` 产物：写 `--output-json` 与 `--per-image-csv`（若给出）并打印报告；**不会改写 `best.pt`
或任何训练产物**。报告固定包含 `checkpoint_selected_by`、`evaluation_space`
（"letterbox 后的 NxN 有效区域"）与免责说明。

### 6.1 INT8 导出与软件对比

在真实训练得到 `best.pt` 后，**只用 train 划分**按固定间距选取 32 张图校准激活尺度，
并导出供 B 组核对的整数参考文件（实际选中标识记录在 spec 中）：

```powershell
py software/role_a/quant.py export `
  --checkpoint runs/role_a_tinyunet/best.pt `
  --split runs/role_a_split/split.json `
  --image-dir data/raw/isic2016/train/images `
  --mask-dir data/raw/isic2016/train/masks `
  --output-dir runs/role_a_int8 --calibration-images 32 --tile-size 8
```

随后在验证集比较 FP32 与 INT8 软件仿真；正式报告可再用含 `test` 角色的划分文件、官方测试图及掩膜，以 `--role test` 单独运行：

```powershell
py software/role_a/quant.py compare `
  --checkpoint runs/role_a_tinyunet/best.pt `
  --spec runs/role_a_int8/model_spec.json `
  --split runs/role_a_split/split.json --role val `
  --image-dir data/raw/isic2016/train/images `
  --mask-dir data/raw/isic2016/train/masks `
  --output-json runs/role_a_int8/val_compare.json
```

顶层另有按 `model_spec.json` 所列层次顺序拼接的 `weights_int8.bin` 与 `bias.bin`，
每层偏移、字节数及两文件 SHA256 都写入 spec。每层目录含 `input.bin`（NCHW INT8）、`weight.bin`（OIHW INT8）、`bias.bin`（小端 INT32）、
`acc_golden.bin`（小端 INT32）、`output_golden.bin`（INT8）、`input_full.bin`、
`output_full_golden.bin`（完整单张图的逐层 INT8 输入/输出）与 `config.json`。`model_spec.json`
记录训练权重 SHA256、校准样本、完整张量形状、每层尺度和 Q24 乘数。激活是每张量对称量化，
权重是每输出通道对称量化；zero point 为 0，数值范围 [-127,127]，整数除法采用 ties-to-even，
再饱和到 INT8。每个卷积入口按该层 `input_scale` 重新量化，包括 skip/concat 之后；B 组实现时
必须使用同样的尺度转换和算子顺序。

小块黄金向量是**独立截取的左上角小块**，在块边界补零，适合验证卷积内核和整数规则；
`input_full.bin` 与 `output_full_golden.bin` 则来自同一张训练图的**完整量化网络前向**，
可用于逐层集成比对。`compare` 是 CPU 上的软件量化仿真；板级一致性、资源和时序
仍需由 B/C 组实测。

交付给 B 组前可独立复核所有层的 SHA256、打包偏移、INT32 累加值与 INT8 小块黄金向量：

```powershell
py software/role_a/verify_export.py `
  --spec runs/role_a_int8/model_spec.json `
  --output-json runs/role_a_int8/verification.json
```

## 7. 已实现 / 未验证（务必区分）

**已实现（代码层面）**

- Tiny U-Net 前向、参数量查询、算子白名单自检、输入合法性检查（错误输入抛异常）。
- 官方命名配对与四类错误路径；固定种子划分与逐字节可复现的划分文件。
- letterbox（双线性图像 / 最近邻掩膜 / 居中补零 / valid 掩膜）与二值化。
- 只统计有效区域的 BCE + soft Dice；逐图 Dice/IoU 与均值/中位数。
- 三个互不混淆的子命令；`train` 拒绝含 test 角色的划分；测试评估只读不写模型。
- 纯 Python 参考计算器 `verify_reference.py`（参数量 / 逐层 shape / MAC），并与
  torch 模型做参数量交叉验证。
- 训练集校准、对称 INT8 逐层导出、INT32 卷积和 Q24 黄金向量、FP32/INT8 软件对比。
- 92 项单元与合成数据端到端测试，已在本机 CUDA 版 PyTorch 环境运行通过。

**真实数据已验证**

- 官方 900 张训练图分 720/180，训练 40 轮，最佳 epoch 40，验证 Dice/IoU 为
  0.8660/0.7800。官方独立测试 379 张 FP32 Dice/IoU 为 0.8756/0.7985。
- 32 张训练图校准的 INT8 软件仿真：验证 Dice/IoU 0.8578/0.7683；官方测试
  Dice/IoU 0.8724/0.7944。15 层权重、偏置、黄金向量经 `verify_export.py` 复核。
- 完整实验来源、结果、限制与 B 组交接格式见 `deliverables/role_a/README.md`。

**仍未验证**

- 尚无 PYNQ-Z2 实测、资源/时序/功耗数据，INT8 结果仅为 CPU 软件仿真。
- `verify_reference.py` 给出的 MAC 与 INT8 权重字节数只是算术估计，**不是**实测资源占用。
- 未做：患者级分组划分、交叉验证、学习率调度、早停、类别重加权、形态学后处理、
  测试时增强、按原始分辨率还原后再评估（当前指标空间是 letterbox 后的有效区域）。
- 未做与论文数字的协议对齐，**不得**把本代码产出的分数直接与论文比较。

## 8. 参考与许可

- 代码为本任务独立实现，**未复制**任何第三方仓库；结构属于 U-Net 系列的一般设计
  （Ronneberger 等, *U-Net: Convolutional Networks for Biomedical Image Segmentation*, MICCAI 2015），
  仅沿用"编码器-解码器 + skip concat + max-pool/upsample"这一公开思路。
- 数据来自 ISIC Archive（<https://challenge.isic-archive.com/landing/2016/37/>）；
  其许可与引用要求需由使用者按该次下载的条款单独确认，本仓库不代为声明。
- 依赖库许可：PyTorch (BSD-3-Clause)、NumPy (BSD-3-Clause)、Pillow (MIT-CMU)。
