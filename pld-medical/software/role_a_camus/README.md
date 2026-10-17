# CAMUS 角色 A：灰度 4 类分割（准备阶段）

数据：**CAMUS_public 官方 NIfTI 版**（Girder collection `6373703d73e9f0047faa1bc8`），
许可 **CC BY-NC-SA 4.0（非商业科研）**，必须引用
**Leclerc et al., IEEE TMI 2019**。官方划分互斥：**400 训练 / 50 开发 / 50 测试**。
`half_sequence` 文件允许存在于患者目录，但**不参与枚举、训练与评估**；测试 GT 默认封条。

结构：

| 模块 | 职责 |
|---|---|
| `geometry.py` | 半像素中心重采样（灰度/ logits 双线性、标签最近邻）、letterbox/valid、原网格重建后 argmax、HD95(mm)、物理面积 |
| `data.py` | 官方名单强绑定（源 SHA 复核）、12 文件名严格解析、官方 `ImageQuality`、原生图像/GT 缓存、测试封条 |
| `model.py` | 单通道 → 4 类 U-Net（Conv/ReLU/MaxPool/**nearest 上采样**/concat/1×1 head，**无 ConvTranspose2d**，无 sigmoid）；参数/MAC/INT8/峰值激活预算 |
| `metrics.py` | 逐类 Dice/召回/覆盖、按帧再按患者等权 macro、逐帧 HD95/面积、低覆盖清单 |
| `train.py` | AdamW + CE/前景 Dice、AMP、native Dice 选模、完整断点（含 RNG/scaler/scheduler/patience/config/源码身份）、原子 best/last、STOP 有界响应 |
| `evaluate.py` | native GT 评估、官方 20-disk 双平面体积/EF（薄封装，数学由 `official_volume.py` 决定） |
| `official_volume.py` | 官方 LV 体积参考实现（**由 Codex 接管数学边界**） |
| `visualize.py` | ED/ES 成对：原生图 + GT（虚线）+ prediction（实线）+ 图例 |
| `prepare.py` | 端点枚举、QC、原生/letterbox 缓存与审计清单 |

## 运行命令（绝对路径）

准备训练/开发缓存（只读 train/dev GT，不碰测试 GT）：

```powershell
E:/嵌赛/pld-medical/.venv/Scripts/python.exe -X utf8 -m software.role_a_camus.prepare `
  --camus-root E:/嵌赛/pld-medical/datasets/camus_public/official/CAMUS_public `
  --out-dir E:/嵌赛/pld-medical/runs/role_a_camus/20261012/data/cache `
  --split training --size 256
```

训练（**必须先有 Codex 批准文件与通过 QC 的缓存**；本轮未运行）：

```powershell
E:/嵌赛/pld-medical/.venv/Scripts/python.exe -X utf8 -m software.role_a_camus.train `
  --approval E:/嵌赛/pld-medical/runs/role_a_camus/20261012/preparation/CODEX_CAMUS_TRAIN_APPROVAL.json `
  --data-manifest E:/嵌赛/pld-medical/runs/role_a_camus/20261012/data/cache/prepare_training.json `
  --out-dir E:/嵌赛/pld-medical/runs/role_a_camus/20261012/runs/baseline01 `
  --epochs 100 --batch-size 8 --patience 15 --device cuda
```

评估（冻结 best 后一次性完整评估；测试集需 owner 放行文件）：

```powershell
E:/嵌赛/pld-medical/.venv/Scripts/python.exe -X utf8 -m software.role_a_camus.evaluate `
  --checkpoint E:/嵌赛/pld-medical/runs/role_a_camus/20261012/runs/baseline01/best.pt `
  --data-manifest E:/嵌赛/pld-medical/runs/role_a_camus/20261012/data/cache/prepare_validation.json `
  --split validation `
  --out-dir E:/嵌赛/pld-medical/runs/role_a_camus/20261012/runs/baseline01/eval_dev --figures
```

合成 CPU 自检（不读真实数据）：

```powershell
E:/嵌赛/pld-medical/.venv/Scripts/python.exe -X utf8 -m unittest discover `
  -s E:/嵌赛/pld-medical/software/role_a_camus/tests -p test_camus_cpu.py -v
```

## 口径与边界

- 主指标：native 网格上三前景结构 Dice，按帧等权后按患者等权；同时报 LV/MYO/LA、2CH/4CH、ED/ES、
  官方 quality（Good/Medium/Poor，Poor 保留）、召回/低覆盖、spacing-aware HD95(mm) 与面积(mm²)。
- 空 GT 且空预测 → `null` 并计数；空 GT 但有预测 → Dice = 0（假阳性不计免）；预测空 → Dice/召回 = 0。
- 几何失败（掩膜空/非单连通、spacing 无效）→ `null` + 原因；**不挑选最大连通域**。
- LV 体积/EF 使用官方 20-disk 双平面方法，**不平均单视图体积**、**不把一个视图的 spacing 用于另一个视图**。
- 未量化、未板端验证，**不得声称临床精度或 PYNQ 适配**；Dice ≠ 发现率。
