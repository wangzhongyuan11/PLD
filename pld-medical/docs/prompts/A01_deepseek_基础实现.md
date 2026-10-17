# A 组基础代码实现任务

你通过 dsh 在 `E:\嵌赛\pld-medical` 工作。请只创建或修改 `software/role_a/` 下的文件，不改动已有文档、教学代码、其他组目录或原始数据。这里已有早期传统图像处理基线；本任务是新增独立的 ISIC 2016 轻量 U-Net 软件基线。不要下载数据、安装依赖、训练模型、声称达成精度或 FPGA 性能。

## 背景与目标

- 目标板卡 PYNQ-Z2；模型最终须由 B 组实现 INT8 RTL。此阶段写可训练的 FP32 基线与可复现的数据划分。
- 数据采用官方 ISIC 2016 Task 1：训练集 900 张 JPEG/对应 PNG 掩膜；官方测试集 379 张。官方测试集只用于最终评估，不参与训练或调参。
- 输入为 RGB、256×256。保持原图宽高比，缩放后居中补零；图像用双线性插值，掩膜用最近邻。输出一个 `valid` 掩膜标出非补边区域。掩膜二值化为 0/1。
- 建议网络为 Tiny U-Net，基通道 8，最多三次 2×2 MaxPool；仅用 Conv 3×3、Conv 1×1、ReLU、MaxPool、nearest Upsample、Concat。不要 BatchNorm、注意力、LayerNorm、转置卷积或复杂激活。输出 logits，不在模型内做 sigmoid。

## 交付文件

1. `software/role_a/model.py`：清晰可读的 TinyUNet，支持 `base_channels=8`，输入 `[N,3,256,256]`，输出 `[N,1,256,256]`；包含参数量查询或便于查询的接口。
2. `software/role_a/data.py`：匹配官方文件名（图片 `ISIC_*.jpg`、掩膜 `ISIC_*_Segmentation.png`）；对缺失、重复、无法配对文件报错。固定种子的训练/验证划分，并把排序后的文件名和划分写入 CSV 或 JSON 以便复现。不要碰官方测试集。
3. `software/role_a/train.py`：CLI 参数指定数据目录、输出目录、epoch、batch、学习率、seed、device。训练损失为只统计有效区域的 BCEWithLogits + soft Dice；验证记录每张图的 Dice/IoU，再计算均值；保存最佳验证 checkpoint 与配置、划分、每轮指标。避免加载任意来源 pickle。测试集评估应是显式单独命令或参数，绝不参与最佳模型选择。
4. `software/role_a/README.md`：给 Windows PowerShell 用户的准确运行命令、目录布局、数据源 URL、依赖、输出说明、已实现与未验证事项。写明仅用于科研/教学，不用于临床诊断。
5. `software/role_a/tests/`：少量真正能抓住错误的测试，至少验证图像/掩膜配对、同一 seed 划分一致、letterbox 的 mask/valid 对齐、网络输入输出 shape、Dice/IoU 边界情形。

## 质量约束

- 类型注解、明确异常、固定随机种子、无硬编码个人路径。
- 不直接复制第三方仓库代码；如借鉴请在 README 中说明来源及许可证。
- `torch`、`Pillow`、`numpy` 可作为依赖。测试若依赖缺失应明确提示，不能假装通过。
- 不要编造训练结果、模型权重、INT8 结果或资源占用。
- 交付后简短列出创建文件、实际运行的命令和结果、未运行原因。

完成代码即可；我会独立检查数值行为、数据泄漏、硬件可映射性和测试覆盖，再做量化与黄金向量实现。
