# A02: 小病灶后训练的基础代码任务（供 dsh / deepseek-flash）

你是受监督的基础代码实现者。工作目录 `E:/嵌赛/pld-medical`。主工程师负责数据选择、实验设计、代码审查、训练、统计检验与最终结论。不要自行下载数据、启动训练、修改既有实验结果或宣称提升。

## 当前事实

- ISIC 2016 Task 1 已有 900 对训练、379 对官方测试，TinyUNet 输入 RGB 256、base_channels=8、三次池化，旧权重 `runs/role_a_tinyunet/best.pt`。原网络与 INT8 接口须保持不变。
- 旧 `data.py` 的面积分层存在错误：五档验证数量为 90/90/0/0/0。不得在新工作中复用此划分逻辑或覆盖历史 `split.json`。
- 用户先关注按病灶面积分组的小病灶识别；诊断类别是后续任务。不能用官方测试集选超参数。
- `data/raw/isic2017_validation/downloads/validation_masks.zip` 是官方 ISIC 2017 Task 1 验证掩膜，150 个 ID，其中 3 个与 2016 train 重合，另 147 个新 ID。官方 API 可按 ID 给出 full JPEG URL 与 size。

## 你的**唯一**任务：写安全的外部数据清单/下载辅助模块

新建 `software/role_a/external2017.py` 与 `software/role_a/tests/test_external2017.py`。不要修改其他文件。实现两个 CLI 子命令：

1. `survey`: 读取本地官方掩膜 ZIP，安全解析 `ISIC_########` ID；对比已有 2016 train/test ID；仅对无重叠 ID 调用官方 `https://api.isic-archive.com/api/v2/images/{id}/` 查询 full JPEG URL、size、license；输出确定性 JSON 清单、各类数量和预计总字节。查询可并发但最多 4 路，超时和失败要清楚报错。只接受 HTTPS 且图片主机为 `isic-archive.s3.amazonaws.com`，不得盲目信任 JSON URL。
2. `download`: 只依据 survey 清单逐个下载 full JPEG；默认总下载上限 200 MiB，可用更小的 `--max-bytes`；如果清单总量超过上限，**在下载任何字节前失败**；下载与掩膜都只能写在 `data/raw/isic2017_validation/` 下。原子写 `.part` 后核对长度、JPEG 解码、尺寸与对应掩膜相符，再改名；计算每张 SHA256 与总实际字节，写 provenance JSON；可恢复中断，已存在文件先重新验证。不要清理、移动或改写 2016 文件。

## 安全与质量

- ZIP 路径须拒绝 `..`、绝对路径、重复 ID、非 PNG 等，不可直接 extractall。
- 只能查询 ZIP 里出现的 ID；下载时只允许清单中通过 survey 验证的官方 URL。
- API 数据与网络不可信：校验响应字段类型、非负 size、license；不要记录令牌、环境变量或私人信息。
- 测试用内存 ZIP 和虚构 API 响应，至少覆盖重叠过滤、重复/路径穿越拒绝、超过容量上限拒绝、非法 URL 拒绝、清单确定性。不做真实网络测试。
- 保持函数小而清晰，使用 Python 3.10+ 标准库与 Pillow；若依赖缺失要报明确信息。
- 完成后列出修改文件与**实际**运行的测试。若当前环境文件写入受限，输出可审查的完整补丁，不要假称已经写入。
