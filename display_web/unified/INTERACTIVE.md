# ROI、点提示、框提示交互补丁（2026-10-16）

适用于已安装双模型统一入口与四宫格的 PYNQ 系统。固定网址仍为 http://192.168.2.99:8765/ 。安装后主页新增“打开 ROI / 点提示 / 框提示交互工作台”。交互页在同一网站的新标签页打开，使用主页当前选中的模型和登录会话。

## 模型核对：无需重复替换权重

此次 Role A 包明确说明未进行训练、校准或重新导出。皮肤镜 spec SHA256 为 fe137ca5b128fc1983fabad9bee39b3208cda4256e0d04c4e27cef5c07666aef，与现有模型一致。CAMUS 当前候选 spec 为 68b4f425cc3612ad72d5e83955c2018981edee912fbc0937e9ac8c1718e7b5ae，也与现有模型一致。包中的 CAMUS 原交付版（26e20066…）不是当前候选，不覆盖它。

仅接入有用的交互逻辑。两个网络均不是点/框条件网络：ROI 在推理前裁剪，重新调用原 FPGA 服务；点、框和阈值使用 FPGA 产生的概率，在 PS/CPU 上图割修正。不能称为“提示直接输入 FPGA 网络”或“PL 图割”。既有位流、模型权重、DMA 驱动和四宫格源码不修改。

## 上传与安装

只上传 PLD_Interactive_20261016_v4.zip 到 Jupyter 根目录。无需删除旧文件，也不要上传整个468MB的 Role A 包。

```bash
cd /home/xilinx/jupyter_notebooks
unzip -n PLD_Interactive_20261016_v4.zip
cd PLD_Interactive_20261016_v4
sudo /usr/local/share/pynq-venv/bin/python3 install_interactive.py --apply
```

安装器先核对现有文件与补丁哈希，并运行微型图割检查 NumPy/Pillow/SciPy 能力。依赖缺失或文件版本不同会在修改前停止。不能跳过该错误强行覆盖，保留终端输出进行核查。

安装器正常停止统一入口和两个后端、确认释放，然后备份并更新9个文件。看到 INTERACTIVE_PATCH_APPLIED 后等待模型初始化，访问原网址并 Ctrl+F5。无需再次运行旧启动脚本。

## 操作

1. 在主页选择皮肤镜或 CAMUS，等就绪后进入交互工作台。
2. 上传 JPG/PNG。交互页长边最多1024像素，模型仍采用原256×256输入。界面坐标/面积对应缩小后的处理图；本页不提供 mm²，以免丢失原始 spacing 后产生虚假物理单位。
3. 可先选 ROI，在原图拖动。更新 ROI 清空旧点和框，区域外标签为背景。
4. 选择前景点/背景点后单击；选择目标框后拖动。点和框需在 ROI 内。CAMUS 可选择左室腔、心肌、左房。
5. 点击“运行 / 更新分割”。同一图片/ROI/模型代次复用概率，修改点或阈值不会再次运行整网；更改 ROI 或模型后重新 FPGA 推理。
6. 四宫格显示原图、增强、所选结构掩膜、叠加。下载交互 JSON（含原图、增强图、标签/掩膜/叠加 PNG 数据及参数）或单独掩膜 PNG。

## 边界与内存

- 只保留一个概率缓存和最近一次交互结果；切换模型清空它。交互执行中拒绝模型切换与普通页的新提交/HDMI控制。
- 图割网格最长边256；近邻矛盾点会报错，建议缩小 ROI 或移动点。提示不保证提高 Dice。
- 皮肤镜现有 FPGA 上传接口只接受 JPEG，ROI 会以 quality100、4:4:4 编码；这并非无损。因此不声称交互入口无提示时与原PNG输入逐像素一致。CAMUS ROI 转换为8位灰度后使用无损PNG；原图为灰度时保留相同像素，彩色输入会发生灰度转换。
- 当前交互入口只接 JPG/PNG；原页面继续保留原医学文件格式。
- 普通分割历史、原ZIP和HDMI仍是后端原始结果；**没有把交互修正覆盖成原始模型结果，也没有接入交互结果HDMI**。交互结果独立下载，防止混淆。
- 后端任务首次需要等待完整原结果生成，当前重点为正确接入，不保证交互首帧速度优化。
- v4 已部署并完成两模型真实 PL 短测；本地49项测试通过。未验收峰值内存、长期稳定性、物理HDMI、交互后精度或浏览器/局域网完整延迟。

## 上传后验收

分别测试两模型：无提示、ROI、前景点、背景点、目标框、撤销/清除、改变类别。确认点处标签符合要求，ROI/框外约束正确；修改提示显示复用概率；更换ROI显示不复用。完成后双向切换模型，再检查普通四宫格和HDMI仍能运行。若模型初始化/内存/图割报错，保留终端和页面错误，不强制杀死硬件进程。

## 回退

安装时会打印 PLD_DualModel/backup_interactive_时间戳。使用该真实路径：

```bash
sudo /usr/local/share/pynq-venv/bin/python3 install_interactive.py --rollback /home/xilinx/jupyter_notebooks/PLD_DualModel/backup_interactive_实际时间戳
```

只有文件仍匹配已安装/备份版本时才会回退；不会删除模型或数据目录。

## 文件筛选

采用：software/interactive_segmentation/contracts.py（参数/坐标）、refinement.py（种子图割）、service.py（ROI/概率/类别处理，已替换推理适配），以及对应API文档作为依据。
新增：interactive_bridge.py（调用原板端推理、恢复概率、缓存、互斥）、interactive.html/js（现有网站交互四宫格）、安装与回滚脚本。
不采用：电脑端 models.py/PyTorch模型加载器、server.py、独立旧网页、训练代码、开发图像/参考标签、原始重复ZIP、旧CAMUS权重。原始压缩包保留在下载目录，不做删除。

来源：用户提供的 role_a_complete_interactive_20261016.zip。保留原算法代码说明；service.py仅适配当前PL入口和现有CAMUS后处理。

## v2 兼容修复

兼容旧 Pillow 的 Image.BILINEAR/NEAREST 常量；兼容不接受 method 关键字的旧 SciPy maximum_flow。无需升级板端系统依赖。v1失败发生在预检阶段，未覆盖旧服务。v2采用独立解压目录，避免 unzip -n 保留旧文件。

## v3 最大流返回值兼容

旧SciPy把有符号流矩阵命名为residual，新版称flow。已依据SciPy v1.5.4官方源码确认语义，统一用容量减流构造剩余容量图。测试同时模拟旧Pillow、无method关键字及仅有residual返回字段，运行真实Edmonds-Karp求解器。无需上传模型或升级系统。


## v4：灰度输入、快速编码与完整计时（已部署）

修复浏览器RGB画布导致CAMUS拒绝输入的问题：发送后端前明确转换为L模式。回归测试把实际POST中的PNG送入冻结CAMUS读取器，避免仅模拟成功结果掩盖格式错误。

五张交互PNG采用与现有CAMUS相同的Sub过滤、RLE/level1无损编码；原图和增强图保留单项缓存，同图的点、框、阈值及ROI更新可复用。切换模型清除缓存。压缩策略不改变像素，当前皮肤镜测试的总PNG字节数从约510KB增加至625KB，换取板端编码时间下降。

页面以普通文本显示请求校验/解码、概率获取、类别及后处理、PS图割、绘制统计、三张结果PNG、原图/增强PNG。概率明细另列输入编码、后端等待及概率恢复；明细已包含在概率获取阶段，不能重复相加。服务端合计包含全部五张PNG与base64，不包含HTTP结果JSON传输、浏览器轮询和绘制；页面另外显示提交到接收结果的墙钟时间。

2026-10-16真实PYNQ短测：皮肤镜ISIC_0000003处理为512×384，CAMUS patient0200_2CH_ED处理为512×423；每种情况一次，测试程序通过板端回环HTTP、100ms轮询收取完整结果。

| 场景 | 皮肤镜优化前 | 皮肤镜v4 | CAMUS v4 |
|---|---:|---:|---:|
| 首次整图请求（模型已经就绪） | 6.172s | 4.240s | 7.093s |
| 复用概率，前景/背景点和框修正 | 5.230s | 2.850s | 2.950s |
| 改变ROI，重新运行PL | 未测 | 3.325s | 5.119s |

皮肤镜前两种情况的五张输出图逐像素相同；两模型均检查种子点、框外背景、ROI外背景、缓存命中/失效。四个新生成的普通结果ZIP CRC通过；CAMUS两张普通四宫格均为640×480。没有将这些结果称作批量性能或医学精度验收。CAMUS v3存在输入错误，没有可比较的成功交互基线。

复测命令：将仓库`scripts/verify_interactive_board.py`上传至板端，以PYNQ Python执行`--model skin|camus --output <结果.json>`；跨模型加`--switch`，旧版对比加`--baseline`。脚本读取原服务依赖路径、复用现有访问码，只将计时与像素哈希写入记录，不导出图像或访问码。

实际回退备份：`/home/xilinx/jupyter_notebooks/PLD_DualModel/backup_interactive_1746558739280287258`。使用v4安装器按上述回退命令执行。完整证据在仓库`docs/validation/interactive_20261016.json`，摘要见`TEST_REPORT.md`。
