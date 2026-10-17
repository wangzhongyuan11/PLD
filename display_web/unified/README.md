> 当前唯一正式版本为 pld_dual_20261017，部署/回退和验证入口见仓库根 README 及 docs/DEPLOYMENT.md。本页独立应用/补丁步骤保留作模块说明，旧目录名和单模型启动方式不作为正式系统入口。

# 双模型统一网页（2026-10-16）

在同一个网页选择「皮肤镜」或「CAMUS」。沿用各自的上传、推理、结果预览、下载和 HDMI 界面；上方增加共同的模型按钮。第一版采用统一外框嵌入原界面，没有重新编写两套结果展示。

## 从仓库生成上传包

在仓库根目录运行，只需要 Python 标准库：

```bash
python -m unittest discover -s display_web/unified/tests -v
node display_web/unified/tests/test_ui.cjs
python scripts/package_dual_model_web.py
```

生成 `out/PLD_DualModel.zip`，只包含统一入口源码、说明和测试，不包含原模型、上传图像、访问码及安装状态。ZIP 内保留 `PLD_DualModel/` 目录，以下板端安装命令不变。无需重新构建 RTL。

安装器同时记录已审查网页源文件的 LF/CRLF 两种精确哈希，兼容 Git 源码与历史 Windows 交付换行；不会跳过内容检查，也不改变冻结运行包的 manifest。

## 先看这几点

- **只上传 `PLD_DualModel.zip`，不删除原 PLD_Web、CAMUS 运行包、权重、bit/hwh、数据目录或已有服务文件。**
- 需要板子已经安装 `pld-web.service` 和 `pld-camus.service`。安装器自动读取当前实际工作目录与模型路径，不写死候选版本。
- 皮肤镜需为已交付的「预览优先 v2」版本；CAMUS 支持交接包内上一采用版与最新候选的网页接口。安装器检查 SHA256；版本不同会停止，未通过时把完整输出发回来，不要绕过检查。
- 本包不重复提供模型和位流，不修改 RTL。原代码不会被覆盖；安装器新增两个 service drop-in 和一个统一入口服务。
- 本地验证及一次 Role-A NPZ 真实 PL 分割通过；详细范围见 `TEST_REPORT.md`。物理 HDMI 和多轮切换后的 CMA 稳定性仍需单独验收。

### Role-A 缓存 NPZ

支持交接代码 `role_a_camus/data.py` 生成的多数组缓存：读取 `native_image`（float32、0～1、原尺寸灰度）与 `spacing_mm`（Y、X），忽略标签、已缩放的 `image` 和对象元数据，始终禁用 pickle。与原灰度读取接口衔接时还原到 0～255 强度域，不把标签作为模型输入。

输入校验在停止 HDMI、重新初始化 FPGA 和推理之前完成。格式、数值或像素间距错误只拒绝本次输入，不将模型标为硬件故障；真正的推理/硬件异常仍保留故障锁定。

## 上传与安装

1. 在网页等待当前任务及完整结果包完成，下载需要保留的结果。关闭 HDMI 显示。
2. 在 Jupyter 文件首页 `/home/xilinx/jupyter_notebooks` 上传 `PLD_DualModel.zip`，完成 Upload 确认。
3. 打开 Jupyter Terminal，逐条运行：

```bash
cd /home/xilinx/jupyter_notebooks
unzip -n PLD_DualModel.zip
sudo systemctl stop pld-web.service pld-camus.service
sudo systemctl show pld-web.service pld-camus.service -p Id -p ActiveState -p MainPID --no-pager
```

两项都应为 `ActiveState=inactive`、`MainPID=0`。如果仍在停止或报错，先查看日志，不要同时启动另一个模型。

```bash
cd /home/xilinx/jupyter_notebooks/PLD_DualModel
sudo /usr/local/share/pynq-venv/bin/python3 install.py
```

出现 `INSTALL_PASS` 后执行：

```bash
sudo systemctl start pld-hub.service
sudo systemctl status pld-hub.service --no-pager -l
```

如果提示 CAMUS 服务尚未安装，先提供下面命令的输出，以现有部署路径完成适配；本包不会猜测路径或下载另一套权重：

```bash
sudo systemctl show pld-camus.service -p LoadState -p WorkingDirectory -p ExecStart --no-pager
```

## 打开和切换

原固定网址保持为：

**http://192.168.2.99:8765/**

前提是板子 IP 仍为 `192.168.2.99`。原带 `#访问码` 的完整链接也能使用。通常浏览器会复用已经保存的访问码；若出现输入框，填原皮肤镜访问码。忘记访问码时，可在板子运行下面命令查看（不要把码公开到 GitHub）：

```bash
cd /home/xilinx/jupyter_notebooks/PLD_DualModel
sudo /usr/local/share/pynq-venv/bin/python3 -c "import json,pathlib; c=json.load(open('config.json')); print(pathlib.Path(c['token_file']).read_text().strip())"
```

首次安装默认启动皮肤镜；后续开机恢复上次成功选择的模型。状态变为「已就绪」后上传图片。点击另一个模型按钮即可切换，顺序为：

1. 拦截新的上传，等待当前推理和完整结果包生成结束。
2. 正常停止 HDMI，退出旧模型及推理子进程，检查服务和进程组已停止。
3. 启动目标模型并初始化 FPGA；准备好后显示该模型的原网页。

切换期间不会同时预热两个模型，切回也需要重新初始化。切换耗时不能套用重复推理的 3～4 秒数据。

切换会重载操作页面。皮肤镜原有本次开机历史机制保持不变；CAMUS 原程序只在进程内保存结果索引，切换后原页面入口不保留，请提前下载。磁盘上的原结果文件不删除。

开机只自动启动 `pld-hub.service`，由它选择启动一个后端；原两个后端不再各自自动启动。**安装后不要另开 Terminal 手动运行 server.py、camus.web 或其他 FPGA 推理/显示程序。** 共用锁和服务互斥保护的是这个统一入口管理的服务。

## 板端验收

建议按「皮肤镜 → CAMUS → 皮肤镜 → CAMUS」测试至少两轮，每次上传对应类别的图片，检查预览、下载包和计时。需要 HDMI 时，每个模型分别显示一次，再切换；查看是否正常退出而非强制结束。

网页顶部「模型切换详情」有切换前、旧模型退出后、目标模型就绪后的 `MemAvailable`、`CmaFree`，便于观察多轮切换是否持续下降。它们是整板可用内存指标，不是单个模型内存；Linux 文件缓存也不等于泄漏。实际 DMA/CMA 回收仍需板端结果确认。

读取诊断信息（不执行推理、不切换、不输出访问码）：

```bash
cd /home/xilinx/jupyter_notebooks/PLD_DualModel
sudo /usr/local/share/pynq-venv/bin/python3 diagnose.py
sudo journalctl -u pld-hub.service -u pld-web.service -u pld-camus.service -n 100 --no-pager
```

停止超时、进程异常或 HDMI 未释放时，统一入口拒绝加载另一个模型。保留报错和日志后检查，不自动强杀或无限重试。

## 回退

先等待任务完成，然后运行：

```bash
sudo systemctl stop pld-hub.service
sudo systemctl show pld-hub.service pld-web.service pld-camus.service -p Id -p ActiveState -p MainPID --no-pager
cd /home/xilinx/jupyter_notebooks/PLD_DualModel
sudo /usr/local/share/pynq-venv/bin/python3 install.py rollback
sudo systemctl start pld-web.service
```

需要三项服务都正常 inactive。回退仅删除本安装器新增且校验未变的配置，恢复原两个服务的开机启用状态，不删除数据和模型。成功后皮肤镜恢复原来的端口配置。安装中断也可用同一个 rollback 命令恢复已写配置。

不要覆盖解压正在运行的统一入口目录；更新前先停止并回退。`installation.json` 是安装归属及回退记录，请保留。

## 代码文件

| 文件 | 用途 |
|---|---|
| index.html | 统一外框、两个模型按钮、状态和登录 |
| hub.py | 8765 入口、认证、地址代理、模型切换状态机 |
| backend_launcher.py | 共用 FPGA 锁、皮肤镜推理进程正常 EOF 退出、CAMUS 任务状态适配 |
| camus_input_adapter.py | Role-A 缓存 NPZ 读取及推理前输入校验，不修改冻结运行包 |
| install.py | 现有部署识别、版本检查、可回退的 systemd 配置 |
| accepted_backends.json | 已审查网页/适配接口的版本哈希 |
| diagnose.py | 读取服务、内存及最近切换状态 |
| tests/ | 本地模拟测试，不替代真实板测 |

新增网关会增加一次板内 HTTP 转发；本版未给出其真实板端性能测量。没有改动模型算法、权重、输入预处理或推理精度。未合并两个模型的结果定义：CAMUS 保持多类别结果，皮肤镜保持二值掩膜与轮廓结果。


## 2026-10-16 加速补丁

CAMUS 在校验后将已解码输入交给同一任务的预处理，避免重复解压/读取；任务结束即释放。切换时缩短就绪状态轮询间隔，在“模型切换详情”中增加等待任务、释放旧模型、初始化新模型三段耗时。仍然同一时间只运行一个模型，不跳过硬件释放检查。实测收益和限制见 TEST_REPORT.md。


## 四宫格展示补丁

2026-10-16新增网页及HDMI四宫格。现有双模型部署使用 `scripts/package_quad_view.py` 生成的专用补丁安装；步骤及验证范围见 [QUAD_VIEW.md](QUAD_VIEW.md)。该补丁同时更新皮肤镜网页/显示子进程和统一目录中的CAMUS显示适配器，不覆盖冻结CAMUS源码或位流。


当前板端已安装四宫格与交互v4；交互入口、更新/回退与计时口径见[INTERACTIVE.md](INTERACTIVE.md)，最新真实板测记录见[TEST_REPORT.md](TEST_REPORT.md)。
