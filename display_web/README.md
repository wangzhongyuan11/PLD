# PLD 网页与显示系统

本项目负责 PYNQ-Z2 皮损分割系统的网页操作、结果读取、轮廓处理、面积统计和 HDMI 显示。模型训练与 FPGA 加速器由同学的 PLD 工程提供，本项目调用其冻结运行包，不包含训练数据、模型权重或硬件比特流。

当前整合了固定网址、浏览器记住访问码以及开机自启。用户已反馈网页分割、HDMI 和重启后直接访问正常；具体测试范围见 [验证记录](docs/VALIDATION.md)。

## 功能

- 上传 JPG，调用板端 FPGA 完成分割。
- 查看原图、切割图、二值掩膜和轮廓描边，调整轮廓线宽。
- 显示面积占比、边界像素与后处理耗时，导出结果包。
- 本次开机历史：显示已完成结果的切割图预览，单独下载切割图或结果包；刷新浏览器和同次开机内重启服务后可恢复。
- 控制 HDMI 显示，支持开发板实体按键。
- 固定网址访问；可安装 systemd 开机自启服务。
- 电脑端提供历史真实掩膜回放，以及图片和配套掩膜的后处理测试。

轮廓处理不会自动修正模型分割错误；目前未实现去噪、填孔、阈值交互或训练。网页服务串行使用计算和显示硬件，不能一边推理一边连续 HDMI 输出。

## 电脑端运行

Windows 双击 `start_windows.cmd`。本机优先使用已有 Python 环境，其他电脑可安装依赖后运行：

```bash
python -m pip install -r requirements.txt
python server.py --open
```

电脑端不执行 FPGA 推理或驱动 HDMI。可以点击“加载真实结果示例”，或上传 `samples/synthetic_original.png` 和 `samples/synthetic_mask.png` 验证操作。合成图片不是模型预测。

## 板端使用

目录保持为 `/home/xilinx/jupyter_notebooks/PLD_Web`。部署步骤、开机自启和故障排查见 [部署说明](docs/DEPLOYMENT.md)。

已配置自启的板子直接打开固定网址：`http://192.168.2.99:8765/`，不要同时手动运行另一份网页或 HDMI 程序。首次使用新浏览器仍需终端/服务日志中的完整访问链接。固定网址的前提是板子 IP 保持不变。

## 文件说明

| 文件或目录 | 用途 |
|---|---|
| `server.py` | HTTP 后端、上传检查、任务管理与硬件互斥 |
| `static/` | 网页 HTML、CSS、JavaScript 和图标，无需 npm 构建 |
| `contour.py` | CPU 轮廓提取，可独立使用 |
| `pld_result.py` | B→C 输出合同验证、掩膜回映射及统计 |
| `infer_then_display.py` | 运行包校验与推理命令构造，也保留独立命令入口 |
| `pld_hdmi.py`、`hdmi_buttons.py`、`hdmi_worker.py` | HDMI 输出、按键和网页控制子进程 |
| `show_result.py` | 独立查看已保存结果；也是现有模块的导入依赖 |
| `start_windows.cmd`、`start_board.sh` | 电脑与板端手动启动 |
| `install_autostart.py` | 板端自启安装与检查 |
| `tests/` | 网页后端、轮廓和访问码保存测试 |
| `samples/` | 小型真实结果与合成测试样例 |
| `docs/` | 部署、模型交接与验证说明 |

## 本次开机历史

历史只显示当前操作系统启动后，由本版本完成的任务；Linux 使用 boot_id，Windows 使用系统启动 GUID。操作系统重启后列表为空，旧版未写入历史索引的结果不会自动导入。历史不包含失败或尚未完成的任务。

现有 data 中的旧文件不会自动删除，因此“本次开机”指列表和访问范围，而不是自动清空磁盘；需要长期连续使用时应另定磁盘清理策略。访问码文件始终保留。此版本仍保留原流程生成的图片和报告，结果包内容不变。

## 测试

在项目根目录运行：

```bash
python -m unittest discover -s tests -v
```

如有 Node.js，可额外运行浏览器访问码逻辑测试（运行网页本身不需要 Node.js）：

```bash
node tests/test_browser_access.cjs
```

## 上传 GitHub

上传本目录的源码及文档即可；无需上传桌面旧压缩包。`.gitignore` 已排除运行生成的 `data/`、访问码、缓存、虚拟环境、日志与本地环境文件。GitHub 网页手动上传不执行忽略规则，因此也应只选择本整理目录，不混入板端 data 文件夹。

若合并到同学的 PLD 仓库，建议放在独立子目录 `display_web/`，部署到板子时仍使用上面约定的 `PLD_Web` 路径。本次整理没有创建远程仓库、提交或上传 GitHub，也未更改开发板上的文件。
