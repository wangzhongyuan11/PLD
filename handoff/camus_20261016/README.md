# CAMUS 接手入口（2026-10-16）

本次按用户要求在最近完成的候选版本收尾。2026-10-16 未连接板卡，未继续优化或追加硬件测试。目标是交给下一位开发者继续处理 CAMUS 的软件性能；**3 秒端到端目标尚未通过**。

GitHub 分支：`codex/camus-handoff-20261016`。现有远端为 `wangzhongyuan11/PLD`；本交接不覆盖远端 main。

## 先选版本

| 版本 | 完整运行包 | 真实板测范围 | 结果 |
|---|---|---|---|
| 上一采用版 | `runtime/stable_20frames.tar.gz` | 同一批 20 张官方开发图，HTTP 上传至 ZIP 可下载 | 中位 3.911460 s，P95 4.378222 s，最慢 4.479964 s，20/20 小于 5 s |
| 最新继续优化候选 | `runtime/candidate_3frames.tar.gz` | 3 张代表图短测，含服务重启后的首帧 | 4.855870 / 3.649589 / 3.870941 s；不能继承上一版的 20 图验收 |

两个版本的 head、原生标签、处理后标签、四张 PNG 解码像素及 ZIP CRC 均在各自测试范围内通过。数据来自开发集，不称独立临床测试。

运行包文件名相对于本地完整交付文件夹。GitHub 只提供源码、交接说明和小型证据；模型、原始样例和二进制运行包留在本地完整交付包，不混入 Git 历史。

## 完整交付文件夹

主机路径：`D:\PLD\out\handoff\CAMUS_20261016`，同级 `.zip` 可直接转交。

| 路径 | 内容 |
|---|---|
| `README.md` | 本接手入口 |
| `NEXT_STEPS.md` | 已实现、耗时解释、剩余优化和验收顺序 |
| `VERIFY.py`、`MANIFEST.json` | 离线逐文件 SHA256 校验 |
| `runtime/` | 精确保留的 20 图采用版与 3 图最新候选，不重新打包 |
| `source/` | 当前源码快照；不含 `.git`，不纳入用户无关改动或已删除的旧 delivery 文件 |
| `source/camus_candidate_hidden_histogram_int8_20261015/` | 原始冻结模型交付，放在源码默认可识别的位置 |
| `source/out/camus/vivado/`、`source/ip_repo/camus/` | 相同构建的 bit/hwh/XSA、报告、构建证书及打包 IP，支持不重建 RTL 的软件再打包 |
| `validation/dev20/` | 与上一采用版完全相同的 20 图输入、期望结果和哈希 |
| `validation/quick3/` | 与最新候选完全相同的 3 图短测样例 |
| `evidence/` | 原始 HTTP 测试 JSON、构建/部署证据、主机离线复核日志 |
| `dependencies/` | nibabel 5.2.1 离线 wheel；其余板端依赖及版本见下文 |

先运行 `python VERIFY.py`。源码快照可用于阅读和测试；需要 Git 历史、提交或再次调用 `camus.package` 时，优先克隆 GitHub 交接分支，再将模型及已认证的 `out/camus/vivado`、`ip_repo/camus` 从交付包复制过去。快照本身不伪造 Git 历史。

构建证书按文件原始字节校验，Git 行尾转换会影响哈希。沿用现有位流重新打包时，还应从 `source/` 快照复制证书所绑定的构建源码，保留其编码/行尾；不要改写 manifest 绕过校验。

## 本地检查与板端运行

主机 Python 3.13 / NumPy 2.3.5 环境：

```powershell
Set-Location <交付文件夹>/source
python -m unittest discover -s camus/tests -p 'test_*.py' -q
```

主机测试覆盖参考/分块整网的 15 层输入输出、预处理、后处理、接口与 ZIP。原生 C 测试类在 Windows 跳过；不能将其解释为硬件通过。

板端实测环境：PYNQ-Z2，PYNQ 3.1.1，Python 3.10.4，NumPy 1.21.5，SciPy 1.8.0，Pillow 9.0.1；NIfTI 使用隔离的 nibabel 5.2.1；原生 C 首次启动需要 gcc、libm。原生编译关闭 fast-math/FMA 合并，并按源码/选项/架构哈希缓存共享库。

板卡当前不在线，以下是接手后执行的命令，不是本次执行记录。认证由接手人通过本机环境或交互配置；交付包不带口令或私钥。

```bash
mkdir -p /home/xilinx/jupyter_notebooks/pld/camus_handoff_candidate_20261016
tar -xzf candidate_3frames.tar.gz -C /home/xilinx/jupyter_notebooks/pld/camus_handoff_candidate_20261016
cd /home/xilinx/jupyter_notebooks/pld/camus_handoff_candidate_20261016
# 同一时刻只允许一个 DMA 拥有者；先停两个网页服务。
sudo systemctl stop pld-camus.service pld-web.service
sudo -E /usr/local/share/pynq-venv/bin/python3 -m unittest camus.tests.test_optimization.NativeOptimizationTests -q
sudo -E /usr/local/share/pynq-venv/bin/python3 -m camus.web --backend pl --runtime . --host 127.0.0.1 --port 8090 --data /home/xilinx/jupyter_notebooks/pld/camus_handoff_data_20261016
```

NIfTI 命令需让 `PYTHONPATH` 指向现有隔离依赖目录，或先把提供的 nibabel wheel 安装到新的隔离目录。SSH 自动化使用登录 shell 初始化 XRT；已有入口为 `scripts/board_ssh.py run --sudo`。

另一个主机终端建立隧道，运行精确同组验证：

```powershell
python -m camus.tunnel --local-port 8091
# 在另一个终端执行，--dataset 指向交付包内 dev20。
python -m camus.board_web_verify --url http://127.0.0.1:8091 --dataset <交付文件夹>/validation/dev20 --output <新的结果文件>.json --frames 20
```

不要并行运行独立板测与网页推理。所有硬件等待必须有界，异常恢复后才能释放活动 DDR。不要通过逐层注入 golden 替代首层输入的整网验证。

## 上次观察到的部署和回退

以下为 2026-10-15 最后一次已观察状态，2026-10-16 未重新确认：

- `pld-camus.service` 指向 `/home/xilinx/jupyter_notebooks/pld/camus_ps_opt_v2_preconv`，板端 loopback 8090。
- 原始 unit 与上一采用版 `camus_ps_opt_release_20261015` 保留；共享历史数据在 `camus_web_data_20261015`。
- 30 号 drop-in 为上一采用版；40 / 50 / 60 号为本轮候选；具体文件名见 `NEXT_STEPS.md`。
- RX 非缓存实验已经回退，失败实验的 drop-in 已移除；最新候选重新使用可缓存 RX。
- 回到 20 图采用版：停服务后，仅移走本轮 40 / 50 / 60 三个候选 drop-in，保留 `30-ps-opt.conf`，再 daemon-reload 并启动。先核查实际文件，勿按旧记录批量删除其他配置。
- 原 ISIC 服务开关仍为 `systemctl start pld-web.service`；不能与 CAMUS 同时拥有 PL。
- 无显示器，物理 HDMI 画面未验收。未追加千帧测试。

本轮保持 CAMUS 及共享 RTL、100 MHz bit/hwh/XSA 不变；不重新综合，也不借用 ISIC 的 125 MHz 时序证据。
