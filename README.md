# PLD：PYNQ-Z2 双模型分割系统

唯一正式交付为 **pld_dual_20261017**，入口清单见 [delivery/current.json](delivery/current.json)、[交付与复现说明](delivery/pld_dual_20261017/README.md)。包含 ISIC 皮肤镜、CAMUS 心脏超声、统一网页和交互分割工作台。旧阶段仅作为历史证据，不是并列交付。

系统用于工程教学和算法演示。数据来自公开开发数据；开发集结果不称独立测试结果，不用于临床诊断。

## 获取与准备

克隆本仓库，从 [唯一正式 Release](https://github.com/wangzhongyuan11/PLD/releases/tag/pld-dual-20261017) 下载 `pld_dual_20261017_assets.zip`，放入 `out/`。这是模型、冻结运行时、配对硬件证据及小型公开回归样本，不能用私人备份替代。

```powershell
python -m pip install -r requirements-reference.txt -r camus/requirements.txt nibabel==5.2.1
python scripts/release_assets.py out/pld_dual_20261017_assets.zip --prepare
python scripts/verify_delivery.py
python -m camus --plan
```

主机量化 JPEG 复算需 Pillow 12.3.0/JPEG8；已有 Conda Pillow12/JPEG9 不可冒充像素一致。板端使用原 PYNQ 环境，不直接安装主机 requirements。完整 FP32 评估需另外按 `requirements-evaluation.txt` 准备 Torch；正常板端推理不需要 Torch。

## 目录

| 路径 | 内容 |
|---|---|
| `rtl/`、`camus/rtl/` | 两套冻结计算核心；`conv_tile_engine.sv` 是皮肤镜必需模块 |
| `board/`、`reference/`、`camus/` | 有界 DMA/驱动、定点参考、调度与验证 |
| `display_web/`、`display_web/unified/` | 普通网页、统一入口、ROI/点/框/阈值交互与四宫格 |
| `sim/`、`camus/tests/`、`display_web/*/tests/` | 数值、协议、服务与交互回归 |
| `vivado/`、`scripts/` | IP/BD 构建、校验、模型准备、部署与板测工具 |
| `delivery/` | 当前版本指针、严格白名单 manifest、小型证据与部署模板 |
| `docs/` | [架构](docs/IMPLEMENTATION.md)、[部署](docs/DEPLOYMENT.md)、[验证](docs/validation/CLEANUP_20261017.md)、历史与需求 |
| `role_a_final/`、`role_a_handoff/`、`camus_candidate_hidden_histogram_int8_20261015/` | 准备命令恢复的模型与旧回归 fixture，忽略展开文件 |
| `数据/` | 本地原始数据及迁入 `validation/` 的验证输入，不公开上传 |
| `out/`、`ip_repo/` | 可再生成产物和本地下载缓存，不作为版本入口 |

## 验证

```powershell
python scripts/verify_a_handoff.py
python scripts/verify_full_reference.py
python -m unittest discover -s sim -p 'test_*.py'
python -m unittest discover -s camus/tests -p 'test_*.py'
cd display_web
python -m unittest discover -s tests -p 'test_*.py'
cd ..
python -m unittest discover -s display_web/unified/tests -p 'test_*.py'
D:\Git\bin\bash.exe scripts/run_sim.sh all
D:\Git\bin\bash.exe camus/run_sim.sh
```

构建使用 Vivado 2026.1 与 Git Bash，见 [工具链说明](docs/validation/VIVADO_ENVIRONMENT.md)。在隔离目录执行 `scripts/build.sh cnn` 和 `camus/build.sh`；新构建必须通过时序、配对、源绑定检查，再独立板测，不能直接覆盖发布位流。

## 板端使用与事实边界

正式目录为 `/home/xilinx/jupyter_notebooks/pld/releases/pld_dual_20261017`，固定入口 `http://192.168.2.99:8765/`。由 `pld-hub.service` 独占协调两个后端；不能同时运行独立 DMA 板测。升级、回退和数据保护见 [部署说明](docs/DEPLOYMENT.md)。凭据由本机环境或交互提供，不写入仓库。

两模型所有网络算术由 PL 执行；原生恢复、显示与提示图割在 PS。交互缓存命中不重新运行 FPGA。交互后输出独立下载，普通历史、ZIP、HDMI 仍显示原任务结果。首次整流程 3 秒和持续 4 FPS 目标尚未通过。物理 HDMI 未接显示器时只验接口，不宣称屏幕目视验收。当前结果和分阶段耗时见 [本次验证记录](docs/validation/CLEANUP_20261017.md)。

协作采用任务分支、聚焦提交、普通 PR，见 [CONTRIBUTING](CONTRIBUTING.md)。不强推，不推送归档/恢复分支，不提交原始数据、访问码或历史任务。

## 可选离线性能评测

[perf_eval 使用说明](perf_eval/README.md)提供CPU/PL延迟与吞吐、输出一致性、Dice/IoU变化、资源/功耗报告和DMA流量分析。先按本页准备唯一正式Release模型；`python -m unittest discover -s perf_eval/tests -q`可在不连接板子、未展开模型的干净克隆中执行。评测工具不修改正式发布产物，板端成绩需另行测量。
