# FPGA 皮肤镜病灶分割与交互式量化系统

## 当前工作区与 GitHub 协作（2026-10-17）

本目录是团队仓库 [wangzhongyuan11/PLD](https://github.com/wangzhongyuan11/PLD) 的 A 组工作区。仓库根目录为本机 `E:\嵌赛`；A 组保留 `pld-medical/` 结构，B/C 组沿用根目录的板端与硬件工程。协作方式、目录对照及大文件边界见 [A 工作区说明](../docs/A_WORKSPACE.md)。

- 当前 A 组状态：[断点与交付记录](docs/A02_当前断点.md)。
- 皮肤病灶 / CAMUS 独立电脑接口：[使用说明](docs/interactive/README.md)。
- 已验收大包及模型的本地位置、SHA256 与 GitHub 发布状态：[交付索引](deliverables/GITHUB_ARTIFACTS.json)。
- PLD 的正式板端版本仍以仓库根 `delivery/current.json` 为准；A 组电脑端验证不代替其板端验收。

源码和文档纳入本仓库；环境、原始数据、GT、完整训练 runs、模型和大压缩包保留本地或按团队 Release 方式准备。首次克隆不会自动附带 A 组本地训练环境或这些大文件，不应据此声称可恢复完整训练。

## 初期说明（2026-09-23 历史记录）

更新于 2026-09-23：A 组 ISIC 2016 训练与 INT8 软件交付已完成，结果和 B 组接口见
[`deliverables/role_a/README.md`](deliverables/role_a/README.md)，可直接传递的压缩包见
[`deliverables/role_a_handoff.zip`](deliverables/role_a_handoff.zip)。A 组代码与复现命令见
[`software/role_a/README.md`](software/role_a/README.md)。板卡选择为 PYNQ-Z2；RTL/上板仍待 B/C 组实现。

B、C 组并行开发及换版边界见 [`deliverables/role_a/B_C_并行开发交接说明.md`](deliverables/role_a/B_C_并行开发交接说明.md)。

以下初审资料保留为项目历史记录：初审截止 2026-09-12，内部定稿 2026-09-11。

## 先读什么

1. `docs/01_初审方案.md`：可继续修改的初审文案。
2. `docs/02_需求与验收.md`：每项赛题要求如何落实，以及哪些还没完成。
3. `docs/03_时间表与分工.md`：三个人在初审前每天做什么。
4. `docs/04_学习路线.md`：从零开始的概念、练习和验收方法。
5. `docs/05_技术架构.md`：电脑与 FPGA 的边界、数据格式和硬件设计约定。
6. `docs/06_数据与实验规范.md`：数据来源、划分和指标，避免把演示当实验。
7. `docs/07_决策与待办.md`：长期项目的已确认决定和待解决问题。

## 现在能运行什么

在本目录打开终端：

```powershell
py software/baseline.py --demo --output runs/demo
py -m unittest discover -s tests -v
```

示例只依赖 Python 标准库，生成合成暗色区域图像，执行整数灰度化、3×3 高斯滤波、固定阈值分割、3×3 开运算和轮廓提取。结果包括 `preview.svg`、PGM 灰度/掩膜文件和 `metrics.json`。SVG 可用浏览器打开。

合成图不是医学影像，所得指标只验证程序流程，不代表 ISIC 精度或 FPGA 性能。
参考程序刻意追求易读，不代表优化后的 CPU 基准。

早期基线的真实 JPEG/PNG 转换需 Pillow，见 `software/README.md`。A 组已在本机下载官方
ISIC 2016 数据并完成训练；原始数据位于被忽略的 `data/raw/`，不包含在交接压缩包中。
FPGA 仿真/综合/上板与完整上位机尚未完成。

## 目录

```text
docs/             初审、学习、技术设计及项目决策
data/             数据来源与划分模板；原始数据不提交版本库
software/         可运行参考算法和真实图像转换工具
hardware/rtl/     入门灰度模块；不是完整 FPGA 系统
hardware/sim/     最小仿真测试台与使用说明
host/             上位机界面与接口设计
tests/            软件行为验证
experiments/      实验记录模板
runs/             本地运行输出（自动生成）
deliverables/     A 组权重、INT8 文件、指标、来源清单与交接压缩包
```

下一次推进优先处理正式报名模板、字数要求、板卡/工具限制与数据许可核查；无需等板卡确定再学习或完善文案。
