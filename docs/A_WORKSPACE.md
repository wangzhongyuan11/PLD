# A 组本地工作与 PLD 协作仓库

2026-10-17，按用户要求，将 `E:\嵌赛` 下的工作关联到团队仓库 [wangzhongyuan11/PLD](https://github.com/wangzhongyuan11/PLD)。本地文件保留，B/C 团队历史和代码接入同一工作区。A 的协作细则只在本地 `pld-medical/AGENTS.md` 维护。

## 先明确一件事：clone 里没有 A 的模型和训练环境

本仓库对 A 组**只跟踪协作说明、接口文档与产物索引**。因此 `git clone` 之后：

- 能看到 A 的接口字段、数值约定、产物清单和许可说明；
- **看不到** A 的训练源码、数据集、`.venv`、模型权重和完整交付 ZIP；
- 不要在仓库内寻找 `pld-medical/training/`、`runtime/` 或 `.venv`，它们本来就不在 Git 里。

A 的完整电脑端接口与三个模型（皮肤 `role_a_final`、CAMUS 原交付、CAMUS 候选）的 FP32/INT8 运行闭包，全部打包在唯一 Release 资产 `pld-dual-20261017` 的附件
`role_a_complete_interactive_20261016.zip`（468,215,138 B，SHA256 `cf6fa99dda170b28df5739f7102790908930dfabd5b013ad7fd0ffdc19a0a794`）中。
**下载后独立解压，在解压出的包根目录运行**，不要尝试在 clone 内直接启动 A 的服务。

## 内容在哪里

| 内容 | 位置 | 先读 |
|---|---|---|
| A 接口契约与启动说明 | 本仓库 `pld-medical/docs/interactive/` | [接口总览](../pld-medical/docs/interactive/README.md)、[协议](../pld-medical/docs/interactive/protocol.md) |
| A 皮肤接口 | 同上 | [skin_api.md](../pld-medical/docs/interactive/skin_api.md) |
| A CAMUS 接口 | 同上 | [camus_api.md](../pld-medical/docs/interactive/camus_api.md) |
| A 产物、模型与 SHA 索引 | 本仓库 `pld-medical/deliverables/GITHUB_ARTIFACTS.json` | 直接查看 |
| A 数据来源与许可 | 本仓库 [docs/A_DATASETS.md](A_DATASETS.md) | 直接查看 |
| A 运行闭包（模型/运行源码/依赖清单） | Release `pld-dual-20261017` 附件 | 解压包内 `README.md`；虚拟环境由使用者在包根目录自行安装 |
| A 训练断点与本地实验 | 只在 A 的机器上，不进 Git | — |
| B/C 板端、驱动、网页与硬件 | 既有根目录 `board/`、`camus/`、`display_web/`、`rtl/`、`vivado/` | 根 [README](../README.md)、[团队协作](../CONTRIBUTING.md) |

根目录正式板端交付继续由 `delivery/current.json` 指向 `pld_dual_20261017`。A 的 `role_a_complete_interactive_20261016.zip` 是电脑侧统一交接包，**不是**新的板端正式版本，也不改变板端验收范围。

## 日常协作

在本地仓库根目录 `E:\嵌赛` 查看状态并获取团队更新：

```powershell
git status --short --branch
git fetch origin
```

已存在工作时先完成当前分支的聚焦提交；切回干净的 `main` 后使用 `git pull --ff-only`，再从 `main` 新建具体任务分支。通过普通 PR 合并。不要用 `--all` / `--mirror` 或强推发布本地归档历史；不要通过 reset/clean 覆盖仍需保留的本地训练工件。

## Git 与资源边界

Git 对 A 组只放协作文档；原始影像、GT、虚拟环境、缓存、完整 runs、训练源码、权重和大包都留在本地或 Release。`.gitignore` 用白名单只放行 `pld-medical/docs/interactive/` 与三个交付索引文件，其余 A 路径一律忽略，避免误加。

需要发布新模型或替换正式资源时，单独核对版本、manifest、数据许可与相应验收，再按团队 Release 方式交付。当前 A 的数据与模型已随 [Release pld-dual-20261017](https://github.com/wangzhongyuan11/PLD/releases/tag/pld-dual-20261017) 发布：13 个附件的下载 URL、字节数、SHA256 与 GitHub 服务端 digest 见 [产物索引](../pld-medical/deliverables/GITHUB_ARTIFACTS.json)；下载、连接分片与校验步骤见同 Release 的 `A_RELEASE_README_20261017.md`。

本次纳入仓库不重训练、不重新校准、不覆盖模型；原有训练 STOP、已关闭许可和已验收 ZIP 保持原值。A 组与 B/C 组的历史软件/板端证据继续按原范围解释。
