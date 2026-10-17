# A 组本地工作与 PLD 协作仓库

2026-10-17，按用户要求，将 `E:\嵌赛` 下的项目工作关联到团队仓库 [wangzhongyuan11/PLD](https://github.com/wangzhongyuan11/PLD)。本地已有文件保留，远端已有团队历史和 B/C 代码接入同一工作区。

## 位置与入口

|内容|本仓库位置|先读|
|---|---|---|
|A 算法、训练、量化、电脑接口|`pld-medical/`|[项目准则](../pld-medical/AGENTS.md)、[当前断点](../pld-medical/docs/A02_当前断点.md)|
|A 皮肤 / CAMUS 交互接口|`pld-medical/host/interactive/`、`pld-medical/software/interactive_segmentation/`|[接口说明](../pld-medical/docs/interactive/README.md)|
|B/C 板端、驱动、统一网页和硬件|既有根目录 `board/`、`camus/`、`display_web/`、`rtl/`、`vivado/`|根 [README](../README.md)、[团队协作](../CONTRIBUTING.md)|
|教材、初审与实验方法|`A组教材.pdf`、`B组教材.pdf`、`pld-medical/docs/`、`report_work/`|各文件原说明|
|A 组已验收大包和模型|本地 `pld-medical/deliverables/` 与已冻结 run|[来源与存储索引](../pld-medical/deliverables/GITHUB_ARTIFACTS.json)|

根目录正式板端交付继续由 `delivery/current.json` 指向 `pld_dual_20261017`。A 组的 `role_a_complete_interactive_20261016.zip` 是电脑侧统一交接包，不能直接等同为新的板端正式版本。

## 日常协作

在本地仓库根目录 `E:\嵌赛` 查看状态和获取团队更新：

```powershell
git status --short --branch
git fetch origin
```

已存在工作时先完成当前分支的聚焦提交；切回干净的 `main` 后使用 `git pull --ff-only`，再从 `main` 新建具体任务分支。通过普通 PR 合并。不要用 `--all` / `--mirror` 或强推发布本地归档历史；不要通过 reset/clean 覆盖仍需保留的本地训练工件。

## Git 与资源边界

Git 共享源码、文档、教材、接口契约和交付校验索引。`.gitignore` 保留原始影像、GT、虚拟环境、缓存、完整 runs、权重及大包；这些文件仍在原位置，关联 Git 不会搬走或重新生成它们。A 组源码初次导入保留原字节，避免改变已有哈希绑定；B/C 原有 LF 和冻结文件规则保持。

PLD 正式模型和运行资源按根 README 的唯一 Release 获取。A 的产物索引明确标记当前仍只在本地的大包，不能把本地路径或 SHA 清单描述为已经上传的附件。需要发布新模型或替换正式资源时，单独核对版本、manifest、数据许可与相应验收，再按团队 Release 方式交付。

首次把本地工作纳入仓库不重训练、不重新校准或覆盖模型；原来的训练 STOP、关闭许可及已验收 ZIP 保持原值。A 组和 B/C 组各自的历史软件 / 板端证据继续按原范围解释。
