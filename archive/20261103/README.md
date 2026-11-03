# 竞赛结束归档 · 2026-11-03

竞赛已经结束。本次按用户授权，将这台电脑上的PLD项目资料归档到同一GitHub仓库，整合查询入口，并在远端完整性核对后清理本地副本。现有主分支的10月30日整合实现保持为代码入口；历史本地A源码保存在本目录，避免旧快照覆盖团队后续实现。

## 从哪里开始

| 需要的内容 | 入口 |
|---|---|
| 最新团队整合代码 | [仓库根目录](../../)与[10月30日说明](../../deployment/repository_20261030/README.md) |
| 本地A源码、实验方法与接口文档 | [a_source](a_source/)、[逐文件源码身份](SOURCE_SNAPSHOT.json)；共309个文本文件 |
| 模型、完整实验记录、数据、历史包与原始目录 | [结束归档Release](https://github.com/wangzhongyuan11/PLD/releases/tag/pld-closeout-20261103)；下载对象包与目录包，按下文恢复 |
| 可直接查看的答辩PPT、Word、PDF和研究报告 | 同Release的 `PLD_Documents_and_Presentations_20261103.zip`；60个资料文件 |
| 本地分支与标签历史 | 同Release的 `PLD_Local_History_20261103.bundle` 与[引用清单](LOCAL_GIT_HISTORY.json) |
| 原正式版本与候选版本 | [原正式入口](../../delivery/current.json)以及各原Release，均保持原来的身份与验收范围 |

本次目录归档保留80297个文件路径、49592个唯一内容对象，原文件合计17765404558 B。相同SHA256的重复文件只存一份内容，恢复时重建各路径。归档附件共14项、12359386061 B，另附校验清单。

## 下载与还原

从结束归档Release下载全部 `PLD_Workspace_Objects_20261103_*.zip`、`PLD_Workspace_Catalog_20261103.zip`、`RESTORE_PLD.py` 和 `SHA256SUMS_20261103.txt`，放到同一目录。对象包不需要手动解压。需要Python 3.10或以上，仅用标准库：

```powershell
python RESTORE_PLD.py --assets . --verify-only
python RESTORE_PLD.py --assets . --target D:/PLD_restored
```

目标必须不存在或为空，脚本拒绝覆盖现有文件。目录包内有原路径、逐文件SHA256、对象定位、包校验值、未上传项目清单及许可证。关联工作副本保存在还原目录 `_linked_worktrees/`，不恢复原机器的绝对Git指针。

需要原本地Git历史时，另下载bundle：

```powershell
git clone PLD_Local_History_20261103.bundle PLD_history
```

原本地工作区HEAD为 `51f85407a079c30a6e545e7f6500081862dff006`，本次整合基于远端 `d3f8845434583817932171d661f8517537b66710`。本地旧合同和旧工作规则按历史快照保留，不提升为新的当前接口。

## 保留与排除的依据

模型、数据、训练记录、失败尝试、历史交付包和研究提案按原身份归档；不因归档而新增功能验收、训练、封存测试或板测结果。研究提案与已实现功能仍须区分。

不上传可重建的原 `.venv`、Python字节码、Git内部元数据、外部依赖目录链接及本地助手会话配置。排除清单与Python包版本保存在目录包中；Git历史由bundle保存。独立安装的Vivado、Vitis、Vitis HLS、Python及其他开发软件不属于清理目标。

数据和第三方代码没有新的统一许可，使用时遵守[原始来源与许可](CITATIONS_AND_LICENSES.md)。ACDC和CAMUS仅限原条款许可的非商业科研；ISIC来源和novice标注归属随原元数据保留。归档操作不重新解析或评分封存数据。

## 完整性

[附件清单](ARCHIVE_MANIFEST.json)、[SHA256校验表](SHA256SUMS_20261103.txt)和[源码身份](SOURCE_SNAPSHOT.json)对应本次归档。发布前逐个读取全部对象进行SHA256校验，并以恢复脚本实际重建小型模型源码和研究Word文件；远端附件还须核对服务端SHA256与大小。机器清理的最终执行结果由本机留下的独立收尾凭证记录，不将“已上传”自动写成“已删除”。
