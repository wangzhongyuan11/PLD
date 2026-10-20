# A 组数据来源与许可

2026-10-17。本文件说明 A 组实际使用过的数据、许可条件，以及这些数据在 Git 与 Release 中的位置。原始影像和标注**不进 Git**，只作为 Release 资产发布。

## 1. 实际使用的数据

| 数据集 | 用途 | 是否进 Git | 许可 |
|---|---|---|---|
| ISIC 2016 Task 1（train 900 / test 379） | 训练与开发来源之一 | 否 | CC0 |
| ISIC 2017 Task 1（training 949 / validation 147 / test 600） | 训练与开发来源；test 为**已暴露**来源 | 否 | CC0 |
| CAMUS 官方公开包 | 心脏超声四类分割（LV / MYO / LA / 背景） | 否 | CC BY-NC-SA 4.0（仅非商业科研） |

**最终实际口径**（依据 `pld-medical/deliverables/role_a_final/DATA_PROVENANCE.md`）：原始 → 冻结保留为 2016 train 900 → **897**、2016 test 379 → **376**、2017 training 949 → **946**、2017 validation 147（官方 150，排除与 2016 重叠 3 张）→ **145**、2017 test 600 → **466 train + 117 dev**（共 583，排除 17 张）。**合并口径 train 2309 / dev 638**。其中 2017 test 属**已暴露**来源，不是独立最终测试集，**不得称为无偏外部验证**；2016 的 900/379 正是最终 897/376 的来源，早期 720/180 划分只是旧基线口径，不代表最终数据角色。

**未纳入**：ISIC 2018 训练/验证/最终测试掩膜（最终测试掩膜未下载、未评分、未用于选模）、novice 试点、IMA++ 候选、任何缓存与 soft target、虚拟环境与训练断点。

## 2. 许可与引用

- **ISIC 2016 / 2017 Task 1**：数据页 <https://challenge.isic-archive.com/data/> 标注 **CC0**。使用时保留来源说明。
- **CAMUS**：官方 `LICENSE_TERMS.md` 为 **CC BY-NC-SA 4.0**（Attribution-NonCommercial-ShareAlike，<https://creativecommons.org/licenses/by-nc-sa/4.0/>）。**仅限非商业科学研究**，须署名、相同方式共享。引用：
  Leclerc et al., *Deep Learning for Segmentation using an Open Large-Scale Dataset in 2D Echocardiography*, IEEE TMI 2019, DOI `10.1109/TMI.2019.2900516`。
- CAMUS：**本轮**只作为原官方 ZIP 的不透明字节发布，**不读取、不解析测试标注，不重新评分或调参**，也不重开任何历史测试指标。历史记录中与此相关的已授权工作保持原样，不因本说明而扩大或重述。

## 3. 发布方式

数据与三个模型已整合到 [Release pld-dual-20261017](https://github.com/wangzhongyuan11/PLD/releases/tag/pld-dual-20261017) 的单一总 ZIP，以四卷传输。完整目录、8 项下载文件和一键还原步骤见 [总包说明](A_RELEASE_BUNDLE.md)，大小及服务端 SHA 见 [产物索引](../pld-medical/deliverables/GITHUB_ARTIFACTS.json)。

- 总包 `datasets/ISIC2016/` 保留四个官方原 ZIP 的字节。
- `datasets/ISIC2017/` 保留来源、影像/标注、冻结划分与逐成员校验原包。
- `datasets/CAMUS/` 已包含合回完整的官方原 ZIP；不再需要单独连接旧 CAMUS 分片。
- `licenses/` 保留 CAMUS 原许可；`metadata/original_release/` 仅供历史追溯。
- 原 13 个散附件已迁入总包并退役，新的下载入口不再依赖其旧直链。

## 4. 边界

数据不随 clone 分发；`git clone` 后拿不到任何影像或标注。A 的电脑端模型与接口运行闭包在 `role_a_complete_interactive_20261016.zip`：**该包不含完整数据集**，只含模型、运行源码、派生结果，以及**两张既有开发演示输入（无参考标注）**。完整数据已放在总包的 `datasets/` 下；来源与冻结角色仍按上述清单核对，不改变任何训练/测试授权。
