# A 组模型与数据集统一总包（2026-10-17）

**本包同时包含模型与数据集。** 它把 A 组此前发布的 13 个附件整合成一个
`PLD_A_Models_and_Datasets_20261017.zip`，没有重新训练、量化、评分或改写模型数值。
GitHub 每个 Release 附件必须小于 2 GiB，所以这个约 6.77 GB 的总 ZIP 以四个字节分卷发布。

## 下载与一键还原

从同一 Release 下载以下 8 个文件，放入同一文件夹：

- `PLD_A_Models_and_Datasets_20261017.zip.001` 至 `.004`，共四卷；
- `A_MODELS_DATASETS_MANIFEST_20261017.json`，大小与 SHA256 清单；
- `RESTORE_A_Models_and_Datasets.py`，跨平台还原工具；
- `RESTORE_A_Models_and_Datasets.cmd`，Windows 一键入口；
- 本说明 `A_MODELS_DATASETS_README_20261017.md`。

已安装 Python 3 的 Windows 电脑双击 `.cmd` 即可。Linux、macOS 或终端执行：

```text
python RESTORE_A_Models_and_Datasets.py
```

工具逐卷核对大小和 SHA256，再核对总 ZIP 的 SHA256；成功后才生成完整 ZIP。
需要额外约 6.77 GB 可用空间。它不会覆盖已有 ZIP、删除分卷、解压数据或运行模型。
只检查下载是否完整时使用 `python RESTORE_A_Models_and_Datasets.py --verify-only`。

## 总 ZIP 内部结构

```text
PLD_A_Models_and_Datasets_20261017/
  README.md
  MANIFEST.json
  SHA256SUMS.txt
  models/                      三个模型及原交接包、验收侧文件
  datasets/ISIC2016/            Task 1 四个原始 ZIP
  datasets/ISIC2017/            既有来源与冻结划分 ZIP
  datasets/CAMUS/              合回完整的官方原 ZIP
  licenses/                    原 CAMUS 许可
  metadata/original_release/   原附件索引、说明与校验表，仅作历史追溯
```

模型原包内含皮肤病灶分割、CAMUS 主模型与候选模型的 FP32/INT8 工件，以及两套点提示、
框提示、ROI 和阈值接口。模型权重和数值规格沿用原验收结果。解压总 ZIP 后，内部各原包按需解压；
运行要求以模型原包 README 为准，不含虚拟环境。

CAMUS 两个旧分片已经在总包内部合为一个完整官方 ZIP，整包 SHA256 为
`862474ff61a660d0d0e9fe9867d5189b8536406e266d983a9914a9cb165c0c35`。
本轮仅复制其不透明压缩字节，没有读取测试标注或重新评估模型。

ISIC 2016/2017 Task 1 沿用 CC0；CAMUS 保留 CC BY-NC-SA 4.0、非商业科研、署名和相同方式共享要求，
原许可在 `licenses/`。引用 Leclerc et al., IEEE TMI 2019, DOI `10.1109/TMI.2019.2900516`。
ISIC 2017 test 已用于开发来源，不能将其称为独立无偏测试集；冻结划分与来源清单保留在原数据包中。

## Release 与仓库分工

- Git 仓库：协作源码、接口与小型说明；最新 `main` 含交互 v5 软件候选，不代表新板端验收。
- 原四个 B/C 附件：正式板端部署、运行资源、冻结源码快照及其说明/校验，内容与哈希保持。
- 本 A 组总包：模型与 ISIC/CAMUS 数据集，按上面步骤还原为一个 ZIP。

原 13 个 A 组散附件在本总包上传并经 GitHub 服务端 size/digest 核验后移除，旧直链不再使用。
总包内 `MANIFEST.json` 记录原附件到新目录的对应与逐文件校验，不依赖旧附件继续在线。

Release：<https://github.com/wangzhongyuan11/PLD/releases/tag/pld-dual-20261017>
GitHub 附件限制：<https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases>
