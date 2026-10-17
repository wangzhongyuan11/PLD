# 数据来源与授权说明（DATA_PROVENANCE）

本文**只依据仓库内已有的记录文件**（`data/raw/*/provenance.json`、`data/raw/*/survey.json`、`data/raw/*/manifest.json`、`runs/role_a_augmented_v4/split.json` 与 `validation_baseline.json`、`runs/role_a_full2017/*exclusions.json`、以及冻结协议的来源规格 `software/role_a/train_overall_v1.py:build_default_spec()`）整理。**本轮最终训练与交付包没有新增任何下载（新增下载 = 0 字节）**，只使用既有本地数据。

## 1. 本轮实际使用的数据源（6 份，全部 ISIC 2016/2017 Task 1 分割）

| 角色 | 名称 | 本地目录 | 原始文件数 | 冻结保留数 | 逐图清单/哈希记录 |
|---|---|---|---|---|---|
| train | `isic2016_train` | `data/raw/isic2016/train/{images,masks}` | 900 | **897** | `data/raw/isic2016/provenance.json`（压缩包级） |
| train | `isic2017_new_train` | `data/raw/isic2017_training/{images,masks}` | 949 | **946** | `data/raw/isic2017_training/provenance.json`（949 项逐图 SHA256） |
| train | `isic2017_exposed_train` | `data/raw/isic2017_test/{images,masks}` | 600 | **466** | `data/raw/isic2017_test/provenance.json`（600 项逐图 SHA256） |
| dev | `isic2016_test` | `data/raw/isic2016/test/{images,masks}` | 379 | **376** | 同上（2016 压缩包级） |
| dev | `isic2017_validation` | `data/raw/isic2017_validation/{images,masks}` | 147 | **145** | `data/raw/isic2017_validation/provenance.json`（147 项逐图 SHA256） |
| dev | `isic2017_exposed_validation` | `data/raw/isic2017_test/{images,masks}` | 600 | **117** | 同上（2017 test 清单） |

合计：**train 2309 / dev 638**（冻结口径；来源规格与数量核对见 `build_default_spec()` 与 `assemble(verify_frozen=True)`）。

## 2. 下载记录路径与哈希

### 2.1 ISIC 2016（压缩包级记录）

`data/raw/isic2016/provenance.json`（1,347 B，SHA256 `dd4a6deb89574bfa…`）：
`source = https://challenge.isic-archive.com/data/`，`retrieved_at_utc = 2026-09-23T07:38:29Z`，`year=2016`，`task = Task 1 lesion segmentation`。

| 归档 | URL | SHA256（记录值） | 文件数 | 解压到 |
|---|---|---|---|---|
| train_images.zip | `https://isic-archive.s3.amazonaws.com/challenges/2016/ISBI2016_ISIC_Part1_Training_Data.zip` | `ad5b48fb7d9073f9…` | 900 | `train/images` |
| train_masks.zip | `…/ISBI2016_ISIC_Part1_Training_GroundTruth.zip` | `81cff572b1cc0e61…` | 900 | `train/masks` |
| test_images.zip | `…/ISBI2016_ISIC_Part1_Test_Data.zip` | `5cb5766aecc0c298…` | 379 | `test/images` |
| test_masks.zip | `…/ISBI2016_ISIC_Part1_Test_GroundTruth.zip` | `065df24f07424aea…` | 379 | `test/masks` |

（完整 64 位哈希见该 JSON；压缩包实体在 `data/raw/isic2016/downloads/`。）

### 2.2 ISIC 2017（逐图记录 + 官方掩膜包）

| 记录 | 字节 | SHA256（前 16 位） | 内容 |
|---|---|---|---|
| `data/raw/isic2017_training/provenance.json` | 263,192 | `24bc36bd57e25d10` | 949 张逐图 `image_sha256`/`mask_sha256`/`area_ratio`；`source_survey_sha256 = 285f2b8022dde48e…` |
| `data/raw/isic2017_training/survey.json` | 457,302 | `285f2b8022dde48e` | 全量 2017 training 掩膜普查与筛选规则 |
| `data/raw/isic2017_training/manifest.json` | 435,056 | `41d7ee3e6cea3ab5` | 官方掩膜包 `source_masks_sha256 = a602bf91be39e327…`，逐项 `license: "CC-0"` |
| `data/raw/isic2017_validation/provenance.json` | 40,990 | `926cc4e6c04ca0b3` | 147 张逐图记录；`source_survey_sha256 = 18d1c3cfeb570d17…` |
| `data/raw/isic2017_validation/survey.json` | 64,267 | `18d1c3cfeb570d17` | 官方验证掩膜包 `source_masks_sha256 = ad7d65cbb1ff5d94…`，`official_mask_count = 150`、排除与 2016 重叠 3 张后 **147** |
| `data/raw/isic2017_test/provenance.json` | 166,512 | `7afacb15b4911d69` | 600 张逐图记录（同时供 `isic2017_exposed_train`/`_validation` 使用） |
| `data/raw/isic2017_test/manifest.json` | 258,702 | `b4d55258fb6bf784` | 官方测试掩膜包 `source_masks_sha256 = 027abe293f3344bf…`，`official_mask_count = 600` |

官方下载 URL（记录在 survey/manifest 内）：
- 2017 training 掩膜：`https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Training_Part1_GroundTruth.zip`
- 2017 validation 掩膜：`https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Validation_Part1_GroundTruth.zip`
- 2017 test 掩膜：`https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip`
- 2017 图像：`https://isic-archive.s3.amazonaws.com/images/{isic_id}.jpg`（逐 ID 下载，非整包）

### 2.3 冻结拆分与排除清单（划分类依据）

| 记录 | 字节 | SHA256（前 16 位） | 作用 |
|---|---|---|---|
| `runs/role_a_augmented_v4/split.json` | 12,401 | `94c3a43b570ff32e` | 冻结划分：`train_ids` 466 + `validation_ids` 117（2017 test 部分），并记录 `source_provenance_sha256 = 7afacb15b4911d69…`、`source_overlap_exclusions_sha256 = 48be5fcd6951d5c5…` |
| `runs/role_a_augmented_v4/validation_baseline.json` | 33,710 | `19d65bba867e3a54` | 冻结开发基线（117 条逐图 `area_ratio`）、`split_sha256 = 94c3a43b570ff32e…`、`baseline_sha256` |
| `runs/role_a_full2017/overlap_exclusions.json` | 1,390 | `35254fb9eb808e55` | 经 dHash + 128 像素 RGB 相似度 + 人工看图的排除清单：`external_validation` 2 张、`supplemental_training` 3 张、`isic2016_training` 等 |
| `runs/role_a_full2017/test2017_overlap_exclusions.json` | 1,607 | `48be5fcd6951d5c5` | 2017 test 与训练来源的重叠排除（`frozen_before_model_evaluation = true`） |

冻结口径摘要：**train 2309 / dev 638**；开发集 ID 摘要 `4ead5256747a87003cb341ce647e2863ec60e5d2d82a737a9572d2963805133d`、训练 ID 摘要 `0b4f6ade53314b206fb8bcccf7d7169c90d2a35fc987dbc70404c9fd4fba2bb0`（与包内 `metrics/comparison.json` 的 identity 一致）。

## 3. 授权（许可）条款

**必须按来源与具体下载包分别核实，不能把整个 ISIC 数据集合称为同一许可。** 仓库内记录能支持的结论如下：

| 来源 | 仓库内记录 | 结论 |
|---|---|---|
| ISIC **2017** Task 1 | `isic2017_test/manifest.json`、`isic2017_validation/survey.json` 记录 `source_license = "https://challenge.isic-archive.com/data/ (2017 Task 1 CC-0)"`；`isic2017_training/manifest.json` 逐项 `license: "CC-0"` | 按仓库记录，**2017 Task 1 训练/验证/测试为 CC-0** |
| ISIC **2016** Task 1 | `isic2016/provenance.json` **只记录 URL、时间、SHA256 与文件数，未保存许可字段**；许可由**本项目于 2026-09-27 核对 ISIC 官方数据页** [`https://challenge.isic-archive.com/data/`](https://challenge.isic-archive.com/data/) 的 "2016 → Task 1" 表格 **License 栏 = CC-0**（同页 "2017 → Task 1" 亦为 CC-0） | 按官方页核对结果，**2016 Task 1 为 CC-0**；本地 provenance 未保存该字段，故本条以官方页为准 |
| 其它 ISIC 年份 / ISIC Archive 全库 | 未被本轮使用（2018 仅下载过元数据/试点，且 2018 最终测试掩膜从未打开） | 不在本说明范围；**不得**据本文推断其许可 |

> 边界限定：上述 CC-0 结论**仅适用于 ISIC 2016 Task 1 与 ISIC 2017 Task 1 这两个具体下载包**（且 2016 的许可来自本项目于 2026-09-27 对 ISIC 官方数据页的核对，而非本地记录）。**不能推广到整个 ISIC Archive、其他年份或其他 Task**；如需引用请以官方页与各下载包内的许可声明为准。本条修改**未下载任何许可文件或原始数据**。

补充（来自仓库既有记录、非本轮新增）：A16 记录显示 IMA++ 元数据为 CC-0 14,639 / CC-BY-NC 328、ISIC 2018 Novice 掩膜为 CC-BY-NC；这些数据**本轮未入训**，因此与本交付包无关。

## 4. 本轮下载与改动声明

- **新增下载 = 0 字节**（未下载任何图像、掩膜、元数据或模型）。
- 未修改 `data/` 下任何原始文件；未解包/未评分 **ISIC 2018 最终测试掩膜**。
- 交付包内的 `best.pt` 与 `int8/**` 均复制自既有已验证产物（来源已在 `MANIFEST.json` 逐条标注）。
