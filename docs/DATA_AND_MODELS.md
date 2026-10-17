# 数据与模型准备

正式模型与小型回归材料由 `pld_dual_20261017_assets.zip` 提供，依照 [manifest](../delivery/pld_dual_20261017/MANIFEST.json) 校验。主机运行 `scripts/release_assets.py ... --prepare` 恢复既有默认路径，不修改参数、定点算术或 frozen golden。

| 路径 | 用途 |
|---|---|
| `role_a_final/int8/` | Skin 当前最终模型，spec SHA256 `fe137ca5b128fc1983fabad9bee39b3208cda4256e0d04c4e27cef5c07666aef`，124成员清单见 `reference/final_manifest.json` |
| `role_a_handoff/role_a/int8/` | 旧 Skin 模型回归 fixture；`sim/test_model_versioning.py`、`sim/test_pl_graph.py` 仍依赖，不是另一正式模型 |
| `camus_candidate_hidden_histogram_int8_20261015/` | 完整 CAMUS 包，spec SHA256 `68b4f425cc3612ad72d5e83955c2018981edee912fbc0937e9ac8c1718e7b5ae`；含 FP32 权重、INT8 导出和200份评估回放，`camus/verify.py` 直接使用 `evaluations/int8/cases` |
| `数据/` | 本地 CAMUS、ISIC、STARE 原始数据与标签，保持私有、不随源码/Release上传 |
| `数据/validation/skin/{data2017,evaluation_638,evaluation_376}` | 从旧 `out/final_model` 迁出的独有验证输入；同级保留 `images_2017.tar.gz` |
| `数据/validation/camus/dev200` | 从旧 `out/camus/dev200_board_v2` 迁出的开发回放；同级保留 `dev200.tar.gz` |
| 附件 `fixtures/camus/validation/dev20/` | 公开开发图及期望结果小样本，便于交付短测；不等于独立测试集 |

`out/` 不能再承担唯一原始数据存储职责。迁出的原数据未重新编码；板端 `final_dataset_638/dataset/images` 仍链接到 `stage5_dataset_20260927/dataset/images`，两个目录一起保留。板端和本机路径分别管理，不以目录时间推断模型版本。

无需像素间距即可分割并报告像素数/占比；填入间距时单位为 mm/pixel，顺序 `(Y, X)`，应使用该图的真实元数据，不能把别人的值套用。NIfTI/MHD 有有效间距时读取其元数据；普通 PNG/JPG 若缺来源，则只报告像素指标。当前单帧面积不能当作体积或临床诊断。

模型与数据的原始来源、许可与 A 方预处理/评估口径保留在完整模型包的 provenance/README；没有替这些资产另行授予新许可证。主机完整 FP32 重评估需要 Torch 和原开发集，当前清理验证不声称重新训练、独立临床测试或全量FP32精度复验。
