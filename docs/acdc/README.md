# ACDC 三方向 2.5D 适配入口

本扩展使用 `ACDC25D-B16-P3-v1`，三个方向分别加载一套权重。新 MRI 分支与已有皮损、CAMUS 分支分开；正式部署指针 `delivery/current.json` 不因本扩展自动切换。

规格见 [板端预适配规格](ACDC25D_B16_P3_v1.md)，机器可读合同见 [MODEL_CONTRACT.json](MODEL_CONTRACT.json)。A 的模型训练及完整 PC/板端适配运行源码保留在 `pld-medical/`，冻结权重、数据来源及运行闭包经 Release 交付；本目录只共享 B/C 对接必要资料。

| 项目 | 固定规则 |
| --- | --- |
| 方向 | axial / coronal / sagittal；同一拓扑，独立权重，依次加载 |
| 输入 | int8 `[1,3,256,256]`，196608 B，NCHW，前一层/中心层/后一层，边界复制 |
| 输出 | int8 `[1,4,256,256]`，262144 B，BG / RV腔 / LV心肌 / LV腔，无输出激活 |
| 深度 | 15 Conv，3 MaxPool，3 nearest×2+CAT，14个融合ReLU |
| CAT | mode4，先全部 skip 通道，再 nearest×2 deep 通道；双LUT，split=64/32/16 |
| 量化 | signed127，零点0，权重OIHW，bias int32LE，正Q24 int32LE，乘积int64，最近偶数舍入 |
| head尺度 | 各模型有一个共享四通道的正scale，实数值随冻结模型发布，不能提前编造 |
| PC | 物理网格重采样、完整视野letterbox、反映射、softmax概率融合、NIfTI/PLY、四面板查看器 |
| 板端 | 只接收已准备的INT8三层堆栈；所有网络算子须由所选设备执行后才能称PL推理 |

## 无模型预适配

从仓库根运行 `python tools/make_acdc_contract_fixtures.py --out <新建的本地目录>`，只需 Python 标准库。它生成三个**合成接口配置**、全零权重/固定偏置参数、已知整网输出、mode4拼接和带符号Q24平局用例。三个方向的固定head不同，能发现切换参数后缓存未失效的问题。

产物全部标记 `synthetic=true` 和 `NOT_A_TRAINED_MODEL`，生产加载器必须拒绝。通过这些用例只证明布局/算术/调度接口，不能证明训练质量、位流可用或板端性能。生成的大文件不提交Git。

## 几何与验收

官方HHP ACDC文件出现 direction-only `sform` 与 `pixdim` 不一致：首例 `sform`列长为1，header间距却为1.5625×1.5625×10mm。实现默认严格拒绝不一致；显式的 `acdc-direction-only-sform-v1` 策略只对 qform_code=0、sform_code=2、单位正交方向列的已识别格式恢复header间距，原始affine/间距和修复依据保留在sidecar。不能把默认affine的1mm当作原始分辨率。原始姿态若缺失，不能据此声称恢复真实采集姿态。

工作网格为2mm、依照文件声明的RAS方向，图像线性插值、标签最近邻，完整视野等比例letterbox到256；非零原始体素强度的1/99分位映射到[-1,1]并截断。训练与推理相同，不依赖GT裁剪。三视图都是同一MRI体数据的重切片，插值不增加真实解剖细节。

100名官方训练患者按五疾病组16/4固定分层，共80训练/20开发；50名官方测试患者封存，未用于选择或校准。评估须分别报告三个方向、2D基线、概率融合，以及FP32/INT8差异；最终分割和体积回到原生网格，距离使用毫米。开发指标不能称独立测试成绩。

截至本接口初次提交，训练和软件验收正在进行。现有CAM2硬件规则支持Ci≤192及输出通道分块，但既有CAMUS加载器固定Ci=1；A提供独立加载适配，不修改旧模型身份。真实bit/HWH配对、加载切换、数值、DDR峰值、故障恢复和时延均须另行板测。

官方来源：[ACDC训练集说明](https://www.creatis.insa-lyon.fr/Challenge/acdc/databasesTraining.html)、[标签与评价口径](https://www.creatis.insa-lyon.fr/Challenge/acdc/evaluation.html)。数据按 CC BY-NC-SA 4.0及严格非商业科研条款使用；引用Bernard等，IEEE TMI 2018，DOI:10.1109/TMI.2018.2837502。
