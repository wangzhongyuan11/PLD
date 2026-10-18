# ACDC 生产数值接口补充定义 v1

版本：2026-10-18；适用 `ACDC25D-B16-P3-v1`，配套 [拓扑合同](MODEL_CONTRACT.json)、[机器可读预处理合同](PREPROCESSING_CONTRACT.json)、[患者划分与时相](PATIENT_SPLIT.json)、[PC/板端交换](BOARD_EXCHANGE.md)。本文件补齐 A→B/C 的数值输入，不修改既有 ABI。

**当前冻结范围：预处理、方向、几何、尺寸、时相接入、尺度传播算法、整数算术、融合和评价规则。最终三方向模型的参数、逐边尺度数值、spec SHA 与最终指标尚待 INT8 验收。首版 minmax INT8 已拒绝，正在修复；不能将失败候选或合成配置作为生产参数。**

| 项目 | 对接状态 |
|---|---|
| 无模型基础设施 | 可依本合同实现、测试；合成用例必须保留 `synthetic=true` |
| 生产参数装载 | 必须取得本方向验收通过的 `model_spec.json`、参数、SHA 和黄金输入/head；缺一拒绝运行 |
| 整网软件数值 | 必须用同一冻结版本逐层、整网核验；训练时 fake quant 不作为精确整数证据 |
| 物理板端生产链路 | 仍需匹配 bit/HWH、实际整网数值、故障恢复与时延验收；本合同不代替板测 |

## 1. 强度标准化

以**一个患者、一个时相的完整原始 3D 体**为统计单位。ED/ES 分别计算，不跨患者/时相汇总，不按切片重算，也不先裁剪 ROI 或按标签选体素。

1. NIfTI 经 `get_fdata(dtype=float32)` 读取（包含文件自身的强度 slope/intercept）。任意 NaN/Inf 直接报错，不填零。
2. 取整个原生体内 `value != 0` 的值，负值也参与；少于 8 个有效值报错。
3. 用 NumPy `percentile(...,[1,99],method="linear")` 得 `lo,hi`。排序后秩为 `(n-1)*p/100`，非整数秩线性插值。`hi <= lo` 的常量/退化体报错。
4. `u = clip(2*(value-lo)/(hi-lo)-1, -1, 1)`。源值 float32，分位数及公式中间计算 float64，最后存 float32。**原始零值也通过该公式，不强制改写为 -1**；只有后续重采样/letterbox 的外部补边固定 -1。
5. 先标准化，再重采样。`geometry.json.normalization` 记录实际 `low/high/method`。不使用均值方差标准化。

生产 PC 路径不往返 float16；历史训练缓存的 float16 存储不能作为另一个生产预处理实现。B 接收最终 `input_int8.bin`，不再标准化。

## 2. 三方向、轴排列与邻层

方向定义基于**文件声明的物理 RAS 坐标系重采样后的工作体**，不是直接对原始数组索引轴命名。RAS 轴正向为 Right / Anterior / Superior；若原始采集姿态本身缺失，只能遵从文件声明，不能声称恢复了真实采集姿态。

工作数组为 `V[X,Y,Z]`，轴 0/1/2 分别为 R/A/S。模型输入中 `H` 是行、`W` 是列：

| 模型 | 中心索引所属轴 | 中心平面 `[H,W]` | 行增加方向 | 列增加方向 | 平面尺寸 |
|---|---:|---|---|---|---|
| axial | 2，Z | `V[:,:,z].T` | +A | +R | `[Ny,Nx]` |
| coronal | 1，Y | `V[:,y,:].T` | +S | +R | `[Nz,Nx]` |
| sagittal | 0，X | `V[x,:,:].T` | +S | +A | `[Nz,Ny]` |

索引从 0 开始；中心层 `j=0..Naxis-1` 全部处理。通道依次为 `max(j-1,0), j, min(j+1,Naxis-1)`，邻层物理距离 2mm，首尾复制。三通道是空间邻层，**不混用 ED/ES 或时间帧**。2mm 插值不会增加原始厚层 MRI 的真实采样信息。

显示层可按放射学/神经学视图约定翻转画面，但不得改变推理张量或回传坐标；翻转必须由显示变换表达。

## 3. 工作网格与原生几何

输入仅限有效 3D NIfTI。原生尺寸 `N=(nx,ny,nz)` 保持；物理 affine 记为 `A`，源 affine/pixdim、采用策略和决定均存入 `geometry.json`。

- 默认 `strict`：affine 三列的范数须与 header pixdim 相符（`rtol=1e-4, atol=1e-5`），否则拒绝。
- 对已识别 ACDC 来源，可显式选 `acdc-direction-only-sform-v1`：仅在 qform_code=0、sform_code=2、三列范数均为1（`atol=1e-6`）、方向正交（`atol=1e-5`）、pixdim 有限且正时，将方向列分别乘 pixdim，保留平移。不是对任意异常 NIfTI 的通用修复。
- 实际 affine/pixdim 已相符则不修复。原始 header 不覆盖；输出和体积计算使用采用后的物理 `A`。

工作网格固定 spacing `[2,2,2]` mm，direction 为 RAS 单位矩阵。**origin 与 size 每例计算，不是全体固定为 256³，也不固定原点为0**：

1. 将原生体八个**体素中心角点** `(0 或 nx-1, 0 或 ny-1, 0 或 nz-1)` 经 `A` 映射到世界坐标。
2. 各轴最小、最大值记 `m,M`；`origin=m`，`size=ceil((M-m)/2)+1`，`working_affine=diag(2,2,2,1)` 并令平移为 `m`。
3. 使用 nibabel 5.4.2 `resample_to_output` / SciPy 一阶插值；输出点先逆映射到原生连续索引，体外常量值 -1。标签仅用于训练/评价，最近邻 `order=0`，体外0。

不做 ROI 裁剪/心脏包围框裁剪。`geometry.json` 必含 `native_shape/native_affine/source_affine/source_pixdim/working_shape/working_affine/working_spacing_mm/normalization/geometry_policy/geometry_decision`，由 SHA 绑定到每个板端请求。总堆栈数为 `Nx+Ny+Nz`，时延须按实际尺寸计，不能按固定层数估算。

## 4. 尺寸处理与输入量化

每个方向的三张原始平面一起做**保持比例的完整视野 letterbox**，目标256×256；2mm等间距工作网格下也保持物理比例。不是分别拉伸到正方形。

对原平面 `H,W`：`r=min(256/H,256/W)`；`H'=max(1,min(256,round(H*r)))`，`W'`同理。`round` 为最近偶数舍入。`top=(256-H')//2`，`left=(256-W')//2`，奇数余量落在下/右侧。

- 图像使用 PyTorch 2.11.0 `interpolate(mode="bilinear",align_corners=False,antialias=False)`。连续源坐标为 `(out+0.5)*in/out_size-0.5`，边缘按该实现处理；标签使用 `mode="nearest"`，不是 `nearest-exact`。
- 图像补边值 -1，标签补边0。保存 `original_hw/resized_hw/top/left/size`；逆变换严格复用这些值。
- 输入量化只在入口执行一次：`q=clip(RNE(float64(u)*127),-127,127).astype(int8)`；scale=`1/127`，zero_point=0，不出现 -128。
- 单堆栈为连续 C-order/NCHW `[1,3,256,256]`，196608 B。板端不再 resize、转置、归一化或量化。
- 注意：网络内部3×3卷积的 `padding=1` 补的是**量化数值0**，不是 letterbox 的 -127。MaxPool 为2×2/stride2，无补边；nearest×2 使用 `source_index=floor(output_index/2)`。

## 5. ED/ES

本版接收两个明确指定的独立3D文件：`--ed ED.nii.gz --es ES.nii.gz`；**不直接接收4D输入，也不自动找 ED/ES**。

官方 ACDC 的 `Info.cfg` 字段 ED/ES 原样对应文件编号：如 ED=1、ES=12，读取 `patient001_frame01.nii.gz` / `patient001_frame12.nii.gz`，不把它们当作 NumPy 从0开始的数组索引。官方100例的准确帧号见 [PATIENT_SPLIT.json](PATIENT_SPLIT.json)。未来若接4D，须新增明确的帧索引合同；当前不能静默减一或猜测。

调用者负责两个文件属于同一患者及对应时相；接口保存两文件 SHA 和显式 ED/ES 标记，不声称自动验证患者身份。两期可有各自几何/标准化参数，分别回到各自原生网格计算容积。

## 6. 连续尺度与量化参数

记 `Si=Cii.output_scale`、`S0=1/127`。每个方向独立一组 `S1..S15`；其**最终数值只能来自验收后的该方向 model_spec.json**。同 shape、同 profile 名不能代替 spec SHA，切换 SHA 时清理权重、bias、乘数及CAT缓存。

21个逻辑节点是15 Conv +3 Pool +3 nearest/CAT；ReLU融合于前14个Conv，nearest融合在CAT逻辑节点内。完整连续关系如下：

| 顺序 | 节点 | 输入尺度 | 输出尺度 |
|---:|---|---|---|
|1|C01|S0|S1|
|2|C02|S1|S2|
|3|Pool0|S2|S2|
|4|C03|S2|S3|
|5|C04|S3|S4|
|6|Pool1|S4|S4|
|7|C05|S4|S5|
|8|C06|S5|S6|
|9|Pool2|S6|S6|
|10|C07|S6|S7|
|11|C08|S7|S8|
|12|CAT0|skip S6 / deep S8|T0=max(S6,S8)|
|13|C09|T0|S9|
|14|C10|S9|S10|
|15|CAT1|skip S4 / deep S10|T1=max(S4,S10)|
|16|C11|T1|S11|
|17|C12|S11|S12|
|18|CAT2|skip S2 / deep S12|T2=max(S2,S12)|
|19|C13|T2|S13|
|20|C14|S13|S14|
|21|C15/head|S14|S15|

每个Conv的 `input_scale` 必须等于上表生产者/转换输出；Pool和nearest本身不改变scale。CAT两路分别转换到common，不允许隐藏额外转换。

整数生成规则（RNE均为带符号最近偶数）：

```text
Sw[o] = max(max(abs(W[o,:,:,:]))/127, 1e-8)
qw[o] = clip(RNE(float64(W[o])/Sw[o]), -127,127)      # OIHW int8
qb[o] = RNE(float64(b[o])/(Sx*Sw[o]))                # int32，单位为累加器单位
M[o]  = RNE((Sx*Sw[o]/Sy)*2^24)                     # 正int32，每输出通道一项
acc[o] = sum(qx*qw[o]) + qb[o]                      # int32
qout[o] = clip(RNE_divide(acc[o]*M[o],2^24),-127,127) # int64乘积
hidden: qout = max(qout,0)                           # C01..C14
head: 不加ReLU/sigmoid/softmax                       # C15
```

`qb/M` 不可溢出；M为0或超过INT32上限则拒绝。逐输出通道证明 `abs(qb)+127*sum(abs(qw)) <= 2^31-1`，再证明最坏乘积不溢出INT64。`RNE_divide` 对正负对称：±1.5→±2，±2.5→±2，不能用统一加半格的近似。bias和M文件为小端int32；交付实际整数参数，B不自行从浮点重新生成。

CAT：`Mskip=RNE(Sskip/T*2^24)`、`Mdeep=RNE(Sdeep/T*2^24)`，分别进行同一Q24舍入/饱和。mode4输入载荷为全skip CHW随后半分辨率deep CHW；输出先skip通道，再nearest×2后的deep。三个split依次64/32/16，通道数192/96/48；两路LUT各256项，索引为 `q & 255`，-128为非法生产输入。

每方向交付必须包含：FP32权重及SHA；15层INT8权重、Sw、int32 bias/M、输入/输出scale、边界证明和独立oracle；三组CAT两路scale/common/M及LUT生成规则；四通道共享head scale；参数文件字节数/SHA、完整图黄金输入/head及spec SHA。CAT LUT可由交付的两路M按相同Q24规则确定性生成。参数以当前包具体文件布局为准，不能只发一张尺度表。

head 为 NCHW int8 `[1,4,256,256]`，262144 B，zero_point=0，共享正scale `S15`。类别0/1/2/3为背景/RV腔/LV心肌/LV腔，颜色黑/红/绿/蓝。**生产scale尚未冻结，训练中的固定fake-quant范围不作为最终参数承诺。**

## 7. 概率恢复与融合

每个方向独立完成以下顺序，不能先argmax、平均类别、平均不同尺度的INT8，或先融合head：

1. `logits=float32(qhead)*该方向S15`，在256×256网格沿四类别轴做稳定softmax。
2. 先裁掉letterbox外边，再用bilinear、align_corners=False恢复到原平面的 `H,W`，以第2节逆转置放回本方向完整工作体。不是先对head插值再softmax。
3. 对每一类别概率，以 `working_affine` 为源、该例 `native_shape/native_affine` 为目标，一阶插值回原生物理网格，体外0。
4. 若某体素四通道概率和≤1e-8，把其背景通道赋1；再除以 `max(重新计算的通道和,1e-8)`。其他通道不额外清零。
5. 三个方向在**同一原生网格**按固定顺序 axial→coronal→sagittal 以float32累加、除以3；权重各1/3，无类别特异权重。没有按有效FOV方向数重新加权的隐含规则。
6. 最后沿类别轴argmax，完全相等选**最小类别编号**，输出uint8原生标签。

工作网格体外MRI补 -1；原生概率体外补0后按上述背景兜底。无额外有效FOV掩码融合。板端缺任一方向或任一中心层、重复层、来源/SHA/尺寸/head_scale不匹配，收集器报错，不用CPU或背景补齐缺失响应。

不做连通域筛选、孔洞填补、形态学平滑、网格平滑或概率阈值二值化；这是四类argmax接口，不沿用皮损/CAMUS阈值。FP32基线与INT8使用同一几何/后处理。浮点softmax可因设备/库产生末位差异，需用绑定版本及实际黄金用例验收，不能将算法相同宣称为跨实现逐位相同。

## 8. 评价与最终交付验收

固定患者级划分见 [PATIENT_SPLIT.json](PATIENT_SPLIT.json)：五组DCM/HCM/MINF/NOR/RV各16训练/4开发，共80/20；同患者ED/ES和全部切片只属于一组。官方50测试患者封存，未用于下载、校准、选模或调参。校准只用固定32训练患者、每期35%/65%两中心层，共128堆栈，不读GT。

- 主指标：20开发患者、每人ED/ES两个原生体；每类 `Dice=2|P∩G|/(|P|+|G|)`，双空为1、单空为0。先平均同患者两期，再平均20患者；主分数再对RV/MYO/LV三类等权平均，不计背景，不按切片/体素数加权。
- 距离：以一次6邻接二值腐蚀、边界外0提取体素表面中心，经完整物理affine转换到mm；双向最近点距离合并，HD为最大值、HD95为合并样本的95分位、ASSD为合并样本均值。双空距离0；单空距离null并明确原因，不填0冒充成功。此为本地口径，不冒充官方挑战成绩。
- 容积：原生标签体素数×`abs(det(A[:3,:3]))/1000`，单位mL；不用显示PLY网格容积。RV/LV EF=`100*(EDV-ESV)/EDV`；仅EDV>0且0≤ESV≤EDV时有效，否则EF为null并保留原始值/异常标记，不裁剪成正常值。
- 分别报告2D基线、三个方向、等权融合的真FP32与精确INT8。FP32关闭autocast/TF32；训练AMP选模分数单列。每份报告绑定患者划分、源权重/spec、运行代码SHA、后处理版本，指标缓存身份不符不得复用。
- 冻结INT8门槛：总体Dice下降≤.01、任一类≤.02、任一ED/ES体三类宏平均下降≤.05；非空GT不得新增相对FP32的全空预测；15层独立整数oracle均零差异。融合也检查同样量化门槛；对2D基线只报告真实增减，不承诺必然改善。

最终Release还须补齐验收模型注册表（每方向实际spec SHA/head_scale/参数SHA）、完整FP32/INT8/融合指标、实际黄金用例，以及中位与最差开发病例。**这些结果仍在产生，当前生产参数验收字段保持未通过，加载器必须拒绝将待验收状态当作生产链路完成。**

当前实现来源及库版本见机器合同的 `implementation_binding`。变更以上规则必须升预处理版本、重新生成黄金输入并重跑相关评价，不能沿用旧指标。

小型无模型检查及可重建合成用例见 [PREPROCESSING_CHECKS.json](PREPROCESSING_CHECKS.json)：覆盖32项规则检查和5个强度/几何/letterbox样例。float32样例散列按IEEE754小端C-order字节计算。该证据仅覆盖所列软件约定，不是生产模型或上板验收。
