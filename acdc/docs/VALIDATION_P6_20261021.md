# ACDC Axial P6 验证与使用记录

日期：2026-10-21。范围为冻结 A Axial B16 模型；没有启用 Coronal/Sagittal 模型或三方向融合。本文记录实际证据，不将尚在运行的补充验证记为通过。

## 当前可用实现

输入同患者 ED/ES 两个3D NIfTI。PC按冻结几何与归一化规则建立RAS 2mm工作体，给每个Axial中心层准备相邻三层INT8堆栈，经SSH隧道交给PYNQ-Z2。PS一次提交整网执行表，PL执行15个Conv、3个Pool及3个虚拟CAT，共21个逻辑节点；head为四类有符号INT8。

PC将head转换为概率，裁letterbox、恢复工作平面并回原生物理网格，最后argmax。体积来自原生标签计数和affine行列式，LV/RV EF来自两期容积；MYO不计算EF。网页提供三个MPR、三维表面、窗宽窗位、联动光标、类别/轮廓/透明度、裁切、ED/ES和导出。查看已有结果不触发新推理。

当前隔离运行根：`/home/xilinx/jupyter_notebooks/pld/releases/acdc_spatial_20261020_p6_rpc_v1`。受管单元为`pld-acdc.service`，worker仅监听板端loopback 8093。PC测试页为`http://127.0.0.1:8092/`，通过本机18093转发板端8093。原`delivery/current.json`保持Skin/CAMUS正式交付指针。

| 身份 | 值 |
|---|---|
| Axial模型spec SHA256 | `fef60178abaf8f81cd3206b2c37e3ca1aef56299ee01c2f8cb3dac3c28bf33be` |
| Runtime manifest SHA256 | `1f0bdf8739980ea8e8e2a06ee349b57ed81d041d87515e4e9400cd30b7ae7747` |
| Runtime ZIP SHA256 | `c61f3b835b66b624a951fd3281f2401dd343dc9df00683aeb40cb01ca904c427` |
| bit SHA256 | `232954378f8c7a42c36589031d9588f50ea2c370f3a29edd40500bab877dc1df` |
| 构建 | `spatial_p6_100_bankstage_v1`，ACD8 / ABI 0x00020000，100MHz |

## 性能与数值

20次预热不计入样本，1000个新鲜帧覆盖21个真实整数输入。每帧head SHA已按独立CPU oracle重新审计，全部相等；另有15个Conv、3个Pool的物理DDR输出全等，虚拟CAT通过下游卷积与loader验证。异常descriptor、非法Ci、活动帧取消后均重新从首层执行并验证head，三种恢复通过。

| 指标 | 实测 |
|---|---:|
| 新鲜1000帧墙钟 | 122.680298 s |
| 持续吞吐，包含比较/报告/任务空隙 | **8.151268 FPS** |
| 完整PS-DMA-PL中位延迟 | 105.822606 ms |
| 完整PS-DMA-PL P95 | **106.257678 ms** |
| 完整PS-DMA-PL最大值 | 116.584966 ms |
| 中位PL执行时间 | 95.540735 ms |
| RSS采样范围 | 87,326,720～87,719,936 B |
| CMA空闲采样范围 | 33,390,592～34,205,696 B |

PS-DMA-PL的4 FPS/250ms及8 FPS/125ms性能门槛均通过。RSS为采样值，不能称为进程峰值或整板峰值；采样包含报告列表的有界增长。连续验证规模为1000帧，不冒充长时间运行或热设计认证。

固定20名开发患者、40个ED/ES体、1616片全部重新由P6推理，每片与CPU参考head比较。所有head、原生标签、affine、容积、EF相等；20个结果ZIP的SHA/CRC通过。与A逐体逐类Dice和预测容积差异为0，宏Dice为0.8823042897748437。该集合参与模型选择，不能称为独立测试。

PC远程请求的1616片中位174.748400ms、P95 219.439400ms、最大247.678500ms。逐患者新鲜推理窗口合计吞吐5.164175 FPS，包含调度、checkpoint和期相间处理；不是完整验证进程墙钟，也不是板内8 FPS口径。20例CLI结果ZIP就绪时间中位16.613500s、范围10.726444～21.859038s，没有上传时间。真正网页上传patient039的82片样例从服务器收输入到ZIP就绪20.306547s；浏览器下载ZIP与服务器SHA一致，PNG导出及ED/ES/MPR/3D显示通过。

独立恢复审计调用A冻结`geometry.py`和Torch softmax，对照B NumPy/Collector，覆盖40体/1616片。原生概率最大误差1.7881393432617188e-7，平面最大误差2.384185791015625e-7；标签差异0。没有把已保存head重算恢复计为新鲜PL样本。

`validate_preprocessing.py`另行直接调用固定Release中的A几何实现，逐体比较采用的原生/工作affine及全部float32工作体，逐片比较相邻三层的INT8量化输入，并核对真实PL checkpoint的输入SHA。40体/1616片全部精确一致；这项只审计预处理，不增加新鲜PL推理数。[独立输入证据](evidence/preprocessing_A_20261021.json)。

## 单层、资源与数据流

`optimization/profile_layers.py`先由真实PL前缀产生该层输入，再独立执行该层的job；输出与独立整数oracle比较，未注入中间golden。每层3次，含本层DMA、虚拟CAT和融合Pool；调试输出复制/比较单列。由于单层边界、缓存/重叠条件与整网不同，不能把这些数相加冒充整网延迟。

| 层 | PL中位 ms | 层 | PL中位 ms | 层 | PL中位 ms |
|---|---:|---|---:|---|---:|
| C01 | 3.539 | C06 | 4.749 | C11 | 14.028 |
| C02 | 6.300 | C07 | 2.522 | C12 | 5.231 |
| C03 | 2.673 | C08 | 5.021 | C13 | 16.258 |
| C04 | 5.281 | C09 | 13.774 | C14 | 6.238 |
| C05 | 2.388 | C10 | 4.721 | C15 | 2.882 |

| 布线指标 | 值 |
|---|---:|
| LUT | 21,143 / 53,200 |
| FF | 28,274 / 106,400 |
| BRAM36等效 | 106.5 / 140 |
| DSP | 210 / 220 |
| WNS / WHS / WPWS | +0.189 / +0.013 / +3.750 ns |
| DRC | 229 Warning、3 Advisory；0 Error、0 Critical Warning |

主要性能来源是P6空间并行、Ci8×Co8归约、精确双INT8乘积DSP映射、四路Q24、分支尺度查表与CAT/Pool融合、双slot预取、输入和参数复用、自主DDR执行表、PS NEON布局复制。逐tap除6已改为递增组索引；BRAM分页/slot选择与空间旋转之间增加弹性寄存器，修复P6时序。CMA arena按实际页对齐申请6,004,736B；该值不是进程全部内存。

模型每片2,674,917,376 MAC，384逻辑乘法通道、100MHz理论38.4 GMAC/s。理想纯MAC下界约69.66ms；实测PL约95.54ms，包含DDR、边界、部分通道和控制开销。完整RTL的事务计数可用于DDR流量估计，不能冒充物理DDR带宽实测。功耗报告如提供，是厂商工具无活动向量估计，不是电源测量，不能用于证明整板能耗。

## 原系统与服务切换

已正常停止ACDC、恢复正式hub，并分别通过Skin两张原JPEG、CAMUS两张原NIfTI的真实PL黄金head、PNG像素/尺寸和ZIP CRC检查；随后正常停止原系统并恢复P6 worker。正式unit、模型、位流和数据未覆盖。本次不声明物理HDMI目视正确。

恢复原hub后必须等待认证`/hub/status`的`phase=ready`，再停止或重新激活ACDC。`systemctl active`和HTTP首页可访问都不足以说明后台模型就绪：本轮曾因此被安全门禁拦下，记录保留，未放宽干净退出条件。所有等待有界，DMA状态不确定时保留拥有者与CMA，不用强杀恢复。

## 复现与使用

在Windows仓库根目录使用独立PC环境，模型及数据按固定A Release SHA导入。手动输入位于`out/acdc/manual_test/patient039`：ED=`patient039_frame01.nii.gz`、ES=`patient039_frame10.nii.gz`。网页结果保存在`out/acdc/manual_pl_jobs_20261020`。下载完整ZIP后可在支持NIfTI/PLY/VTP的软件中检查标签和表面。

```powershell
$AxialSha='fef60178abaf8f81cd3206b2c37e3ca1aef56299ee01c2f8cb3dac3c28bf33be'
# 在另一个终端建立SSH隧道，凭据交互输入，不写脚本。
ssh -N -L 127.0.0.1:18093:127.0.0.1:8093 xilinx@192.168.2.99
& acdc/.venv/Scripts/python.exe -m acdc web --backend remote --model out/acdc/axial_release_v2 --model-spec-sha $AxialSha --board-url http://127.0.0.1:18093 --geometry-policy acdc-direction-only-sform-v1 --output out/acdc/manual_pl_jobs_20261020 --port 8092
```

板端从配对、校验过的冻结运行包根执行。首次install不能覆盖已活动ACDC，换版本先正常停止；启动前等待原hub模型ready，停止后再确认全部旧服务及其子进程退出。激活后另查8093 `/health`的真实PL/model/runtime身份。

```bash
sudo -E /usr/local/share/pynq-venv/bin/python3 -m acdc.deploy.manage install --release /home/xilinx/jupyter_notebooks/pld/releases/acdc_spatial_20261020_p6_rpc_v1 --spec-sha fef60178abaf8f81cd3206b2c37e3ca1aef56299ee01c2f8cb3dac3c28bf33be
sudo -E /usr/local/share/pynq-venv/bin/python3 -m acdc.deploy.manage activate
# 正常释放ACDC并恢复原hub：
sudo -E /usr/local/share/pynq-venv/bin/python3 -m acdc.deploy.manage stop
# 撤销最后一次受管unit安装、保留所有版本目录：
sudo -E /usr/local/share/pynq-venv/bin/python3 -m acdc.deploy.manage rollback
```

主要证据为本目录`evidence/p6_board_20261020.json`、`development_PL_20261021.json`、`restoration_A_20261020.json`、`board_regression_20261021.json`、`web_P6_20261021.json`；其中原始大报告有路径和SHA。最终运行包携带冻结源码，当前Git中的新增诊断/文档不冒充该运行包已执行的文件。

## 尚未关闭的补充验证

完整尺寸RTL与真实板测已通过。P6 OOC功能网表有59147实例，xsim BASIC限制50000，启动失败，不能记为网表数值通过。已用MSYS2官方仓库的Icarus 12.0作为补充；其不支持原DDR模型的SV关联数组，因此增加有界可移植存储。该存储在xsim上以同一A参数、272 job/16768写回/313888周期与原模型一致。Icarus网表运行在首次读写前因未知控制状态失败；基础厂商LUT/FDRE/GSR探针通过，原因仍待定位，不以真实板测成功掩盖此失败。

交付包及最终验收状态需在实际打包、SHA/CRC、源码/运行包身份和此补充验证结果核对后更新。工程教学与算法演示，不作为临床诊断依据；A数据/模型许可及引用随原Release保留。

补充定位：相同问题在全展平网表下仍出现；VPI只读观察表明复位释放后状态最初正确，首次descriptor握手时才变成X。最小`sim/tb_unisim_initialization.sv`复现Icarus对声明初值的事件调度差异：原厂LUT6的显式输入敏感进程可能漏掉时刻0的声明赋值。网表TB现改为时刻1ns、仍保持复位时驱动静态输入，未修改RTL、综合网表、原厂模型或期望数据。完整自校验重跑结果尚待完成，最小用例通过不替代整网通过。

RC包复现（输入清单固定每个运行包/报告/结果的SHA；不上传、不部署）：

```powershell
& acdc/.venv/Scripts/python.exe -m acdc.package_delivery create --inputs acdc/delivery/inputs_rc1.json --output out/acdc/delivery/acdc_axial_p6_20261021_rc1
& acdc/.venv/Scripts/python.exe -m acdc.package_delivery verify out/acdc/delivery/acdc_axial_p6_20261021_rc1
```

RC清单中的`acceptance_complete=false`保留网表缺口。包内含源码、冻结运行包、原始板测与40体评估报告、全部20例结果ZIP及浏览器示例；原始训练数据不重复分发，按原Release SHA取得。
