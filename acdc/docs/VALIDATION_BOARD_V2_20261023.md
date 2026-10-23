# ACDC 独立板端 V2 最终部署与验证

2026-10-23。当前板端为 **C12 免访问码版**，本机浏览器实际容量边界任务上传至完整可交互 **38.521 秒**。用户要求优先本机验收、取消第二设备测试并尽快收尾，已停止后续重复推理。机器可读记录见[最终证据](evidence/appliance_final_20261023.json)。

## 当前使用

打开 **http://192.168.2.99:8092/**，无需访问码。板卡显示“PL 就绪”后，选择同一患者的 ED、ES 三维 `.nii` / `.nii.gz`，点击“开始完整分析”。网页显示两期完成后，可切换 ED/ES、联动 MPR、调窗宽窗位/标签透明度、显示类别/轮廓、旋转缩放三维、切换原始/平滑表面、线框/半透明、裁切及下载 ZIP。

电脑只需要浏览器和原始文件；不依赖本机 Python、模型或 SSH 隧道。上传和结果浏览仍需电脑与板卡网络连通。V2 的访问码界面、认证 Cookie、session 接口及读写访问码文件逻辑均已移除；旧 V1、Skin/CAMUS 的设置保持原样。

开机或重启服务时，ARM 加载模型、核验身份、准备原生库/网格并执行真实 PL 自检，完成前禁用上传。C12 本次 engine 初始化 **56.241 秒**；该时间单列，不能声称开机到结果低于40秒。服务已设置开机启动，本轮未另做断电冷启动。

## 实现与数据流

1. 浏览器上传两期原文件；板端 Web 校验长度、文件类型及任务预约。
2. ARM 读取 NIfTI 和几何，按 A 的冻结口径独立归一化、RAS 2mm 重采样，组装相邻三片、letterbox 和 INT8 RNE 量化。
3. 唯一 engine 调度 DDR/DMA/PL；P6 执行完整固定 Axial 网络。没有用 CPU 参考替代实际推理。
4. ARM 对四类输出执行 Softmax、撤销补边、恢复原空间概率后 argmax，输出原空间标签；计算 LV/RV/MYO 体积与 LV/RV EF。
5. ARM 生成原始/平滑网格、板端 MPR、CSV/JSON/NIfTI/PLY/VTP 和完整 ZIP。平滑只改变展示网格，不改变标签、体积或 EF。
6. 浏览器接收两期初始 MPR 和平滑索引网格，上传 GPU 并绘制首帧。计时从上传开始，直到两期完整资源、ZIP 可下载和 GPU 首帧全部就绪。

实现位于 `acdc/appliance/`。`pipeline.py` 负责单 PL 拥有者、切片输入预取和概率恢复重叠、ES 预处理预取；`full_pipeline.py` 使 ED 展示生成与 ES 推理重叠。`native/image.c` 实现精确数值操作与 MPR 坐标复用；`png.py` 生成无损 Sub/RLE PNG；`archive.py` 以256KiB块流式生成 ZIP_STORED，同时计算 CRC/SHA。`web.py` 的两体 MPR 缓存初始化计入结果资源加载，不从40秒终点扣除。缓存/任务队列有界，不同时保存两期完整概率体。

## 实测结果与证据范围

| 验证 | 结果 |
|---|---|
| C11 连续40次：20位开发患者各两次 | 3,232次新鲜 PL；每片输入/head 与独立整数参考一致 |
| 原空间恢复 | 全40任务 ED/ES 标签零差异，affine、mm单位、体积/EF及ZIP SHA/CRC/全部成员一致 |
| 40次上传至完整初始资源下载 | 最短12.059s，P50 18.554s，P95 23.272s，最长27.223s；这组不冒充浏览器GPU计时 |
| C12 实际本机浏览器容量边界 | 两期112片，上传至完整可交互38.521s；服务器36.166s |
| C12 该任务的独立审计 | 112片输入/head、原空间标签/affine/单位/统计、98个清单成员与ZIP校验通过 |
| C11 单片 PS-DMA-PL | P50 112.609ms，P95 131.519ms，最大231.509ms；不据此把整卷链路称为8FPS |
| C12 容量任务单片 | P95 139.732ms，最大157.995ms |
| C11 空闲 MPR，30个不命中PNG缓存的位置 | 400×400，P95 179.520ms |
| C11 新鲜推理期间 MPR，18次 | P95 328.619ms，达到忙时500ms目标 |
| 本地验证 | 87项Python测试与实际前端加载/切期/异步GPU所有权JS检查通过 |

容量样例每期原始512×384×10、工作306×229×56；从公开ACDC样例调整得到，**仅用于性能和容量验证，不用于医学准确率判断**。C12 与 C11 的数值/PL/MPR实现相同，最后修改仅为移除访问码、界面与验证客户端认证依赖；40次证据属于C11，C12补一次实际全链路验证，没有声称C12另跑40/60次。

原计划60次的原始报告保留 `complete=false`：在两轮共40次已完整审计后，按用户新要求主动结束第三轮，不把它改成60次通过。跨设备实测同样由用户取消，不写成已验证另一台设备。

## 资源、旧版与限制

C11 连续窗口采集1,198份资源样本，覆盖约1,199.9秒；engine最大RSS190.68MiB、VmHWM200.56MiB，Web最大RSS45.54MiB。MemAvailable最低188.19MiB；engine cgroup峰值300.16MiB包括可回收文件缓存和子进程，不等于主进程RSS。CMA最低空闲104KiB；发生swap活动，未出现OOM，不能声称零swap或仍有大量CMA余量。该40次窗口启动ID不变；此前C9整板重启原因仍未知，本次短时通过不证明任意长期运行均稳定。

旧V1已实际切回并用新鲜输入核对整数head；Skin/CAMUS各两例原链路和导出回归通过，随后恢复V2。这些是C9阶段的回退证据；C10～C12没有修改旧组件。本轮未改 `board/`、`camus/`、`display_web/`、`rtl/` 或 `delivery/current.json`。P6已有布线/板测证据继续保留；功能网表仿真受xsim BASIC实例数限制的历史缺口没有由本轮软件测试关闭。没有进行新综合。

40秒结论适用于服务READY、单计算任务、当前输入容量限制和已测样例。每期原始≤2,000,000体素、工作≤4,000,000体素，两期上传≤32MiB；更大文件拒绝。其它输入、较慢网络、浏览器或额外负载不保证40秒。MPR三平面来自同一Axial分割体；没有新增冠状/矢状模型，也不会补造短轴采集视野之外的解剖。

浏览器自动化的下载事件接口未完成验证；ZIP已通过直接HTTP下载和全部内容校验。截图保存按钮沿用已测试源码，浏览器自动下载落盘不列为本轮通过项。

## 版本、复现和回退

- 当前目录：`/home/xilinx/acdc_board_v2/releases/candidate12_20261023`。
- 应用SHA256：`9459bbad2a814b54d28748b22c824910c9451e484b25a125e7c8116f1d2405c3`；37个应用文件与提交源码逐字节核对一致。
- 模型：`fef60178abaf8f81cd3206b2c37e3ca1aef56299ee01c2f8cb3dac3c28bf33be`。
- P6核心运行清单：`1f0bdf8739980ea8e8e2a06ee349b57ed81d041d87515e4e9400cd30b7ae7747`；位流不变。
- 服务：`pld-acdc-v2-engine.service`、`pld-acdc-v2-web.service`，均active/enabled；旧服务停用，文件和数据保留。
- 数据：`/home/xilinx/acdc_board_v2/final_data_20261022`；应用部署记录和前次unit位于`/var/lib/acdc-board-v2/deployment.json`。

复现入口（验证报告输出路径必须为新目录）：

```text
python -m pytest acdc/tests -q
node acdc/tests/test_appliance_viewer.cjs
python acdc/appliance/validate_http.py --results <独立参考结果> --output <新目录> --patient patient057
python -m acdc.appliance.audit_http --results <独立参考结果> --download <上一步目录> --output <新审计.json>
python -m acdc.optimization.measure_appliance_mpr --job <已完成任务ID> --output <新报告.json>
```

验证客户端无需SSH凭据。原生模块在Windows编译时将实际GCC目录置于PATH前端；ARM按既有构建脚本和固定sysroot构建，不升级系统PYNQ/NumPy。

回退须先停止V2 Web和engine，确认engine进程退出、DMA释放，再按保留的V1部署管理器启动旧release；不要同时启动两个FPGA服务。V2前次unit和各候选目录均保留。用户要求通过PR交付，本次不打交付包、不发Release、不合并PR。
