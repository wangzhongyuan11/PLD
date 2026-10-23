# ACDC Board Standalone V2（实施中）

目标和完整验收范围见[独立板端计划](../docs/BOARD_STANDALONE_V2_PLAN_20261021.md)。V1 保留在提交 `69e18ec`，V2 工作位于独立任务分支。现已实现独立板端 Web、真实 PL、网格、MPR、统计和导出，正在优化并进行完整验收。首例真实浏览器为 30.72 s；首轮 20 例中 patient057 约 54 s，触发后续优化，不能声称全部输入均已低于40秒。

## 已实现

- 原生 C 的归一化、Torch 等价双线性缩放、INT8 RNE 量化、SciPy 等价三维线性重采样。
- 三切片 LRU；经过参考环境生成的 Softmax 查表；单期概率体与分块原空间恢复，不提前 argmax。
- 轴对齐坐标缓存、一次检查后复用的采样器，避免分块反复扫描完整概率体。
- `NumericalPipeline` 提供 ED/ES 数值链路、单硬件执行者、输入预取/输出恢复重叠、有界 future、取消和失败状态。当前输出状态为 `numerics_completed`，明确不等于最终完整任务完成。
- 板端MPR采样、窗宽窗位、标签叠加/轮廓与PNG生成已接入LAN网页。浏览器仅显示PNG、绘制十字线及渲染三维网格，不执行医学空间采样。
- 全20例/40体/1616片开发机输入/标签对照；板端ARM首期46片核对。证据见[原生模块记录](../docs/evidence/appliance_native_20261021.json)。这里使用保存的head验证后处理，新增PL推理次数为0。
- 已在独立候选完成真实P6两期92片：输入/head逐片与独立参考一致，原空间标签/affine/统计一致，数值链路19.483 s，单片P95 119.363 ms。见[两期PL记录](../docs/evidence/appliance_numerical_PL_20261021.json)；不包含初始化/上传/网格/导出/浏览器，尚非40秒整体验收。

板上候选目录与旧部署分离，PYNQ/系统NumPy未升级。候选使用隔离的 Nibabel 5.2.1（需要 NumPy>=1.20），板端已有 NumPy1.21.5。PC参考继续使用原固定环境；实际实现版本和原生库SHA另行记录，不伪造原A绑定。

## 已有验证入口

在仓库根目录、原固定PC环境中执行；路径需替换为实际工件位置：

```text
python -m pytest acdc/tests/test_appliance_native.py acdc/tests/test_appliance_pipeline.py -q
python -m acdc.appliance.qualify --model <冻结模型> --model-sha <spec_SHA> --results <已有40体结果根> --output <新报告.json>
python -m acdc.appliance.assets --model <冻结模型> --model-sha <spec_SHA> --output <新派生常量目录>
```

`qualify`核对MRI/原结果manifest及检查点身份，比较实际原生工作体、全部整数输入与原空间标签。它不执行PL，不能用于板端FPS或完整任务速度验收。

`benchmark_replay`在板端读取已校验的单期fixture：原始MRI、冻结exp表、原始工作体/输入/标签SHA及保存的head。fixture必须由PC参考准备；它不连接远程推理或用CPU代替PL。

```text
python -m acdc.appliance.benchmark_replay --fixture <fixture目录> --output <新ARM报告.json>
```

编译器默认 `gcc`，可设 `ACDC_IMAGE_CC`；首次构建在系统临时目录生成按源码、编译器、flags和机器身份绑定的共享库，以后核验并复用。禁用fast-math/FMA收缩；ARM使用Cortex-A9/NEON hard-float选项。编译缓存准备时间单列，不纳入已就绪数值阶段。

精简VTK已在真实ARM运行，无GUI/Python VTK。展示参数为不细分、40次Windowed Sinc、passband 0.02、Nuttall；全40体与V1展示表面对照通过，标签与统计不变。[几何证据](../docs/evidence/appliance_surface_profile_20261021.json)。

隔离构建入口 `native/mesh/build_vtk.sh` 可在板卡编译，但内存和时间开销较大。实际采用 Linux/WSL、GCC11 armhf 和板端实际sysroot交叉构建：`snapshot_sysroot.py`读取板端头文件/libc/C++库，`build_cross.sh`重建headless VTK与网格程序。工具链排除交叉发行版的新libc头，避免GLIBC2.39符号混入。host-tools package仅解除主机工具位数比较，不改变ARM ABI。构建后必须在板上运行`--version`和几何对照。不会修改系统PYNQ/NumPy。

## 板端使用与数据流

同一局域网的任意电脑访问 `http://192.168.2.99:8092/`，输入本板访问码，选择同一患者的ED/ES三维NIfTI。待顶部显示PL就绪后开始分析；电脑无需Python、模型或SSH隧道。访问码仅在板卡`data/access_code.txt`，不随源码共享。网络用于上传、下载展示资源及MPR交互。

```text
浏览器 ED/ES 上传
 → ARM：校验几何、分期归一化、RAS 2mm重采样
 → ARM：相邻三片缓存、letterbox、INT8量化
 → DDR/DMA → PL：冻结P6完整Axial网络
 → ARM：Softmax、撤销补边、概率恢复到原空间、argmax
 → ARM：原空间标签、体积/EF、原始/平滑表面、MPR、CSV/JSON/ZIP
 → 浏览器：两期资源预取、PNG显示、GPU网格旋转/缩放/裁切
```

三幅MPR来自同一个Axial分割体，不代表三方向模型。冠状/矢状厚度受原始短轴采集覆盖限制，显示插值不能补造视野之外的结构。平滑仅改变展示表面；定量使用原空间标签。

| 文件 | 职责 |
|---|---|
| `__main__.py`、`assets.py`、`identity.py` | 预载、真实PL启动自检、派生常量与独立应用身份 |
| `engine.py`、`web.py` | 单硬件拥有者、上传预约/取消/状态、LAN网页、板端MPR API |
| `pipeline.py`、`reconstruct.py` | 一个预取、一个平面恢复、单PL调用者；原空间双核有界slab、ES预处理预取 |
| `surface.py`、`native/mesh/` | 物理ROI、原始/平滑PLY/VTP及索引网格 |
| `full_pipeline.py`、`archive.py` | 两期完整性、ED展示与ES推理重叠、统计、公共资产、流式ZIP、CRC与内联SHA |
| `web_assets/` | ED/ES切换、MPR参数、类别、三维样式、截图及客户端计时 |
| `deploy.py` | 独立V2服务、保留前次unit及旧部署 |
| `validate_http.py`、`monitor.py` | 真实LAN任务和板端资源证据 |

ZIP包括原空间NIfTI标签、几何/数值记录、统计CSV/JSON、原始和平滑PLY/VTP、索引网格、初始MPR及SHA清单。灰度/标签NPY仅留在板端供交互，不公开下载。失败或取消不会发布完整结果。

`assets_ready`只在两期全部资源及ZIP完成后发布；网页以`performance.now()`从上传开始测至两期资源接收、GPU上传和首帧绘制。启动预载另列，初始化期间禁用开始。每期原始体素≤2,000,000、工作体素≤4,000,000，两期上传硬限制32MiB；性能包络见计划。结果限额4GiB，不自动删除旧任务。

存储用量在Web启动时盘点，之后只更新刚结束/中断的任务及客户端计时文件，避免每次上传扫描全部历史。手工移动或清理V2结果后重启Web以重新盘点。验证客户端显式直连LAN，避免系统代理干扰。默认MPR光标定位于分割区域，三维镜头按分割网格适配；默认裁切范围覆盖完整体素表面，不改变标签与定量。

部署复制认证P6核心到`/home/xilinx/acdc_board_v2/releases/<版本>`，添加appliance、`v2_assets`、隔离Nibabel5.2.1的`v2_deps`、网格二进制和配置。保持原runtime manifest，新增应用manifest。使用PYNQ venv Python和登录shell初始化XRT：

```text
python -m acdc.appliance.identity <新release>
sudo -E python -m acdc.appliance.deploy --release <新release> --app-sha <应用SHA>
sudo systemctl start pld-acdc-v2-engine.service pld-acdc-v2-web.service
python acdc/appliance/validate_http.py --results <参考结果> --output <新板测目录> --repeat 3
```

回退先停止V2两个服务并确认进程和DMA退出，再通过V1原管理器激活旧release。`/var/lib/acdc-board-v2/deployment.json`保留前次unit；旧核心、原服务文件及历史数据均保留。不可并发启动两个FPGA服务。

## 尚待完成

2026-10-23 MPR候选：保留C9通用斜切采样，增加轴对齐坐标/权重缓存、不可变体数据一次校验，并将现有PLD固定Sub/RLE无损PNG策略扩展到RGBA。ARM两种体数据18帧与C9逐像素一致，暖缓存微基准中位239ms降至83ms；这是MPR/PNG微基准，尚不能当作HTTP交互或整卷验收。此变化不修改模型、PL位流、标签与定量。

2026-10-22：跨期预处理/展示重叠和ZIP_STORED已实现。容量边界两期112片单次完整资源下载39.32s，输入/head/原空间标签/几何/统计一致；这不是连续验收结论。ZIP保留全部文件，较DEFLATE版本更大，采用256KiB流式复制、CRC和同步SHA，避免二次全文件读取。最终候选20例各3次新鲜任务、资源/取消/故障恢复、开机readiness、V1/Skin/CAMUS回归及实际回退。最终按浏览器上传开始到两期完整结果、平滑展示、ZIP和首帧全部可用验收 `<40 s`。
