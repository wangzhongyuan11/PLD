# ACDC Axial 2.5D

独立于 Skin/CAMUS 的 Axial-only 工程。输入显式 ED、ES 两个3D NIfTI；按A的生产数值文字合同预处理，逐Axial中心片执行15Conv/21节点，恢复原生概率与标签，提供三平面MPR、原始/平滑3D表面、LV/RV/MYO容积、LV/RV EF、异步网页与结果包。平滑表面由电脑端生成，仅用于展示；定量依据仍为原空间标签。

A真实研究Release已接入：原spec/参数身份保留，独立参考、Torch整数见证和tile模拟器的21节点精确一致。固定20名开发患者40体的软件Axial Dice为0.8823042898，与A逐体逐类Dice及容积差异均为0。当前事实和证据见[实施状态](docs/IMPLEMENTATION_STATUS.md)，范围以[最终优化计划](docs/FINAL_ALIGNMENT_OPTIMIZATION_PLAN_20261019.md)为准。

P6@100MHz已完成真实1000帧、逐节点数值和3类恢复板测：持续8.151268 FPS、P95 106.257678ms；布线WNS +0.189ns。2026-10-21又完成40体/1616片真实PL、15层实测、网页和原Skin/CAMUS恢复回归。当前P6网页为`http://127.0.0.1:8092/`，详细结果、固定身份和启停步骤见[验证与使用记录](docs/VALIDATION_P6_20261021.md)。交付包及补充Icarus网表结果待完成；xsim BASIC受实例数限制没有通过P6网表。旧[性能计划](docs/OPTIMIZATION_4FPS_8FPS.md)与[组件记录](optimization/README.md)保留历史版本范围。

当前硬件路径为`optimization/rtl`、`optimization/vivado/build_spatial.tcl`、`optimization/device.py`和`optimization/board_verify.py`（ACD8）；运行时按配对证书选择。下文ACD3/ACD4构建命令属于历史兼容路径，不能继承ACD8板测结果。运行包、模型和病例在Release或被忽略的`out/acdc`中。

当前手测：打开`http://127.0.0.1:8092/`，在ED选择`out/acdc/manual_test/patient039/patient039_frame01.nii.gz`，在ES选择同目录`patient039_frame10.nii.gz`，点击“开始完整分析”。右上角应为REMOTE/AXIAL，表示通过本机18093隧道调用板端真实PL。已有任务可继续浏览；模型与服务身份、恢复后的单片检查和临时暂停perf10的记录见上方验证文档。重启后必须重新确认实际FPGA拥有者，不要同时启动另一路推理。

## PC安装与直接演示

已有任务无需重新推理即可查看。没有板卡或模型连接时，在独立PC环境运行：

```powershell
& acdc/.venv/Scripts/python.exe -m acdc view --output out/acdc/manual_pl_jobs_20261020 --port 8092
```

打开 `http://127.0.0.1:8092/?job=<已有任务编号>`。右上角显示“历史结果 / 不执行推理”，上传与开始按钮禁用；这与调用板端的REMOTE模式不同。已有服务占用8092时可选择其他端口。

三维区标注“电脑端生成”，支持原始/平滑几何、表面/线框/半透明、三维透明度与RAS方向。平滑采用物理空间一次线性细分、80次Windowed Sinc（pass band 0.005）和重新计算法线；透明模式按观察方向排序各类别三角形。RV红、MYO绿、LV蓝对应不变。算法说明见[VTK官方文档](https://vtk.org/doc/nightly/html/classvtkWindowedSincPolyDataFilter.html)。

平滑副本先核对源标签与结果manifest的SHA，再写入`_presentations/<job>/<source+settings SHA>/<phase>/`缓存。原任务的标签、统计、计时、原始网格、manifest与ZIP均不修改；完整结果ZIP仍保留原始表面。截图附带“电脑端生成”、所用几何和截断提示。首次生成可能包含VTK导入耗时，之后同一标签与设置复用缓存，不执行PL。

9层、10mm层间距的数据可以改善外观，但平滑不能恢复未采集细节或证明解剖准确。触及视野边界仍提示截断。后续目标是板端独立任务服务和跨设备浏览；预处理/统计迁移及板端表面重建资源评估尚未实现。

在仓库根目录执行。PC依赖放入独立 `acdc/.venv`，不改变原工程环境。需要uv和Python3.13；安装脚本可以通过`-PythonPath`指定解释器。

```powershell
& acdc/install_pc.ps1
& acdc/.venv/Scripts/python.exe -m acdc plan
& acdc/.venv/Scripts/python.exe -m acdc phantom out/acdc/my_phantom
& acdc/.venv/Scripts/python.exe -m acdc web --backend demo --output out/acdc/jobs --port 8092
```

打开 `http://127.0.0.1:8092/`，上传刚生成的ED.nii.gz、ES.nii.gz。网页显示明确的DEMO/SYNTHETIC标志。已完成任务刷新后可恢复浏览，ED/ES分别加载各自几何；三个MPR共用RAS世界坐标光标，3D可旋转/缩放/裁切，类别显隐和alpha只改变显示。

直接CLI运行与断点续算：

```powershell
& acdc/.venv/Scripts/python.exe -m acdc run --backend demo --ed out/acdc/my_phantom/ED.nii.gz --es out/acdc/my_phantom/ES.nii.gz --output out/acdc/example
& acdc/.venv/Scripts/python.exe -m acdc run --backend demo --ed out/acdc/my_phantom/ED.nii.gz --es out/acdc/my_phantom/ES.nii.gz --output out/acdc/resumed --resume-from out/acdc/example
```

每次输出目录必须不存在；不会覆盖历史。checkpoint保存完整head和输入/模型/几何/backend身份；续算必须身份与字节hash全部一致。命中checkpoint或worker响应缓存不计新增推理样本/FPS。ZIP不含内部checkpoint目录。

## PC整数参考与可选合成演示

```powershell
& acdc/.venv/Scripts/python.exe -m acdc synthetic-model out/acdc/my_synthetic_model
& acdc/.venv/Scripts/python.exe -m acdc verify-model out/acdc/my_synthetic_model --allow-synthetic --full
& acdc/.venv/Scripts/python.exe -m acdc web --backend emulator --model out/acdc/my_synthetic_model --allow-synthetic --output out/acdc/neural_demo_jobs
```

`reference`执行独立完整INT64参考；`emulator`使用成熟分块设备，实际计算Conv/Pool/CAT，不读取golden代替前向。`--full`比较完整256×256输入下全部21节点。合成模型只用于数值/容量/软件链路测试，不适合展示真实分割质量。

真实Axial研究包已按固定A Release导入`out/acdc/axial_release_v2`，原spec SHA为`fef60178abaf8f81cd3206b2c37e3ca1aef56299ee01c2f8cb3dac3c28bf33be`。模型导入与[内部交换格式](docs/MODEL_BUNDLE.md)适配保留A的原始参数身份；按外部指定SHA核验：

```text
python -m acdc verify-model <model_directory> --model-spec-sha <A_approved_spec_sha> --full
python -m acdc run --backend reference --model <model_directory> --model-spec-sha <sha> --ed ED.nii.gz --es ES.nii.gz --output <new_output>
```

研究包核验完整参数/尺度、A软件资格、独立oracle、21节点见证、FP32源权重和实现绑定；错方向、错SHA或缺少数值见证均拒绝。A的软件资格与B的物理板测是分开的证据，不能由其中一项推导另一项。当前冻结模型、真实MRI输入、板端结果已完成对应检查，最终状态见验证记录。

## 几何与数值

- 默认strict几何：明确mm单位、可逆affine、affine列范数与pixdim一致。qform/sform的选用与代码均记录。
- ACDC原数据使用`--geometry-policy acdc-direction-only-sform-v1`和冻结A实现绑定，对未知单位及direction-only sform逐项校验；操作者声明或单独传`--recognized-acdc`不足以授权未知单位。strict仍拒绝该类输入，不自动修复任意NIfTI。40体工作网格与1616片整数输入已直接对照A全等。
- 原生整卷非零体素1%/99%线性分位数，float64中间运算后float32；原始零值也按公式变换。NaN/Inf、少于8个非零值、常量体拒绝。
- RAS2mm工作体、八个体素中心角点包围盒；Axial=`V[:,:,z].T`，首尾邻层clamp。层数=`Nz_ED+Nz_ES`，不是固定24或1048。
- 保比例letterbox用Torch2.11.0 bilinear/align_corners=False/antialias=False；量化值-127用于外部补边，网络卷积padding为0。
- 四类logits先softmax，再裁补边、恢复平面、回原生网格、按A规则概率兜底与归一化，最后argmax。无三方向融合、阈值、形态学或网格平滑。
- 标签输出使用采用后的物理affine；体积=体素数×abs(det(A))/1000mL。EF只在EDV>0且0≤ESV≤EDV时有效，否则null。

## 当前板端路径与历史兼容构建

当前P6使用ACD8（ID `0x41434438`，ABI `0x00020000`），PS一次提交整网DDR执行表，PL运行15Conv、3Pool及虚拟CAT。构建与仿真入口见[组件说明](optimization/README.md)，板测入口为`python -m acdc.optimization.board_verify`。同一P6运行包已通过1000帧/18物理节点/3类恢复及40体真实PL；仍保留P6功能网表终态缺口。以下ACD3/ACD4命令仅用于重现历史兼容实现，不代表当前部署。

兼容位流ACD3，ID=0x41434433；128MAC候选ACD4，ID=0x41434434；命令ABI=0x00030000。兩者沿用CAM2模式0～4、NCHW/OIHW、容量/Co分块和数值合同，使用独立RTL/IP/BD。驱动复用原有有界DMA/native搬运实现，硬件ID和runtime证书单独检查；PC导入ACDC模块不会加载PYNQ。

```powershell
$env:PROCESSOR_ARCHITECTURE='AMD64'
& D:/Git/bin/bash.exe acdc/run_sim.sh
& D:/Git/bin/bash.exe acdc/build.sh
& D:/Git/bin/bash.exe acdc/performance_ip/run_sim.sh
& D:/Git/bin/bash.exe acdc/performance_ip/run_tile_sim.sh
& D:/Git/bin/bash.exe acdc/build_fast.sh 100
```

脚本默认Vivado2026.1，支持VIVADO_BIN/ACDC_PYTHON。产物分别在out/acdc/vivado和out/acdc/vivado_fast100。当前兼容100MHz与128MAC候选100MHz通过布线；128MAC的125MHz版本时序失败，不能用于运行包。构建、时序、配对与source certificate都通过后才允许打运行包；运行包不代表板测通过。

```text
python -m acdc runtime --build <certified_build> --model <approved_model> --model-spec-sha <sha> --output <new_runtime_directory>
# 仅在真实PYNQ上执行；先停止hub及Skin/CAMUS后端，保持单DMA所有者。
sudo -E python3 -m acdc worker --backend pl --model <runtime>/model --model-spec-sha <sha> --runtime <runtime> --port 8093
```

worker只绑定板端loopback。PC建立SSH隧道到板卡loopback8093，再使用：

```text
python -m acdc web --backend remote --board-url http://127.0.0.1:18093 --model <same_approved_model> --model-spec-sha <sha> --geometry-policy acdc-direction-only-sform-v1
```

生产remote只接受真实backend=pl响应；无自动CPU回退。`deploy/render_unit.py`仅渲染服务模板；`deploy/manage.py`已用于真实systemd隔离部署及原hub回归，检查受管unit hash和干净退出。`active`不表示worker已就绪，必须再查`/health`身份。新出现的外部服务不自动加入停止/恢复清单；本次perf10由用户明确选择临时暂停，其异常退出与受控重配置均保留记录。安装核验模型/runtime，不接受合成包用于正式切换；已有活动ACDC时不要重复install。

```text
# 新版本人工复制到独立目录并校验后，在板端该目录执行：
sudo -E python3 -m acdc.deploy.manage install --release /home/xilinx/jupyter_notebooks/pld/releases/<new_acdc_version> --spec-sha <sha>
sudo -E python3 -m acdc.deploy.manage activate
# 用PC remote握手或本机 /health 确认真实就绪，systemd active本身不代表验收。
sudo -E python3 -m acdc.deploy.manage stop
# 需要撤销本工具安装的单元时：python3 -m acdc.deploy.manage rollback
```

历史ACD3/ACD4的`python -m acdc.board_verify`入口保留。当前ACD8使用`python -m acdc.optimization.board_verify`，可按`--cases`和`--node-oracle`校验首层起算、18个物理节点和重复推理；虚拟CAT通过下游及loader验证。当前真实故障恢复为descriptor tag、非法Ci和活动帧取消三类，不将旧入口的短包注入描述冒充新硬件已测用例。运行包和报告均记录synthetic身份，数值/调试抓取与性能计时分别记录。

```text
# 当前ACD8离线板测，先停止网页worker并确认DMA/FPGA单拥有者：
sudo -E python3 -m acdc.optimization.board_verify --runtime <runtime> --model <runtime>/model --model-spec-sha <sha> --cases <cases.json> --node-oracle <nodes.npz> --node-oracle-sha <nodes_sha> --warmup 20 --repeat 1000 --recovery-checks --output <new_report.json>
```

## 输出、测试与PR交付

结果目录包含labels/ED.nii.gz和ES.nii.gz、各期geometry、statistics.json/csv、performance.json、quality.json、VTK表面与浏览器二进制、manifest和ZIP。completion.json记录服务端上传读取至ZIP就绪的完整时间；ZIP中performance的pre_export_seconds不冒充最终导出时间。浏览器渲染及实际下载另报。

```powershell
& acdc/.venv/Scripts/python.exe -m pytest acdc/tests -q
node --check acdc/web_assets/app.js
node acdc/tests/test_viewer.cjs
```

本轮通过GitHub PR交付源码、测试、脚本、文档与小型证据，不再生成压缩包。只读复用组件及hash清单保留；虚拟环境、模型、病例或生成位流不进入PR。历史RC保留原路径及SHA，不替代当前PR状态。PC浏览器使用自带Canvas MPR与WebGL表面显示，无CDN依赖；VTK生成无平滑VTP/PLY表面。真实模型开发集质量、板端性能/内存/1000帧稳定性和浏览器导出已有证据；物理HDMI目视不在本ACDC网页验收范围，P6功能网表尚待终态，候选包不冒充最终验收。

有真实GT后，使用`python -m acdc evaluate --help`。病例清单schema为`pld-acdc-evaluation-cases-v1`，`cases`每项包含`patient`、`phase`、`result_dir`、`gt`、`gt_sha256`；每患者必须ED/ES配对，路径相对清单目录。预测标签按结果manifest核验hash，GT也必须固定hash，原生物理网格必须一致。输出逐类Dice、HD95/ASSD、容积和EF误差；该命令不替代A的FP32/INT8量化质量验收。
