# ACDC Axial 2.5D

独立于 Skin/CAMUS 的 Axial-only 工程。输入显式 ED、ES 两个3D NIfTI；按A的生产数值文字合同预处理，逐Axial中心片执行15Conv/21节点，恢复原生概率与标签，提供三平面MPR、无平滑3D表面、LV/RV/MYO容积、LV/RV EF、异步网页与结果包。

A真实研究Release已接入：原spec/参数身份保留，独立参考、Torch整数见证和tile模拟器的21节点精确一致。固定20名开发患者40体的软件Axial Dice为0.8823042898，与A逐体逐类Dice及容积差异均为0。当前事实和证据见[实施状态](docs/IMPLEMENTATION_STATUS.md)，范围以[最终优化计划](docs/FINAL_ALIGNMENT_OPTIMIZATION_PLAN_20261019.md)为准。

P6@100MHz已完成真实1000帧、逐节点数值和3类恢复板测：持续8.151268 FPS、P95 106.257678ms；布线WNS +0.189ns。2026-10-21又完成40体/1616片真实PL、15层实测、网页和原Skin/CAMUS恢复回归。当前P6网页为`http://127.0.0.1:8092/`，详细结果、固定身份和启停步骤见[验证与使用记录](docs/VALIDATION_P6_20261021.md)。交付包及补充Icarus网表结果待完成；xsim BASIC受实例数限制没有通过P6网表。旧[性能计划](docs/OPTIMIZATION_4FPS_8FPS.md)与[组件记录](optimization/README.md)保留历史版本范围。

当前硬件路径为`optimization/rtl`、`optimization/vivado/build_spatial.tcl`、`optimization/device.py`和`optimization/board_verify.py`（ACD8）；运行时按配对证书选择。下文ACD3/ACD4构建命令属于历史兼容路径，不能继承ACD8板测结果。运行包、模型和病例在Release或被忽略的`out/acdc`中。

## PC安装与直接演示

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

## 真正的完整整数网络

```powershell
& acdc/.venv/Scripts/python.exe -m acdc synthetic-model out/acdc/my_synthetic_model
& acdc/.venv/Scripts/python.exe -m acdc verify-model out/acdc/my_synthetic_model --allow-synthetic --full
& acdc/.venv/Scripts/python.exe -m acdc web --backend emulator --model out/acdc/my_synthetic_model --allow-synthetic --output out/acdc/neural_demo_jobs
```

`reference`执行独立完整INT64参考；`emulator`使用成熟分块设备，实际计算Conv/Pool/CAT，不读取golden代替前向。`--full`比较完整256×256输入下全部21节点。合成模型只用于数值/容量/软件链路测试，不适合展示真实分割质量。

收到生产Axial包后，以[内部交换格式](docs/MODEL_BUNDLE.md)导入，并通过外部指定SHA固定身份：

```text
python -m acdc verify-model <model_directory> --model-spec-sha <A_approved_spec_sha> --full
python -m acdc run --backend reference --model <model_directory> --model-spec-sha <sha> --ed ED.nii.gz --es ES.nii.gz --output <new_output>
```

生产包要求完整参数/尺度、质量验收、独立oracle、21节点golden、FP32源权重和实际实现绑定；未验收状态、错方向、缺SHA/golden等直接拒绝。A最终原始包布局和机器合同未收到，目前不能声称已经兼容尚不存在的具体导出格式。

## 几何与数值

- 默认strict几何：明确mm单位、可逆affine、affine列范数与pixdim一致。未知单位拒绝；qform/sform按Nibabel选用并记录代码，取得A实现绑定后再核对其具体策略。
- 受限修复仅在操作者确认ACDC来源并传入 `--geometry-policy acdc-direction-only-sform-v1 --recognized-acdc` 时启用，仍检查全部header条件，不自动修复任意NIfTI。
- 原生整卷非零体素1%/99%线性分位数，float64中间运算后float32；原始零值也按公式变换。NaN/Inf、少于8个非零值、常量体拒绝。
- RAS2mm工作体、八个体素中心角点包围盒；Axial=`V[:,:,z].T`，首尾邻层clamp。层数=`Nz_ED+Nz_ES`，不是固定24或1048。
- 保比例letterbox用Torch2.11.0 bilinear/align_corners=False/antialias=False；量化值-127用于外部补边，网络卷积padding为0。
- 四类logits先softmax，再裁补边、恢复平面、回原生网格、按A规则概率兜底与归一化，最后argmax。无三方向融合、阈值、形态学或网格平滑。
- 标签输出使用采用后的物理affine；体积=体素数×abs(det(A))/1000mL。EF只在EDV>0且0≤ESV≤EDV时有效，否则null。

## 板端路径

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
python -m acdc web --backend remote --board-url http://127.0.0.1:8093 --model <same_approved_model> --model-spec-sha <sha>
```

生产remote只接受真实backend=pl响应；无自动CPU回退。`deploy/render_unit.py`仅渲染服务模板；`deploy/manage.py`提供板端本地install/activate/stop/rollback，检查受管单元hash，停止hub及旧后端后才启动ACDC，退出清理成功后才能恢复原hub。脚本尚未在真实systemd/板卡验收，本轮没有部署。安装会核验生产模型与runtime；不接受合成包用于正式切换。

```text
# 未来人工复制已批准运行包到独立版本目录后，在板端该目录执行：
sudo -E python3 -m acdc.deploy.manage install --release /home/xilinx/jupyter_notebooks/pld/releases/acdc_20261018_v1 --spec-sha <sha>
sudo -E python3 -m acdc.deploy.manage activate
# 用PC remote握手或本机 /health 确认真实就绪，systemd active本身不代表验收。
sudo -E python3 -m acdc.deploy.manage stop
# 需要撤销本工具安装的单元时：python3 -m acdc.deploy.manage rollback
```

`python -m acdc.board_verify`提供首层起算、完整21节点比较及重复板测入口。每个输入先运行一次独立CPU oracle和PL逐节点比较，之后计时独立的完整PS-DMA-PL前向；磁盘读入、oracle与结果比较不计入slice_seconds，总wall_seconds单独报告。可通过`--inputs`指定完整量化首层输入目录；每文件196608字节，不是逐层注入。报告含P50/P95、FPS、逐节点延迟、PL周期、RSS/CMA。可选`--recovery-checks`注入短包、缺包超时和计算中复位，每次恢复均要求下一次完整网络head与独立oracle一致；当前仅检查了控制流程，真实故障恢复尚未上板执行。运行包和报告均记录synthetic状态。

```text
sudo -E python3 -m acdc.board_verify --runtime <runtime> --model <runtime>/model --model-spec-sha <sha> --inputs <first_layer_inputs> --repeat 1000 --recovery-checks --output <new_report.json>
```

## 输出、测试、打包

结果目录包含labels/ED.nii.gz和ES.nii.gz、各期geometry、statistics.json/csv、performance.json、quality.json、VTK表面与浏览器二进制、manifest和ZIP。completion.json记录服务端上传读取至ZIP就绪的完整时间；ZIP中performance的pre_export_seconds不冒充最终导出时间。浏览器渲染及实际下载另报。

```powershell
& acdc/.venv/Scripts/python.exe -m pytest acdc/tests -q
node --check acdc/web_assets/app.js
node acdc/tests/test_viewer.cjs
& acdc/.venv/Scripts/python.exe -m acdc package out/acdc/source_package.zip
```

源码包包含只读复用组件及hash清单，不带虚拟环境、模型、病例或生成位流。PC浏览器使用自带Canvas MPR与WebGL表面显示，无CDN依赖；VTK在PC侧生成标准表面文件。生产参数、真实精度、板端FPS/内存/稳定性和物理显示仍需真实模型与设备验收。

有真实GT后，使用`python -m acdc evaluate --help`。病例清单schema为`pld-acdc-evaluation-cases-v1`，`cases`每项包含`patient`、`phase`、`result_dir`、`gt`、`gt_sha256`；每患者必须ED/ES配对，路径相对清单目录。预测标签按结果manifest核验hash，GT也必须固定hash，原生物理网格必须一致。输出逐类Dice、HD95/ASSD、容积和EF误差；该命令不替代A的FP32/INT8量化质量验收。
