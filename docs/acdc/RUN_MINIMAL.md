# ACDC 最小运行示例

下载入口、完整ZIP名、大小及SHA-256见 [MINIMAL_RUNTIME.json](MINIMAL_RUNTIME.json)。它使用已验收的同一套模型，只保留运行所需资源；完整100名训练患者数据仍在 [原研究总包](RESEARCH_RELEASE.json)。

ZIP包含三方向FP32/INT8模型及黄金输入/head、完整运行依赖源码、patient039的ED/ES两个真实MRI输入、一套已生成的INT8四面板预览、许可、依赖清单、逐文件校验及`run_demo.py`。没有训练源码、完整数据集、虚拟环境或已验收的板端bit/HWH。默认是电脑CPU推理。

先用本仓库固定提交中`MINIMAL_RUNTIME.json`的`archive_sha256`核对ZIP，再解压。只看已有结果时，直接打开`preview/index.html`，无需安装Python。

运行真实模型需要64位Python 3.12。下面在解压后的包根执行，Windows PowerShell：

```powershell
py -3.12 -m venv ..\.acdc-demo-venv
..\.acdc-demo-venv\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
..\.acdc-demo-venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
..\.acdc-demo-venv\Scripts\python.exe run_demo.py --verify-only
..\.acdc-demo-venv\Scripts\python.exe run_demo.py
```

环境放在包外的`.acdc-demo-venv`，保持交付目录清单不变。已存在兼容环境时可直接使用其Python，无需重复安装。CPU轮子来自 [PyTorch官方索引](https://download.pytorch.org/whl/cpu/torch/)。环境安装需联网，ZIP不包含离线依赖轮子。

默认实际运行两个时相的三方向FP32模型，完成物理网格预处理、概率融合、NIfTI标签、PLY表面、原生容积/EF和四面板HTML。输出位于包的同级`<包名>_results`，打开其中`index.html`。结果目录已存在时会拒绝覆盖，可传`--output <新的外部目录>`。CPU INT8参考可用`--backend int8`，耗时通常更长；电脑整数参考不代表板卡加速。

在clone里调用同一入口时使用`python tools/run_acdc_minimal.py --root <解压包绝对路径> --output <新的外部目录>`。自有数据用`--ed <3D NIfTI> --es <3D NIfTI>`显式指定，两者必须同时提供；使用严格几何检查，不默认修复未知MRI头信息。ACDC示例帧号来自原始Info：ED=01、ES=10。

模型输入为`[1,3,256,256]`，输出为四类`[1,4,256,256]`；三方向尺度不同，融合按已冻结数值合同执行。示例来自已用于选模的开发患者，不是独立测试；官方50测试患者未包含。数据仅用于所附条款允许的非商业科学研究，保留Bernard等2018论文署名与引用。

物理板端验收尚未完成。PNG预览已验证，内置浏览器原生下载完成事件仍未验证；本示例不把电脑推理或现成预览声称为物理PL运行。
