# A 组统一交付包：皮肤病灶与 CAMUS 交互分割

2026-10-16。本版把已交付的 A 组模型、两套独立交互界面及其 HTTP/Python 接口合并到一个可移动的交付目录。模型和量化参数保持原验收版本，没有新训练、校准或导出。

## 解压后启动

在解压得到的本包根目录打开 PowerShell。建议使用 Python 3.12，首次配置本包自己的环境：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B verify_package.py
```

然后分别双击根目录的 `start_skin.cmd` 和 `start_camus.cmd`，或在两个窗口运行：

```powershell
.\start_skin.cmd
```

```powershell
.\start_camus.cmd
```

|模型|网页|预测 API|默认行为|
|---|---|---|---|
|皮肤病灶分割|http://127.0.0.1:8766/skin/|POST /api/skin/predict|INT8，阈值 0.45|
|CAMUS 心脏超声|http://127.0.0.1:8767/camus/|POST /api/camus/predict|INT8，保留多类 argmax；可启用独立阈值|

如果本机已运行前一版同端口服务，先关闭那两个服务再启动本包。也可在本包根目录用 `--port` 指定其他端口：

```powershell
.\.venv\Scripts\python.exe -B -m software.interactive_segmentation.server --model skin --port 8876
.\.venv\Scripts\python.exe -B -m software.interactive_segmentation.server --model camus --port 8877
```

两个命令各占一个窗口；浏览器地址同步使用对应端口。服务仅监听本机，按 Ctrl+C 关闭。本包不附带虚拟环境或 Python 安装器；有可用 Python 环境时，也可使用该环境的 Python 执行上述模块。PyTorch 的 CPU/CUDA 构建需与目标设备匹配，源环境版本另见 `ENVIRONMENT.json`。

## 功能与使用

- 两套界面分别运行，支持图像上传、前景/背景点、目标框、ROI 裁剪、阈值调整、撤销/清除及原图尺寸结果下载。
- 皮肤输出病灶分割，CAMUS 可选择左室腔、心肌或左房。CAMUS 原交付为默认，隐藏层直方图候选作为可选项单独保留。
- 点击“载入开发样例”可读取本包附带的两张既有开发输入；其中没有参考标注。也可自行上传图像；CAMUS 支持 2D 单帧 NIfTI。
- 点、框由冻结模型概率、图像和额外图割算法处理，原网络没有经过提示式训练；矩形不会直接填成分割掩膜。当前是电脑软件接口，交互精度与 FPGA 板端性能没有新增验收结论。

## 内容导航

|位置|内容|
|---|---|
|`host/interactive/`|两套页面、脚本和启动入口|
|`software/interactive_segmentation/`|独立控制器、参数校验、ROI / 阈值 / 图割、HTTP 服务及 16 项回归|
|`docs/interactive/`|两套 API 文档、通用协议和 15 个请求示例|
|`deliverables/role_a_final/`|皮肤原主模型 FP32、完整 INT8 与原 B/C 说明|
|`deliverables/role_a_camus_opt_20261013/`|CAMUS 主模型 FP32、完整 INT8 与原说明|
|`runs/role_a_camus/20261013_auto03/selection/`|候选对应的已冻结 FP32 权重|
|`runs/role_a_camus/20261014_auto10/exports/hidden_histogram/`|候选完整 INT8 导出；保留原相对路径以兼容接口|
|`original_deliveries/`|皮肤、CAMUS 主交付、CAMUS 候选的三个原 ZIP，原字节和历史证据完整保留|
|`evidence/`|接口验收、真实推理结果、16 项回归、最终截图及六组基准标签|
|`DATA_NOTICE.md`|开发演示输入的来源、范围与原项目授权记录位置|
|`verify_package.py`、`MANIFEST.json`|标准库完整性校验与逐文件 SHA256 / 字节数 / 来源|

`original_deliveries/` 中的历史文档记录原开发路径和实验。运行本新版请使用顶层本说明与 `docs/interactive/`，无需把文件放回原来的 E 盘目录。运行模块只对 CAMUS 旧源码中的两处项目根目录定位做了相对路径适配，数值计算代码、模型与量化文件不变；详情见 `PORTABILITY.json`。

## B / C 组集成

完整字段与错误码见 `docs/interactive/protocol.md`；皮肤与 CAMUS 分别见 `skin_api.md`、`camus_api.md`。坐标为原图像素，矩形右下边界不包含。ROI 在推理前裁剪，再映射回完整原图；区域外输出背景。

API 下载的 `labels_png` 使用类别编号：皮肤 0/1，CAMUS 0/1/2/3；`target_mask_png` 使用 0/255。历史后处理 CLI 的可视 PNG 使用 0/85/170/255，接入时须按所用接口明确编码。硬件整数布局、尺度和 Q24 规则继续以各模型的 `model_spec.json`、逐层黄金及原 B/C 文档为准。

## 验证范围

并包前的 16 项合成 / HTTP 回归、六个 FP32/INT8 版本组合及 ROI / 点框约束均已验收。新版交付另在独立解压目录检查完整性、模块来源、两套 HTTP 服务、实际模型与交互请求；最终结果在本 ZIP 旁的 `.acceptance.json`。软件测试证明接口与既有模型可运行，不把开发样例结果当作新增准确率。

可自行运行不读取数据集的 16 项回归：

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s software/interactive_segmentation/tests -v
```
