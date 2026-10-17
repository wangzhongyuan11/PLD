# CAMUS 心脏超声专用接口

入口 `http://127.0.0.1:8767/camus/`；预测 `POST http://127.0.0.1:8767/api/camus/predict`。

接入既有 CAMUS 原交付，以及已单独交付的隐藏层直方图候选。输出类别 0 背景 / 1 左室腔 / 2 心肌 / 3 左房。默认 `delivery + int8 + target_class:1 + threshold:null`。选择目标决定提示、二值掩膜和叠加显示针对哪个结构；完整类别 PNG 仍输出全部类别。

默认多类 argmax 加固定 `island5_hole1`，保持原交付流程。网页默认不启用独立阈值。给定数值时，所选结构概率达到阈值就先判为该类，其余位置取概率最高的非目标类别，再做固定后处理。有点/框时最后执行图割；前景点属于当前结构，背景点只表示“不属于当前结构”。

可使用普通图像或 2D 单帧 NIfTI。坐标对应解码后的原图，不是原始 NIfTI 文件索引顺序；网页会显示宽高。完整字段、PNG 输出和错误码见 [接口约定](protocol.md)，分别可复制的请求见 [CAMUS 示例](camus_api_examples.json)。示例坐标假定图像至少 230×240，需换成实际坐标与 Base64。

Python 调用（**在 Release ZIP 解压出的包根目录**运行；先按包内 README 执行 `python -m venv .venv` 与 `pip install -r requirements.txt`，ZIP 不附带已装好的环境）：

```python
from pathlib import Path
import base64
from software.interactive_segmentation.images import decode_image
from software.interactive_segmentation.camus_api import CamusAPI

path = Path("patient0200_2CH_ED.nii.gz")
image = decode_image("camus", base64.b64encode(path.read_bytes()).decode(), path.name)
api = CamusAPI()
result = api.predict(image, {
    "variant": "delivery", "backend": "int8",
    "target_class": 1, "threshold": None
})
```

|版本|冻结权重|模型规格|
|---|---|---|
|delivery|`deliverables/role_a_camus_opt_20261013/weights/model_state_fp32.pt`|`deliverables/role_a_camus_opt_20261013/int8_export/model_spec.json`|
|candidate|`runs/role_a_camus/20261013_auto03/selection/model_state_fp32.pt`|`runs/role_a_camus/20261014_auto10/exports/hidden_histogram/model_spec.json`|

原交付权重 SHA256 `74d0da6f13b48528b221e6c295d90de73b6e16f1320a860bc086a05e3c2b774f`，规格 `26e200660f91d228afe7127e046f4d69d52000e4a240c9fc8f7068baf074cd1a`；候选权重 `3d8f268910783ea524513b3506f4425aec991a54dcb23bf25cff5598fb942378`，规格 `68b4f425cc3612ad72d5e83955c2018981edee912fbc0937e9ac8c1718e7b5ae`。加载时验证身份。切换版本只选择已有工件，不重训练或替换原交付。
