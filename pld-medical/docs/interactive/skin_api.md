# 皮肤病灶分割专用接口

入口 `http://127.0.0.1:8766/skin/`；预测 `POST http://127.0.0.1:8766/api/skin/predict`。

接入现有 `deliverables/role_a_final` 病灶分割模型，RGB 输入，类别 0 背景 / 1 病灶；默认 `delivery + int8 + target_class:1 + threshold:0.45`。这项接口输出病灶范围，不输出疾病分类诊断。

点、框、ROI、阈值可单独使用或组合；先 ROI 推理，再按阈值判断，有点/框时执行模型辅助图割。完整字段、PNG 输出和错误码见 [接口约定](protocol.md)，分别可复制的请求见 [皮肤示例](skin_api_examples.json)。示例坐标假定图像至少 420×340，使用前替换为实际图像坐标和 Base64。

Python 调用（项目根目录、现有 `.venv`）：

```python
import numpy as np
from PIL import Image
from software.interactive_segmentation.skin_api import SkinAPI

image = np.asarray(Image.open("skin.png").convert("RGB"))
api = SkinAPI()
result = api.predict(image, {
    "backend": "int8", "threshold": 0.45,
    "points": [{"x": 200, "y": 170, "label": "foreground"}]
})
# result["outputs"]["target_mask_png"] 为原图尺寸 PNG data URL。
```

冻结权重 `deliverables/role_a_final/best.pt` SHA256 为 `f488719fe6d4d09c9b86be84d06b19d34241d6c14060b87da7bab25d1367eb77`；规格 `deliverables/role_a_final/int8/model_spec.json` SHA256 为 `fe137ca5b128fc1983fabad9bee39b3208cda4256e0d04c4e27cef5c07666aef`。加载时校验，文件身份不符则报错。
