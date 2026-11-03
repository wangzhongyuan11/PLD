"""CAMUS 优化包：连通性后处理 + 三前景 boundary loss 微调（control20 / boundary20）。

模块：

* :mod:`postprocess` — LV-only 8 连通后处理（island5 / island5_hole1，纯预测几何、不用 GT）
* :mod:`boundary_loss` — 三前景 signed-distance boundary loss（距离按 256 canvas 对角线归一化）
* :mod:`train` — 复用原成熟训练/缓存/几何/指标/模型接口的优化臂训练（λ 感知、完整断点、STOP 安全）
* :mod:`evaluate_postprocess` — 需 owner 许可的 dev 后处理对比评估（不重推理、不读 test）

原 ``software/role_a_camus`` 与旧 run/交付**只读**；本包不 monkeypatch、不修改原模块。
"""
from __future__ import annotations

__all__ = ["postprocess", "boundary_loss", "train", "evaluate_postprocess"]
__version__ = "0.1.0-opt01"
