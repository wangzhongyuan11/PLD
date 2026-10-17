"""CAMUS 角色 A 实现包：灰度单通道 → 4 类（背景/LV 腔/心肌/左心房）分割基线。

模块划分：

* :mod:`geometry` — NIfTI 轴/物理 spacing、letterbox、原尺寸重建、HD95、面积
* :mod:`data` — 官方 400/50/50 患者名单、NIfTI 读取、测试封条
* :mod:`model` — 单通道 4 输出 U-Net（Conv/ReLU/MaxPool/nearest-up/concat），预算函数
* :mod:`metrics` — 逐类 Dice/召回/覆盖、按帧与按患者等权、物理面积、spacing-aware HD95
* :mod:`train` — AdamW + CE/前景 Dice、AMP、完整断点、原子保存与 STOP
* :mod:`evaluate` — native grid 评估与 LV 体积/EF（参考官方公开公式）
* :mod:`visualize` — ED/ES 轮廓与面积对比

数据：CAMUS_public 官方 NIfTI 版，许可 CC BY-NC-SA 4.0（非商业科研），
必须引用 TMI2019（Leclerc et al. 2019）。本轮**未训练、未推理真实数据**。
"""
from __future__ import annotations

__all__ = ["geometry", "data", "model", "metrics", "train", "evaluate", "visualize"]

__version__ = "0.1.0-preparation"
