"""CAMUS INT8（软件整数模拟）包。

* :mod:`integer` — 显式 INT8 整数图算子（15 Conv / skip-first concat / 整数 pool / nearest /
  Q24 ties-even 重定标 / 共享 head scale），只复用只读 ``software.role_a.quant_math``；
* :mod:`pipeline` — 校准（固定 128 帧 roster、只读 image 键）→ CAMUS 独立 v1 导出 → 校验。

**声明**：卷积使用 FP64 对整数乘加的软件模拟（要求中间整数 < 2^53 精确），
**不代表原生 INT8 速度或板端性能**；所有结论以 dev FP32/INT8 实测差异为准，不夸大。
"""
from __future__ import annotations

__all__ = ["integer", "pipeline"]
__version__ = "0.1.0-int8-01"
