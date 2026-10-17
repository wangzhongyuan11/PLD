"""CAMUS INT8 整数推理图（只复用只读 ``software.role_a.quant_math``）。

设计（与 `PROTOCOL.json` 的 ``quantization`` 字段一致）：

* 特征图显式 **INT8 NCHW**，零点 0、范围 ``[-127, 127]``；
* **15 个 Conv**（OIHW s8、per-output-channel 权重 scale、INT32 bias），
  hidden 卷积后 **fused ReLU**（clamp min 0）；**skip-first 拼接**（先 skip，再最近邻上采样分支）；
* 重定标：**正 Q24 int32 乘子**，int64 乘积、**ties-even** 舍入、饱和到 ±127；
* MaxPool 用**整数 reshape-max**（保持 scale）；nearest 上采样用**整数 repeat**（保持 scale）；
  concat 前两支各自按 ``max(source scales)`` 做 **Q24 整数重定标**（ties-even）；
* head 4 通道**共享同一输出 scale**，**无 sigmoid/softmax**；
* 入口：float 归一化 ``[0,1]`` 仅在图入口用 **float64 ties-even** 量化（``scale = 1/127``）；
  内部**禁止**任何 float 特征图或"重新量化代理"；
* 输出：用共享 scale 反量化得到 4 通道 logits，供既有 ``geometry`` 的
  native 通道双线性 + argmax 流程使用。

**性能声明**：卷积用既有 FP64 对整数乘加的快速数值模拟（要求中间整数精确可表示，``< 2^53``），
并逐层用 NumPy 整数 oracle（``quant_math.conv2d_int32``）交叉验证。
**软件模拟 ≠ 原生 INT8 速度，也不代表板端性能。**
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

PROJECT = Path(r"E:/嵌赛/pld-medical")
_QUANT_SOURCE = PROJECT / "software/role_a"
if str(_QUANT_SOURCE) not in sys.path:
    sys.path.insert(0, str(_QUANT_SOURCE))

import quant_math as qm  # noqa: E402  只读复用：数值内核

QMIN = qm.QMIN
QMAX = qm.QMAX
QFRAC = qm.QFRAC
INT32_MAX = 2 ** 31 - 1
INT32_MIN = -(2 ** 31)
INT64_MAX = 2 ** 63 - 1
FP64_EXACT_INT = 2 ** 53
INPUT_SCALE = 1.0 / 127.0


class IntegerGraphError(ValueError):
    """整数图的前置条件/数值不变量被破坏。"""


@dataclass(frozen=True)
class Int8Feature:
    """显式 INT8 特征图与其激活 scale。"""

    data: np.ndarray          # int8, NCHW
    scale: float

    def check(self) -> "Int8Feature":
        if self.data.dtype != np.int8:
            raise IntegerGraphError(f"特征图必须是 int8，收到 {self.data.dtype}")
        if self.data.ndim != 4 or min(self.data.shape)<1:
            raise IntegerGraphError(f"特征图必须是 NCHW，收到 shape={self.data.shape}")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise IntegerGraphError(f"非法 scale: {self.scale!r}")
        if self.data.size and (self.data.min() < QMIN or self.data.max() > QMAX):
            raise IntegerGraphError("特征图取值越界")
        return self


@dataclass(frozen=True)
class QuantLayer:
    """一层量化卷积及其先验边界证明。"""

    name: str
    q_weight: np.ndarray       # int8 OIHW
    q_bias: np.ndarray         # int32
    multiplier: np.ndarray     # int32 Q24（正）
    input_scale: float
    output_scale: float
    padding: int
    kernel: int
    conservative_bound: np.ndarray   # int64 per output channel
    bound_ok: bool
    weight_scales: np.ndarray | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"name": self.name, "weight_shape": list(self.q_weight.shape),
                "padding": self.padding, "kernel": self.kernel,
                "input_scale": float(self.input_scale), "output_scale": float(self.output_scale),
                "multiplier_q24": self.multiplier.tolist(),
                "conservative_bound": self.conservative_bound.tolist(),
                "bound_ok": bool(self.bound_ok), "weight_scales": self.weight_scales.tolist() if self.weight_scales is not None else None}


def conservative_bound(q_weight: np.ndarray, q_bias: np.ndarray) -> np.ndarray:
    """逐输出通道保守上界：``|bias| + 127 * sum(|q_weight|)``（int64）。"""
    weight = np.asarray(q_weight, dtype=np.int64)
    bias = np.asarray(q_bias, dtype=np.int64)
    if weight.ndim != 4:
        raise IntegerGraphError("权重必须是 OIHW")
    if weight.shape[0] != bias.shape[0]:
        raise IntegerGraphError("bias 数量与输出通道不一致")
    return np.abs(bias) + 127 * np.abs(weight).sum(axis=(1, 2, 3))


def require_bounds(q_weight: np.ndarray, q_bias: np.ndarray) -> np.ndarray:
    """计算并**拒绝**任何超过 INT32 的通道（先验证明）。"""
    bound = conservative_bound(q_weight, q_bias)
    bad = np.nonzero(bound > INT32_MAX)[0]
    if bad.size:
        raise IntegerGraphError(f"通道 {bad.tolist()} 的保守上界超过 INT32："
                                f"{bound[bad].tolist()}")
    return bound


def quantize_input_float64(image: np.ndarray, *, scale: float = INPUT_SCALE) -> Int8Feature:
    """入口量化：float ``[0,1]`` 图像用 **float64 ties-even** 量化为 INT8。

    ``scale`` 固定 ``1/127``；越界立即拒绝（入口数据应为 ``[0,1]``）。
    """
    if scale != INPUT_SCALE:
        raise IntegerGraphError("input scale is fixed at 1/127")
    array = np.asarray(image, dtype=np.float64)
    if array.ndim != 2:
        raise IntegerGraphError(f"期望二维灰度图，收到 shape={array.shape}")
    if not np.isfinite(array).all():
        raise IntegerGraphError("输入含 NaN/Inf")
    if array.min() < 0.0 or array.max() > 1.0:
        raise IntegerGraphError(f"输入超出 [0,1]: min={array.min()} max={array.max()}")
    quantized = np.clip(np.rint(array / scale), QMIN, QMAX).astype(np.int8)
    return Int8Feature(quantized[None, None], float(scale)).check()


def check_layer(layer: QuantLayer) -> None:
    w,b,m=layer.q_weight,layer.q_bias,layer.multiplier
    if w.dtype!=np.int8 or w.ndim!=4 or min(w.shape)<1 or np.any(w<QMIN):
        raise IntegerGraphError("weight must be nonempty OIHW signed127 int8")
    if b.dtype!=np.int32 or b.shape!=(w.shape[0],) or m.dtype!=np.int32 or m.shape!=b.shape or np.any(m<1):
        raise IntegerGraphError("bias/multiplier must be per-channel int32 and multipliers positive")
    if layer.kernel not in (1,3) or w.shape[2:]!=(layer.kernel,layer.kernel) or layer.padding != (layer.kernel//2):
        raise IntegerGraphError("only same-shape 1x1 or 3x3 convolution supported")
    if not np.isfinite(layer.input_scale) or not np.isfinite(layer.output_scale) or min(layer.input_scale,layer.output_scale)<=0:
        raise IntegerGraphError("invalid layer scales")
    bound=require_bounds(w,b)
    if not layer.bound_ok or not np.array_equal(bound,layer.conservative_bound):
        raise IntegerGraphError("missing or incorrect accumulator bound proof")


def conv2d_int8_float64(feature: Int8Feature, layer: QuantLayer, *,
                        relu: bool, oracle: bool = True,
                        capture: dict | None = None) -> Int8Feature:
    """FP64 exact-integer convolution simulator, guarded before any narrowing/product."""
    feature.check();check_layer(layer)
    if feature.data.shape[0]!=1 or feature.data.shape[1]!=layer.q_weight.shape[1] or feature.scale!=layer.input_scale:
        raise IntegerGraphError(f"{layer.name}: shape or predecessor scale mismatch")
    # Every absolute partial sum is bounded by the per-channel sum of absolute
    # products, including bias. The <=INT32 proof is stronger than <2**53.
    exact_bound=max(int(v) for v in layer.conservative_bound)
    if exact_bound>=FP64_EXACT_INT:
        raise IntegerGraphError("integer sum exceeds FP64 exact representation")
    source=torch.from_numpy(feature.data.astype(np.float64))
    weight=torch.from_numpy(layer.q_weight.astype(np.float64))
    bias=torch.from_numpy(layer.q_bias.astype(np.float64))
    acc=F.conv2d(source,weight,bias,padding=layer.padding)
    if not torch.isfinite(acc).all() or not torch.equal(acc,torch.round(acc)):
        raise IntegerGraphError("FP64 simulator produced a noninteger/nonfinite accumulator")
    if float(acc.min())<INT32_MIN or float(acc.max())>INT32_MAX:
        raise IntegerGraphError("actual accumulator exceeds INT32")
    integers=acc.to(torch.int64)
    if max(abs(int(integers.min())),abs(int(integers.max())))*int(layer.multiplier.max())>INT64_MAX:
        raise IntegerGraphError("requantization product exceeds INT64")
    products=integers*torch.from_numpy(layer.multiplier.astype(np.int64)).view(1,-1,1,1)
    q=_round_divide_even_int64(products).clamp(QMIN,QMAX)
    if relu:q=q.clamp(min=0)
    output=Int8Feature(q.to(torch.int8).numpy(),layer.output_scale).check()
    if capture is not None:
        capture.update(input_int8=feature.data.copy(),acc_int32=integers.numpy().astype('<i4'),output_int8=output.data.copy(),relu=relu,input_scale=feature.scale,output_scale=layer.output_scale)
    if oracle:
        ref=qm.requantize_int8(qm.conv2d_int32(feature.data,layer.q_weight,layer.q_bias,layer.padding),layer.multiplier)
        if relu:ref=np.maximum(ref,0).astype(np.int8)
        if not np.array_equal(output.data,ref):raise IntegerGraphError("NumPy integer oracle mismatch")
    return output


def _oracle_conv(feature: np.ndarray, q_weight: np.ndarray, q_bias: np.ndarray,
                 padding: int) -> np.ndarray:
    return qm.conv2d_int32(feature, q_weight, q_bias, padding)


def _round_divide_even_int64(products: torch.Tensor, shift: int = QFRAC) -> torch.Tensor:
    """int64 除以 ``2**shift`` 的 ties-even 舍入（与 ``quant_math.round_divide_even`` 同语义）。"""
    if shift < 1 or shift > 62:
        raise IntegerGraphError(f"shift 非法: {shift}")
    if products.numel() and bool((products == -(2 ** 63)).any()):
        raise IntegerGraphError("出现 INT64_MIN，无法取绝对值")
    magnitude = products.abs()
    quotient = magnitude >> shift
    remainder = magnitude & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    increment = (remainder > half) | ((remainder == half) & ((quotient & 1) == 1))
    rounded = quotient + increment.to(torch.int64)
    return torch.where(products < 0, -rounded, rounded)


def maxpool_int8(feature: Int8Feature, *, kernel: int = 2, stride: int = 2) -> Int8Feature:
    """整数 reshape-max 下采样（scale 不变）。"""
    feature.check()
    if kernel != 2 or stride != 2:
        raise IntegerGraphError("only kernel2 stride2 pool supported")
    if feature.data.shape[2] % kernel or feature.data.shape[3] % kernel:
        raise IntegerGraphError(f"pool 尺寸不能整除: {feature.data.shape}")
    data = feature.data
    pooled = data.reshape(data.shape[0], data.shape[1], data.shape[2] // kernel, kernel,
                          data.shape[3] // kernel, kernel).max(axis=(3, 5))
    if stride != kernel:
        raise IntegerGraphError("仅支持 stride == kernel 的整数 pool")
    return Int8Feature(pooled.astype(np.int8), feature.scale).check()


def nearest_upsample_int8(feature: Int8Feature, *, factor: int = 2) -> Int8Feature:
    """整数 repeat 上采样（scale 不变）。"""
    feature.check()
    if factor != 2:raise IntegerGraphError("only nearest factor2 supported")
    data = np.repeat(np.repeat(feature.data, factor, axis=2), factor, axis=3)
    return Int8Feature(data.astype(np.int8), feature.scale).check()


def rescale_int8(feature: Int8Feature, target_scale: float) -> Int8Feature:
    """把特征图整数重定标到 ``target_scale``（Q24 ties-even，饱和）。"""
    feature.check()
    if not np.isfinite(target_scale) or target_scale <= 0:
        raise IntegerGraphError(f"非法目标 scale: {target_scale!r}")
    multiplier = int(np.rint(feature.scale / target_scale * (1 << QFRAC)))
    if multiplier < 1 or multiplier > INT32_MAX:
        raise IntegerGraphError(f"重定标乘子越界: {multiplier}")
    products = feature.data.astype(np.int64) * multiplier
    if products.size and np.abs(products).max() > INT64_MAX:
        raise IntegerGraphError("重定标乘积超出 INT64")
    tensor = torch.from_numpy(products)
    requantized = _round_divide_even_int64(tensor).clamp(QMIN, QMAX)
    return Int8Feature(requantized.to(torch.int8).numpy(), target_scale).check()


def concat_int8(branches: Sequence[Int8Feature]) -> tuple[Int8Feature, dict[str, Any]]:
    """先按 ``max(source scales)`` 把各分支整数重定标，再沿通道拼接。"""
    if len(branches) < 2:
        raise IntegerGraphError("concat 至少需要两个分支")
    for branch in branches:
        branch.check()
    common = max(branch.scale for branch in branches)
    rescaled = [rescale_int8(branch, common) if branch.scale != common else branch
                for branch in branches]
    data = np.concatenate([branch.data for branch in rescaled], axis=1)
    info = {"source_scales": [float(branch.scale) for branch in branches],
            "common_scale": float(common), "output_channels": int(data.shape[1])}
    return Int8Feature(data.astype(np.int8), common).check(), info


def build_layer(conv: torch.nn.Conv2d, name: str, *, input_scale: float,
                output_scale: float) -> QuantLayer:
    """由 FP32 Conv2d 构造量化层并做先验边界证明（per-output-channel 权重 scale）。"""
    if conv.groups != 1 or conv.stride != (1, 1) or conv.dilation != (1, 1):
        raise IntegerGraphError(f"{name}: 仅支持 groups=1/stride=1/dilation=1")
    if conv.kernel_size not in ((1, 1), (3, 3)) or conv.padding not in ((0, 0), (1, 1)):
        raise IntegerGraphError(f"{name}: 仅支持 1x1/3x3 且 padding 0/1")
    q_weight, weight_scales = qm.quantize_weights(conv.weight.detach().cpu().numpy())
    bias = (conv.bias.detach().cpu().numpy() if conv.bias is not None
            else np.zeros(conv.out_channels, dtype=np.float64))
    q_bias = qm.quantize_bias(bias, input_scale, weight_scales)
    multiplier = qm.q24_multipliers(input_scale, weight_scales, output_scale)
    bound = require_bounds(q_weight, q_bias)
    return QuantLayer(name=name, q_weight=q_weight, q_bias=q_bias, multiplier=multiplier,
                      input_scale=float(input_scale), output_scale=float(output_scale),
                      padding=conv.padding[0], kernel=conv.kernel_size[0],
                      conservative_bound=bound, bound_ok=True, weight_scales=weight_scales)


def dequantize(feature: Int8Feature, scale: float) -> np.ndarray:
    """用给定 scale 反量化到 float（仅用于最终 logits 输出）。"""
    feature.check()
    if not np.isfinite(scale) or scale <= 0 or scale != feature.scale:
        raise IntegerGraphError(f"非法输出 scale: {scale!r}")
    return feature.data.astype(np.float64) * scale


def verify_against_float64_graph(layer: QuantLayer, feature: Int8Feature, *,
                                 relu: bool) -> dict[str, Any]:
    """独立 oracle 报告（供 verify 使用）：与 NumPy 整数实现逐元素比较。"""
    output = conv2d_int8_float64(feature, layer, relu=relu, oracle=False)
    reference = qm.requantize_int8(_oracle_conv(feature.data, layer.q_weight, layer.q_bias,
                                               layer.padding), layer.multiplier)
    if relu:
        reference = np.clip(reference.astype(np.int16), 0, QMAX).astype(np.int8)
    matches = bool(np.array_equal(output.data, reference))
    return {"layer": layer.name, "matches_oracle": matches,
            "mismatch_count": int((output.data != reference).sum())}
