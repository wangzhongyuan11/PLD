"""A→B 交接用的确定性对称 INT8 卷积参考。

张量均为 NCHW；权重 OIHW；有符号 INT8 范围 [-127, 127]，零点 0。
累加和 bias 用 INT32，逐输出通道 Q24 整数乘数，除以 2**24 时采用最近偶数舍入。
本模块只定义算术，不负责训练或声称 FPGA 已实现同一规则。
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

QMIN, QMAX, QFRAC = -127, 127, 24
_INT32 = np.iinfo(np.int32)
_INT64 = np.iinfo(np.int64)


def symmetric_scale(max_abs: float) -> float:
    if not np.isfinite(max_abs) or max_abs < 0:
        raise ValueError("max_abs must be finite and nonnegative")
    return max(float(max_abs) / QMAX, 1e-8)


def quantize_int8(values: np.ndarray, scale: float) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    return np.clip(np.rint(np.asarray(values, dtype=np.float64) / scale), QMIN, QMAX).astype(np.int8)


def quantize_weights(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 4 or not np.isfinite(weights).all():
        raise ValueError("weights must be finite OIHW array")
    scales = np.maximum(np.max(np.abs(weights), axis=(1, 2, 3)) / QMAX, 1e-8)
    quantized = np.clip(np.rint(weights / scales[:, None, None, None]), QMIN, QMAX).astype(np.int8)
    return quantized, scales.astype(np.float64)


def quantize_bias(bias: np.ndarray, input_scale: float, weight_scales: np.ndarray) -> np.ndarray:
    values = np.rint(np.asarray(bias, dtype=np.float64) /
                     (input_scale * np.asarray(weight_scales, dtype=np.float64)))
    if not np.isfinite(values).all() or np.any(values < _INT32.min) or np.any(values > _INT32.max):
        raise OverflowError("quantized bias does not fit INT32")
    return values.astype(np.int32)


def q24_multipliers(input_scale: float, weight_scales: np.ndarray, output_scale: float) -> np.ndarray:
    if input_scale <= 0 or output_scale <= 0:
        raise ValueError("scales must be positive")
    values = np.rint(input_scale * np.asarray(weight_scales, dtype=np.float64) /
                     output_scale * (1 << QFRAC))
    if not np.isfinite(values).all() or np.any(values < 1) or np.any(values > _INT32.max):
        raise OverflowError("Q24 multiplier is zero or exceeds INT32")
    return values.astype(np.int32)


def round_divide_even(products: np.ndarray, shift: int = QFRAC) -> np.ndarray:
    """带符号整数除以 2**shift，恰好半格时舍入到偶数。"""
    if shift < 1 or shift > 62:
        raise ValueError("shift out of range")
    values = np.asarray(products, dtype=np.int64)
    if np.any(values == _INT64.min):
        raise OverflowError("cannot take abs(INT64_MIN)")
    magnitude = np.abs(values)
    quotient = magnitude >> shift
    remainder = magnitude & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    increment = (remainder > half) | ((remainder == half) & ((quotient & 1) == 1))
    rounded = quotient + increment.astype(np.int64)
    return np.where(values < 0, -rounded, rounded)


def conv2d_int32(input_nchw: np.ndarray, weight_oihw: np.ndarray,
                 bias: np.ndarray, padding: int) -> np.ndarray:
    """小型黄金向量专用精确卷积；batch=1、stride=1。"""
    x = np.asarray(input_nchw, dtype=np.int8)
    w = np.asarray(weight_oihw, dtype=np.int8)
    b = np.asarray(bias, dtype=np.int32)
    if x.ndim != 4 or x.shape[0] != 1 or w.ndim != 4 or x.shape[1] != w.shape[1]:
        raise ValueError("expected NCHW batch=1 and matching OIHW weights")
    if w.shape[2] != w.shape[3] or w.shape[2] not in (1, 3) or padding not in (0, 1):
        raise ValueError("supported kernels: 1x1 or 3x3 with padding 0 or 1")
    if b.shape != (w.shape[0],):
        raise ValueError("bias shape must match output channels")
    padded = np.pad(x[0].astype(np.int64), ((0, 0), (padding, padding), (padding, padding)))
    windows = sliding_window_view(padded, (w.shape[2], w.shape[3]), axis=(1, 2))
    acc = np.einsum("cxyij,ocij->oxy", windows, w.astype(np.int64), optimize=True)
    acc += b.astype(np.int64)[:, None, None]
    if np.any(acc < _INT32.min) or np.any(acc > _INT32.max):
        raise OverflowError("convolution accumulator exceeds INT32")
    return acc[None].astype(np.int32)


def requantize_int8(accumulator: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    acc = np.asarray(accumulator, dtype=np.int64)
    mult = np.asarray(multipliers, dtype=np.int64)
    if acc.ndim != 4 or mult.shape != (acc.shape[1],):
        raise ValueError("expected NCHW accumulator and one multiplier per output channel")
    if np.max(np.abs(acc)) * np.max(mult) > _INT64.max:
        raise OverflowError("requantization product exceeds INT64")
    products = acc * mult[None, :, None, None]
    return np.clip(round_divide_even(products), QMIN, QMAX).astype(np.int8)
