"""Exact NumPy integer operators for the frozen CAMUS candidate.

The operators deliberately mirror the reviewed handoff arithmetic: INT8
activations/weights, INT32 accumulation, signed INT64 Q24 products, ties-even
division and symmetric saturation.  This module is an independent numerical
oracle and does not imply PL execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

QMIN = -127
QMAX = 127
QFRAC = 24
INT32_MAX = np.iinfo(np.int32).max
INT64_MAX = np.iinfo(np.int64).max


def q24_lut(multiplier):
    if not isinstance(multiplier, (int, np.integer)) or not 0 < multiplier < (1 << 31):
        raise IntegerOperatorError('invalid LUT multiplier')
    raw = np.arange(256, dtype=np.uint8).view(np.int8).astype(np.int64)
    return np.clip(round_divide_even(raw * int(multiplier)), -127, 127).astype(np.int8)


class IntegerOperatorError(ValueError):
    """An integer tensor or operator violates the frozen contract."""


@dataclass(frozen=True)
class Int8Feature:
    data: np.ndarray
    scale: float

    def __post_init__(self) -> None:
        self.check()

    def check(self) -> None:
        data = np.asarray(self.data)
        if data.ndim != 4 or data.shape[0] != 1 or min(data.shape) < 1:
            raise IntegerOperatorError(f"expected NCHW batch=1, got {data.shape}")
        if data.dtype != np.int8:
            raise IntegerOperatorError(f"expected int8 data, got {data.dtype}")
        if data.size and int(np.min(data)) < QMIN:
            raise IntegerOperatorError("INT8 tensor contains -128")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise IntegerOperatorError(f"invalid scale {self.scale!r}")


def round_divide_even(products: np.ndarray, shift: int = QFRAC) -> np.ndarray:
    """Divide signed INT64 values by 2**shift with round-half-to-even."""
    if shift < 1 or shift > 62:
        raise IntegerOperatorError("shift out of range")
    raw = np.asarray(products)
    if not np.issubdtype(raw.dtype, np.signedinteger):
        raise IntegerOperatorError("rounding requires signed integers")
    values = raw.astype(np.int64)
    if values.size and np.any(values == np.iinfo(np.int64).min):
        raise IntegerOperatorError("abs(INT64_MIN) is unsupported")
    magnitude = np.abs(values)
    quotient = magnitude >> shift
    remainder = magnitude & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    increment = (remainder > half) | ((remainder == half) & ((quotient & 1) == 1))
    rounded = quotient + increment.astype(np.int64)
    return np.where(values < 0, -rounded, rounded)


def requantize_int8(accumulator: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    acc = np.asarray(accumulator)
    mult = np.asarray(multipliers)
    if not all(np.issubdtype(a.dtype, np.signedinteger) for a in (acc, mult)):
        raise IntegerOperatorError("accumulator/multipliers require signed integers")
    if acc.ndim != 4 or mult.ndim != 1 or mult.shape[0] != acc.shape[1]:
        raise IntegerOperatorError("requantize expects NCHW and per-output multipliers")
    if min(acc.shape) < 1 or np.any(acc < -(1 << 31)) or np.any(acc > INT32_MAX):
        raise IntegerOperatorError("accumulator outside nonempty INT32 contract")
    acc = acc.astype(np.int64)
    mult = mult.astype(np.int64)
    if mult.size and (np.any(mult <= 0) or np.any(mult > INT32_MAX)):
        raise IntegerOperatorError("Q24 multiplier outside positive INT32")
    if acc.size and int(np.max(np.abs(acc))) * int(np.max(mult)) > INT64_MAX:
        raise IntegerOperatorError("Q24 product exceeds INT64")
    products = acc * mult[None, :, None, None]
    rounded = round_divide_even(products)
    return np.clip(rounded, QMIN, QMAX).astype(np.int8)


def apply_relu_int8(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data)
    if array.dtype != np.int8:
        raise IntegerOperatorError("ReLU expects INT8")
    return np.maximum(array, np.int8(0))


def conv2d_int32(
    input_nchw: np.ndarray,
    weight_oihw: np.ndarray,
    bias: np.ndarray,
    *,
    padding: int,
) -> np.ndarray:
    """Exact stride-1 convolution; tile callers pass local padding=0."""
    x = np.asarray(input_nchw)
    w = np.asarray(weight_oihw)
    b = np.asarray(bias)
    if x.ndim != 4 or x.shape[0] != 1 or w.ndim != 4:
        raise IntegerOperatorError("convolution tensors must be NCHW/OIHW")
    if x.dtype != np.int8 or w.dtype != np.int8 or b.dtype != np.int32:
        raise IntegerOperatorError("convolution dtypes must be int8/int8/int32")
    if min(x.shape) < 1 or min(w.shape) < 1 or x.shape[1] != w.shape[1] or b.shape != (w.shape[0],):
        raise IntegerOperatorError("convolution channel/bias shape mismatch")
    if np.any(x == -128) or np.any(w == -128):
        raise IntegerOperatorError("convolution does not accept -128")
    kernel = w.shape[2]
    if w.shape[2] != w.shape[3] or kernel not in (1, 3) or padding not in (0, 1):
        raise IntegerOperatorError("only 1x1/3x3 and padding 0/1 are frozen")
    if padding and min(x.shape[2:]) < 1:
        raise IntegerOperatorError("empty convolution input")
    if not padding and min(x.shape[2:]) < kernel:
        raise IntegerOperatorError("input smaller than unpadded kernel")
    padded = np.pad(
        x[0].astype(np.int64),
        ((0, 0), (padding, padding), (padding, padding)),
    )
    oh, ow = padded.shape[1] - kernel + 1, padded.shape[2] - kernel + 1
    acc = np.broadcast_to(b.astype(np.int64)[:, None, None], (w.shape[0], oh, ow)).copy()
    for y in range(kernel):
        for z in range(kernel):
            acc += np.einsum("oc,chw->ohw", w[:, :, y, z].astype(np.int64),
                             padded[:, y:y + oh, z:z + ow], optimize=False)
    if np.any(acc < -(1 << 31)) or np.any(acc > INT32_MAX):
        raise IntegerOperatorError("convolution accumulator exceeds INT32")
    return acc[None].astype(np.int32)


def maxpool2_int8(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim != 4 or array.dtype != np.int8:
        raise IntegerOperatorError("maxpool expects NCHW int8")
    if array.shape[0] != 1:
        raise IntegerOperatorError("maxpool expects batch=1")
    if min(array.shape[2:]) < 2 or array.shape[2] % 2 or array.shape[3] % 2:
        raise IntegerOperatorError("maxpool requires even H/W")
    pooled = array.reshape(
        array.shape[0], array.shape[1],
        array.shape[2] // 2, 2, array.shape[3] // 2, 2,
    ).max(axis=(3, 5))
    return np.ascontiguousarray(pooled)


def nearest_upsample2_int8(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim != 4 or array.dtype != np.int8 or array.shape[0] != 1:
        raise IntegerOperatorError("upsample expects NCHW int8 batch=1")
    return np.ascontiguousarray(np.repeat(np.repeat(array, 2, axis=2), 2, axis=3))


def rescale_int8(data: np.ndarray, multiplier_q24: int) -> np.ndarray:
    mult = int(multiplier_q24)
    if mult <= 0 or mult > INT32_MAX:
        raise IntegerOperatorError("rescale multiplier outside positive INT32")
    array = np.asarray(data)
    if array.dtype != np.int8:
        raise IntegerOperatorError("rescale expects INT8")
    if np.any(array == -128):
        raise IntegerOperatorError("rescale does not accept -128")
    if array.size and int(np.max(np.abs(array.astype(np.int64)))) * mult > INT64_MAX:
        raise IntegerOperatorError("rescale product exceeds INT64")
    rounded = round_divide_even(array.astype(np.int64) * mult)
    return np.clip(rounded, QMIN, QMAX).astype(np.int8)


def concat_int8(branches: Iterable[np.ndarray]) -> np.ndarray:
    arrays = tuple(np.asarray(branch) for branch in branches)
    if len(arrays) != 2:
        raise IntegerOperatorError("the CAMUS concat has exactly two branches")
    if any(a.dtype != np.int8 or a.ndim != 4 for a in arrays):
        raise IntegerOperatorError("concat branches must be NCHW int8")
    if any(a.shape[0] != 1 for a in arrays):
        raise IntegerOperatorError("concat branches must have batch=1")
    if arrays[0].shape[2:] != arrays[1].shape[2:]:
        raise IntegerOperatorError("concat spatial shapes differ")
    return np.ascontiguousarray(np.concatenate(arrays, axis=1))


def quantize_letterbox_input(canvas_float32: np.ndarray) -> np.ndarray:
    """Quantize the normalized canvas with the exact float64 entry contract."""
    array = np.asarray(canvas_float32)
    if array.dtype != np.float32 or array.ndim != 2:
        raise IntegerOperatorError("letterbox canvas must be 2-D float32")
    if not array.size or not np.all(np.isfinite(array)) or np.any(array < 0) or np.any(array > 1):
        raise IntegerOperatorError("letterbox canvas must be finite in [0,1]")
    values = np.rint(array.astype(np.float64) / (1.0 / 127.0))
    return np.clip(values, QMIN, QMAX).astype(np.int8)
