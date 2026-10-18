"""Independent full-tensor INT64 oracle; no tiled-device arithmetic is imported."""
from __future__ import annotations

import numpy as np

from .contract import ContractError, checked_int8


def round_q24(value):
    a = np.asarray(value)
    if not np.issubdtype(a.dtype, np.signedinteger):
        raise ContractError("Q24 requires signed integers")
    a = a.astype(np.int64)
    if np.any(a == np.iinfo(np.int64).min):
        raise ContractError("INT64_MIN cannot be negated")
    mag = np.abs(a)
    base, rem = mag // (1 << 24), mag % (1 << 24)
    base += ((rem > (1 << 23)) | ((rem == (1 << 23)) & (base % 2 == 1)))
    return np.where(a < 0, -base, base)


def requantize(acc, mult):
    a = np.asarray(acc)
    m = np.asarray(mult)
    if not np.issubdtype(a.dtype, np.signedinteger) or not np.issubdtype(m.dtype, np.signedinteger):
        raise ContractError("integer accumulator/multiplier required")
    if np.any(a < -(1 << 31)) or np.any(a > (1 << 31) - 1):
        raise ContractError("INT32 accumulator overflow")
    if np.any(m < 1) or np.any(m > (1 << 31) - 1):
        raise ContractError("invalid Q24 multiplier")
    return np.clip(round_q24(a.astype(np.int64) * m.astype(np.int64)), -127, 127).astype(np.int8)


def conv_full(x, weight, bias, mult, padding, relu):
    x = checked_int8(x)
    w = checked_int8(weight)
    b = np.asarray(bias)
    m = np.asarray(mult)
    if (x.ndim != 4 or x.shape[0] != 1 or w.ndim != 4 or w.shape[1] != x.shape[1]
            or b.dtype != np.int32 or m.dtype != np.int32
            or b.shape != (w.shape[0],) or m.shape != b.shape
            or w.shape[2] != w.shape[3] or w.shape[2] not in (1, 3)
            or padding not in (0, 1)):
        raise ContractError("invalid convolution")
    k = w.shape[2]
    p = np.pad(x[0].astype(np.int64), ((0, 0), (padding, padding), (padding, padding)))
    oh, ow = p.shape[1] - k + 1, p.shape[2] - k + 1
    acc = np.broadcast_to(b.astype(np.int64)[:, None, None], (w.shape[0], oh, ow)).copy()
    # Zero-channel elimination is exact for synthetic sparse fixtures, never a CPU fallback.
    for o in range(w.shape[0]):
        for ci, ky, kx in np.argwhere(w[o] != 0):
            acc[o] += int(w[o, ci, ky, kx]) * p[ci, ky:ky+oh, kx:kx+ow]
    out = requantize(acc[None], m[None, :, None, None])
    return np.maximum(out, 0) if relu else out


def pool(x):
    x = checked_int8(x)
    n, c, h, w = x.shape
    if n != 1 or h % 2 or w % 2:
        raise ContractError("pool requires even NCHW")
    return x.reshape(n, c, h//2, 2, w//2, 2).max(axis=(3, 5)).copy()


def concat(skip, deep, multipliers):
    skip, deep = checked_int8(skip), checked_int8(deep)
    if skip.shape[2:] != tuple(2*v for v in deep.shape[2:]):
        raise ContractError("CAT spatial mismatch")
    deep = deep.repeat(2, axis=2).repeat(2, axis=3)
    return np.concatenate([requantize(skip.astype(np.int32), multipliers[0]),
                           requantize(deep.astype(np.int32), multipliers[1])], axis=1)
