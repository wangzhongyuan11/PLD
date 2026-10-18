"""Reversible internal packing. External model tensors remain NCHW / OIHW."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


LAYOUT = 'acdc-c8hw8-w8x8-v1'


def _int8(value, rank):
    a = np.asarray(value)
    if a.dtype != np.int8 or a.ndim != rank or any(n < 1 for n in a.shape):
        raise ValueError(f'expected nonempty rank-{rank} INT8 tensor')
    if np.any(a == -128):
        raise ValueError('production tensors forbid -128')
    return a


def pack_feature(chw):
    a = _int8(chw, 3)
    ci, h, w = a.shape
    padded = np.zeros((math.ceil(ci/8)*8, h, w), dtype=np.int8)
    padded[:ci] = a
    return np.ascontiguousarray(padded.reshape(-1, 8, h, w).transpose(0, 2, 3, 1))


def unpack_feature(blocks, channels):
    a = _int8(blocks, 4)
    if a.shape[-1] != 8 or not 1 <= channels <= a.shape[0]*8:
        raise ValueError('invalid C8HW8 shape/channels')
    return np.ascontiguousarray(a.transpose(0, 3, 1, 2).reshape(-1, *a.shape[1:3])[:channels])


def pack_weights(oihw):
    a = _int8(oihw, 4)
    co, ci, kh, kw = a.shape
    if kh != kw or kh not in (1, 3):
        raise ValueError('only frozen 1x1/3x3 kernels are supported')
    padded = np.zeros((math.ceil(co/8)*8, math.ceil(ci/8)*8, kh, kw), dtype=np.int8)
    padded[:co, :ci] = a
    return np.ascontiguousarray(padded.reshape(-1, 8, math.ceil(ci/8), 8, kh, kw)
                               .transpose(0, 2, 4, 5, 1, 3))


def unpack_weights(blocks, co, ci):
    a = _int8(blocks, 6)
    if (a.shape[-2:] != (8, 8) or a.shape[2] != a.shape[3]
            or a.shape[2] not in (1, 3) or not 1 <= co <= a.shape[0]*8
            or not 1 <= ci <= a.shape[1]*8):
        raise ValueError('invalid W8x8 shape/channels')
    return np.ascontiguousarray(a.transpose(0, 4, 1, 5, 2, 3)
                               .reshape(a.shape[0]*8, a.shape[1]*8, *a.shape[2:4])[:co, :ci])


def pack_parameter_groups(oihw, bias, multipliers):
    """Derived Co8 DMA packets: little-endian bias[8], M[8], then W8x8 taps."""
    w = _int8(oihw, 4)
    b, m = np.asarray(bias), np.asarray(multipliers)
    if (b.dtype != np.int32 or m.dtype != np.int32 or b.shape != (w.shape[0],)
            or m.shape != b.shape or np.any(m <= 0)):
        raise ValueError('INT32 Co-sized bias and positive multipliers required')
    blocks = pack_weights(w)
    packets = []
    for group in range(blocks.shape[0]):
        count = min(8, w.shape[0]-group*8)
        bp = np.zeros(8, dtype='<i4'); mp = np.ones(8, dtype='<i4')
        bp[:count] = b[group*8:group*8+count]
        mp[:count] = m[group*8:group*8+count]
        packets.append(bp.tobytes()+mp.tobytes()+blocks[group].tobytes())
    return packets


@dataclass(frozen=True)
class BankGeometry:
    """One read of 8 Ci per spatial bank; address in 64-bit words."""
    p: int
    ci: int
    height: int
    width: int
    words_per_bank: int

    def __post_init__(self):
        if self.p not in (4, 6) or not 1 <= self.ci <= 192:
            raise ValueError('unsupported spatial/Ci geometry')
        if min(self.height, self.width, self.words_per_bank) < 1:
            raise ValueError('empty bank geometry')
        if self.used_words > self.words_per_bank:
            raise ValueError('tile exceeds physical per-bank capacity')

    @property
    def columns(self):
        return math.ceil(self.width/self.p)

    @property
    def used_words(self):
        return math.ceil(self.ci/8)*self.height*self.columns

    def address(self, block, y, x):
        if not (0 <= block < math.ceil(self.ci/8) and 0 <= y < self.height and 0 <= x < self.width):
            raise ValueError('bank coordinate outside tile')
        return x % self.p, (block*self.height+y)*self.columns+x//self.p

    def encode(self, chw):
        a = _int8(chw, 3)
        if a.shape != (self.ci, self.height, self.width):
            raise ValueError('bank input shape mismatch')
        blocks = pack_feature(a)
        banks = np.zeros((self.p, self.words_per_bank, 8), dtype=np.int8)
        for x in range(self.width):
            bank = x % self.p
            for block in range(blocks.shape[0]):
                for y in range(self.height):
                    _, address = self.address(block, y, x)
                    banks[bank, address] = blocks[block, y, x]
        return banks
