"""Dense deterministic first-principles compute vectors; no FPGA output as oracle."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def packed(values, bits):
    return sum((int(v) & ((1 << bits) - 1)) << (i * bits)
               for i, v in enumerate(np.asarray(values).ravel()))


def quant(value, multiplier, relu):
    # Magnitude division deliberately differs from RTL's signed floor-bit path.
    product = int(value) * int(multiplier)
    magnitude, fraction = divmod(abs(product), 1 << 24)
    if fraction > (1 << 23) or (fraction == (1 << 23) and magnitude & 1):
        magnitude += 1
    result = -magnitude if product < 0 else magnitude
    result = min(127, max(-127, result))
    return max(0, result) if relu else result


def generate(output: Path, p: int):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0xACDC2564)
    inputs, expected = [], []
    for group in range(96):
        count = (1, 2, 9, 18, 36, 72, 144, 216)[group % 8]
        relu = bool(group % 3 == 0)
        bias = rng.integers(-32768, 32768, 8, dtype=np.int64)
        mult = rng.choice(np.array([1, 1 << 16, 1 << 22, 1 << 23,
                                   1 << 24, (1 << 31)-1]), 8)
        mask = np.ones(p * 8, dtype=np.uint8)
        if group % 4 == 0:
            mask[-(group % (p*8) + 1):] = 0
        tie_case = group % 13 == 0
        overflow_case = group % 17 == 1
        if tie_case:
            bias = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.int64)
            mult[:] = 1 << 23
        if overflow_case:
            bias[:] = (1 << 31)-1
        acc = np.broadcast_to(bias, (p, 8)).copy()
        overflow = False
        tag = 0xACDC0000 + group
        for tap in range(count):
            x = rng.integers(-127, 128, (p, 8), dtype=np.int64)
            w = rng.integers(-127, 128, (8, 8), dtype=np.int64)
            if tie_case:
                w[:] = 0
            if overflow_case:
                x[:] = 127
                w[:] = 127
            acc += x @ w.T
            overflow |= bool(np.any((acc < -(1 << 31)) | (acc >= (1 << 31))))
            acc = ((acc + (1 << 31)) % (1 << 32)) - (1 << 31)
            inputs.append(f'{int(tap==0):x} {int(tap==count-1):x} {int(relu):x} '
                          f'{tag:08x} {packed(mask,1):x} {packed(bias,32):064x} '
                          f'{packed(mult,32):064x} {packed(x,8):0{p*16}x} {packed(w,8):0128x}')
        values = [quant(acc.ravel()[j], mult[j % 8], relu) for j in range(p*8)]
        for start in range(0, p*8, 4):
            token = (tag << 7) | (int(start == p*8-4) << 6) | (int(start == 0) << 5)
            token |= int(overflow) << 4 | packed(mask[start:start+4], 1)
            expected.append(f'{((token<<32)|packed(values[start:start+4],8)):018x}')
    (output/'input.txt').write_text(str(len(inputs))+'\n'+'\n'.join(inputs)+'\n', encoding='ascii')
    (output/'expected.txt').write_text(str(len(expected))+'\n'+'\n'.join(expected)+'\n', encoding='ascii')
    return dict(p=p, groups=96, beats=len(inputs), output_beats=len(expected))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--p', type=int, choices=(4, 6), default=4)
    args = parser.parse_args()
    print(generate(args.output, args.p))
