"""CAM2 ABI shape rules, shared by scheduling, emulator and real MMIO driver."""
from __future__ import annotations
import math
import numpy as np


def shape(mode, h, w, ci, co=0, kernel=1, split=0, *, relu=False):
    values = (mode, h, w, ci, co, kernel, split)
    if not isinstance(relu, (bool, np.bool_)):
        raise ValueError('ReLU must be boolean')
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in values):
        raise ValueError("integer CAM2 dimensions required")
    if mode not in range(5) or not 1 <= ci <= 192 or min(h, w) < 1:
        raise ValueError("invalid CAM2 mode/input channels/dimensions")
    if mode != 0 and relu:
        raise ValueError("CAM2 ReLU flag is defined only for convolution")
    plane = h * w
    if mode == 0:
        if not 1 <= co <= 64 or kernel not in (1, 3) or min(h, w) < kernel or max(h, w) > 258:
            raise ValueError("invalid CAM2 convolution")
        if math.ceil(co / 8) * math.ceil(ci / 4) * kernel * kernel > 2048:
            raise ValueError("CAM2 weight bank exceeds 2048 entries")
        oh, ow, oc = h - kernel + 1, w - kernel + 1, co
    else:
        if max(h, w) > 256 or kernel != 1 or co != 0:
            raise ValueError("invalid CAM2 tensor shape")
        if mode in (2, 4) and (h % 2 or w % 2):
            raise ValueError("pool/concat dimensions must be even")
        if mode == 4 and not 0 < split < ci:
            raise ValueError("invalid CAM2 skip-first split")
        oh, ow, oc = (h // 2, w // 2, ci) if mode == 2 else (h * 2, w * 2, ci) if mode == 3 else (h, w, ci)
    if mode != 4 and split != 0:
        raise ValueError("split only belongs to concat")
    occupancy = math.ceil(ci / 4) * plane
    input_bytes = ci * plane
    if mode == 4:
        occupancy = math.ceil(split / 4) * plane + math.ceil((ci - split) / 4) * (plane // 4)
        input_bytes = split * plane + (ci - split) * (plane // 4)
    output_occupancy = math.ceil(oc / 8) * oh * ow
    if max(oh, ow) > 256 or min(oh, ow) < 1 or occupancy > 32768 or output_occupancy > 8192:
        raise ValueError("CAM2 tile exceeds feature banks")
    return dict(mode=int(mode), h=int(h), w=int(w), ci=int(ci), co=int(co), kernel=int(kernel),
                split=int(split), oh=int(oh), ow=int(ow), oc=int(oc), relu=bool(relu),
                input_bytes=int(input_bytes), output_bytes=int(oc * oh * ow),
                parameter_bytes=int(co * ci * kernel * kernel + 8 * co) if mode == 0 else 0,
                lut_bytes=512 if mode == 4 else 256,
                input_occupancy=int(occupancy), output_occupancy=int(output_occupancy))


def tensor_tile_shape(mode, channels, height, width, split=0):
    """Choose OUTPUT tile dimensions; nearest-2 boundaries remain aligned."""
    candidates = []
    for h in {height} | {1 << p for p in range(height.bit_length())}:
        for w in {width} | {1 << p for p in range(width.bit_length())}:
            if h > height or w > width or (mode in (3, 4) and (h % 2 or w % 2)):
                continue
            ih, iw = (h * 2, w * 2) if mode == 2 else (h // 2, w // 2) if mode == 3 else (h, w)
            try:
                shape(mode, ih, iw, channels, split=split)
            except ValueError:
                continue
            candidates.append((math.ceil(height / h) * math.ceil(width / w), -h * w, -w, h, w))
    if not candidates:
        raise ValueError("no CAM2 tensor tile fits")
    return min(candidates)[-2:]
