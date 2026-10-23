"""Materialize the derived DDR representation without changing A's tensors."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from acdc.contract import INPUT_SHAPE, SHAPES, checked_int8
from acdc.integer import requantize
from .layout import pack_feature, pack_parameter_groups, unpack_feature


@dataclass(frozen=True)
class Segment:
    name: str
    address: int
    data: bytes

    @property
    def sha256(self):
        return hashlib.sha256(self.data).hexdigest()


def parameter_image(model):
    if len(model.layers)!=15 or len(model.concats)!=3:
        raise ValueError('frozen 15 Conv / 3 CAT model required')
    packets=[]
    for i,(layer,(ci,co,h,k)) in enumerate(zip(model.layers,SHAPES)):
        if (layer.index!=i or layer.weight.shape!=(co,ci,k,k) or layer.relu is not (i<14)
                or layer.full_input_shape!=(1,ci,h,h)):
            raise ValueError('model geometry differs from frozen production architecture')
        packets.extend(pack_parameter_groups(layer.weight,layer.bias,layer.multipliers))
    data=b''.join(packets)
    if len(data)!=493120:
        raise ValueError('derived parameter size mismatch')
    return data


def table_image(model):
    if len(model.concats)!=3:
        raise ValueError('three CAT tables required')
    tables=[]
    for cat in model.concats:
        table=np.zeros((2,256),dtype=np.int8)
        for branch,m in enumerate(cat.multiplier_q24):
            values=np.arange(-127,128,dtype=np.int32)
            table[branch,values&255]=requantize(values,int(m))
        tables.append(table.tobytes())
    return b''.join(tables)


def prepare_segments(model, program, input_tensor):
    """Return only initialized regions; outputs/intermediates come from PL.

    Caller owns CMA, validates physical ranges, flushes these CPU writes, then
    starts the matching hardware profile. Epoch must be fresh for changed data.
    ``load_bundle`` remains responsible for production hash/contract admission.
    """
    x=checked_int8(input_tensor,INPUT_SHAPE)
    if program['schema']!='acdc-spatial-program-v1':
        raise ValueError('wrong execution program ABI')
    addr=program['addresses']
    return (Segment('input',addr['input'],pack_feature(x[0]).tobytes()),
            Segment('parameters',addr['parameters'],parameter_image(model)),
            Segment('luts',addr['luts'],table_image(model)),
            Segment('program',addr['program'],program['program']))


def decode_head(data):
    if len(data)!=8*256*256:
        raise ValueError('wrong physical C8 head size')
    packed=np.frombuffer(data,dtype=np.int8).reshape(1,256,256,8)
    if np.any(packed[...,4:]!=0):
        raise ValueError('head padding lanes were not zeroed')
    return unpack_feature(packed,4)[None].copy()
