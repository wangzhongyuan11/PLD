"""Independent exact FP64/Torch integer witness for the frozen A arithmetic.

Does not import B's integer operators, graph executor or tile emulator. FP64
represents every integer product/sum exactly under the loader's INT32 bound.
This generates B-side cross-reference evidence, not A-delivered node files.
"""
from __future__ import annotations

import numpy as np


def forward(model, first):
    import torch
    import torch.nn.functional as f
    from .contract import checked_int8
    q=torch.from_numpy(checked_int8(first,model.input_shape).copy())
    captures={};skips={};index=0

    def requant(acc,m):
        value=acc.to(torch.int64)*m.to(torch.int64)
        mag=value.abs();base=mag>>24;rem=mag&((1<<24)-1)
        rounded=base+((rem>(1<<23))|((rem==(1<<23))&((base&1)!=0)))
        return (rounded*value.sign()).clamp(-127,127).to(torch.int8)

    def conv(value):
        nonlocal index
        layer=model.layers[index];index+=1
        acc=f.conv2d(value.to(torch.float64),torch.from_numpy(layer.weight.copy()).to(torch.float64),
                     torch.from_numpy(layer.bias.copy()).to(torch.float64),padding=layer.padding)
        result=requant(acc,torch.from_numpy(layer.multipliers.copy()).view(1,-1,1,1))
        if layer.relu:result=result.clamp_min(0)
        captures[layer.name]=result.numpy().copy()
        return result

    with torch.inference_mode():
        for level in range(4):
            q=conv(conv(q))
            if level<3:
                skips[level]=q;n,c,h,w=q.shape
                q=q.reshape(n,c,h//2,2,w//2,2).amax((3,5))
                captures[f'P{level}']=q.numpy().copy()
        for j,cat in enumerate(model.concats):
            skip=requant(skips.pop(cat.level),torch.tensor(cat.multiplier_q24[0]))
            deep=requant(q,torch.tensor(cat.multiplier_q24[1])).repeat_interleave(2,2).repeat_interleave(2,3)
            q=torch.cat((skip,deep),dim=1);captures[f'CAT{j+1}']=q.numpy().copy()
            q=conv(conv(q))
        q=conv(q)
    return captures
