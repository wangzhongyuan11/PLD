"""Dense synthetic 15-convolution / 21-node DDR execution oracle.

The default grid is 32x32, preserving all frozen channel counts and kernels.
Only external input, weights and LUTs initialize test DDR. Intermediates must
come from RTL writes. Work A/B addresses are reused across decoder layers.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from acdc.contract import SHAPES
from acdc.integer import conv_full, pool, concat, requantize
from .layout import pack_feature, pack_parameter_groups
from .program import Arena, Job, KERNEL3, RELU, CAT, POOL, BARRIER
from .planner import plan


def generate(output, p, divisor=8, model=None):
    if divisor not in (1,2,4,8):
        raise ValueError('unsupported test spatial divisor')
    output.mkdir(parents=True,exist_ok=True)
    rng = np.random.default_rng(0xACDCED25)
    initial,expected,jobs = {},[],[]
    versions = defaultdict(int)
    arena = Arena(0x10000000)
    addresses = {n:arena.allocate(n,s) for n,s in
                 [('input',524288),('a',1048576),('b',1048576),('skip0',1048576),
                  ('skip1',524288),('skip2',262144),('pool0',262144),('pool1',131072),
                  ('pool2',65536),('params',524288),('luts',1536),('head',524288)]}

    def initialize(address, data):
        for off in range(0,len(data),8):
            if address+off in initial:
                raise AssertionError('overlapping fixture memory')
            initial[address+off] = int.from_bytes(data[off:off+8],'little')

    def oracle(address, tensor):
        data = pack_feature(tensor[0]).tobytes()
        for off in range(0,len(data),8):
            a = address+off
            expected.append(f'{a:08x} {versions[a]:08x} {int.from_bytes(data[off:off+8],"little"):016x}')
            versions[a] += 1

    h = 256//divisor
    if model is None:
        x = rng.integers(-127,128,(1,3,h,h),dtype=np.int16).astype(np.int8)
    else:
        from acdc.contract import safe_member
        first=np.fromfile(safe_member(model.root,model.spec['golden']['input']['path']),np.int8).reshape(1,3,256,256)
        x=first[:,:,::divisor,::divisor].copy()
    initialize(addresses['input'],pack_feature(x[0]).tobytes())
    sources = ['input','a','pool0','a','pool1','a','pool2','a','skip2','a','skip1','a','skip0','a','b']
    targets = ['a','skip0','a','skip1','a','skip2','a','b','a','b','a','b','a','b','head']
    producer0 = [0,1,3,4,6,7,9,10,8,13,5,16,2,19,20]
    tensors = {'input':x}
    param_offset = 0
    schedule = plan('P256' if p==4 else 'P384') if divisor==1 else None
    for i,(ci,co,original_h,k) in enumerate(SHAPES):
        h = original_h//divisor
        cat = i in (8,10,12);pooled = i in (1,3,5)
        split = {8:64,10:32,12:16}.get(i,0)
        lut = addresses['luts']+(i-8)//2*512 if cat else 0
        if cat:
            cm = (np.array([1<<24,12582912-(i-8)*65536],dtype=np.int32) if model is None
                  else np.array(model.concats[(i-8)//2].multiplier_q24,dtype=np.int32))
            table = np.zeros((2,256),dtype=np.int8)
            for branch in range(2):
                table[branch,np.arange(-127,128)&255] = requantize(np.arange(-127,128,dtype=np.int32),cm[branch])
            initialize(lut,table.tobytes())
            x = concat(tensors[sources[i]],tensors['b'],cm)
        else:
            x = tensors[sources[i]]
        if model is None:
            w = rng.integers(-3,4,(co,ci,k,k),dtype=np.int16).astype(np.int8)
            bias = rng.integers(-300,301,co,dtype=np.int32)
            mult = rng.integers(1<<17,1<<20,co,dtype=np.int32)
        else:
            layer=model.layers[i];w=layer.weight;bias=layer.bias;mult=layer.multipliers
        result = conv_full(x,w,bias,mult,padding=k//2,relu=i<14)
        packets = pack_parameter_groups(w,bias,mult)
        initialize(addresses['params']+param_offset,b''.join(packets))
        oracle(addresses[targets[i]],result)
        tensors[targets[i]] = result
        pool_dest = addresses['pool'+str((i-1)//2)] if pooled else 0
        if pooled:
            tensors['pool'+str((i-1)//2)] = pool(result)
            oracle(pool_dest,tensors['pool'+str((i-1)//2)])
        first = len(jobs)
        tile_h,tile_w = schedule['layers'][i]['tile_hw'] if schedule else (8,8)
        for y in range(0,h,tile_h):
            for xx in range(0,h,tile_w):
                th,tw = min(tile_h,h-y),min(tile_w,h-xx)
                for group in range(len(packets)):
                    flags = (KERNEL3 if k==3 else 0)|(RELU if i<14 else 0)|(CAT if cat else 0)|(POOL if pooled else 0)
                    if len(jobs)==first:flags |= BARRIER
                    job = Job(addresses[sources[i]],addresses['b'] if cat else 0,
                              addresses['params']+param_offset+group*len(packets[0]),lut,
                              addresses[targets[i]]+(group*h*h+y*h+xx)*8,h*8,
                              pool_dest+(group*(h//2)**2+y//2*(h//2)+xx//2)*8 if pooled else 0,
                              h//2*8 if pooled else 0,h,h,y,xx,th,tw,ci,min(8,co-group*8),split,flags,
                              len(jobs),producer0[i],{8:11,10:14,12:17}.get(i,0),1).validate('P256' if p==4 else 'P384')
                    jobs.append(job)
        param_offset += sum(map(len,packets))
    (output/'source_memory.txt').write_text(str(len(initial))+'\n'+''.join(f'{a:08x} {v:016x}\n' for a,v in sorted(initial.items())),encoding='ascii')
    (output/'memory_expected.txt').write_text(str(len(expected))+'\n'+'\n'.join(expected)+'\n',encoding='ascii')
    (output/'jobs.txt').write_text(str(len(jobs))+'\n'+''.join(f'{int.from_bytes(j.pack(),"little"):0128x}\n' for j in jobs),encoding='ascii')
    if model is not None and divisor==1:
        golden=np.fromfile(safe_member(model.root,model.spec['golden']['head']['path']),np.int8).reshape(1,4,256,256)
        if not np.array_equal(result,golden):raise ValueError('full real-model DDR oracle differs from A head')
    metadata = dict(synthetic=model is None or model.synthetic,original_grid=divisor==1,spatial_divisor=divisor,p=p,
                    model_spec_sha256=model.model_sha256 if model else None,
                    input_kind='A golden input' if model and divisor==1 else 'spatially reduced A input' if model else 'dense random',
                    convolutions=15,logical_nodes=21,jobs=len(jobs),initialized_words=len(initial),
                    expected_writes=len(expected),intermediate_golden_injected=False,parameter_bytes=param_offset)
    (output/'fixture.json').write_text(json.dumps(metadata,indent=2)+'\n',encoding='utf-8')
    return metadata


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--p',type=int,choices=(4,6),default=4)
    parser.add_argument('--divisor',type=int,choices=(1,2,4,8),default=8)
    parser.add_argument('--model',type=Path);parser.add_argument('--model-spec-sha')
    args=parser.parse_args()
    model=None
    if args.model:
        from acdc.model import load_bundle
        model=load_bundle(args.model,expected_sha=args.model_spec_sha)
    print(generate(args.output,args.p,args.divisor,model))
