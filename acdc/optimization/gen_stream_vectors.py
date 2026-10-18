"""First-principles signed pool and frozen-Q24 lookup stream vectors."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .gen_compute_vectors import packed, quant


def generate(output):
    output.mkdir(parents=True, exist_ok=True)
    rng=np.random.default_rng(0x2516ACDC)
    cases=[(1,2,2),(1,6,10),(1,4,256),(1,32,32),(1,16,64),(1,256,2),(0,2,4)]
    commands=[str(len(cases))];raw=[];pooled=[]
    for enabled,h,w in cases:
        a=rng.integers(-127,128,(8,h,w),dtype=np.int16).astype(np.int8)
        commands.append(f'{enabled:x} {h:x} {w:x}')
        for y in range(h):
            for x in range(w):
                word=f'{packed(a[:,y,x],8):016x}'
                commands.append(word);raw.append(word)
        if enabled:
            result=a.reshape(8,h//2,2,w//2,2).max(axis=(2,4))
            for y in range(h//2):
                for x in range(w//2):
                    pooled.append(f'{((y<<72)|(x<<64)|packed(result[:,y,x],8)):020x}')
    (output/'pool_input.txt').write_text('\n'.join(commands)+'\n',encoding='ascii')
    (output/'pool_raw_expected.txt').write_text(str(len(raw))+'\n'+'\n'.join(raw)+'\n',encoding='ascii')
    (output/'pool_expected.txt').write_text(str(len(pooled))+'\n'+'\n'.join(pooled)+'\n',encoding='ascii')
    multipliers=(1<<24,1<<23)
    lut=[quant(v if v<128 else v-256,m,False) for m in multipliers for v in range(256)]
    inputs=[];expected=[]
    for j in range(2048):
        values=[(j*8+i)%255-127 for i in range(8)]
        lookup=j%3!=0;branch=j%2;zero=j%17==0;mask=j%256
        result=[0 if zero or not mask&(1<<i) else
                quant(v,multipliers[branch],False) if lookup else v
                for i,v in enumerate(values)]
        inputs.append(f'{int(lookup):x} {branch:x} {int(zero):x} {mask:02x} {j:08x} {packed(values,8):016x}')
        expected.append(f'{((j<<64)|packed(result,8)):024x}')
    (output/'lut.txt').write_text('\n'.join(f'{v&255:02x}' for v in lut)+'\n',encoding='ascii')
    (output/'transform_input.txt').write_text(str(len(inputs))+'\n'+'\n'.join(inputs)+'\n',encoding='ascii')
    (output/'transform_expected.txt').write_text('\n'.join(expected)+'\n',encoding='ascii')
    print(dict(pool_cases=len(cases),raw_words=len(raw),pool_words=len(pooled),transform_words=len(inputs)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    generate(p.parse_args().output)
