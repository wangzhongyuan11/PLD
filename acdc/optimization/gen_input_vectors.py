"""Independent full-image/virtual-CAT fixtures for the DMA halo loader."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from .gen_compute_vectors import packed, quant
from .layout import BankGeometry, pack_feature


def generate(output,p):
    output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(0x1ACDC251)
    # ci, split, H, W, tile_y, tile_x, tile_h, tile_w, kernel, CAT
    cases=[(3,0,32,32,0,0,5,7,3,0),(16,0,32,32,27,25,5,7,3,0),
           (192,0,32,32,4,7,8,12,3,0),(16,0,32,256,2,0,4,256,1,0),
           (192,64,32,32,0,0,8,12,3,1),(96,32,32,32,1,3,7,11,3,1),
           (48,16,64,64,56,51,8,13,3,1),(48,16,32,256,0,1,6,31,3,1)]
    memory=[];expected=[];commands=[]
    for case,(ci,split,h,w,ty,tx,th,tw,k,cat) in enumerate(cases):
        base0=0x02000000+case*0x200000;base1=base0+0x100000
        if cat:
            skip=rng.integers(-127,128,(split,h,w),dtype=np.int16).astype(np.int8)
            deep=rng.integers(-127,128,(ci-split,h//2,w//2),dtype=np.int16).astype(np.int8)
            # Oracle uses exact dyadic floating RNE, distinct from LUT generation.
            aligned=np.rint(deep.astype(np.float64)*.5).astype(np.int8)
            logical=np.concatenate([skip,np.repeat(np.repeat(aligned,2,1),2,2)])
            sources=[(base0,pack_feature(skip)),(base1,pack_feature(deep))]
        else:
            logical=rng.integers(-127,128,(ci,h,w),dtype=np.int16).astype(np.int8)
            sources=[(base0,pack_feature(logical))]
        for base,source in sources:
            for offset,pixel in enumerate(source.reshape(-1,8)):
                memory.append(f'{base+offset*8:08x} {packed(pixel,8):016x}')
        pad=k//2
        tile=np.pad(logical,((0,0),(pad,pad),(pad,pad)))[:,ty:ty+th+2*pad,tx:tx+tw+2*pad]
        geometry=BankGeometry(p,ci,tile.shape[1],tile.shape[2],4608 if p==4 else 3584)
        blocks=pack_feature(tile)
        for cg in range(blocks.shape[0]):
            for y in range(tile.shape[1]):
                for x in range(tile.shape[2]):
                    bank,address=geometry.address(cg,y,x)
                    key=(case<<20)|(bank<<16)|address
                    expected.append(f'{key:08x} {packed(blocks[cg,y,x],8):016x}')
        xs=range(max(0,tx-pad),min(w,tx+tw+pad));ys=range(max(0,ty-pad),min(h,ty+th+pad))
        read_words=((ci+7)//8)*len(xs)*len(ys)
        if cat:
            read_words=(split//8)*len(xs)*len(ys)+((ci-split)//8)*len({x//2 for x in xs})*len({y//2 for y in ys})
        commands.append(f'{base0:x} {base1:x} {h:x} {w:x} {ty:x} {tx:x} {th:x} {tw:x} {ci:x} {split:x} {int(k==3):x} {cat:x} {case%2:x} {read_words:x}')
    (output/'input_cases.txt').write_text(str(len(commands))+'\n'+'\n'.join(commands)+'\n',encoding='ascii')
    (output/'banks_expected.txt').write_text(str(len(expected))+'\n'+'\n'.join(expected)+'\n',encoding='ascii')
    lut=[quant(v if v<128 else v-256,m,False) for m in (1<<24,1<<23) for v in range(256)]
    for offset in range(0,512,8):
        memory.append(f'{0x01000000+offset:08x} {packed(lut[offset:offset+8],8):016x}')
    (output/'source_memory.txt').write_text(str(len(memory))+'\n'+'\n'.join(memory)+'\n',encoding='ascii')
    (output/'lut.txt').write_text('\n'.join(f'{v&255:02x}' for v in lut)+'\n',encoding='ascii')
    print(dict(p=p,cases=len(cases),source_words=len(memory),bank_writes=len(expected)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    parser.add_argument('--p',type=int,choices=(4,6),default=4);args=parser.parse_args()
    generate(args.output,args.p)
