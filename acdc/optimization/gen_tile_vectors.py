"""Dense Co8 tile vectors, with a separate full-Ci NumPy convolution oracle."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from acdc.integer import conv_full
from acdc.contract import SHAPES
from .gen_compute_vectors import packed
from .layout import BankGeometry, pack_weights, pack_parameter_groups


def generate(output, p):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0xACDC5016)
    commands, expected, writer_expected, source_memory = [], [], [], []
    shapes = [(3,8,5,7,3), (16,4,7,11,1), (192,8,4,9,3),
              (48,7,13,17,3), (128,8,12,16,3), (16,4,32,32,1)]
    shapes += [(ci, min(co,8), 5, 7, k) for ci,co,h,k in SHAPES]
    shapes += [(192,8,24,24,3), (16,4,32,256,1)]
    for case, (ci,co,oh,ow,k) in enumerate(shapes):
        slot = case % 2
        x = rng.integers(-127,128,(ci,oh,ow),dtype=np.int16).astype(np.int8)
        x = np.pad(x,((0,0),(k//2,k//2),(k//2,k//2)))
        w = rng.integers(-12,13,(co,ci,k,k),dtype=np.int16).astype(np.int8)
        b = rng.integers(-1000,1001,co,dtype=np.int32)
        m = rng.integers(1<<15,1<<18,co,dtype=np.int32)
        relu = case % 2 == 0
        result = conv_full(x[None],w,b,m,padding=0,relu=relu)[0]
        packet=pack_parameter_groups(w,b,m)[0]
        for offset in range(0,len(packet),8):
            source_memory.append(f'{0x08000000+case*0x10000+offset:08x} {int.from_bytes(packet[offset:offset+8],"little"):016x}')
        for yy in range(oh):
            for xx in range(ow):
                pixel=np.zeros(8,dtype=np.int8);pixel[:co]=result[:,yy,xx]
                address=0x10000000+case*0x10000+(yy*(ow+4)+xx)*8
                writer_expected.append(f'{address:08x} {packed(pixel,8):016x}')
        if oh%2==0 and ow%2==0:
            pooled=result.reshape(co,oh//2,2,ow//2,2).max(axis=(2,4))
            for yy in range(oh//2):
                for xx in range(ow//2):
                    pixel=np.zeros(8,dtype=np.int8);pixel[:co]=pooled[:,yy,xx]
                    address=0x18000000+case*0x10000+(yy*(ow//2+3)+xx)*8
                    writer_expected.append(f'{address:08x} {packed(pixel,8):016x}')
        geometry = BankGeometry(p,ci,x.shape[1],x.shape[2],4608 if p==4 else 3584)
        banks = geometry.encode(x)
        for bank in range(p):
            for address in range(geometry.used_words):
                commands.append(f'0 {slot:x} {bank:x} {address:x} {packed(banks[bank,address],8):016x}')
        wp = pack_weights(w)[0].reshape(-1,8,8)
        for address, block in enumerate(wp):
            for bank in range(8):
                commands.append(f'1 {slot:x} {bank:x} {address:x} {packed(block[bank],8):016x}')
        bp = np.zeros(8,dtype=np.int32);mp = np.ones(8,dtype=np.int32)
        bp[:co]=b;mp[:co]=m
        commands.append(f'2 {slot:x} {packed(bp,32):064x} {packed(mp,32):064x}')
        commands.append(f'3 {ci:x} {co:x} {oh:x} {ow:x} {int(k==3):x} {int(relu):x} {slot:x} {slot:x}')
        for y in range(oh):
            for gx in range(0,ow,p):
                for start in range(0,p*8,4):
                    data,keep=0,0
                    for lane in range(4):
                        pixel,channel=divmod(start+lane,8)
                        if gx+pixel<ow and channel<co:
                            data |= (int(result[channel,y,gx+pixel])&255) << (lane*8)
                            keep |= 1 << lane
                    tag=(y<<8)|gx
                    token=(tag<<7)|(int(start==p*8-4)<<6)|(int(start==0)<<5)|keep
                    expected.append(f'{((token<<32)|data):018x}')
    (output/'commands.txt').write_text(str(len(commands))+'\n'+'\n'.join(commands)+'\n',encoding='ascii')
    (output/'tile_expected.txt').write_text(str(len(expected))+'\n'+'\n'.join(expected)+'\n',encoding='ascii')
    (output/'writer_expected.txt').write_text(str(len(writer_expected))+'\n'+'\n'.join(writer_expected)+'\n',encoding='ascii')
    (output/'source_memory.txt').write_text(str(len(source_memory))+'\n'+'\n'.join(source_memory)+'\n',encoding='ascii')
    return dict(p=p,tiles=len(shapes),commands=len(commands),outputs=len(expected))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--p',type=int,choices=(4,6),default=4)
    args=parser.parse_args()
    print(generate(args.output,args.p))
