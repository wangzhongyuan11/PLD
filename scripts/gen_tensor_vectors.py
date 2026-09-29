"""PL non-convolution vectors: A layer transitions and independent edge cases."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference.tinyunet import Model, apply_lut, maxpool2, upsample2, rescale_lut, require_equal


def emit(f, op, first, tables, expected, second=None, repeat=False, reset=False):
    _, ca, h, w = first.shape
    channels = ca if second is None else ca+second.shape[1]
    f.write(f"0 {h:x} {w:x} {channels:x} {op:x} {ca if second is not None else 0:x}\n")
    for table, lut in enumerate(tables):
        for idx,value in enumerate(lut):
            f.write(f"2 {table*256+idx:x} {int(value)&255:x} 0 0 0\n")
    for data, base in ((first,0), (second,((ca+3)//4)*h*w)):
        if data is None:
            continue
        _, c, ih, iw = data.shape
        for ch in range(c):
            for y in range(ih):
                for x in range(iw):
                    addr = base+(ch//4)*ih*iw+y*iw+x
                    f.write(f"1 {ch%4:x} {addr:x} {int(data[0,ch,y,x])&255:x} 0 0\n")
    for run in range(2 if repeat else 1):
        f.write(f"3 {int(reset and run==0):x} 0 0 0 0\n")
        _, c, oh, ow = expected.shape
        for ch in range(c):
            for y in range(oh):
                for x in range(ow):
                    addr = (ch//8)*oh*ow+y*ow+x
                    f.write(f"4 {ch%8:x} {addr:x} {int(expected[0,ch,y,x])&255:x} 0 0\n")
    return (2 if repeat else 1), expected.size*(2 if repeat else 1)


def main():
    model = Model(Path("role_a_final/int8"))
    path = Path("out/vectors/tensor.txt")
    path.parent.mkdir(parents=True,exist_ok=True)
    runs = checks = 0
    with path.open("w",encoding="ascii") as f:
        for idx in range(1,15):
            layer, previous = model.layers[idx], model.layers[idx-1]
            main_lut = rescale_lut(previous["output_scale"],layer["input_scale"])
            expected = model.golden(idx,"input")[:,:,:8,:8]
            if idx in (8,10,12):
                skip = {8:5,10:3,12:1}[idx]
                first = model.golden(skip,"output")[:,:,:8,:8]
                second = model.golden(idx-1,"output")[:,:,:4,:4]
                skip_lut = rescale_lut(model.layers[skip]["output_scale"],layer["input_scale"])
                actual = np.concatenate((apply_lut(first,skip_lut),upsample2(apply_lut(second,main_lut))),axis=1)
                require_equal(actual,expected,f"transition{idx}")
                n, size = emit(f,3,first,(skip_lut,main_lut),expected,second)
            else:
                pool = idx in (2,4,6)
                length = 16 if pool else 8
                first = model.golden(idx-1,"output")[:,:,:length,:length]
                require_equal(apply_lut(maxpool2(first) if pool else first,main_lut),expected,f"transition{idx}")
                n, size = emit(f,int(pool),first,(main_lut,),expected,repeat=idx==1,reset=idx==1)
            runs+=n; checks+=size
        rng = np.random.default_rng(20260926)
        identity=np.arange(256,dtype=np.uint8).view(np.int8)
        invert=np.clip(-identity.astype(np.int16),-127,127).astype(np.int8)
        for op in range(4):
            first=rng.integers(-127,128,(1,5,6,10),dtype=np.int8)
            second=None
            if op==0: expected=apply_lut(first,invert)
            elif op==1: expected=apply_lut(maxpool2(first),invert)
            elif op==2: expected=apply_lut(upsample2(first),invert)
            else:
                second=rng.integers(-127,128,(1,7,3,5),dtype=np.int8)
                expected=np.concatenate((apply_lut(first,invert),upsample2(second)),axis=1)
            n,size=emit(f,op,first,(invert,identity),expected,second)
            runs+=n; checks+=size
        # Input bank last address and output bank last address in separate cases.
        for op,ci in ((1,16),(0,8)):
            first=rng.integers(-127,128,(1,ci,64,128),dtype=np.int8)
            expected=maxpool2(first) if op else first
            n,size=emit(f,op,first,(identity,),expected)
            runs+=n; checks+=size
        invalid=((0,8,8,0,0),(5,8,8,1,0),(8,7,8,3,4),(8,8,8,3,0),
                 (8,8,8,3,8),(256,256,8,0,0),(129,128,8,2,0),(64,129,16,1,0),
                 (65,128,8,0,0),(8,8,97,0,0))
        for h,w,c,op,split in invalid:
            f.write(f"0 {h:x} {w:x} {c:x} {op:x} {split:x}\n3 0 1 0 0 0\n")
        f.write(f"7 {runs:x} {len(invalid):x} {checks:x} 0 0\n")
    print(f"TENSOR_VECTORS A_transitions=14 runs={runs} bytes={checks} rejected={len(invalid)}")


if __name__=="__main__":
    main()
