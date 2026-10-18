"""Synthetic 15-layer-shape + CAT/error AXIS vectors; independent integer oracle."""
from pathlib import Path
import json
import numpy as np

from camus.gen_vectors import Writer
from camus.device import output_channel_chunks
from .contract import SHAPES
from .integer import conv_full,requantize,pool,concat


def main(root=Path('out/acdc/vectors'),device_id=0x41434433):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261018)
    rows=[]
    with (root/'accel.txt').open('w',encoding='ascii') as stream:
        wr=Writer(stream)
        wr.line(5,0,device_id,0xffffffff);wr.line(5,4,0x00030000,0xffffffff)
        def cfg(mode,h,w,ci,co=0,k=1,split=0,relu=False):
            wr.configure(mode,h,w,ci,co,k,split);wr.write(0x5c,int(relu))
        def conv(x,w,b,m,relu):
            cfg(0,*x.shape[2:],x.shape[1],w.shape[0],w.shape[2],relu=relu)
            wr.load(1,w.tobytes()+b.astype('<i4').tobytes()+m.astype('<i4').tobytes())
            wr.load(2,x.tobytes());wr.run(conv_full(x,w,b,m,0,relu))
        for i,(ci,co,hw,k) in enumerate(SHAPES):
            for chunk in output_channel_chunks(ci,co,k):
                x=rng.integers(-30,31,size=(1,ci,4+k-1,6+k-1),dtype=np.int8)
                w=rng.integers(-2,3,size=(chunk,ci,k,k),dtype=np.int8)
                b=rng.integers(-100,101,size=chunk,dtype=np.int32)
                m=np.full(chunk,1<<20,np.int32)
                conv(x,w,b,m,i<14)
                rows.append(dict(layer=i+1,ci=ci,co_chunk=chunk))
        for channels,split in [(192,64),(96,32),(48,16),(13,5)]:
            skip=rng.integers(-127,128,(1,split,4,6),dtype=np.int8)
            deep=rng.integers(-127,128,(1,channels-split,2,3),dtype=np.int8)
            cfg(4,4,6,channels,split=split)
            raw=np.arange(256,dtype=np.uint8).view(np.int8).astype(np.int32)
            lut=lambda m:requantize(raw,np.array(m,np.int32)).tobytes()
            wr.load(3,lut(1<<23)+lut(1<<24));wr.load(2,skip.tobytes()+deep.tobytes())
            wr.run(concat(skip,deep,(1<<23,1<<24)))
        x=rng.integers(-127,128,(1,16,32,256),dtype=np.int8)
        cfg(2,32,256,16);wr.load(3,lut(1<<24));wr.load(2,x.tobytes());wr.run(pool(x))
        x=np.array([-127,-5,-3,-1,1,3,5,127],np.int8).reshape(1,1,1,8)
        conv(x,np.ones((1,1,1,1),np.int8),np.zeros(1,np.int32),np.array([1<<23],np.int32),False)
        # Head negative bias survives; repeated RUN and random backpressure exercise stable output.
        conv(np.zeros((1,16,2,3),np.int8),np.zeros((4,16,1,1),np.int8),
             np.array([-7,2,-1,3],np.int32),np.full(4,1<<24,np.int32),False)
        # Actual large spatial tiles stress 15-bit input / 13-bit output addressing,
        # Co16 tail masks, full Co64 bank capacity, and full 256-pixel head width.
        for ci,co,h,w,k,relu in [(64,64,34,34,3,True),(128,56,30,34,3,True),
                                 (192,32,20,34,3,True),(16,4,32,256,1,False)]:
            x=rng.integers(-127,128,(1,ci,h,w),dtype=np.int8)
            weight=np.zeros((co,ci,k,k),np.int8)
            for o in range(co):weight[o,(o*7)%ci,o%k,(o//k)%k]=1 if o%2 else -1
            b=np.arange(co,dtype=np.int32)-co//2
            conv(x,weight,b,np.full(co,1<<24,np.int32),relu)
        for mode,ci,co,k in [(0,193,1,1),(0,192,64,3),(5,1,1,1)]:
            cfg(mode,8,8,ci,co,k);wr.start(1);wr.wait(error=1);wr.reset()
        cfg(1,2,3,1);wr.start(2);wr.packet(1,bytes(5));wr.wait(error=2,received=5);wr.reset()
        cfg(0,4,4,3,16,3);wr.start(4);wr.wait(error=4);wr.reset()
        wr.line(10,wr.runs,wr.loads,wr.errors)
        (root/'coverage.json').write_text(json.dumps(dict(synthetic=True,production_model=False,
            layers=rows,runs=wr.runs,loads=wr.loads,expected_errors=wr.errors,seed=20261018),indent=2)+'\n')
    print(f'ACDC_SYNTHETIC_VECTORS runs={wr.runs} loads={wr.loads} errors={wr.errors}')


if __name__=='__main__':main()
