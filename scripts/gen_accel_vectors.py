"""Command-level vectors. Layer goldens are comparison oracles, not PL results."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference.tinyunet import Model, conv2d, rescale_lut, apply_lut
from reference.tiling import choose_tile, choose_fused_decoder_tile, tiles, input_tile, deep_input_tile


class Writer:
    def __init__(self,stream):
        self.stream,self.job,self.frame,self.runs,self.loads,self.errors=stream,0,0x12345678,0,0,0

    def line(self,op,*args):
        fields=[op,*args]
        fields += [0]*(8-len(fields))
        self.stream.write(" ".join(f"{int(value):x}" for value in fields)+"\n")

    def write(self,addr,value,strobe=15,response=0):
        self.line(0,addr,value,strobe,response)

    def configure(self,mode,h,w,ci,co=0,k=1,split=0):
        for addr,value in ((0x18,mode),(0x1c,h),(0x20,w),(0x24,ci),(0x28,co),(0x2c,k),(0x30,split),(0x34,self.frame)):
            self.write(addr,value)

    def start(self,command):
        self.job+=1
        self.write(0x14,command)
        self.write(0x38,self.job)
        self.write(8,1)

    def packet(self,op,data):
        data=bytes(data)
        for pos in range(0,len(data),8):
            part=data[pos:pos+8]
            self.line(op,int.from_bytes(part,"little"),(1<<len(part))-1,int(pos+len(part)==len(data)))

    def wait(self,error=0,received=0,sent=0):
        if error: self.errors+=1
        self.line(2,error,self.job,self.frame,received,sent)

    def load(self,command,data):
        self.start(command)
        self.packet(1,data)
        self.wait(received=len(data))
        self.loads+=1

    def run(self,expected):
        self.packet(3,expected.tobytes())
        self.start(4)
        self.wait(sent=expected.size)
        self.runs+=1

    def reset(self):
        self.write(8,2)
        self.line(5,0x0c,0,0x7f)


def main():
    model=Model(Path("role_a_final/int8"))
    path=Path("out/vectors/accel.txt")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",encoding="ascii") as f:
        w=Writer(f)
        w.line(5,0,0x504c4431,0xffffffff)
        w.line(5,4,0x00010002,0xffffffff)
        # All real convolution parameter shapes, bias and per-channel multiplier.
        for idx,(layer,cfg,params) in enumerate(zip(model.layers,model.configs,model.parameters)):
            weights,bias,mult=params
            folder=model.root/layer["directory"]
            data=np.fromfile(folder/"input.bin",dtype=np.int8).reshape(cfg["input_shape"])
            expected=np.fromfile(folder/"output_golden.bin",dtype=np.int8).reshape(cfg["output_shape"])
            p=cfg["padding"]
            data=np.pad(data,((0,0),(0,0),(p,p),(p,p)))
            co,ci,k,_=weights.shape
            w.configure(0,*data.shape[2:],ci,co,k)
            packed=weights.tobytes()+bias.astype("<i4").tobytes()+mult.astype("<i4").tobytes()
            w.load(1,packed)
            w.load(2,data.tobytes())
            w.run(expected)
            if idx==0:
                w.run(expected)  # retained input and parameters, new job id
            if idx==7:
                w.start(4)
                w.line(6,20)
                w.reset()  # interrupt a multi-beat convolution before any result
        # Non-multiple input/output groups expose masked lanes and packet tails.
        rng=np.random.default_rng(20260927)
        for ci,co,h,width,k in ((5,9,5,7,3),(1,1,1,1,1)):
            data=rng.integers(-127,128,(1,ci,h+k-1,width+k-1),dtype=np.int8)
            weights=rng.integers(-7,8,(co,ci,k,k),dtype=np.int8)
            bias=rng.integers(-100,101,co,dtype=np.int32)
            mult=np.full(co,1<<20,dtype=np.int32)
            w.configure(0,*data.shape[2:],ci,co,k)
            w.load(1,weights.tobytes()+bias.astype("<i4").tobytes()+mult.astype("<i4").tobytes())
            w.load(2,data.tobytes())
            w.run(conv2d(data,weights,bias,mult,0))
        # A's 14 layer transitions, including two independently scaled branches.
        for idx in range(1,15):
            layer,previous=model.layers[idx],model.layers[idx-1]
            main=rescale_lut(previous["output_scale"],layer["input_scale"])
            expected=model.golden(idx,"input")[:,:,:8,:8]
            if idx in (8,10,12):
                skip={8:5,10:3,12:1}[idx]
                first=model.golden(skip,"output")[:,:,:8,:8]
                second=model.golden(idx-1,"output")[:,:,:4,:4]
                skip_lut=rescale_lut(model.layers[skip]["output_scale"],layer["input_scale"])
                w.configure(4,8,8,expected.shape[1],split=first.shape[1])
                tables=skip_lut.tobytes()+main.tobytes()
                payload=first.tobytes()+second.tobytes()
            else:
                pool=idx in (2,4,6)
                length=16 if pool else 8
                first=model.golden(idx-1,"output")[:,:,:length,:length]
                w.configure(2 if pool else 1,length,length,first.shape[1])
                tables=main.tobytes(); payload=first.tobytes()
            w.load(3,tables); w.load(2,payload); w.run(expected)
        # Fused loader: non-monotonic signed mapping, odd groups, tails,
        # both kernels and every real plain transition. Raw outputs stay raw.
        raw=np.arange(-127,128,dtype=np.int8).reshape(1,1,1,255)
        table=rng.integers(-127,128,256,dtype=np.int8)
        w.configure(5,1,255,1,1,1)
        w.load(1,b"\x01"+bytes(4)+(1<<24).to_bytes(4,"little"))
        w.load(3,table.tobytes()); w.load(2,raw.tobytes())
        w.run(apply_lut(raw,table))
        for ci,co,h,width,k in ((5,9,5,7,3),(1,1,1,7,1)):
            raw=rng.integers(-127,128,(1,ci,h+k-1,width+k-1),dtype=np.int8)
            table=rng.integers(-127,128,256,dtype=np.int8)
            weights=rng.integers(-7,8,(co,ci,k,k),dtype=np.int8)
            bias=rng.integers(-100,101,co,dtype=np.int32)
            mult=np.full(co,1<<20,dtype=np.int32)
            w.configure(5,*raw.shape[2:],ci,co,k)
            w.load(1,weights.tobytes()+bias.astype('<i4').tobytes()+mult.astype('<i4').tobytes())
            w.load(3,table.tobytes()); w.load(2,raw.tobytes())
            expected=conv2d(apply_lut(raw,table),weights,bias,mult,0)
            w.run(expected); w.run(expected)
            # Replacing a LUT must invalidate an already mapped input.
            w.load(3,table.tobytes()); w.start(4); w.wait(error=4); w.reset()
        for idx in (1,3,5,7,9,11,13,14):
            layer=model.layers[idx]; weights,bias,mult=model.parameters[idx]
            co,ci,k,_=weights.shape
            raw=model.golden(idx-1,'output')[:,:,:10,:10]
            table=rescale_lut(model.layers[idx-1]['output_scale'],layer['input_scale'])
            expected=conv2d(apply_lut(raw,table),weights,bias,mult,0)
            w.configure(5,*raw.shape[2:],ci,co,k)
            w.load(1,weights.tobytes()+bias.astype('<i4').tobytes()+mult.astype('<i4').tobytes())
            w.load(3,table.tobytes()); w.load(2,raw.tobytes()); w.run(expected)
        # Missing LUT is rejected before input bytes can reach mapped RAM.
        w.reset(); w.configure(5,1,1,1,1,1); w.start(2); w.wait(error=4); w.reset()
        # Mode 6 fused decoder-input convolution: first/middle/last tile of
        # every decoder entry, against the frozen per-layer output goldens.
        for idx in (8,10,12):
            layer=model.layers[idx]; weights,bias,mult=model.parameters[idx]
            co,ci,k,_=weights.shape
            skip_index={8:5,10:3,12:1}[idx]
            skip_all=model.golden(skip_index,"output"); deep_all=model.golden(idx-1,"output")
            expected_all=model.golden(idx,"output")
            height,width=expected_all.shape[2:]
            th,tw=choose_fused_decoder_tile(ci,skip_all.shape[1],co,height,width)
            lut0=rescale_lut(model.layers[skip_index]["output_scale"],layer["input_scale"])
            lut1=rescale_lut(model.layers[idx-1]["output_scale"],layer["input_scale"])
            tile_list=list(tiles(height,width,th,tw))
            params=weights.tobytes()+bias.astype('<i4').tobytes()+mult.astype('<i4').tobytes()
            first=True
            for tile in (tile_list[0],tile_list[len(tile_list)//2],tile_list[-1]):
                skip_patch=input_tile(skip_all,tile,k,layer["padding"])
                deep_patch=deep_input_tile(deep_all,tile,k)
                golden=expected_all[:,:,tile.y:tile.y+tile.height,tile.x:tile.x+tile.width]
                w.configure(6,*skip_patch.shape[2:],ci,co,k,split=skip_patch.shape[1])
                if first:
                    w.load(1,params); first=False
                w.load(3,lut0.tobytes()+lut1.tobytes())
                w.load(2,skip_patch.tobytes()+deep_patch.tobytes())
                w.run(golden)
        # Random mode 6 case: negative bytes, arbitrary dual LUTs, odd output
        # lanes, two deep groups, x/y parity, repeated frames and LUT reload.
        ci,split,ih,iw,co=12,4,6,6,9
        skip_raw=rng.integers(-127,128,(1,split,ih,iw),dtype=np.int8)
        deep_raw=rng.integers(-127,128,(1,ci-split,ih//2+1,iw//2+1),dtype=np.int8)
        lut0=rng.integers(-127,128,256,dtype=np.int8)
        lut1=rng.integers(-127,128,256,dtype=np.int8)
        weights=rng.integers(-7,8,(co,ci,3,3),dtype=np.int8)
        bias=rng.integers(-100,101,co,dtype=np.int32)
        mult=np.full(co,1<<20,dtype=np.int32)
        rows=(np.arange(ih)+1)//2; cols=(np.arange(iw)+1)//2
        expanded=apply_lut(deep_raw,lut1)[:,:,rows][:,:,:,cols]
        expected=conv2d(np.concatenate((apply_lut(skip_raw,lut0),expanded),axis=1),
                        weights,bias,mult,0)
        params=weights.tobytes()+bias.astype('<i4').tobytes()+mult.astype('<i4').tobytes()
        payload=skip_raw.tobytes()+deep_raw.tobytes()
        w.configure(6,ih,iw,ci,co,3,split)
        w.load(1,params); w.load(3,lut0.tobytes()+lut1.tobytes()); w.load(2,payload)
        w.run(expected); w.run(expected)
        # Interrupt a fused convolution mid-accumulation, then fully recover.
        w.start(4); w.line(6,20); w.reset()
        w.load(1,params); w.load(3,lut0.tobytes()+lut1.tobytes()); w.load(2,payload)
        w.run(expected)
        # Reloading either LUT half invalidates a mapped mode 6 input.
        w.load(3,lut0.tobytes()+lut1.tobytes()); w.start(4); w.wait(error=4); w.reset()
        # Missing LUT is rejected before mapped input bytes reach the banks.
        w.configure(6,ih,iw,ci,co,3,split); w.start(2); w.wait(error=4); w.reset()
        # Shape/contract violations fail before or during the packet.
        for args in ((6,7,6,8,8,3,4),(6,6,7,8,8,3,4),(6,6,6,8,8,1,4),
                     (6,6,6,8,8,3,2),(6,6,6,8,8,3,0),(6,6,6,8,8,3,8),
                     (6,6,6,9,8,3,4),(6,258,258,8,8,3,4)):
            w.configure(*args); w.start(2); w.wait(error=1); w.reset()
        w.configure(6,4,4,8,8,3,4)
        w.load(3,lut0.tobytes()+lut1.tobytes())
        # TLAST one beat too early (100-byte contract) is error 2; RESET
        # clears the LUT, so reload it before the keep and -128 probes.
        w.start(2); w.line(1,0x0101010101010101,255,1); w.wait(error=2,received=8); w.reset()
        w.load(3,lut0.tobytes()+lut1.tobytes())
        w.start(2); w.line(1,0,5,1); w.wait(error=3); w.reset()
        w.load(3,lut0.tobytes()+lut1.tobytes())
        w.start(2); w.line(1,0x0101010101010180,255,0); w.wait(error=7,received=1); w.reset()
        # Busy writes must respond SLVERR without modifying the active shape.
        w.configure(0,1,1,1,1,1)
        w.start(2)
        w.write(0x1c,123,response=2); w.write(8,1,response=2)
        w.line(5,0x1c,1,0xffffffff)
        w.packet(1,b"\x01"); w.wait(received=1)
        # Missing model data, malformed lengths/keep, numeric range, invalid shape.
        w.reset(); w.start(4); w.wait(error=4)
        w.write(8,1,response=2); w.reset()
        w.start(1); w.packet(1,b"\x01"); w.wait(error=2,received=1); w.reset()
        w.start(2); w.line(1,0,5,1); w.wait(error=3); w.reset()
        w.start(2); w.line(1,1,1,0); w.wait(error=3); w.reset()
        w.start(2); w.packet(1,b"\x80"); w.wait(error=7,received=1); w.reset()
        w.configure(0,0,1,1,1,1); w.start(2); w.wait(error=1); w.reset()
        # A full final beat without TLAST must fail at the expected byte count.
        w.configure(0,1,8,1,1,1); w.start(2); w.line(1,0x0101010101010101,255,0)
        w.wait(error=2,received=8); w.reset()
        # Overflow is surfaced as a failed job, never returned as valid output.
        w.configure(0,1,1,1,1,1)
        params=b"\x7f"+(0x7fffffff).to_bytes(4,"little")+(1<<24).to_bytes(4,"little")
        w.load(1,params); w.load(2,b"\x7f"); w.start(4); w.wait(error=5); w.reset()
        # Recover from faults, then hold the last result beat: DONE must wait.
        params=b"\x02"+bytes(4)+(1<<24).to_bytes(4,"little")
        w.load(1,params); w.load(2,b"\x03")
        w.line(7,0) # force downstream backpressure
        w.packet(3,b"\x06"); w.start(4)
        w.line(8) # wait for output TVALID without consuming it
        w.line(5,0x0c,1,7) # BUSY=1,DONE=0,ERROR=0
        w.line(7,2) # restore seeded random TREADY
        w.wait(sent=1); w.runs+=1
        w.line(10,w.runs,w.loads,w.errors)
    print(f"ACCEL_VECTORS runs={w.runs} loads={w.loads} jobs={w.job}")


if __name__=="__main__":
    main()
