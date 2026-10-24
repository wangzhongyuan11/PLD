"""Driver ownership/layout tests with fake MMIO, never a hardware benchmark."""
import json
from pathlib import Path
import numpy as np
from board_image import build_image
from frame_driver import FrameSession
from budget import ROOT


class Buffer(np.ndarray):
    def __new__(cls,size,address):
        obj=np.zeros(size,np.uint8).view(cls);obj.physical_address=address
        obj.freed=False;obj.flushes=0;obj.invalidates=0;return obj
    def flush(self):self.flushes+=1
    def invalidate(self):self.invalidates+=1
    def freebuffer(self):self.freed=True


class MMIO:
    def __init__(self):self.status=0;self.count=0;self.polls=0;self.stuck=False
    def write(self,address,value):
        if address==0:self.status=2;self.polls=0
        elif address==0x14:self.count=value
    def read(self,address):
        if address==4:return 0x534b3800
        if address==0:
            if self.status==2:
                self.polls+=1
                if not self.stuck and self.polls>2:self.status=4
            return self.status
        if address==0x1c:return self.count
        if address==0x18:return 123456
        return 0


def main():
    reports={}
    for name in ('skin','camus'):
        allocated=[];address=0x1000000
        def allocate(shape,dtype):
            nonlocal address
            buf=Buffer(shape[0],address);address+=(shape[0]+4095)//4096*4096
            allocated.append(buf);return buf
        mmio=MMIO();session=FrameSession(ROOT/'out/compiled_models'/name,mmio,allocate)
        shape=tuple(session.manifest['layers'][0]['input_shape']);x=np.zeros(shape,np.int8)
        x[:,0,0]=np.arange(shape[0])+1
        for invalid_timeout in (0,-1,float('nan'),float('inf')):
            try:session.run(x,timeout=invalid_timeout)
            except ValueError:pass
            else:raise AssertionError('unbounded/invalid timeout accepted')
        assert mmio.status==0,'invalid timeout must not submit hardware work'
        session.buffers['head'][:]=7
        head,timing=session.run(x)
        assert np.all(head==7) and timing['hardware_cycles']==123456
        packed=session.buffers['input'].reshape(-1,shape[1],shape[2],8)
        assert np.array_equal(packed[0,0,0,:shape[0]],x[:,0,0])
        assert np.all(packed[:,:,:,shape[0]:]==0)
        session.buffers['head'][:]=9
        assert np.all(head==7),'result must own its bytes'
        mmio.stuck=True
        try:session.run(x,timeout=0.001)
        except TimeoutError:pass
        else:raise AssertionError('missing timeout')
        try:session.close()
        except RuntimeError:pass
        else:raise AssertionError('active DMA buffers freed')
        assert not any(b.freed for b in allocated)
        mmio.status=0 # Explicitly simulate external verified hardware reset.
        session.release_after_verified_reset()
        assert all(b.freed for b in allocated)
        regions={'input':(0x1000000,0x100000),'arena':(0x2000000,0x1000000),'weights':(0x4000000,0x100000),
                 'luts':(0x5000000,0x10000),'descriptors':(0x6000000,0x100000),'head':(0x7000000,0x100000)}
        for bad in ({**regions,'head':regions['input']},{**regions,'descriptors':(0x6000008,0x100000)},
                    {**regions,'weights':(0x4000000,64)},{**regions,'input':(0x1fffffc0,0x100000)}):
            try:build_image(ROOT/'out/compiled_models'/name,bad)
            except ValueError:pass
            else:raise AssertionError('unsafe region accepted')
        reports[name]={'layout':True,'owned_head':True,'timeout_retains_buffers':True,'explicit_reset_release':True,'invalid_regions':4}
        print(name,'driver ownership/layout passed')
    (ROOT/'evidence/frame_driver.json').write_text(json.dumps({'kind':'fake_mmio_not_board','models':reports},indent=2))


if __name__=='__main__':main()
