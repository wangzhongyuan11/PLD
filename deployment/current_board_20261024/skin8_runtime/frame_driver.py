"""Experimental PYNQ driver for board_top; no service or overlay changes here.

Caller supplies an already verified MMIO instance and PYNQ allocate function.
Timeout deliberately retains all buffers until hardware is confirmed quiescent.
"""
import json
import time
from pathlib import Path
import numpy as np
from board_image import build_image


_quarantined_sessions=[]


class FrameSession:
    def __init__(self, directory, mmio, allocate):
        self.directory=Path(directory);self.mmio=mmio;self.buffers={};self.active=False;self.closed=False
        self.faulted=False
        if mmio.read(4)!=0x534b3800:raise RuntimeError('wrong Skin8 register ABI')
        if mmio.read(0)&2:raise RuntimeError('FPGA is already busy')
        # Preflight frozen artifacts before making physical allocations.
        dummy={'input':(0x1000000,0x100000),'arena':(0x2000000,0x1000000),'weights':(0x4000000,0x100000),
               'luts':(0x5000000,0x10000),'descriptors':(0x6000000,0x100000),'head':(0x7000000,0x100000)}
        image=build_image(self.directory,dummy)
        try:
            for name,size in image['required_sizes'].items():
                self.buffers[name]=allocate(shape=(size,),dtype=np.uint8)
            regions={name:(int(buf.physical_address),int(buf.nbytes)) for name,buf in self.buffers.items()}
            image=build_image(self.directory,regions)
            self.count=image['count'];self.manifest=image['manifest']
            for name,buf in self.buffers.items():
                buf[:]=0
                if name in ('weights','luts','descriptors'):buf[:]=np.frombuffer(image[name],np.uint8)
                buf.flush()
            mmio.write(0x10,regions['descriptors'][0]);mmio.write(0x14,self.count)
        except BaseException:
            for buf in self.buffers.values():buf.freebuffer()
            self.closed=True
            raise

    def run(self, quantized, timeout=5.0):
        if self.closed or self.faulted or self.active:raise RuntimeError('session is not available')
        if timeout<=0:raise ValueError('timeout must be positive')
        shape=tuple(self.manifest['layers'][0]['input_shape'])
        x=np.asarray(quantized)
        if x.dtype!=np.int8 or x.shape!=shape:raise ValueError('expected frozen quantized CHW int8 input')
        if np.any(x==-128):raise ValueError('frozen model input range is -127..127')
        if self.mmio.read(0)&2:raise RuntimeError('FPGA is already busy')
        begin=time.perf_counter()
        c,h,w=shape
        packed=self.buffers['input'].view(np.int8).reshape(-1,h,w,8)
        packed[:]=0
        for channel in range(c):packed[channel//8,:,:,channel%8]=x[channel]
        self.buffers['input'].flush()
        prepared=time.perf_counter()
        self.active=True
        try:
            self.mmio.write(0,1)
            deadline=prepared+timeout
            while True:
                status=self.mmio.read(0)
                if status&4 and not status&2:break
                if time.perf_counter()>=deadline:raise TimeoutError('FPGA timeout; retain allocations and recover hardware before closing')
                time.sleep(0.0001)
            self.active=False
            if status&8:raise RuntimeError('FPGA returned an execution error; partial frame discarded')
            if self.mmio.read(0x1c)!=self.count:raise RuntimeError('incomplete descriptor list')
            hardware_cycles=self.mmio.read(0x18)
            finished=time.perf_counter()
            self.buffers['head'].invalidate()
            co,oh,ow=self.manifest['layers'][-1]['output_shape']
            head=self.buffers['head'].view(np.int8).reshape(-1,oh,ow,8).transpose(0,3,1,2).reshape(-1,oh,ow)[:co].copy()
            return head,{'prepare_seconds':prepared-begin,'submit_to_done_seconds':finished-prepared,
                         'input_to_owned_head_seconds':time.perf_counter()-begin,'hardware_cycles':hardware_cycles}
        except BaseException:
            self.faulted=True
            if self.active and self not in _quarantined_sessions:_quarantined_sessions.append(self)
            raise

    def close(self):
        if self.closed:return
        if self.active or self.mmio.read(0)&2:
            raise RuntimeError('cannot free buffers while hardware activity is unresolved')
        for buf in self.buffers.values():buf.freebuffer()
        self.closed=True

    def release_after_verified_reset(self):
        """Call only after an external full hardware reset/reconfiguration is verified."""
        if self.mmio.read(0)&2:raise RuntimeError('hardware still reports busy')
        self.active=False
        self.close()
        if self in _quarantined_sessions:_quarantined_sessions.remove(self)
