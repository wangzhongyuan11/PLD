"""Bounded ACD8 whole-frame PYNQ owner with explicit cache/DMA lifetime.

The GPIO control/status ABI belongs to build_spatial.tcl. No legacy SimpleDMA
object is created: the PL issues all DataMover commands autonomously.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time
import os
import signal
import sys
import numpy as np

from acdc.contract import INPUT_SHAPE, ContractError, checked_int8
from acdc.hardware import Lease
from .image import parameter_image,table_image,decode_head
from .layout import pack_feature,unpack_feature
from .program import compile_program

DEVICE_ID=0x41434438
ABI=0x00020000
_QUARANTINE=[]


class UnsafeDMAState(RuntimeError):
    pass


def validate_overlay(core,gpio):
    if (core.get('type')!='user:pld:spatial_accel:1.0' or core.get('phys_addr')!=0x43c00000
            or gpio.get('phys_addr')!=0x41200000 or gpio.get('type')!='xilinx.com:ip:axi_gpio:2.0'):
        raise ContractError('not the paired ACD8/GPIO overlay')


class SpatialDevice:
    def __init__(self,runtime,model,*,timeout=5.):
        if not math.isfinite(timeout) or timeout<=0:raise ValueError('positive finite timeout required')
        from acdc.package import verify_runtime
        runtime=verify_runtime(runtime,model)
        record=json.loads((runtime/'overlay/build_manifest.json').read_text('utf-8'))
        if record.get('hardware_id')!=DEVICE_ID or record.get('abi')!=ABI:
            raise ContractError('runtime is not ACD8')
        self.timeout=timeout;self.model=model;self.arena=None;self.unsafe=False;self.closed=False
        self.epoch=0;self.last_timing={};self.lease=Lease()
        try:
            from pynq import Overlay,MMIO,allocate,Clocks
            import pyxrt
            self.to_device=pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
            self.from_device=pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
            self.overlay=Overlay(str(runtime/'overlay/dma_top.bit'),download=False)
            validate_overlay(self.overlay.ip_dict.get('spatial_accel_0',{}),self.overlay.ip_dict.get('dma_control',{}))
            self.overlay.download()
            self.core=MMIO(0x43c00000,256);self.gpio=MMIO(0x41200000,65536)
            self.gpio.write(4,0);self.gpio.write(12,0xffffffff)
            # GPIO reset default is zero; no transfer exists at first release.
            self.gpio.write(0,0);time.sleep(.001);self.gpio.write(0,1)
            if self.core.read(0)!=DEVICE_ID or self.core.read(4)!=ABI:raise ContractError('ACD8 identity mismatch')
            p=self.core.read(8)&255
            if p not in (4,6) or p!=record['spatial_p']:raise ContractError('spatial lane mismatch')
            self.profile='P256' if p==4 else 'P384'
            self.clock_mhz=float(Clocks.fclk0_mhz)
            if abs(self.clock_mhz-record['clock_mhz'])>.1 or self.core.read(0x3c)!=round(self.clock_mhz*1e6):
                raise ContractError('FCLK/build/core clock mismatch')
            self.arena=allocate(shape=(16*1024*1024,),dtype=np.uint8,cacheable=True)
            self.base=int(self.arena.device_address)
            if self.base+self.arena.nbytes>0x20000000:raise ContractError('CMA outside declared HP DDR aperture')
            self.program=compile_program(self.profile,base=self.base,epoch=1)
            self.view=np.asarray(self.arena)
            # Eliminate dirty initialization cachelines before PL owns workspace.
            self.arena.flush()
            for name,data in [('parameters',parameter_image(model)),('luts',table_image(model)),('program',self.program['program'])]:
                self._write(name,data);self._sync(name,self.to_device)
            self.program_words=self._region('program').view('<u4').reshape(-1,16)
        except BaseException:
            self.close();raise

    def _region(self,name):
        b=self.program['arena'].buffers[name]
        return self.view[b['offset']:b['offset']+b['bytes']]

    def _write(self,name,data):
        target=self._region(name)
        if len(data)!=target.nbytes:raise ContractError('CMA segment size mismatch: '+name)
        target[:]=np.frombuffer(data,np.uint8)

    def _sync(self,name,direction):
        b=self.program['arena'].buffers[name]
        start=b['offset']//64*64;end=(b['offset']+b['bytes']+63)//64*64
        if end>self.arena.nbytes:raise ContractError('cache sync outside owned allocation')
        self.arena.bo.sync(direction,end-start,start)

    def snapshot(self):
        return dict(registers={f'{a:02x}':int(self.core.read(a)) for a in range(0,0x40,4)},
                    dma_status=int(self.gpio.read(8)),epoch=self.epoch)

    def recover(self):
        """HALT both actual masters, then common reset. Never free on failure."""
        self.gpio.write(0,3)
        deadline=time.monotonic()+self.timeout
        while self.gpio.read(8)&3!=3:
            if time.monotonic()>=deadline:
                self.unsafe=True
                raise UnsafeDMAState('DataMover HALT not confirmed; CMA and lease retained')
            time.sleep(.0005)
        self.gpio.write(0,0);time.sleep(.001);self.gpio.write(0,1)
        if self.core.read(0x10)!=0:raise RuntimeError('ACD8 common reset did not clear state')

    def _feature(self,name,channels,height):
        self._sync(name,self.from_device)
        size=((channels+7)//8)*height*height*8
        packed=self._region(name)[:size].view(np.int8).reshape(-1,height,height,8)
        return unpack_feature(packed,channels)[None]

    def capture_nodes(self,first):
        """Debug-only prefix replays, always starting from the original input.

        Captures 15 Conv + 3 Pool physical DDR outputs. Virtual CAT is covered
        by its downstream convolution and loader RTL, never fabricated here.
        """
        nodes={};self.capture_timings=[]
        for i in range(15):
            prefix,timing=self.forward(first,debug_layer=i);nodes.update(prefix)
            self.capture_timings.append(dict(last_layer=i+1,**timing))
        return nodes

    def forward(self,first,*,cancel=None,debug_layer=None):
        if debug_layer is not None and (type(debug_layer) is not int or not 0<=debug_layer<15):
            raise ValueError('invalid debug layer')
        if self.closed or self.unsafe:raise RuntimeError('device is closed or quarantined')
        started=time.perf_counter();x=checked_int8(first,INPUT_SHAPE)
        if self.core.read(0x10)&5:raise RuntimeError('device busy/fault before frame')
        if self.epoch==0xffffffff:self.core.write(0x0c,2);self.epoch=0
        self.epoch+=1
        stage=time.perf_counter()
        self._write('input',pack_feature(x[0]).tobytes())
        self.program_words[:,15]=self.epoch
        converted=time.perf_counter()
        self._sync('input',self.to_device);self._sync('program',self.to_device)
        synced=time.perf_counter()
        self.core.write(0x14,self.program['addresses']['program'])
        n=(self.program['layers'][debug_layer]['first_job']+self.program['layers'][debug_layer]['jobs']
           if debug_layer is not None else len(self.program['jobs']))
        self.core.write(0x18,n);self.core.write(0x1c,self.epoch)
        self.core.write(0x0c,1)
        deadline=time.monotonic()+self.timeout
        try:
            while True:
                status=int(self.core.read(0x10))
                if status&4:raise RuntimeError('ACD8 fault '+str(self.snapshot()))
                if status&2:break
                if cancel is not None and cancel.is_set():raise InterruptedError('frame cancelled')
                if time.monotonic()>=deadline:raise TimeoutError('ACD8 completion timeout '+str(self.snapshot()))
                time.sleep(.0002)
            waited=time.perf_counter()
            cycles=int(self.core.read(0x28))
            counts=int(self.core.read(0x2c))
            if counts!=((n<<16)|n):raise RuntimeError('ACD8 incomplete task counts')
            if debug_layer is None:
                self._sync('head',self.from_device)
                head=decode_head(self._region('head').tobytes())
            else:
                from acdc.contract import SHAPES
                target=['work_a','skip0','work_a','skip1','work_a','skip2','work_a','work_b',
                        'work_a','work_b','work_a','work_b','work_a','work_b','head'][debug_layer]
                _,co,h,_=SHAPES[debug_layer]
                head={f'C{debug_layer+1:02}':self._feature(target,co,h)}
                if debug_layer in (1,3,5):
                    level=(debug_layer-1)//2
                    head[f'P{level}']=self._feature(f'pool{level}',co,h//2)
            self.core.write(0x0c,4)
        except BaseException:
            self.last_fault_snapshot=self.snapshot()
            self.recover();raise
        finished=time.perf_counter()
        self.last_timing=dict(slice_seconds=finished-started,input_validation_seconds=stage-started,
            input_layout_seconds=converted-stage,flush_seconds=synced-converted,
            submit_wait_seconds=waited-synced,head_seconds=finished-waited,pl_cycles=cycles,
            pl_seconds=cycles/(self.clock_mhz*1e6),jobs=n,epoch=self.epoch,debug_prefix_layer=debug_layer)
        return head,self.last_timing.copy()

    def close(self):
        if self.closed:return
        if hasattr(self,'gpio') and hasattr(self,'core'):
            if self.core.read(0x10)&5 and not self.unsafe:
                try:self.recover()
                except BaseException:self.unsafe=True
            if not self.unsafe:self.gpio.write(0,0)
        if self.unsafe:
            _QUARANTINE.append(self)
            # A global reference only retains XRT buffers while this process
            # lives. An unhandled exception in a CLI would otherwise free the
            # BO at process exit while the master might still write to it.
            # Suspend the real board owner before unwinding. No DMA polling or
            # retry occurs; the manager must refuse handoff while PID/cgroup
            # and the FPGA lease remain live. Resume only after externally
            # confirming the masters stopped (normally a board reset).
            if sys.platform=='linux' and self.arena is not None:
                print('ACD8_UNSAFE_DMA_QUARANTINE: suspending owner; CMA/lease held; do not kill or resume before master-stop confirmation',file=sys.stderr,flush=True)
                os.kill(os.getpid(),signal.SIGSTOP)
            raise UnsafeDMAState('DMA state uncertain; retained allocation and exclusive lease')
        if self.arena is not None:self.arena.freebuffer();self.arena=None
        self.lease.close();self.closed=True
