"""Optional CAM2 command batching over existing PYNQ MMIO mappings.

The library never allocates/frees DDR or recovers hardware. SimpleDMA remains
the single transaction owner; Python's existing reset/reload path handles any
native failure before a buffer can be reused or freed.
"""
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np


class NativeControl:
    def __init__(self, path):
        path=Path(path).resolve()
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        self.info={'kind':'native_packet','abi':2,'library_sha256':digest(path)}
        self.library=ct.CDLL(str(path))
        self.library.tensor_control_abi.restype=ct.c_uint32
        if self.library.tensor_control_abi()!=2:
            raise ValueError('native control ABI mismatch')
        self.configure_fn=self.library.tensor_configure
        self.configure_fn.argtypes=[ct.c_void_p,ct.c_uint32,ct.POINTER(ct.c_uint32)]
        self.configure_fn.restype=ct.c_int
        self.packet_fn=self.library.tensor_packet
        self.packet_fn.argtypes=[ct.c_void_p,ct.c_uint32,ct.c_void_p,ct.c_uint32,
                                ct.c_uint32,ct.c_uint32,ct.c_uint32,ct.c_uint64,
                                ct.c_uint32,ct.c_uint64,ct.POINTER(ct.c_uint32)]
        self.packet_fn.restype=ct.c_int
        self.wait_fn=self.library.tensor_wait_core
        self.wait_fn.argtypes=[ct.c_void_p,ct.c_uint32,ct.c_uint32,ct.c_uint32,
                              ct.c_uint32,ct.c_uint32,ct.c_uint64,ct.POINTER(ct.c_uint32)]
        self.wait_fn.restype=ct.c_int
        self.result=(ct.c_uint32*8)()
        self.core=None; self.dma=None

    def bind(self, core, dma):
        arrays=[np.asarray(core.array),np.asarray(dma.mmio.array)]
        for array in arrays:
            if (array.dtype!=np.uint32 or array.ndim!=1 or not array.flags.c_contiguous or
                    not array.flags.writeable or array.nbytes<0x60 or array.ctypes.data%4):
                raise ValueError('native control requires writable UINT32 MMIO arrays')
        self.core,self.dma=core,dma
        self._arrays=arrays  # Retain the actual mapped allocations, not just addresses.
        self.core_pointer,self.dma_pointer=(a.ctypes.data for a in arrays)
        self.core_bytes,self.dma_bytes=(a.nbytes for a in arrays)

    def configure(self, config, frame):
        values=(ct.c_uint32*9)(*(config[k] for k in ('mode','h','w','ci','co','kernel','split')),frame,int(config['relu']))
        status=self.configure_fn(self.core_pointer,self.core_bytes,values)
        if status:
            raise RuntimeError(f'native configure failed code={status}')

    def wait_core(self, device, received, sent):
        if (received>0)==(sent>0):
            raise ValueError('exactly one command direction required')
        status=self.wait_fn(self.core_pointer,self.core_bytes,4 if sent else 2,
                            device.frame,device.job,sent or received,
                            max(1,int(device.timeout*1e9)),self.result)
        if status==9:
            raise TimeoutError('core completion timeout (native)')
        if status:
            raise RuntimeError(f'native core wait failed code={status} core_error={self.result[6]}')
        return {'core_cycles':int(self.result[0]),'compute_cycles':int(self.result[1])}

    def execute(self, device, command, buffer, length):
        dma=self.dma
        if device.config is None or command not in (1,2,3,4):
            raise ValueError('configured CAM2 command required')
        if dma.active:
            raise RuntimeError('previous transfer still active; reset required')
        timeout=device.timeout
        if not math.isfinite(timeout) or not 0<timeout<18446744073:
            raise ValueError('invalid native control timeout')
        receive=command==4
        dma._validate(buffer,length,dma.align[int(receive)])
        if not 0<=device.frame<=0xffffffff:
            raise ValueError('frame ID outside UINT32')
        begin=time.perf_counter()
        buffer.flush()  # RX dirty lines must be written back before device access.
        flush_seconds=time.perf_counter()-begin
        device.job=(device.job+1)&0xffffffff or 1
        dma._pending=(None if receive else buffer,buffer if receive else None,
                      length if receive else None,(0x30,) if receive else (0,))
        dma.active=True
        status=self.packet_fn(self.core_pointer,self.core_bytes,self.dma_pointer,self.dma_bytes,
                              command,device.frame,device.job,int(buffer.physical_address),length,
                              max(1,int(timeout*1e9)),self.result)
        if status:
            raise RuntimeError(f'native packet failed code={status} core_status={self.result[2]:#x} '
                               f'dma_status={self.result[3]:#x} core_error={self.result[6]}')
        if receive:
            buffer.invalidate()
        dma.active=False
        dma._pending=None
        return {'core_cycles':int(self.result[0]),'compute_cycles':int(self.result[1])}, (
            flush_seconds+(int(self.result[4])+(int(self.result[5])<<32))/1e9)
