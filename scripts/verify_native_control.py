"""Fault-test the real command/deadline code against an independent MMIO model.

Requires a separate library compiled with -DPLD_CONTROL_TEST; the production
library has no test callbacks. No hardware is accessed by this verifier.
"""
import argparse
import ctypes as ct
from pathlib import Path
import time


CORE, DMA = 0x100000, 0x200000
READ = ct.CFUNCTYPE(ct.c_uint32, ct.c_size_t, ct.c_uint32)
WRITE = ct.CFUNCTYPE(None, ct.c_size_t, ct.c_uint32, ct.c_uint32)
CLOCK = ct.CFUNCTYPE(ct.c_uint64)


class DeviceModel:
    def __init__(self, fault=None):
        self.fault=fault; self.writes=[]; self.ticks=1; self.clock_calls=0
        self.core={0x0c:0,0x34:23,0x10:0}
        self.dma={4:0x1002,0x34:0x1002}  # Both contain a stale completion.
        self.started=False; self.polls=0; self.length=0; self.base=0
        self.read_cb=READ(self.read); self.write_cb=WRITE(self.write); self.clock_cb=CLOCK(self.clock)
        if fault=='busy': self.core[0x0c]=1
        if fault=='faulted': self.core[0x0c]=0x44
        if fault=='halted_before': self.dma[4]=1
        if fault=='error_before': self.dma[4]=0x1072

    def clock(self):
        self.clock_calls+=1
        if self.fault=='clock_before' or (self.fault=='clock_after' and self.clock_calls>2): return 0
        self.ticks+=10000
        return self.ticks

    def write(self, address, offset, value):
        self.writes.append((address,offset,value))
        if address==CORE:
            self.core[offset]=value
            if offset==8 and value==1:
                self.core[0x0c]=1
        elif offset in (4,0x34):
            self.dma[offset]&=~value  # W1C, never an ordinary RAM assignment.
        else:
            self.dma[offset]=value
            if offset in (0x28,0x58):
                self.started=True; self.base=offset-0x28; self.length=value

    def finish_core(self):
        command=self.core[0x14]
        self.core.update({0x0c:2,0x3c:23,0x40:self.core[0x38],0x44:456,0x58:123,
                          0x48:self.length if command!=4 else 0,0x4c:self.length if command==4 else 0,
                          0x50:self.length if command!=4 else 0,0x54:self.length if command==4 else 0})
        corrupt={'stale_job':0x40,'stale_frame':0x3c,'received':0x50,'sent':0x54,
                 'expected_input':0x48,'expected_output':0x4c}
        if self.fault in corrupt: self.core[corrupt[self.fault]]+=1
        if self.fault=='busy_done': self.core[0x0c]=3
        if self.fault=='core_timeout': self.core[0x0c]=1

    def read(self, address, offset):
        if address==CORE:
            return self.core.get(offset,0)
        if self.started and offset==self.base+4:
            self.polls+=1
            if self.fault=='dma_error': return 0x1072
            if self.fault=='dma_halt': return 1
            if self.fault=='core_error':
                self.core[0x0c]=6; self.core[0x10]=2
                return 0
            if self.fault in ('dma_timeout','clock_after'): return 0
            if self.polls>=3:
                self.finish_core()
                if self.base and self.fault=='short_rx': self.dma[0x58]=self.length-1
                return 0x1002
        return self.dma.get(offset,0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library',type=Path)
    parser.add_argument('--production-library',type=Path,help='also verify real monotonic deadlines on ordinary RAM')
    args=parser.parse_args()
    library=ct.CDLL(str(args.library.resolve()))
    library.tensor_control_abi.restype=ct.c_uint32
    assert library.tensor_control_abi()==1
    library.control_test_hooks.argtypes=[READ,WRITE,CLOCK]
    packet=library.tensor_packet
    packet.argtypes=[ct.c_void_p,ct.c_uint32,ct.c_void_p,ct.c_uint32,
                     ct.c_uint32,ct.c_uint32,ct.c_uint32,ct.c_uint64,ct.c_uint32,
                     ct.c_uint64,ct.POINTER(ct.c_uint32)]
    packet.restype=ct.c_int
    configure=library.tensor_configure
    configure.argtypes=[ct.c_void_p,ct.c_uint32,ct.POINTER(ct.c_uint32)]
    configure.restype=ct.c_int
    wait=library.tensor_wait_core
    wait.argtypes=[ct.c_void_p,ct.c_uint32,ct.c_uint32,ct.c_uint32,ct.c_uint32,
                   ct.c_uint32,ct.c_uint64,ct.POINTER(ct.c_uint32)]
    wait.restype=ct.c_int
    count=0

    def call(fault=None,command=2,**changes):
        nonlocal count
        model=DeviceModel(fault)
        library.control_test_hooks(model.read_cb,model.write_cb,model.clock_cb)
        result=(ct.c_uint32*8)()
        values=dict(core=CORE,core_bytes=4096,dma=DMA,dma_bytes=4096,command=command,
                    frame=23,job=41,address=0x10000000,length=64,timeout=1000000)
        values.update(changes)
        status=packet(*values.values(),result)
        count+=1
        return status,model,list(result)

    for command in (1,2,3,4):
        status,model,result=call(command=command)
        assert status==0 and result[:2]==[456,123] and model.polls==3,(command,status,result)
        base=0x30 if command==4 else 0
        assert model.writes==[(CORE,0x14,command),(CORE,0x38,41),(CORE,8,1),
                              (DMA,base+4,0x7000),(DMA,base+0x18,0x10000000),
                              (DMA,base+0x1c,0),(DMA,base+0x28,64)]
    expected={'busy':2,'faulted':2,'halted_before':3,'error_before':4,
              'dma_error':4,'dma_halt':5,'dma_timeout':6,'short_rx':7,'core_error':8,
              'core_timeout':9,'stale_job':10,'stale_frame':10,'busy_done':10,
              'received':10,'sent':10,'expected_input':10,'expected_output':10,
              'clock_before':11,'clock_after':11}
    for fault,code in expected.items():
        status,model,result=call(fault,4 if fault in ('short_rx','expected_output') else 2)
        assert status==code,(fault,status,code)
        assert model.clock_calls<110,(fault,'deadline unbounded')
        if fault in ('busy','faulted','halted_before','error_before','clock_before'):
            assert model.writes==[],fault
    for changes in ({'core':0},{'dma':DMA+1},{'core_bytes':88},{'dma_bytes':88},
                    {'command':0},{'command':5},{'job':0},{'length':0},{'length':0x4000000},
                    {'address':0x10000001},{'address':1<<32},{'address':0xfffffff8}, {'timeout':0}):
        status,model,_=call(**changes)
        assert status==1 and not model.writes,changes
    for fault in (None,'busy','faulted'):
        model=DeviceModel(fault)
        library.control_test_hooks(model.read_cb,model.write_cb,model.clock_cb)
        values=(ct.c_uint32*8)(0,34,258,3,8,3,0,23)
        status=configure(CORE,4096,values)
        if fault: assert status==2 and not model.writes
        else: assert status==0 and model.writes==[(CORE,0x18+4*i,int(v)) for i,v in enumerate(values)]
        count+=1
    model=DeviceModel('busy')
    library.control_test_hooks(model.read_cb,model.write_cb,model.clock_cb)
    result=(ct.c_uint32*8)()
    assert wait(CORE,4096,2,23,41,64,1000000,result)==9
    assert not model.writes and model.clock_calls<110
    count+=1
    print(f'NATIVE_CONTROL_MODEL_PASS cases={count} stale_IOC lengths faults deadlines write_order')
    if args.production_library is not None:
        production=ct.CDLL(str(args.production_library.resolve()))
        production.tensor_wait_core.argtypes=wait.argtypes
        production.tensor_wait_core.restype=ct.c_int
        production.tensor_packet.argtypes=packet.argtypes
        production.tensor_packet.restype=ct.c_int
        # These are private ordinary arrays, never mapped device memory.
        core=(ct.c_uint32*1024)(); dma=(ct.c_uint32*1024)(); result=(ct.c_uint32*8)()
        core[3]=1
        start=time.perf_counter()
        status=production.tensor_wait_core(core,4096,2,23,41,64,20000000,result)
        elapsed=time.perf_counter()-start
        assert status==9 and .019<=elapsed<2,(status,elapsed)
        core[3]=0
        start=time.perf_counter()
        status=production.tensor_packet(core,4096,dma,4096,2,23,41,0x10000000,64,20000000,result)
        elapsed=time.perf_counter()-start
        assert status==6 and .019<=elapsed<2,(status,elapsed)
        print('NATIVE_CONTROL_CLOCK_PASS real_20ms_core_and_DMA_deadlines')


if __name__=='__main__': main()
