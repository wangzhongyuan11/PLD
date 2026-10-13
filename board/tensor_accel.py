"""ABI-1 PYNQ driver. All CNN arithmetic executes in the PL tile engines."""
import time
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from simple_dma import SimpleDMA


class AcceleratorFault(RuntimeError):
    def __init__(self,reason,state,recovered):
        self.state,self.recovered=state,recovered
        super().__init__(f"{reason}; recovered={recovered}; state={state}")


def _record_stage(timing, name, started, close=False):
    elapsed=time.perf_counter()-started
    if not math.isfinite(elapsed) or elapsed<0:
        raise RuntimeError("non-finite accelerator timing measurement")
    timing["close_stages" if close else "init_stages"][name]=elapsed


def expected_fclk_mhz(hwh):
    values=ET.parse(hwh).getroot().findall('.//PARAMETER[@NAME="PCW_ACT_FPGA0_PERIPHERAL_FREQMHZ"]')
    if len(values)!=1:
        raise ValueError("HWH must identify exactly one PS FCLK0 frequency")
    frequency=float(values[0].attrib["VALUE"])
    if not math.isfinite(frequency) or not any(abs(frequency-known)<0.01 for known in (100.,125.)):
        raise ValueError("unsupported HWH FCLK0 frequency")
    return frequency


def shape(mode,h,w,ci,co=0,kernel=1,split=0):
    values=(mode,h,w,ci,co,kernel,split)
    if any(not isinstance(v,(int,np.integer)) for v in values):
        raise ValueError("integer dimensions required")
    if mode not in range(7) or not 1<=ci<=96 or min(h,w)<1:
        raise ValueError("invalid mode/input shape")
    plane=h*w
    if mode in (0,5):
        if not 1<=co<=64 or kernel not in (1,3) or min(h,w)<kernel or max(h,w)>258:
            raise ValueError("invalid convolution shape")
        oh,ow,oc=h-kernel+1,w-kernel+1,co
    elif mode==6:
        if not 1<=co<=64 or kernel!=3 or min(h,w)<3 or max(h,w)>258:
            raise ValueError("invalid fused decoder convolution shape")
        if h%2 or w%2:
            raise ValueError("fused decoder patch dimensions must be even")
        if not 0<split<ci or split%4 or (ci-split)%4:
            raise ValueError("fused decoder split must align input groups to one branch")
        oh,ow,oc=h-kernel+1,w-kernel+1,co
    else:
        if max(h,w)>256:
            raise ValueError("invalid tensor input shape")
        if mode in (2,4) and (h%2 or w%2):
            raise ValueError("pool/concat dimensions must be even")
        if mode==4 and not 0<split<ci:
            raise ValueError("invalid concatenation split")
        oh,ow,oc=(h//2,w//2,ci) if mode==2 else (h*2,w*2,ci) if mode==3 else (h,w,ci)
    deep_plane=(h//2+1)*(w//2+1) if mode==6 else plane//4
    input_bank=((split+3)//4)*plane+((ci-split+3)//4)*deep_plane if mode in (4,6) else ((ci+3)//4)*plane
    output_bank=((oc+7)//8)*oh*ow
    if max(oh,ow)>256 or input_bank>32768 or output_bank>8192:
        raise ValueError("tile exceeds PL bank capacity")
    return {"mode":int(mode),"h":int(h),"w":int(w),"ci":int(ci),"co":int(co),
            "kernel":int(kernel),"split":int(split),"oh":oh,"ow":ow,"oc":oc,
            "input_bytes":split*plane+(ci-split)*deep_plane if mode in (4,6) else ci*plane,
            "parameter_bytes":co*ci*kernel*kernel+8*co if mode in (0,5,6) else 0,
            "lut_bytes":512 if mode in (4,6) else 256,"output_bytes":oc*oh*ow}


class TileAccelerator:
    def __init__(self,bit,timeout=2.0,native_copy=None,tx_range_sync=False,native_control=None):
        init_started=time.perf_counter()
        self.timing={"timing_clock":"time.perf_counter wall seconds",
                     "scope":"TileAccelerator cold initialization and device close; init/close stages do not overlap",
                     "init_stages":{},"close_stages":{},"elapsed":{}}
        if not math.isfinite(timeout) or timeout<=0:
            raise ValueError("timeout must be finite and positive")
        self.native_copy=None
        self.copy_info={"kind":"numpy"}
        self.native_control=None
        self.control_info={"kind":"python"}
        if native_control is not None:
            started=time.perf_counter()
            from native_control import NativeControl
            self.native_control=NativeControl(native_control)
            self.control_info=self.native_control.info
            _record_stage(self.timing,"native_control_init",started)
        self.tx_sync_direction=None
        self.tx_sync_info={"kind":"pynq_full"}
        if tx_range_sync:
            started=time.perf_counter()
            import pyxrt
            import hashlib
            self.tx_sync_direction=pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
            self.tx_sync_info={"kind":"xrt_tx_prefix_64", "source_sha256":hashlib.sha256(
                Path(__file__).with_name("cache_sync.py").read_bytes()).hexdigest()}
            _record_stage(self.timing,"xrt_import_and_tx_sync_init",started)
        if native_copy is not None:
            started=time.perf_counter()
            from native_copy import NativeCopy
            self.native_copy=NativeCopy(native_copy,verify_build=True)
            self.copy_info=self.native_copy.info
            _record_stage(self.timing,"native_copy_init",started)
        started=time.perf_counter()
        from pynq import Overlay,MMIO,allocate
        _record_stage(self.timing,"pynq_import",started)
        started=time.perf_counter()
        self.expected_clock_mhz=expected_fclk_mhz(Path(bit).with_suffix('.hwh'))
        _record_stage(self.timing,"hwh_clock_parse",started)
        self.timeout=timeout
        self.job=0; self.frame=0; self.config=None; self.events=[]
        self.tx=None; self.rx=None; self.closed=False
        self._tx_view=None; self._rx_view=None
        self.mmio_type=MMIO
        started=time.perf_counter()
        self.overlay=Overlay(str(bit),download=False)
        _record_stage(self.timing,"overlay_metadata_parse",started)
        info=self.overlay.ip_dict.get("axis_tensor_accel_0",{})
        dma_info=self.overlay.ip_dict.get("axi_dma",{})
        if info.get("phys_addr")!=0x43c00000 or info.get("addr_range")!=4096 or \
                info.get("type")!="user:pld:axis_tensor_accel:1.0" or dma_info.get("phys_addr")!=0x40400000:
            raise ValueError("overlay does not match the ABI-1 core/DMA address map")
        p={key.lower():value for key,value in dma_info["parameters"].items()}
        for key in ("c_m_axi_mm2s_data_width","c_m_axi_s2mm_data_width","c_m_axis_mm2s_tdata_width"):
            if int(p[key])!=64:
                raise ValueError("overlay requires 64-bit DMA datapaths")
        started=time.perf_counter()
        self.overlay.download()
        _record_stage(self.timing,"bitstream_download",started)
        self._attach()
        try:
            # SimpleDMA flushes before device access and invalidates RX after
            # completion. Cacheable mappings make the PS gather/copy practical.
            started=time.perf_counter()
            self.tx=allocate(shape=(131072,),dtype=np.uint8,cacheable=True)
            self.rx=allocate(shape=(65536+64,),dtype=np.uint8,cacheable=True)
            # Ordinary ndarray views avoid PynqBuffer metadata reconstruction
            # on every slice/ufunc. DMA still owns the original buffers.
            self._tx_view=np.asarray(self.tx)
            self._rx_view=np.asarray(self.rx)
            _record_stage(self.timing,"dma_buffer_allocation",started)
        except BaseException:
            self.close()
            raise
        self.timing["elapsed"]["tile_accelerator_init"]=time.perf_counter()-init_started

    def _attach(self,record_init=True):
        started=time.perf_counter() if record_init else None
        from pynq import Clocks
        self.clock_mhz=float(Clocks.fclk0_mhz)
        if abs(self.clock_mhz-self.expected_clock_mhz)>0.1:
            raise ValueError(f"expected HWH FCLK0 {self.expected_clock_mhz} MHz, got {self.clock_mhz}")
        core_info=self.overlay.ip_dict["axis_tensor_accel_0"]
        dma_info=self.overlay.ip_dict["axi_dma"]
        self.core=self.mmio_type(core_info["phys_addr"],core_info["addr_range"])
        self.abi_version=int(self.core.read(4))
        if self.core.read(0)!=0x504c4431 or self.abi_version not in (0x00010000,0x00010001,0x00010002):
            raise ValueError("core ID/ABI mismatch")
        self.dma=SimpleDMA(self.mmio_type(dma_info["phys_addr"],dma_info["addr_range"]),dma_info["parameters"],self.timeout)
        if started is not None:
            _record_stage(self.timing,"attach_setup",started)
        self.recover(record_init=record_init)
        if self.native_control is not None:
            started=time.perf_counter() if record_init else None
            self.native_control.bind(self.core,self.dma)
            if started is not None:
                _record_stage(self.timing,"native_control_bind",started)

    def snapshot(self):
        return {"core":{name:int(self.core.read(addr)) for name,addr in (
            ("status",0x0c),("error",0x10),("command",0x14),("mode",0x18),
            ("frame",0x3c),("job",0x40),("expected_in",0x48),("expected_out",0x4c),
            ("received",0x50),("sent",0x54))},"dma":self.dma.snapshot()}

    def recover(self,record_init=False):
        # Stop DDR access first. The CNN BD has no external stream FIFO; core
        # RESET flushes both byte adapters, engines, and all data-valid tags.
        started=time.perf_counter() if record_init else None
        self.dma.reset()
        self.core.write(8,2)
        deadline=time.monotonic()+self.timeout
        while self.core.read(0x0c)&0x7f:
            if time.monotonic()>=deadline:
                raise TimeoutError("core reset timeout")
        self.dma.start()
        if started is not None:
            _record_stage(self.timing,"initial_recovery",started)

    def _guard(self,operation):
        try:
            return operation()
        except BaseException as exc:
            state=self.snapshot()
            recovered=False
            try:
                self.recover()
                recovered=True
            except BaseException as reset_error:
                state["reset_error"]=str(reset_error)
                try:
                    self.overlay.download()
                    self._attach(record_init=False)
                    recovered=True
                except BaseException as reload_error:
                    state["reload_error"]=str(reload_error)
            raise AcceleratorFault(str(exc),state,recovered) from exc

    def configure(self,mode,h,w,ci,co=0,kernel=1,split=0,frame=None):
        if self.closed:
            raise RuntimeError("accelerator is closed")
        cfg=shape(mode,h,w,ci,co,kernel,split)
        if mode==5 and self.abi_version<0x00010001:
            raise ValueError("mapped convolution requires ABI 1.1")
        if mode==6 and self.abi_version<0x00010002:
            raise ValueError("fused decoder convolution requires ABI 1.2")
        if self.native_control is None and self.core.read(0x0c)&0x45:
            raise RuntimeError("core busy/faulted; finish or recover before configuration")
        if frame is not None:
            if not 0<=int(frame)<=0xffffffff:
                raise ValueError("frame ID outside UINT32")
            self.frame=int(frame)
        if self.native_control is not None:
            self.native_control.configure(cfg,self.frame)
        else:
            for addr,key in ((0x18,"mode"),(0x1c,"h"),(0x20,"w"),(0x24,"ci"),(0x28,"co"),(0x2c,"kernel"),(0x30,"split")):
                self.core.write(addr,cfg[key])
            self.core.write(0x34,self.frame)
        self.config=cfg
        return cfg

    def _begin(self,command):
        if self.config is None or self.core.read(0x0c)&0x45:
            raise RuntimeError("configure an idle, healthy core before starting")
        self.job=(self.job+1)&0xffffffff or 1
        self.core.write(0x14,command)
        self.core.write(0x38,self.job)
        self.core.write(8,1)

    def _wait(self,received,sent):
        native=getattr(self,"native_control",None)
        if native is not None:
            return native.wait_core(self,received,sent)
        deadline=time.monotonic()+self.timeout
        while True:
            status=self.core.read(0x0c)
            if status&4:
                raise RuntimeError(f"core error {self.core.read(0x10)}")
            if status&2:
                if status&1 or self.core.read(0x40)!=self.job or self.core.read(0x3c)!=self.frame:
                    raise RuntimeError("stale or inconsistent completion")
                if self.core.read(0x50)!=received or self.core.read(0x54)!=sent:
                    raise RuntimeError("core byte counters differ from this job")
                return {"core_cycles":int(self.core.read(0x44)),"compute_cycles":int(self.core.read(0x58))}
            if time.monotonic()>=deadline:
                raise TimeoutError("core completion timeout")

    def _load(self,command,payload,expected):
        data=np.frombuffer(payload,dtype=np.uint8)
        return self._load_parts(command,(data,),expected)

    def _execute(self,command,buffer,length):
        native=getattr(self,"native_control",None)
        if native is not None:
            return native.execute(self,command,buffer,length)
        self._begin(command)
        start=time.perf_counter()
        if command==4: self.dma.receive(buffer,length)
        else: self.dma.send(buffer,length)
        dma_seconds=time.perf_counter()-start
        result=self._wait(0,length) if command==4 else self._wait(length,0)
        if self.core.read(0x4c if command==4 else 0x48)!=length:
            raise RuntimeError("hardware packet length disagrees with driver")
        return result,dma_seconds

    def _load_parts(self,command,parts,expected):
        length=sum(part.nbytes for part in parts)
        if length!=expected or length>self.tx.nbytes:
            raise ValueError("payload length does not match configured command")
        begin=time.perf_counter()
        offset=0
        # Finish every preflight/copy before any MMIO or DMA operation. Native
        # gather may write TX while detecting an invalid byte; it must never
        # cause a partially validated packet to be sent or a hardware reset.
        native=getattr(self,"native_copy",None)
        for part in parts:
            target=self._tx_view[offset:offset+part.nbytes].view(part.dtype).reshape(part.shape)
            if native is not None and part.dtype==np.int8:
                native.copy(target,part)
            else:
                np.copyto(target,part,casting='no')
            offset+=part.nbytes
        source=self.tx
        direction=getattr(self,"tx_sync_direction",None)
        if direction is not None:
            from cache_sync import XrtTXPrefix
            source=XrtTXPrefix(self.tx,length,direction)
        def operation():
            result,dma_seconds=self._execute(command,source,length)
            self.events.append(dict(result,command=command,mode=self.config["mode"],bytes=length,
                                    dma_seconds=dma_seconds,seconds=time.perf_counter()-begin))
        self._guard(operation)

    def load_parameters(self,weights,bias,multipliers):
        c=self.config
        weights=np.asarray(weights)
        bias=np.asarray(bias,dtype=np.int64); multipliers=np.asarray(multipliers,dtype=np.int64)
        if c is None or c["mode"] not in (0,5,6) or weights.shape!=(c["co"],c["ci"],c["kernel"],c["kernel"]):
            raise ValueError("parameter shape mismatch")
        if weights.dtype!=np.int8 or np.any(weights==-128) or bias.shape!=(c["co"],) or multipliers.shape!=(c["co"],):
            raise ValueError("parameter dtype/range mismatch")
        if np.any(bias<-(1<<31)) or np.any(bias>(1<<31)-1) or np.any(multipliers<0) or np.any(multipliers>(1<<31)-1):
            raise ValueError("INT32 parameter out of range")
        blob=weights.tobytes()+bias.astype("<i4").tobytes()+multipliers.astype("<i4").tobytes()
        self._load(1,blob,c["parameter_bytes"])

    def load_input(self,data,second=None):
        c=self.config
        if c is None:
            raise ValueError("configuration required")
        data=np.asarray(data)
        channels=c["split"] if c["mode"] in (4,6) else c["ci"]
        native=getattr(self,"native_copy",None)
        if data.dtype!=np.int8 or data.shape!=(1,channels,c["h"],c["w"]) or (native is None and np.any(data==-128)):
            raise ValueError("input must match configured NCHW INT8 shape/range")
        parts=[data]
        if c["mode"] in (4,6):
            second=np.asarray(second)
            deep_h=c["h"]//2+1 if c["mode"]==6 else c["h"]//2
            deep_w=c["w"]//2+1 if c["mode"]==6 else c["w"]//2
            if second.dtype!=np.int8 or second.shape!=(1,c["ci"]-channels,deep_h,deep_w) or (native is None and np.any(second==-128)):
                raise ValueError("deep branch shape/range mismatch")
            parts.append(second)
        elif second is not None:
            raise ValueError("second branch only supported by CONCAT_UP/fused decoder convolution")
        self._load_parts(2,parts,c["input_bytes"])

    def load_lut(self,first,second=None):
        c=self.config
        if c is None or c["mode"]==0:
            raise ValueError("LUT requires a tensor operation")
        tables=[np.asarray(first)]
        if c["mode"] in (4,6): tables.append(np.asarray(second))
        elif second is not None: raise ValueError("second LUT only supported by CONCAT_UP/fused decoder convolution")
        if any(t.dtype!=np.int8 or t.shape!=(256,) or np.any(t==-128) for t in tables):
            raise ValueError("each LUT must contain 256 signed INT8 values in [-127,127]")
        self._load(3,b"".join(t.tobytes() for t in tables),c["lut_bytes"])

    def run(self):
        c=self.config
        if c is None:
            raise ValueError("configuration required")
        def operation():
            begin=time.perf_counter(); n=c["output_bytes"]
            self._rx_view[:]=0xa5
            result,dma_seconds=self._execute(4,self.rx,n)
            if not np.all(self._rx_view[n:]==0xa5):
                raise RuntimeError("output length/tail guard mismatch")
            # Return owned memory, never a view that a subsequent DMA overwrites.
            output=self._rx_view[:n].copy().view(np.int8).reshape(1,c["oc"],c["oh"],c["ow"])
            self.events.append(dict(result,command=4,mode=c["mode"],bytes=n,
                                    dma_seconds=dma_seconds,seconds=time.perf_counter()-begin))
            return output
        return self._guard(operation)

    def close(self):
        if self.closed:
            return
        close_started=time.perf_counter()
        try:
            started=time.perf_counter()
            try:
                self.dma.reset()
                self.core.write(8,2)
            except BaseException:
                # Reconfiguration must succeed before active DDR can be released.
                _record_stage(self.timing,"close_dma_and_core_reset",started,close=True)
                started=time.perf_counter()
                try:
                    self.overlay.download()
                finally:
                    _record_stage(self.timing,"close_error_overlay_download",started,close=True)
            else:
                _record_stage(self.timing,"close_dma_and_core_reset",started,close=True)
            started=time.perf_counter()
            try:
                for buffer in (self.rx,self.tx):
                    if buffer is not None:
                        buffer.freebuffer()
            finally:
                _record_stage(self.timing,"close_dma_buffer_free",started,close=True)
            self.rx=None; self.tx=None; self._rx_view=None; self._tx_view=None; self.closed=True
        finally:
            self.timing["elapsed"]["device_close_total"]=time.perf_counter()-close_started

    def __enter__(self): return self
    def __exit__(self,*args): self.close()
