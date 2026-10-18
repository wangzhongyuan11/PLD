"""Independent ACD3 MMIO identity; mature bounded CAM2 tile operations are read-only donors."""
from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np

DEVICE_ID=0x41434433
ABI_VERSION=0x00030000
VLNV='user:pld:axis_tensor_accel_acdc:1.0'


def validate_overlay(info,dma):
    if (info.get('type')!=VLNV or info.get('phys_addr')!=0x43c00000 or info.get('addr_range')!=4096
            or dma.get('phys_addr')!=0x40400000):
        raise ValueError('overlay is not the independent ACD3 accelerator')
    params={k.lower():v for k,v in dma.get('parameters',{}).items()}
    for key in ('c_m_axi_mm2s_data_width','c_m_axi_s2mm_data_width','c_m_axis_mm2s_tdata_width'):
        if int(params.get(key,0))!=64:raise ValueError('64-bit simple AXI DMA required')
    if int(params.get('c_include_sg',0))!=0:raise ValueError('SimpleDMA cannot operate scatter-gather')


class Lease:
    def __init__(self):
        if sys.platform!='linux':raise RuntimeError('PL requires PYNQ Linux, no CPU fallback')
        import fcntl
        self.handles=[]
        try:
            for unit in ('pld-hub.service','pld-web.service','pld-camus.service'):
                r=subprocess.run(['systemctl','is-active',unit],capture_output=True,text=True,timeout=5)
                if r.returncode not in (3,4) or r.stdout.strip() not in ('inactive','failed','unknown'):
                    raise RuntimeError(unit+' may own FPGA; stop it through the release manager first')
            # Include the CAMUS legacy lease and a shared future lease, without changing its service.
            for name in ('pld-fpga.lock','pld-camus-pl.lock'):
                handle=open(Path(tempfile.gettempdir())/name,'a+b')
                self.handles.append(handle);fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:self.close();raise

    def close(self):
        for h in self.handles:h.close()
        self.handles=[]


def _classes():
    # Lazy import: PC-only operation never imports PYNQ or builds an ARM native library.
    from camus.accelerator import CamusMMIO,CamusTileAccelerator
    from camus.device import EmulatedTileAccelerator
    from camus.native import load_native
    from board.tensor_accel import expected_fclk_mhz
    from board.simple_dma import SimpleDMA

    class MMIO(CamusMMIO):
        def __init__(self,bit,expected_id=DEVICE_ID,timeout=5):
            if not math.isfinite(timeout) or timeout<=0:raise ValueError('bounded positive timeout required')
            from pynq import Overlay,MMIO,allocate
            import pyxrt
            self.timeout,self.job,self.frame=timeout,0,0
            if expected_id not in (0x41434433,0x41434434):raise ValueError('unknown ACDC hardware capability')
            self.expected_device_id=expected_id
            self.config,self.events,self.closed=None,[],False
            self.tx=self.rx=self._tx_view=self._rx_view=None
            self.native_control=self.native_copy=load_native()
            self.tx_sync_direction=pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
            self._config_cache={};self.copy_info={'kind':'numpy'};self.control_info={'kind':'native'}
            self.tx_sync_info={'kind':'prefix'}
            self.timing=dict(init_stages={},close_stages={},elapsed={})
            self.mmio_type=MMIO;self.expected_clock_mhz=expected_fclk_mhz(Path(bit).with_suffix('.hwh'))
            self.overlay=Overlay(str(bit),download=False)
            validate_overlay(self.overlay.ip_dict.get('axis_tensor_accel_acdc_0',{}),self.overlay.ip_dict.get('axi_dma',{}))
            self.overlay.download()
            try:
                self._attach()
                self.tx=allocate(shape=(131072,),dtype=np.uint8,cacheable=True)
                self.rx=allocate(shape=(65536+64,),dtype=np.uint8,cacheable=True)
                self._tx_view,self._rx_view=np.asarray(self.tx),np.asarray(self.rx)
            except BaseException:
                if hasattr(self,'dma'):self.close()
                raise

        def _attach(self,record_init=True):
            from pynq import Clocks
            self.clock_mhz=float(Clocks.fclk0_mhz)
            if abs(self.clock_mhz-self.expected_clock_mhz)>.1:raise ValueError('ACD3 HWH/FCLK mismatch')
            core=self.overlay.ip_dict['axis_tensor_accel_acdc_0'];dma=self.overlay.ip_dict['axi_dma']
            self.core=self.mmio_type(core['phys_addr'],core['addr_range'])
            self.abi_version=int(self.core.read(4))
            if self.core.read(0)!=self.expected_device_id or self.abi_version!=ABI_VERSION:raise ValueError('ACDC ID/ABI mismatch')
            self.dma=SimpleDMA(self.mmio_type(dma['phys_addr'],dma['addr_range']),dma['parameters'],self.timeout)
            self.native_control.bind(self.core,self.dma)
            self.recover(record_init=False)

    class Device(CamusTileAccelerator):
        def __init__(self,runtime,model):
            from .package import verify_runtime
            runtime=verify_runtime(runtime,model)
            EmulatedTileAccelerator.__init__(self)
            self.lease=Lease()
            try:
                import json
                identity=json.loads((runtime/'runtime_manifest.json').read_text('utf-8'))
                from dataclasses import replace
                self.contract=replace(self.contract,device_id=identity['hardware_id'],abi_version=ABI_VERSION,vlnv=VLNV)
                self.io=MMIO(runtime/'overlay/dma_top.bit',expected_id=identity['hardware_id'])
                self._conv_plans={};self._lut_cache={};self._parameter_blobs={}
            except BaseException:self.lease.close();raise
    return Device


class TileAccelerator:
    def __new__(cls,runtime,model):
        if runtime is None:raise ValueError('source-bound --runtime is required for PL')
        return _classes()(runtime,model)
