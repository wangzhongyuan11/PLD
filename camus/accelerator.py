"""Real CAM2 driver. Imports PYNQ only for an explicit, validated PL session."""
from __future__ import annotations
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from .contract import shape
from .device import ABI_VERSION, DEVICE_ID, VLNV, EmulatedTileAccelerator, output_channel_chunks, choose_camus_tile, ConvOperationRecord, input_bank_usage, output_bank_usage, weight_bank_usage
from reference.tiling import tiles
from .native import load_native, TXPrefix
from .integer import q24_lut as lut

# Reuse the bounded DMA/cache/recovery implementation, not its ABI-1 constructor.
_board = str(Path(__file__).resolve().parents[1] / 'board')
if _board not in sys.path:
    sys.path.insert(0, _board)
from tensor_accel import TileAccelerator, expected_fclk_mhz
from simple_dma import SimpleDMA


class HardwareLease:
    """CAMUS-process lock plus a read-only check of the existing web owner."""
    def __init__(self):
        if sys.platform != 'linux':
            raise RuntimeError('PL backend requires the PYNQ Linux board')
        import fcntl
        import tempfile
        self.handle = None
        check = subprocess.run(['systemctl', 'is-active', 'pld-web.service'],
                               capture_output=True, text=True, timeout=5)
        if check.returncode not in (3, 4) or check.stdout.strip() not in ('inactive', 'failed', 'unknown'):
            raise RuntimeError('pld-web.service may own DMA; CAMUS refuses to acquire PL')
        self.handle = open(Path(tempfile.gettempdir()) / 'pld-camus-pl.lock', 'a+b')
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.handle.close(); self.handle = None
            raise RuntimeError('another CAMUS process owns PL')

    def close(self):
        if self.handle is not None:
            self.handle.close(); self.handle = None


class CamusMMIO(TileAccelerator):
    def __init__(self, bit, timeout=3.0):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('timeout must be finite positive')
        from pynq import Overlay, MMIO, allocate
        self.timeout, self.job, self.frame = timeout, 0, 0
        self.config, self.events, self.closed = None, [], False
        self.tx = self.rx = self._tx_view = self._rx_view = None
        self.native_control = self.native_copy = load_native()
        import pyxrt
        self.tx_sync_direction = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        self._config_cache = {}
        self.copy_info = {'kind': 'numpy'}; self.control_info = {'kind': 'python'}
        self.tx_sync_info = {'kind': 'pynq_full'}
        self.timing = dict(init_stages={}, close_stages={}, elapsed={})
        self.mmio_type = MMIO
        self.expected_clock_mhz = expected_fclk_mhz(Path(bit).with_suffix('.hwh'))
        self.overlay = Overlay(str(bit), download=False)
        info = self.overlay.ip_dict.get('axis_tensor_accel_camus_0', {})
        dma = self.overlay.ip_dict.get('axi_dma', {})
        validate_overlay_metadata(info, dma)
        self.overlay.download()
        self._attach()
        try:
            self.tx = allocate(shape=(131072,), dtype=np.uint8, cacheable=True)
            self.rx = allocate(shape=(65536+64,), dtype=np.uint8, cacheable=True)
            self._tx_view, self._rx_view = np.asarray(self.tx), np.asarray(self.rx)
        except BaseException:
            self.close()
            raise

    def _attach(self, record_init=True):
        from pynq import Clocks
        self.clock_mhz = float(Clocks.fclk0_mhz)
        if abs(self.clock_mhz - self.expected_clock_mhz) > .1:
            raise ValueError('CAM2 HWH/FCLK mismatch')
        core = self.overlay.ip_dict['axis_tensor_accel_camus_0']
        dma = self.overlay.ip_dict['axi_dma']
        self.core = self.mmio_type(core['phys_addr'], core['addr_range'])
        self.abi_version = int(self.core.read(4))
        if int(self.core.read(0)) != DEVICE_ID or self.abi_version != ABI_VERSION:
            raise ValueError('CAM2 ID/ABI mismatch; ABI-1 skin core is incompatible')
        self.dma = SimpleDMA(self.mmio_type(dma['phys_addr'], dma['addr_range']), dma['parameters'], self.timeout)
        self.native_control.bind(self.core,self.dma)
        self.recover(record_init=False)

    def recover(self,record_init=False):
        try:return super().recover(record_init=record_init)
        finally:self.config=None

    def configure(self, mode, h, w, ci, co=0, kernel=1, split=0, frame=None, *, relu=False):
        key=(mode,h,w,ci,co,kernel,split,relu)
        cache=getattr(self,'_config_cache',{})
        cfg=cache.get(key)
        if cfg is None:
            cfg=shape(mode,h,w,ci,co,kernel,split,relu=relu);cache[key]=cfg
        native=getattr(self,'native_control',None)
        if self.closed or native is None and self.core.read(0x0c) & 0x45:
            raise RuntimeError('CAM2 is closed, busy or faulted')
        old_frame=self.frame
        if frame is not None:
            if not isinstance(frame, int) or not 0 <= frame <= 0xffffffff:
                raise ValueError('frame outside UINT32')
            self.frame = frame
        if native is not None:
            if cfg!=self.config or old_frame!=self.frame:native.configure(cfg,self.frame)
        else:
            for address, field in ((0x18,'mode'), (0x1c,'h'), (0x20,'w'), (0x24,'ci'),
                                 (0x28,'co'), (0x2c,'kernel'), (0x30,'split')):
                self.core.write(address, cfg[field])
            self.core.write(0x5c, int(relu)); self.core.write(0x34, self.frame)
        self.config = cfg
        return cfg

    def load_parameters(self, weights, bias, multipliers):
        if np.asarray(bias).dtype != np.int32 or np.asarray(multipliers).dtype != np.int32:
            raise ValueError('CAM2 bias and multipliers must be INT32 before any MMIO')
        return super().load_parameters(weights, bias, multipliers)

    def _load_parts(self,command,parts,expected):
        # The shared driver imports its cache wrapper only when enabled. Use
        # the independent CAM2 wrapper here, leaving the ISIC runtime intact.
        direction=self.tx_sync_direction
        self.tx_sync_direction=None
        original=self.tx
        self.tx=TXPrefix(original,expected,direction)
        # Copy staging still needs allocation capacity, while DMA uses prefix.
        try:return super()._load_parts(command,parts,expected)
        finally:self.tx=original;self.tx_sync_direction=direction

    def load_gathered(self,length,started):
        if length!=self.config['input_bytes']:raise ValueError('gathered input length mismatch')
        def operation():
            result,dma_seconds=self._execute(2,TXPrefix(self.tx,length,self.tx_sync_direction),length)
            self.events.append(dict(result,command=2,mode=0,bytes=length,dma_seconds=dma_seconds,seconds=time.perf_counter()-started))
        self._guard(operation)

    def run_into(self,destination,tile):
        def operation():
            started=time.perf_counter();n=self.config['output_bytes'];self._rx_view[:]=0xa5
            result,dma_seconds=self._execute(4,self.rx,n)
            if not np.all(self._rx_view[n:]==0xa5):raise RuntimeError('output length/tail guard mismatch')
            self.native_control.scatter(destination,self._rx_view[:n],tile)
            self.events.append(dict(result,command=4,mode=0,bytes=n,dma_seconds=dma_seconds,seconds=time.perf_counter()-started))
        self._guard(operation)


def validate_overlay_metadata(info, dma):
    if info.get('type') != VLNV or info.get('phys_addr') != 0x43c00000 or info.get('addr_range') != 4096 or dma.get('phys_addr') != 0x40400000:
        raise ValueError('HWH is not the independent CAM2/DMA address map')
    params = {k.lower(): v for k, v in dma.get('parameters', {}).items()}
    for key in ('c_m_axi_mm2s_data_width', 'c_m_axi_s2mm_data_width', 'c_m_axis_mm2s_tdata_width'):
        if int(params.get(key, 0)) != 64:
            raise ValueError('CAM2 requires 64-bit DMA')


class CamusTileAccelerator(EmulatedTileAccelerator):
    """The same tiled graph as the emulator, with every tile executed on PL."""
    def __init__(self, runtime, model_root, timeout=3.0):
        super().__init__()
        from .package import verify_runtime
        runtime = verify_runtime(runtime, model_root)
        self.lease = HardwareLease()
        try:
            self.io = CamusMMIO(runtime / 'overlay/dma_top.bit', timeout)
            self._conv_plans={}
            self._lut_cache={}
            self._parameter_blobs={}
        except BaseException:
            self.lease.close()
            raise

    def begin_frame(self):
        super().begin_frame()
        self.io.events.clear()
        self.events = self.io.events

    def _prepare_conv(self, weight, bias, mult, h, w, relu):
        co, ci, k, _ = weight.shape
        cfg=self.io.configure(0, h, w, ci, co, k, frame=self.frame, relu=relu)
        frozen=not any(a.flags.writeable for a in (weight,bias,mult))
        key=(weight.ctypes.data,weight.shape,bias.ctypes.data,mult.ctypes.data)
        if frozen and key in self._parameter_blobs:self.io._load(1,self._parameter_blobs[key],cfg['parameter_bytes'])
        else:
            self.io.load_parameters(weight, bias, mult)
            if frozen:self._parameter_blobs[key]=weight.tobytes()+bias.astype('<i4').tobytes()+mult.astype('<i4').tobytes()

    def conv(self,data,weight,bias,multipliers,*,padding,relu):
        x=np.asarray(data);w=np.asarray(weight)
        if x.dtype!=np.int8 or x.ndim!=4 or x.shape[0]!=1 or not x.flags.c_contiguous:
            raise ValueError('CAM2 graph convolution requires contiguous NCHW INT8')
        co,ci,k,kw=w.shape
        if ci!=x.shape[1] or kw!=k or padding not in (0,1) or w.dtype!=np.int8:
            raise ValueError('invalid CAM2 convolution shape')
        oh,ow=x.shape[2]+2*padding-k+1,x.shape[3]+2*padding-k+1
        key=(w.ctypes.data,w.shape,bias.ctypes.data,multipliers.ctypes.data,x.shape,padding,relu)
        plan=self._conv_plans.get(key)
        if plan is None:
            plan=[];offset=0
            for chunk in output_channel_chunks(ci,co,k):
                th,tw=choose_camus_tile(ci,chunk,oh,ow,k)
                record=ConvOperationRecord(ci,chunk,k,(th,tw),math.ceil(oh/th)*math.ceil(ow/tw),
                    input_bank_usage(ci,th,tw,k),output_bank_usage(chunk,th,tw),weight_bank_usage(chunk,ci,k),oh*ow*chunk*ci*k*k)
                params=(w[offset:offset+chunk],bias[offset:offset+chunk],multipliers[offset:offset+chunk])
                plan.append((offset,chunk,th,tw,tuple(tiles(oh,ow,th,tw)),params,record));offset+=chunk
            self._conv_plans[key]=plan
        output=np.empty((1,co,oh,ow),np.int8)
        for offset,chunk,th,tw,schedule,params,record in plan:
            self._prepare_conv(*params,th+k-1,tw+k-1,relu)
            destination=output[:,offset:offset+chunk]
            for tile in schedule:
                self.io.configure(0,tile.height+k-1,tile.width+k-1,ci,chunk,k,frame=self.frame,relu=relu)
                started=time.perf_counter()
                length=self.io.native_control.gather(self.io._tx_view,x,tile,k,padding)
                self.io.load_gathered(length,started)
                self.io.run_into(destination,tile)
            self.conv_records.append(record)
        return output

    def _conv_tile(self, data, weight, bias, mult, relu):
        co, ci, k, _ = weight.shape
        self.io.configure(0, *data.shape[2:], ci, co, k, frame=self.frame, relu=relu)
        self.io.load_input(data)
        return self.io.run()

    def _tensor_tile(self, cfg, first, second, multipliers, load_lut):
        self.io.configure(cfg['mode'], cfg['h'], cfg['w'], cfg['ci'], split=cfg['split'], frame=self.frame)
        if load_lut:
            key=tuple(multipliers)
            if key not in self._lut_cache:self._lut_cache[key]=tuple(lut(m) for m in key)
            self.io.load_lut(*self._lut_cache[key])
        self.io.load_input(first, second)
        return self.io.run()

    def close(self):
        try:
            self.io.close()
        finally:
            self.lease.close()
