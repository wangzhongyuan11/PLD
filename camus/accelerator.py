"""Real CAM2 driver. Imports PYNQ only for an explicit, validated PL session."""
from __future__ import annotations
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

from .contract import shape
from .device import ABI_VERSION, DEVICE_ID, VLNV, EmulatedTileAccelerator
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
        self.native_copy = self.native_control = self.tx_sync_direction = None
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
        self.recover(record_init=False)

    def configure(self, mode, h, w, ci, co=0, kernel=1, split=0, frame=None, *, relu=False):
        cfg = shape(mode, h, w, ci, co, kernel, split, relu=relu)
        if self.closed or self.core.read(0x0c) & 0x45:
            raise RuntimeError('CAM2 is closed, busy or faulted')
        if frame is not None:
            if not isinstance(frame, int) or not 0 <= frame <= 0xffffffff:
                raise ValueError('frame outside UINT32')
            self.frame = frame
        for address, key in ((0x18,'mode'), (0x1c,'h'), (0x20,'w'), (0x24,'ci'),
                             (0x28,'co'), (0x2c,'kernel'), (0x30,'split')):
            self.core.write(address, cfg[key])
        self.core.write(0x5c, int(relu)); self.core.write(0x34, self.frame)
        self.config = cfg
        return cfg

    def load_parameters(self, weights, bias, multipliers):
        if np.asarray(bias).dtype != np.int32 or np.asarray(multipliers).dtype != np.int32:
            raise ValueError('CAM2 bias and multipliers must be INT32 before any MMIO')
        return super().load_parameters(weights, bias, multipliers)


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
        except BaseException:
            self.lease.close()
            raise

    def begin_frame(self):
        super().begin_frame()
        self.io.events.clear()
        self.events = self.io.events

    def _prepare_conv(self, weight, bias, mult, h, w, relu):
        co, ci, k, _ = weight.shape
        self.io.configure(0, h, w, ci, co, k, frame=self.frame, relu=relu)
        self.io.load_parameters(weight, bias, mult)

    def _conv_tile(self, data, weight, bias, mult, relu):
        co, ci, k, _ = weight.shape
        self.io.configure(0, *data.shape[2:], ci, co, k, frame=self.frame, relu=relu)
        self.io.load_input(data)
        return self.io.run()

    def _tensor_tile(self, cfg, first, second, multipliers, load_lut):
        self.io.configure(cfg['mode'], cfg['h'], cfg['w'], cfg['ci'], split=cfg['split'], frame=self.frame)
        if load_lut:
            self.io.load_lut(lut(multipliers[0]), lut(multipliers[1]) if len(multipliers) == 2 else None)
        self.io.load_input(first, second)
        return self.io.run()

    def close(self):
        try:
            self.io.close()
        finally:
            self.lease.close()
