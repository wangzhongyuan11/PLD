import unittest
from unittest.mock import patch,Mock
from types import SimpleNamespace

from acdc.contract import ContractError
from acdc.optimization.device import SpatialDevice,UnsafeDMAState,validate_overlay


class SpatialDeviceTest(unittest.TestCase):
    def test_interface_identity_does_not_accept_cam2_same_abi(self):
        core=dict(type='user:pld:spatial_accel:1.0',phys_addr=0x43c00000)
        gpio=dict(type='xilinx.com:ip:axi_gpio:2.0',phys_addr=0x41200000)
        validate_overlay(core,gpio)
        with self.assertRaises(ContractError):validate_overlay(dict(core,type='user:pld:axis_tensor_accel_camus:1.0'),gpio)

    def test_recovery_requires_both_masters_halted_before_reset(self):
        writes=[];reads=iter([1,3])
        device=object.__new__(SpatialDevice)
        device.timeout=1;device.unsafe=False
        device.gpio=SimpleNamespace(write=lambda a,v:writes.append((a,v)),read=lambda _:next(reads))
        device.core=SimpleNamespace(read=lambda _:0)
        with patch('acdc.optimization.device.time.sleep'):device.recover()
        self.assertEqual(writes,[(0,3),(0,0),(0,1)])
        self.assertFalse(device.unsafe)

    def test_halt_timeout_never_asserts_reset_or_frees_memory(self):
        writes=[];device=object.__new__(SpatialDevice);device.timeout=1;device.unsafe=False
        device.gpio=SimpleNamespace(write=lambda a,v:writes.append((a,v)),read=lambda _:1)
        with patch('acdc.optimization.device.time.monotonic',side_effect=[0,2]):
            with self.assertRaises(UnsafeDMAState):device.recover()
        self.assertTrue(device.unsafe);self.assertEqual(writes,[(0,3)])

    def test_unsafe_cli_owner_cannot_unwind_and_release_xrt_buffer(self):
        device=object.__new__(SpatialDevice)
        device.closed=False;device.unsafe=True;device.arena=Mock();device.lease=Mock()
        with patch('acdc.optimization.device.sys.platform','linux'), \
                patch('acdc.optimization.device.signal.SIGSTOP',19,create=True), \
                patch('acdc.optimization.device.os.getpid',return_value=123), \
                patch('acdc.optimization.device.os.kill') as suspend, \
                patch('acdc.optimization.device._QUARANTINE',[]):
            with self.assertRaises(UnsafeDMAState):device.close()
        suspend.assert_called_once_with(123,19)
        device.arena.freebuffer.assert_not_called();device.lease.close.assert_not_called()
