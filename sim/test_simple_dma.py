"""Host regression of failure paths, independent of PYNQ/hardware."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from simple_dma import DMAFault, SimpleDMA

PARAMETERS = {"c_include_sg": "0", "c_addr_width": "32", "c_sg_length_width": "26",
              "c_m_axi_mm2s_data_width": "64", "c_m_axi_s2mm_data_width": "64"}


class Buffer:
    physical_address = 0x100000
    nbytes = 128

    def __init__(self, events):
        self.events = events

    def flush(self):
        self.events.append("flush")

    def invalidate(self):
        self.events.append("invalidate")


class MMIO:
    def __init__(self, status=0x1002, received=64):
        self.status, self.received = status, received
        self.events = []

    def read(self, address):
        if address in (4, 0x34):
            return self.status
        return self.received if address == 0x58 else 0

    def write(self, address, value):
        self.events.append((address, value))


class Tests(unittest.TestCase):
    def setup_dma(self, status=0x1002, received=64):
        mmio = MMIO(status, received)
        return mmio, SimpleDMA(mmio, PARAMETERS, timeout=0.001), Buffer(mmio.events), Buffer(mmio.events)

    def test_completion_rx_first_cache_order(self):
        mmio, dma, src, dst = self.setup_dma()
        dma.transfer(src, dst, 64)
        self.assertEqual(mmio.events[:2], ["flush", "flush"])
        self.assertLess(mmio.events.index((0x58, 64)), mmio.events.index((0x28, 64)))
        self.assertEqual(mmio.events[-1], "invalidate")
        self.assertFalse(dma.active)

    def test_stale_idle_does_not_complete(self):
        mmio, dma, src, dst = self.setup_dma(status=2)
        with self.assertRaisesRegex(DMAFault, "transfer timeout"):
            dma.transfer(src, dst, 64)
        self.assertNotIn("invalidate", mmio.events)
        with self.assertRaisesRegex(RuntimeError, "still active"):
            dma.transfer(src, dst, 64)

    def test_dma_error_dominates_done(self):
        mmio, dma, src, dst = self.setup_dma(status=0x1072)
        with self.assertRaises(DMAFault) as caught:
            dma.transfer(src, dst, 64)
        self.assertEqual(caught.exception.reason, "DMA error")
        self.assertEqual(caught.exception.registers["S2MM_SR"], "0x00001072")
        self.assertNotIn("invalidate", mmio.events)

    def test_short_packet_rejected(self):
        _, dma, src, dst = self.setup_dma(received=7)
        with self.assertRaisesRegex(DMAFault, "short packet"):
            dma.transfer(src, dst, 64)

    def test_invalid_buffer_never_programs_hardware(self):
        for length, address in ((0, 0x1000), (129, 0x1000), (8, 0x1001), (8, 1 << 32)):
            mmio, dma, src, dst = self.setup_dma()
            src.physical_address = address
            with self.assertRaises(ValueError):
                dma.transfer(src, dst, length)
            self.assertEqual(mmio.events, [])

    def test_reset_requires_both_halted(self):
        _, dma, _, _ = self.setup_dma(status=0)
        with self.assertRaisesRegex(DMAFault, "reset timeout"):
            dma.reset()

    def test_start_has_deadline(self):
        _, dma, _, _ = self.setup_dma(status=1)
        with self.assertRaisesRegex(DMAFault, "start timeout"):
            dma.start()

    def test_invalid_timeout(self):
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                SimpleDMA(MMIO(), PARAMETERS, timeout)

    def test_independent_lengths(self):
        mmio, dma, src, dst = self.setup_dma(received=24)
        dma.transfer(src,dst,send_nbytes=64,receive_nbytes=24)
        self.assertLess(mmio.events.index((0x58,24)),mmio.events.index((0x28,64)))
        self.assertEqual(mmio.events[-1],"invalidate")
        self.assertIsNone(dma._pending)

    def test_send_only_ignores_unused_halted_receiver(self):
        mmio, dma, src, _ = self.setup_dma()
        original_read = mmio.read
        mmio.read = lambda address: 1 if address==0x34 else original_read(address)
        dma.send(src,64)
        self.assertIn((0x28,64),mmio.events)
        self.assertNotIn("invalidate",mmio.events)
        self.assertFalse(any(isinstance(event,tuple) and event[0]>=0x30 for event in mmio.events))

    def test_receive_only_ignores_unused_halted_sender(self):
        mmio, dma, _, dst = self.setup_dma()
        original_read = mmio.read
        mmio.read = lambda address: 1 if address==4 else original_read(address)
        dma.receive(dst,64)
        self.assertIn((0x58,64),mmio.events)
        self.assertEqual(mmio.events[-1],"invalidate")
        self.assertFalse(any(isinstance(event,tuple) and event[0]<0x30 for event in mmio.events))

    def test_submit_retains_buffers_until_wait(self):
        mmio, dma, src, dst = self.setup_dma()
        dma.submit(src,dst,send_nbytes=64,receive_nbytes=64)
        self.assertTrue(dma.active)
        self.assertIs(dma._pending[0],src)
        self.assertIs(dma._pending[1],dst)
        self.assertNotIn("invalidate",mmio.events)
        with self.assertRaisesRegex(RuntimeError,"still active"):
            dma.send(src,8)
        dma.wait()
        self.assertFalse(dma.active)
        with self.assertRaisesRegex(RuntimeError,"no pending"):
            dma.wait()

    def test_failed_wait_retains_buffers_until_successful_reset(self):
        mmio, dma, src, dst = self.setup_dma(status=2)
        dma.submit(src,dst,send_nbytes=64,receive_nbytes=64)
        with self.assertRaises(DMAFault):
            dma.wait()
        self.assertIs(dma._pending[1],dst)
        with self.assertRaises(DMAFault):
            dma.reset()
        self.assertTrue(dma.active)
        self.assertIsNotNone(dma._pending)
        mmio.status=1
        dma.reset()
        self.assertFalse(dma.active)
        self.assertIsNone(dma._pending)

    def test_legacy_receive_only_remains_timeout_injection(self):
        mmio,dma,src,dst=self.setup_dma()
        original_read=mmio.read
        mmio.read=lambda address: 2 if address==4 else original_read(address)
        with self.assertRaisesRegex(DMAFault,"transfer timeout"):
            dma.transfer(src,dst,64,receive_only=True)
        self.assertIn((0x58,64),mmio.events)
        self.assertNotIn((0x28,64),mmio.events)

    def test_new_invalid_arguments_do_not_touch_hardware(self):
        mmio,dma,src,dst=self.setup_dma()
        for kwargs in ({}, {"source":src}, {"send_nbytes":8},
                       {"source":src,"send_nbytes":1.5},
                       {"source":src,"send_nbytes":8,"receive_nbytes":8}):
            with self.assertRaises(ValueError):
                dma.submit(**kwargs)
            self.assertEqual(mmio.events,[])
            self.assertFalse(dma.active)
        with self.assertRaises(ValueError):
            dma.transfer(src,dst,64,send_nbytes=64)


if __name__ == "__main__":
    unittest.main()
