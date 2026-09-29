"""An active DMA must stop before its physical buffers can be freed."""
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "board"))
from simple_dma import DMAFault
from verify_overlay import Loopback


class CleanupTests(unittest.TestCase):
    def run_fault(self, reset_fails):
        events = []

        class Buffer(np.ndarray):
            def freebuffer(self):
                events.append("free")

        def allocate(shape, dtype):
            return np.zeros(shape, dtype=dtype).view(Buffer)

        class DMA:
            def transfer(self, *args, **kwargs):
                events.append("active")
                raise DMAFault("transfer timeout", {})

            def reset(self):
                events.append("reset")
                if reset_fails:
                    raise DMAFault("reset timeout", {})

        loop = Loopback.__new__(Loopback)
        loop.mode = "fifo"
        loop.dma = DMA()
        loop.overlay = types.SimpleNamespace(download=lambda: events.append("download"))
        with patch.dict(sys.modules, {"pynq": types.SimpleNamespace(allocate=allocate)}):
            with self.assertRaises(DMAFault):
                loop.packet(np.arange(64, dtype=np.uint8))
        return events

    def test_reset_before_free(self):
        self.assertEqual(self.run_fault(False), ["active", "reset", "free", "free"])

    def test_reset_failure_reconfigures_before_free(self):
        self.assertEqual(self.run_fault(True), ["active", "reset", "download", "free", "free"])


if __name__ == "__main__":
    unittest.main()
