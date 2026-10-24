"""Host-only lifecycle tests for HDMI admission and stale-overlay protection."""
import unittest
import threading
from types import SimpleNamespace
from hdmi_policy import guard_camus,guard_skin
from web_entry import Core

class Tests(unittest.TestCase):
    def test_camus_admission(self):
        class Server:
            fault=False
            def __init__(self):self.busy=threading.BoundedSemaphore(1);self.calls=0
            def start_hdmi(self,*args):self.calls+=1;return 'showing'
        guard_camus(SimpleNamespace(CamusServer=Server))
        server=Server();server.busy.acquire()
        self.assertEqual(server.start_hdmi('job','quad'),'showing')
        self.assertEqual(server.calls,1);server.busy.release()
        self.assertTrue(server.busy.acquire(False));server.busy.release()
        server.fault=True
        with self.assertRaises(ValueError):server.start_hdmi('job','quad')
        self.assertEqual(server.calls,1)

    def test_stale_overlay_close_never_reads_registers(self):
        core=Core.__new__(Core)
        core.overlay=SimpleNamespace(is_loaded=lambda:False)
        def unsafe_close():raise AssertionError('stale MMIO touched')
        core.session=SimpleNamespace(closed=False,close=unsafe_close)
        with self.assertRaisesRegex(RuntimeError,'configuration changed'):core.close()
        core.session.closed=True
        core.close()

if __name__=='__main__':unittest.main()
