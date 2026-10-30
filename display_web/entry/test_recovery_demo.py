"""A real finite software watchdog around a simulated device, not board evidence."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from test_entry import entry, Board
from recovery_demo import RecoveryDemo, SimulatedDevice


class DemoTests(unittest.TestCase):
    def demo(self, root=None, **kwargs):
        demo = RecoveryDemo(root, timeout=.02, step_seconds=.002, **kwargs)
        self.addCleanup(demo.close)
        return demo

    def finish(self, demo):
        demo.thread.join(2)
        self.assertFalse(demo.thread.is_alive(), 'software demo exceeded bound')
        return demo.snapshot()

    def test_deadline_recovery_independent_output_check_and_persistent_evidence(self):
        with tempfile.TemporaryDirectory() as root:
            demo=self.demo(root); demo.start(); result=self.finish(demo)
            self.assertEqual(result['state'], 'done')
            self.assertEqual([e['code'] for e in result['events']],
                             ['SIM_START','SIM_TIMEOUT','SIM_RESET','SIM_RELOAD','SIM_VERIFIED','SIM_OK'])
            self.assertGreaterEqual(result['events'][1]['elapsed_seconds'], .02)
            self.assertTrue(all(e['simulated'] for e in result['events']))
            self.assertEqual(json.loads((Path(root)/'recovery_demo.json').read_text(encoding='utf-8')), result)

    def test_wrong_recovered_response_fails_without_another_retry(self):
        demo=self.demo(); demo.start('recovery_failure'); result=self.finish(demo)
        self.assertEqual(result['state'], 'failed')
        codes=[e['code'] for e in result['events']]
        self.assertNotIn('SIM_OK',codes)
        self.assertNotIn('SIM_VERIFIED',codes)
        self.assertEqual(codes.count('SIM_RESET'),1)
        self.assertIn('校验失败',result['error'])

    def test_duplicate_start_is_rejected_and_shutdown_cancels(self):
        demo=self.demo(); demo.start()
        with self.assertRaises(RuntimeError):demo.start()
        demo.close(); self.assertEqual(demo.snapshot()['state'],'cancelled')
        with self.assertRaises(RuntimeError):demo.start()

    def test_invalid_scenario_and_unwritable_log_do_not_start_a_worker(self):
        demo=self.demo()
        with self.assertRaises(ValueError):demo.start('real_dma')
        self.assertIsNone(demo.thread)
        with tempfile.TemporaryDirectory() as root:
            demo=self.demo(Path(root)/'missing_directory')
            with self.assertRaises(OSError):demo.start()
            self.assertIsNone(demo.thread)
            self.assertEqual(demo.snapshot()['state'],'failed')

    def test_simulated_memory_cannot_be_mapped_before_reset(self):
        device=SimulatedDevice()
        self.assertIsNone(device.poll())
        with self.assertRaises(RuntimeError):device.reload_and_map()
        device.reset_then_release(); self.assertIsNone(device.poll())
        device.reload_and_map(); self.assertIsInstance(device.poll(),bytes)

    def test_controller_never_calls_board_or_changes_display_and_boot_preferences(self):
        board=Board(); controller=entry.Controller(board)
        self.addCleanup(controller.recovery_demo.close)
        with self.assertRaises(RuntimeError):controller.start_recovery_demo({})
        controller.phase='ready';controller.model='skin'
        before=controller.snapshot()
        with patch.object(board,'request',side_effect=AssertionError('demo touched hardware')):
            result=controller.start_recovery_demo({})
            controller.recovery_demo.thread.join(3)
        self.assertTrue(result['recovery_demo']['simulated'])
        after=controller.snapshot()
        for key in ('settings','display','phase','model','boot'):
            self.assertEqual(before[key],after[key])


class DemoGateway(unittest.TestCase):
    def setUp(self):
        self.controller=entry.Controller(Board());self.controller.phase='ready';self.controller.model='skin'
        self.controller.recovery_demo.close()
        self.controller.recovery_demo=RecoveryDemo(timeout=.02,step_seconds=.002)
        self.server=entry.ThreadingHTTPServer(('127.0.0.1',0),entry.Handler);self.server.controller=self.controller
        self.thread=threading.Thread(target=self.server.serve_forever);self.thread.start()
        self.conn=http.client.HTTPConnection(*self.server.server_address,timeout=3)

    def tearDown(self):
        self.conn.close();self.controller.recovery_demo.close()
        self.server.shutdown();self.server.server_close();self.thread.join()

    def post(self,body,origin=None):
        self.conn.request('POST','/entry/recovery-demo',json.dumps(body),
            {'Origin':origin or 'http://127.0.0.1:'+str(self.server.server_port)})
        response=self.conn.getresponse();return response.status,json.loads(response.read())

    def test_same_origin_and_strict_body_only(self):
        self.assertEqual(self.post({},'http://unrelated.invalid')[0],403)
        self.assertEqual(self.post([])[0],400)
        self.assertEqual(self.post(dict(hardware=True))[0],400)
        code,body=self.post(dict(scenario='recover'))
        self.assertEqual(code,202)
        self.assertTrue(body['recovery_demo']['simulated'])
        self.controller.recovery_demo.thread.join(2)
        self.conn.request('GET','/entry/status');response=self.conn.getresponse()
        self.assertEqual(json.loads(response.read())['recovery_demo']['state'],'done')


if __name__ == '__main__':unittest.main()
