"""Read-only metrics routes must also work while model preparation is idle."""
import http.client
import json
import os
import threading
import unittest
from unittest.mock import patch
from test_entry import entry,Board


class MetricGatewayTests(unittest.TestCase):
    def test_view_and_missing_evidence_are_available_without_hardware_or_boot(self):
        controller=entry.Controller(Board())
        server=entry.ThreadingHTTPServer(('127.0.0.1',0),entry.Handler);server.controller=controller
        thread=threading.Thread(target=server.serve_forever);thread.start()
        connection=http.client.HTTPConnection(*server.server_address,timeout=3)
        try:
            with patch.dict(os.environ,{'PLD_DEMO_METRICS':''}),patch.object(controller.board,'request',side_effect=AssertionError('viewer touched hardware')):
                for path in ('/entry/metrics','/entry/metrics?model=skin'):
                    connection.request('GET',path);r=connection.getresponse()
                    self.assertEqual(r.status,200)
                    self.assertEqual(json.loads(r.read()),dict(status='no_evidence',records=[]))
                connection.request('GET','/entry/metrics?model=unknown');r=connection.getresponse()
                self.assertEqual(r.status,400);r.read()
                connection.request('GET','/entry/metrics/view');r=connection.getresponse()
                self.assertEqual(r.status,200);self.assertIn('CPU / 软件基线'.encode(),r.read())
        finally:
            connection.close();controller.recovery_demo.close();server.shutdown();server.server_close();thread.join()


if __name__=='__main__':unittest.main()
