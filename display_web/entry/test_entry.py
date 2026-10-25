"""Lifecycle regressions without FPGA access."""
import importlib.util
import http.client
from http.server import ThreadingHTTPServer
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, patch

spec = importlib.util.spec_from_file_location('entry', Path(__file__).with_name('entry.py'))
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


class Board:
    cookie = 'session=private; HttpOnly'

    def __init__(self):
        self.states = {unit: False for unit in (entry.ENGINE, entry.WEB, entry.HUB,
                      'pld-web.service', 'pld-camus.service', 'pld-acdc.service')}
        self.states[entry.ENGINE] = self.states[entry.WEB] = True
        self.events = []
        self.model = 'skin'
        self.busy = False
        self.fail_stop = False

    def running(self, unit):
        return self.states[unit]

    def start(self, unit):
        if unit == entry.WEB:
            assert not self.states[entry.HUB]
            self.states[entry.ENGINE] = True
        if unit == entry.HUB:
            assert not self.states[entry.ENGINE]
        self.states[unit] = True
        self.events.append(('start', unit))

    def stop(self, unit):
        self.events.append(('stop', unit))
        if self.fail_stop:
            raise RuntimeError('release failed')
        self.states[unit] = False

    def health(self):
        return dict(ready=True, backend='pl', active_job='upload' if self.busy else None)

    def hub(self):
        return dict(phase='ready', model=self.model, generation=7, backend={})

    def prepare_dual(self, target):
        self.model = target

    def request(self, port, path, body):
        assert path == '/hub/switch'
        self.model = body['model']


class Lifecycle(unittest.TestCase):
    def test_prestart_only_uses_inactive_nonconflicting_units(self):
        board=entry.Board.__new__(entry.Board)
        with patch.object(board,'props',return_value=dict(ActiveState='inactive',Conflicts='')),patch.object(board,'start') as start:
            board.prestart_dual('skin');start.assert_called_once_with('pld-web.service')
        for state in [dict(ActiveState='active',Conflicts=''),dict(ActiveState='inactive',Conflicts=entry.ENGINE)]:
            with patch.object(board,'props',return_value=state),patch.object(board,'start') as start:
                board.prestart_dual('skin');start.assert_not_called()

    def test_default_adopts_heart_without_reloading_engine(self):
        board = Board()
        controller = entry.Controller(board)
        controller._switch('acdc')
        self.assertEqual(controller.snapshot()['phase'], 'ready')
        self.assertFalse(any(event[0] == 'stop' for event in board.events))

    def test_roundtrip_three_models_has_one_hardware_owner(self):
        board = Board()
        controller = entry.Controller(board)
        for target in ('skin', 'camus', 'acdc'):
            controller._switch(target)
            self.assertEqual(controller.model, target)
            self.assertEqual(controller.phase, 'ready')
            self.assertFalse(board.running(entry.HUB) and board.running(entry.ENGINE))
        self.assertEqual(controller.url, '8092/')

    def test_failed_release_prevents_loading_target(self):
        board = Board()
        board.fail_stop = True
        controller = entry.Controller(board)
        controller._switch('camus')
        self.assertEqual(controller.phase, 'error')
        self.assertFalse(board.running(entry.HUB))

    def test_busy_timeout_leaves_current_service_running(self):
        board = Board()
        board.busy = True
        controller = entry.Controller(board)
        with patch.object(controller, 'wait', side_effect=RuntimeError('task still active')):
            controller._switch('skin')
        self.assertTrue(board.running(entry.ENGINE))
        self.assertEqual(board.events, [])

    def test_transient_hub_startup_connection_is_retried(self):
        board = Board()
        controller = entry.Controller(board)
        original = board.hub
        attempts = []
        def hub():
            attempts.append(1)
            if len(attempts) == 1:
                raise ConnectionRefusedError()
            return original()
        board.hub = hub
        with patch.object(entry.time, 'sleep'):
            controller._switch('camus')
        self.assertEqual(controller.phase, 'ready')
        self.assertGreaterEqual(len(attempts), 2)

    def test_interactive_task_finishes_before_heart_switch(self):
        board = Board()
        board.states[entry.ENGINE] = board.states[entry.WEB] = False
        board.states[entry.HUB] = True
        original = board.hub
        board.hub = lambda: dict(original(), interactive_busy=True)
        controller = entry.Controller(board)
        def wait(check, seconds=300):
            if board.events:
                self.assertTrue(check())
                return
            self.assertFalse(check())
            self.assertEqual(board.events, [])
            board.hub = original
            self.assertTrue(check())
        with patch.object(controller, 'wait', side_effect=wait):
            controller._switch('acdc')
        self.assertEqual(controller.phase, 'ready')

    def test_resident_roundtrip_keeps_software_and_serializes_hardware(self):
        class ResidentBoard(Board):
            def __init__(self):super().__init__();self.heart_paused=False;self.hub_paused=False
            def can_park_heart(self):return True
            def health(self):return dict(ready=not self.heart_paused,backend='pl',active_job=None,
                suspended=self.heart_paused,hardware_released=self.heart_paused,hdmi='off',
                lifecycle='pause-resume-v1',resident_handoff='resident-hub-v1')
            def hub(self):return dict(phase='paused' if self.hub_paused else 'ready',model=self.model,
                generation=7,backend={},lifecycle='resident-hub-v1',hardware_released=self.hub_paused)
            def start(self,unit):
                if unit==entry.HUB:assert self.heart_paused
                if unit==entry.WEB:assert self.hub_paused
                self.states[unit]=True;self.events.append(('start',unit))
            def request(self,port,path,body):
                if port==8092:
                    if body['action']=='resume':assert self.hub_paused
                    self.heart_paused=body['action']=='pause'
                elif path=='/hub/lifecycle':
                    if body['action']=='resume':assert self.heart_paused;self.model=body['model']
                    self.hub_paused=body['action']=='pause'
                else:self.model=body['model']
        board=ResidentBoard();controller=entry.Controller(board)
        for model in ('skin','camus','acdc','skin'):
            controller._switch(model)
            self.assertEqual(controller.phase,'ready',controller.error)
            self.assertEqual(controller.model,model)
        self.assertFalse(any(event[0]=='stop' for event in board.events))
        self.assertTrue(board.states[entry.ENGINE] and board.states[entry.HUB])


class Gateway(unittest.TestCase):
    def setUp(self):
        self.controller = entry.Controller(Board())
        self.controller.model = 'skin'
        self.controller.phase = 'ready'
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), entry.Handler)
        self.server.controller = self.controller
        self.thread = threading.Thread(target=self.server.serve_forever)
        self.thread.start()
        self.connection = http.client.HTTPConnection(*self.server.server_address, timeout=3)

    def tearDown(self):
        self.connection.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_binary_upload_query_and_download_are_preserved(self):
        upstream = Mock()
        upstream.getresponse.return_value.status = 200
        headers = {'Content-Type': 'application/zip', 'Content-Length': '4',
                   'Content-Disposition': 'attachment; filename=result.zip'}
        upstream.getresponse.return_value.getheader.side_effect = headers.get
        upstream.getresponse.return_value.read.side_effect = [b'PK\x00\xff', b'']
        factory = entry.http.client.HTTPConnection
        with patch.object(entry.http.client, 'HTTPConnection', side_effect=lambda host, port=None, **kw:
                          upstream if port == 8765 else factory(host, port, **kw)):
            origin = 'http://127.0.0.1:' + str(self.server.server_port)
            self.connection.request('POST', '/skin/g/7/api/infer?download=1', b'\x00\xffimage',
                                    {'Origin': origin, 'Content-Type': 'multipart/form-data; boundary=sample'})
            response = self.connection.getresponse()
            self.assertEqual(response.read(), b'PK\x00\xff')
            self.assertEqual(response.getheader('Content-Disposition'), headers['Content-Disposition'])
        method, path, body, sent = upstream.request.call_args.args
        self.assertEqual((method, path, body), ('POST', '/skin/g/7/api/infer?download=1', b'\x00\xffimage'))
        self.assertEqual(sent['Origin'], 'http://127.0.0.1:8765')
        self.assertEqual(sent['Cookie'], 'session=private')

    def test_old_model_page_is_rejected_after_switch(self):
        self.connection.request('GET', '/camus/g/2/api/status')
        response = self.connection.getresponse()
        self.assertEqual(response.status, 409)
        response.read()

    def test_interactive_routes_rewrite_html_js_and_keep_json_body(self):
        upstream = Mock()
        upstream.getresponse.return_value.status = 200
        response = upstream.getresponse.return_value
        factory = entry.http.client.HTTPConnection
        with patch.object(entry.http.client, 'HTTPConnection', side_effect=lambda host, port=None, **kw:
                          upstream if port == 8765 else factory(host, port, **kw)):
            for path, kind, raw, expected in (
                ('interactive', 'text/html', b'<script src="/interactive.js"></script>', b'"/skin/g/7/interactive.js"'),
                ('interactive.js', 'application/javascript', b"api('/hub/status');api('/interactive/jobs');", b"api('/skin/g/7/interactive/jobs')"),
                ('interactive/images', 'application/json', b'{"image_token":"same"}', b'"same"'),
            ):
                response.getheader.side_effect = {'Content-Type': kind, 'Content-Length': str(len(raw))}.get
                response.read.side_effect = [raw, b'']
                body = b'{"model":"skin","generation":7}' if path.endswith('images') else None
                self.connection.request('POST' if body else 'GET', '/skin/g/7/'+path, body,
                                        {'Origin': 'http://127.0.0.1:'+str(self.server.server_port)})
                received = self.connection.getresponse()
                self.assertEqual(received.status, 200)
                self.assertIn(expected, received.read())
                self.assertEqual(upstream.request.call_args.args[1], '/'+path)
                self.assertEqual(upstream.request.call_args.args[2], body)

    def test_stale_interactive_generation_is_rejected(self):
        self.connection.request('GET', '/skin/g/6/interactive')
        response = self.connection.getresponse()
        self.assertEqual(response.status, 409)
        response.read()


if __name__ == '__main__':
    unittest.main()
