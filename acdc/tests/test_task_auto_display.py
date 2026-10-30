"""Per-task opt-out preserves the existing browser display default."""
import http.client
from http.server import ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from acdc.appliance.engine import JobManager
from acdc.appliance.web import WebApplication, handler


@pytest.mark.parametrize('header,automatic', [(None, True), ('true', True), ('false', False)])
def test_upload_display_option_reaches_engine_without_changing_default(tmp_path, monkeypatch, header, automatic):
    completion = Mock()
    monkeypatch.setattr('acdc.appliance.engine.Completion', completion)
    backend = SimpleNamespace(kind='pl', synthetic=False, model_sha='model', runtime_sha='runtime')
    def run(ed, es, output, **kwargs):
        assert ed.read_bytes() == b'ED'
        assert es.read_bytes() == b'ES'
        kwargs['progress'](dict(run_id=output.name, status='assets_ready'))
    manager = JobManager(SimpleNamespace(backend=backend, run=run), tmp_path)
    class Client:
        def request(self, **message): return manager.dispatch(message)
    app = WebApplication(Client(), tmp_path, None)
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler(app))
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .02}, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(*server.server_address, timeout=3)
    try:
        headers = {'X-ED-Bytes': '2'}
        if header is not None: headers['X-Auto-Display'] = header
        connection.request('POST', '/api/v2/jobs?ed=ed.nii&es=es.nii', b'EDES', headers)
        response = connection.getresponse()
        assert response.status == 202
        job = json.loads(response.read())['run_id']
        manager.thread.join(2)
        assert manager.status(job)['status'] == 'assets_ready'
        assert manager.health()['task_auto_display'] is True
        if automatic:
            completion.assert_called_once()
            completion.return_value.finish.assert_called_once_with(True)
        else:
            completion.assert_not_called()
    finally:
        connection.close(); server.shutdown(); server.server_close(); thread.join(2); manager.close()


def test_invalid_display_header_is_rejected_before_reserving(tmp_path):
    client = Mock()
    server = ThreadingHTTPServer(('127.0.0.1', 0), handler(SimpleNamespace(root=tmp_path, client=client)))
    thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .02}, daemon=True)
    thread.start()
    connection = http.client.HTTPConnection(*server.server_address, timeout=3)
    try:
        connection.request('POST', '/api/v2/jobs?ed=ed.nii&es=es.nii', b'EDES',
                           {'X-ED-Bytes': '2', 'X-Auto-Display': 'invalid'})
        response = connection.getresponse()
        assert response.status == 400
        response.read()
        client.request.assert_not_called()
    finally:
        connection.close(); server.shutdown(); server.server_close(); thread.join(2)
