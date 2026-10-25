"""The board-local pause/resume command must use the POST control path."""
import http.client
from http.server import ThreadingHTTPServer
import json
from types import SimpleNamespace
import threading
from unittest.mock import Mock

from acdc.appliance.web import handler


def test_post_lifecycle_dispatches_to_engine():
    client=Mock();client.request.return_value=dict(suspended=True)
    server=ThreadingHTTPServer(('127.0.0.1',0),handler(SimpleNamespace(client=client)))
    thread=threading.Thread(target=server.serve_forever);thread.start()
    connection=http.client.HTTPConnection(*server.server_address,timeout=3)
    try:
        connection.request('POST','/api/v2/lifecycle',json.dumps(dict(action='pause')),
                           {'Content-Type':'application/json'})
        response=connection.getresponse()
        assert response.status==202
        assert json.loads(response.read())['suspended']
        client.request.assert_called_once_with(op='lifecycle',action='pause')
    finally:
        connection.close();server.shutdown();server.server_close();thread.join()
