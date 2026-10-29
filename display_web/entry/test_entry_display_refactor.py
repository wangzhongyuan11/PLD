"""Entry display retries and read concurrency, without board or socket access."""
import io
import json
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from test_entry import entry, Board


class DisplayRefactorTests(unittest.TestCase):
    def controller(self):
        c = entry.Controller(Board()); c.model = 'skin'; c.phase = 'ready'
        return c

    def begin(self, c, job='a', request_id='a'*32):
        body = dict(action='begin', model='skin', job=job, view='outlined', request_id=request_id)
        return body, c.display.task(body)['token']

    def handler(self, c, method='GET', path='/skin/g/7/api/status', body=b'{}'):
        h = entry.Handler.__new__(entry.Handler)
        h.server = SimpleNamespace(controller=c); h.command=method; h.path=path
        h.headers={'Host':'board:8088', 'Origin':'http://board:8088', 'Content-Length':str(len(body))}
        h.rfile=io.BytesIO(body); h.wfile=io.BytesIO()
        h.send=Mock(); h.send_response=Mock(); h.send_header=Mock(); h.end_headers=Mock()
        return h

    def test_get_bypasses_display_lock_but_post_is_serialized(self):
        c=self.controller()
        for method in ('GET', 'POST'):
            h=self.handler(c,method)
            conn=Mock(); response=conn.getresponse.return_value
            response.status=200; response.getheader.return_value=None
            response.read.side_effect=[b'{}',b'']
            c.proxy_lock.acquire(); finished=threading.Event()
            def run():
                try: h.proxy()
                finally: finished.set()
            with patch.object(entry.http.client,'HTTPConnection',return_value=conn):
                t=threading.Thread(target=run); t.start()
                try: self.assertEqual(finished.wait(.2), method=='GET')
                finally: c.proxy_lock.release(); t.join(2)
            self.assertFalse(t.is_alive()); self.assertEqual(h.wfile.getvalue(), b'{}')

    def test_enable_only_applies_to_future_completion(self):
        c=self.controller(); c.settings['auto_hdmi']=False
        c.display.restore=Mock()
        h=self.handler(c,'POST','/entry/settings',b'{"auto_hdmi":true}')
        h.do_POST()
        self.assertTrue(c.settings['auto_hdmi']); c.display.restore.assert_not_called()

    def test_duplicate_begin_does_not_cancel_newer_task(self):
        c=self.controller(); old,token=self.begin(c)
        self.assertEqual(c.display.task(old)['token'], token)
        _,new=self.begin(c,'b','b'*32)
        self.assertEqual(c.display.task(old)['token'],token)
        self.assertEqual(c.display.current['token'],new)
        c.display.task(dict(action='done',model='skin',job='a',token=token))
        self.assertIsNone(c.display.pending)

    def test_duplicate_done_does_not_queue_again(self):
        c=self.controller(); _,token=self.begin(c)
        body=dict(action='done',model='skin',job='a',token=token)
        self.assertTrue(c.display.task(body)['accepted'])
        c.display.pending=None
        self.assertTrue(c.display.task(body)['accepted'])
        self.assertIsNone(c.display.pending)

    def queue(self,c):
        _,token=self.begin(c)
        c.display.task(dict(action='done',model='skin',job='a',token=token))

    def test_timeout_reconciles_same_job_without_second_show(self):
        c=self.controller(); self.queue(c); shows=[]; status={}
        def request(port,path,body=None):
            if body is None:return status
            shows.append(body)
            status.update(hdmi='starting',hdmi_job='a',hdmi_view='outlined')
            raise TimeoutError('reply lost')
        c.board.request=request
        c.display.dispatch(); self.assertIsNotNone(c.display.pending)
        c.display.pending['retry_at']=0
        c.display.dispatch()
        self.assertIsNone(c.display.pending); self.assertEqual(len(shows),1)
        self.assertTrue(c.display.last_request['reconciled'])

    def test_repeated_timeout_is_bounded_and_visible(self):
        c=self.controller(); self.queue(c)
        c.board.request=Mock(side_effect=TimeoutError('unreachable'))
        c.display.dispatch(); c.display.pending['retry_at']=0; c.display.dispatch()
        self.assertIsNone(c.display.pending); self.assertIn('unreachable',c.display.error)
        self.assertEqual(c.board.request.call_count,2)

    def test_old_show_failure_cannot_drop_new_intent(self):
        c=self.controller(); self.queue(c)
        def request(port,path,body=None):
            if body is None: return {}
            self.begin(c,'b','b'*32)
            raise TimeoutError('late old reply')
        c.board.request=request; c.display.dispatch()
        self.assertEqual(c.display.current['job'],'b'); self.assertEqual(c.display.error,'')


if __name__=='__main__': unittest.main()
