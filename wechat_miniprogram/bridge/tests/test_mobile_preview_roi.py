"""Mobile-only regression coverage; no board, model or HDMI changes."""
import copy
import io
import json
import socket
import sys
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from bridge import Bridge
from board_client import BoardClient
from preprocess import prepare_skin_image
from wechat_api import WechatApiError, WechatCloud
from tests.mocks import FakeCloud


def png(size=(100, 60)):
    out = io.BytesIO()
    Image.new('RGB', size, (40, 80, 120)).save(out, 'PNG')
    return out.getvalue()


class MobileTests(unittest.TestCase):
    def setUp(self):
        self.cloud = FakeCloud()
        self.b = Bridge(self.cloud, None, {'async_results': False}, sleep=lambda _: None)
        self.task = {'_id': 'mobile', 'key': 'mobile', 'type': 'skin', 'status': 'pending'}
        self.cloud.add_doc('tasks', copy.deepcopy(self.task))

    def tearDown(self):
        self.b.close()

    def test_preview_visible_while_exports_are_still_running(self):
        stages = iter([{'state': 'running', 'preview_ready': True, 'preview_views': ['outlined']},
                       {'state': 'done', 'views': ['outlined']}])
        reads = []
        def read(job, name):
            reads.append(name)
            self.assertEqual(name, 'preview_outlined.png')
            return png((1600, 900))
        def poll():
            row = next(stages)
            if row['state'] == 'done':
                result = self.cloud.collections['tasks']['mobile']['result']
                self.assertTrue(result['previews'])
                self.assertNotIn('zip', result)
                self.assertEqual(reads, ['preview_outlined.png'])
            return row
        self.b.board = SimpleNamespace(skin_file=read)
        self.b._poll_with_retry('mobile', poll, lambda r: r['state']=='done', lambda r: False,
            lambda r: '', time.monotonic()+5,
            on_update=lambda r: self.b._early_preview(self.task, 'skin', 'j', r))
        data = self.cloud.download_bytes(self.task['result']['previews'][0]['fileID'])
        with Image.open(io.BytesIO(data)) as im:
            self.assertEqual(im.format, 'JPEG')
            self.assertLessEqual(max(im.size), 768)
        self.b._finish_task(self.task, [('result.zip', 'zip', b'ZIP')], [])
        result = self.cloud.collections['tasks']['mobile']['result']
        self.assertTrue(result['previews'])
        self.assertTrue(result['zip'])

    def test_preview_failure_does_not_fail_computation(self):
        self.b.board = SimpleNamespace(skin_file=Mock(return_value=b'bad image'))
        self.b._early_preview(self.task, 'skin', 'j', {'preview_ready': True, 'preview_views': ['outlined']})
        self.assertEqual(self.cloud.collections['tasks']['mobile']['status'], 'pending')
        self.assertNotIn('_preview_done', self.task)

    def test_acdc_reads_quick_preview_before_assets(self):
        self.task['type'] = 'acdc'
        self.b.board = SimpleNamespace(acdc_preview=Mock(return_value=png()))
        self.b._early_preview(self.task, 'acdc', 'j', {'quick_preview_ready': True, 'status': 'infer_restore'})
        self.b.board.acdc_preview.assert_called_once_with('j', quick=True)
        self.assertIn('中心切片', self.task['result']['previews'][0]['label'])

    def test_acdc_preview_uses_readonly_endpoint(self):
        c = BoardClient('unused')
        c._request = Mock(return_value=(200, {}, b'PNG'))
        self.assertEqual(c.acdc_preview('job', quick=True), b'PNG')
        self.assertEqual(c._request.call_args.args[1:3], ('GET', '/api/v2/jobs/job/quick-preview/axial.png'))

    def test_roi_is_applied_after_exif_and_full_image_path_stays_identical(self):
        out = io.BytesIO()
        image = Image.new('RGB', (100, 60))
        exif = image.getexif(); exif[274] = 6
        image.save(out, 'JPEG', exif=exif)
        raw = out.getvalue()
        self.assertEqual(prepare_skin_image(raw), raw)
        cropped = prepare_skin_image(raw, {'x0': .1, 'y0': .2, 'x1': .9, 'y1': .8})
        with Image.open(io.BytesIO(cropped)) as im:
            self.assertEqual(im.size, (48, 60))
            self.assertNotIn(274, im.getexif())

    def test_roi_rejects_invalid_coordinates(self):
        for roi in ({'x0': 0, 'y0': 0, 'x1': float('nan'), 'y1': 1},
                    {'x0': .9, 'y0': 0, 'x1': .1, 'y1': 1},
                    {'x0': 0, 'y0': 0, 'x1': .01, 'y1': .01}):
            with self.assertRaises(ValueError): prepare_skin_image(png(), roi)

    def test_roi_uses_normal_pl_submission_and_roi_area_label(self):
        fid = self.cloud.upload_bytes('input.png', png())
        self.task.update(inputs=[{'fileID': fid, 'name': 'photo.png'}],
            params={'roi': {'x0': .1, 'y0': .1, 'x1': .9, 'y1': .9}})
        def submit(raw, **kwargs):
            with Image.open(io.BytesIO(raw)) as im: self.assertEqual(im.size, (80, 48))
            return {'id': 'roi-job'}
        self.b.board = SimpleNamespace(skin_submit=Mock(side_effect=submit),
            skin_job=lambda _: {'state': 'done', 'views': ['mask'], 'statistics': {'area_percent': 25}},
            skin_file=lambda *args: png())
        self.b._run_skin(self.task)
        self.b.board.skin_submit.assert_called_once()
        result = self.cloud.collections['tasks']['mobile']['result']
        self.assertEqual(result['metrics'][0], {'label': 'ROI 内病变面积占比', 'value': '25.00%'})

    def test_claim_ack_loss_reconciles_without_replaying_claim(self):
        update = self.cloud.update_where
        def lost(*args):
            update(*args)
            raise WechatApiError('lost reply', kind='network')
        self.cloud.update_where = Mock(side_effect=lost)
        self.assertTrue(self.b.claim_task(self.task))
        self.assertEqual(self.cloud.update_where.call_count, 1)

    def test_accepted_job_is_checkpointed_and_recovered_without_reinference(self):
        self.b._accepted(self.task, 'existing')
        self.cloud.update_doc('tasks', 'mobile', {'status': 'running', 'bridge_session': 'old'})
        self.b.recover_stale_tasks(force=True)
        saved = self.cloud.collections['tasks']['mobile']
        self.assertEqual(saved['resume_job'], {'model': 'skin', 'id': 'existing'})
        self.b.board = SimpleNamespace(skin_submit=Mock(side_effect=AssertionError('must not resubmit')),
            skin_job=lambda _: {'state': 'done', 'views': ['mask']}, skin_file=lambda *a: png())
        self.b._run_skin(copy.deepcopy(saved))
        self.b.board.skin_submit.assert_not_called()


class NetworkRecoveryTests(unittest.TestCase):
    def test_query_reconnects_once_but_mutation_is_never_replayed(self):
        calls = []
        class Handler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *args): pass
            def do_POST(self):
                self.rfile.read(int(self.headers.get('Content-Length', '0')))
                calls.append(self.path)
                if len(calls) == 1:
                    self.close_connection = True
                    self.connection.shutdown(socket.SHUT_RDWR)
                    return
                body = b'{"errcode":0,"data":[]}'
                self.send_response(200); self.send_header('Content-Length', str(len(body)))
                self.end_headers(); self.wfile.write(body)
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .02}, daemon=True)
        thread.start()
        cloud = WechatCloud('a', 'b', 'c', api_base='http://127.0.0.1:%s' % server.server_port)
        try:
            self.assertEqual(cloud._api_post_json('/tcb/databasequery', {})['data'], [])
            self.assertEqual(len(calls), 2)
            cloud.close(); calls.clear()
            with self.assertRaises(WechatApiError): cloud._api_post_json('/tcb/databaseupdate', {})
            self.assertEqual(len(calls), 1)
        finally:
            cloud.close(); server.shutdown(); server.server_close(); thread.join(2)


if __name__ == '__main__': unittest.main()
