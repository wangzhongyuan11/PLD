import json
from pathlib import Path
import tempfile
import threading
import unittest
import zipfile
from unittest.mock import Mock, patch

import numpy as np

import server
import test_web
from pld_result import load_result


class ProcessingOnlyTests(unittest.TestCase):
    setUp = test_web.WebTests.setUp
    tearDown = test_web.WebTests.tearDown
    wait = test_web.WebTests.wait

    def test_five_views_keep_pixels_and_zip_without_comparison(self):
        self.app.processing_only = True
        rgb = np.full((32, 40, 3), 100, np.uint8)
        mask = np.zeros((32, 40), np.uint8)
        mask[8:24, 10:30] = 255
        job = self.wait(self.app.submit(dict(mode='mask', image=test_web.encoded(rgb),
                                             mask=test_web.encoded(mask)))['id'])
        self.assertEqual(job['state'], 'done')
        self.assertEqual(job['views'], ['original', 'mask', 'contour', 'outlined', 'cutout'])
        folder = Path(self.temp.name) / job['id']
        self.assertFalse((folder / 'comparison.png').exists())
        for key in ('view_composition', 'png_batch_wall', 'runtime_verify'):
            self.assertNotIn(key, job['statistics']['final_stage_seconds'])
        expected = server.process(mask, rgb)
        for key in ('mask', 'contour', 'outlined', 'cutout'):
            pixels = expected[key]
            if pixels.dtype == bool:
                pixels = pixels.astype(np.uint8) * 255
            with server.Image.open(folder / (key + '.png')) as image:
                np.testing.assert_array_equal(np.asarray(image), pixels)
        with zipfile.ZipFile(folder / 'results.zip') as archive:
            self.assertEqual(set(archive.namelist()),
                             {view + '.png' for view in job['views']} | {'report.json'})

    def test_warmup_does_not_validate_runtime(self):
        self.app.processing_only = True
        self.app.board = True
        self.app.runtime_ok = True
        self.app.persistent_b_worker = True
        self.app.runtime = Path(self.temp.name)
        self.app.b_worker = Mock()
        with patch.object(self.app, 'verify_deployment', side_effect=AssertionError('runtime validation')):
            with patch('server.inference_command', return_value=['python', 'infer.py', 'unused']):
                self.app.begin_warmup()
                self.app.warmup_thread.join(3)
        self.assertEqual(self.app.warmup_state, 'ready')

    def test_postprocess_reads_result_without_hash_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'head.json'
            head = np.full((1, 1, 256, 256), -2, np.int8)
            head[0, 0, 40:80, 10:30] = 0
            metadata = dict(valid_rect=[0, 32, 256, 192],
                            processing_only=True,
                            input_metadata=dict(original_width=1024, original_height=768))
            path.write_text(json.dumps(metadata), encoding='utf-8')
            path.with_suffix('.bin').write_bytes(head.tobytes())
            with patch('pld_result.hashlib.sha256', side_effect=AssertionError('content hash')):
                result = load_result(path)
            self.assertEqual(result.stats['mask_pixels'], 800)
            self.assertEqual(result.stats['valid_pixels'], 49152)
            self.assertEqual(result.original_mask().shape, (768, 1024))

    def test_waiting_request_returns_when_preview_is_published(self):
        self.app.jobs['waiting'] = dict(id='waiting', state='running')
        replies = []
        waiting = threading.Thread(target=lambda: replies.append(self.app.wait_job('waiting', 'preview')))
        waiting.start()
        self.app.update('waiting', preview_ready=True)
        waiting.join(1)
        self.assertFalse(waiting.is_alive())
        self.assertTrue(replies[0]['preview_ready'])
        self.assertEqual(replies[0]['state'], 'running')

    def test_original_png_reaches_board_worker_without_jpeg_conversion(self):
        self.app.processing_only = True
        self.app.board = self.app.runtime_ok = self.app.persistent_b_worker = True
        self.app.runtime = Path(self.temp.name)
        self.app.b_worker = Mock()
        rgb = np.full((32, 40, 3), 100, np.uint8)
        encoded = test_web.encoded(rgb)

        def run(argv, log):
            import base64
            source = Path(argv[0])
            self.assertEqual(source.read_bytes(), base64.b64decode(encoded))
            self.assertEqual(source.suffix, '.png')
            output = Path(argv[argv.index('--output')+1]); output.mkdir()
            head = np.full((1, 1, 256, 256), -2, np.int8)
            head[0, 0, 40:80, 10:30] = 0
            (output/'frame_000001.bin').write_bytes(head.tobytes())
            (output/'frame_000001.json').write_text(json.dumps(dict(
                processing_only=True,valid_rect=[0,25,256,205],online_b_seconds=.01,
                input_metadata=dict(original_width=40,original_height=32))))
            return dict(model_reused=True, hardware_reused=True, startup_seconds=0.,
                        request_seconds=.01, worker_pid=1, alive_after_reply=True)

        self.app.b_worker.run.side_effect = run
        with patch('server.inference_command', side_effect=lambda runtime, source, dest, portable:
                   ['python','infer.py',str(source),'--output',str(dest)]):
            job = self.wait(self.app.submit(dict(mode='infer',image=encoded))['id'])
        self.assertEqual(job['state'], 'done', job.get('error'))
