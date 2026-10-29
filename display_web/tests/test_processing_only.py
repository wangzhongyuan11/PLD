import json
from pathlib import Path
import tempfile
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
