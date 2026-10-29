"""Workbench contract: separate baseline, baseline cache, image token, replay."""
import base64
import hashlib
import io
import json
import sys
import threading
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'display_web/unified'))
from prompt_seg.service import SegmentationService, pixel_sha256, png_data
from interactive_bridge import Manager, Predictor, REPLAY_SCHEMA, SPECS


class FakePredictor:
    def __init__(self, cached=False):
        self.cached = cached
        self.calls = 0

    def predict(self, image, model, *args):
        self.calls += 1
        p = np.zeros((2 if model == 'skin' else 4, *image.shape[:2]), np.float32)
        p[0] = .2
        p[1] = .8
        return p, {'prediction_cache_hit': self.cached, 'model': model}


def labels(result):
    return np.array(Image.open(io.BytesIO(base64.b64decode(result['outputs']['labels_png'].split(',')[1]))))


class BaselineTests(unittest.TestCase):
    def test_baseline_separate_and_not_changed_by_prompts(self):
        svc = SegmentationService(FakePredictor())
        image = np.zeros((32, 40, 3), np.uint8)
        plain = svc.process('skin', image, {})
        self.assertTrue(plain['reused_baseline_as_final'])
        self.assertEqual(plain['outputs']['labels_png'], plain['baseline']['outputs']['labels_png'])
        self.assertEqual(plain['statistics']['changed_pixels_by_refinement'], 0)
        prompted = svc.process('skin', image, {'points': [dict(x=8, y=8, label='background')]})
        self.assertEqual(prompted['baseline']['outputs']['labels_png'], plain['baseline']['outputs']['labels_png'])
        self.assertNotEqual(prompted['label_hashes']['final'], prompted['label_hashes']['baseline'])
        self.assertGreater(prompted['statistics']['changed_pixels_by_refinement'], 0)
        self.assertEqual(prompted['baseline']['statistics']['changed_pixels_by_refinement'], 0)
        self.assertFalse(prompted['reused_baseline_as_final'])

    def test_baseline_cache_reused_for_prompt_changes_only(self):
        svc = SegmentationService(FakePredictor())
        image = np.zeros((24, 32, 3), np.uint8)
        first = svc.process('skin', image, {'points': [dict(x=4, y=4, label='foreground')]})
        second = svc.process('skin', image, {'points': [dict(x=6, y=6, label='foreground')]})
        self.assertFalse(first['baseline']['cache_hit'])
        self.assertTrue(second['baseline']['cache_hit'])
        self.assertEqual(first['baseline']['outputs'], second['baseline']['outputs'])
        third = svc.process('skin', image, {'roi': dict(x0=2, y0=2, x1=20, y1=18)})
        self.assertFalse(third['baseline']['cache_hit'])
        fourth = svc.process('skin', image, {'threshold': .6})
        self.assertFalse(fourth['baseline']['cache_hit'])
        fifth = svc.process('skin', image, {}, scope=(9, 'spec-a'))
        sixth = svc.process('skin', image, {}, scope=(9, 'spec-b'))
        self.assertFalse(fifth['baseline']['cache_hit'])
        self.assertFalse(sixth['baseline']['cache_hit'])

    def test_prompt_only_change_keeps_probability_identity(self):
        svc = SegmentationService(FakePredictor(cached=True))
        image = np.zeros((16, 16, 3), np.uint8)
        row = svc.process('skin', image, {})
        self.assertTrue(row['identity']['prediction_cache_hit'])


class TokenTests(unittest.TestCase):
    def manager(self):
        manager = Manager.__new__(Manager)
        manager.controller = SimpleNamespace(generation=3, model='skin', lock=threading.RLock(),
                                             interactive_busy=False, phase='ready', stopping=False)
        manager.guard = threading.Lock()
        manager.job = None
        manager.image_slot = None
        manager.display_cache_key = None
        manager.display_cache = None
        manager.service = SegmentationService(FakePredictor())
        manager.predictor = Predictor(manager)
        return manager

    def encoded(self, shape=(20, 24, 3)):
        buffer = io.BytesIO()
        Image.fromarray(np.zeros(shape, np.uint8)).save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()

    def test_token_roundtrip_and_scope(self):
        manager = self.manager()
        uploaded = manager.upload({'model': 'skin', 'generation': 3, 'image': self.encoded()})
        token = uploaded['image_token']
        self.assertRegex(token, '^[0-9a-f]{32}$')
        self.assertEqual(uploaded['sha256'], pixel_sha256(np.zeros((20, 24, 3), np.uint8)))
        image, echoed = manager._image_for({'image_token': token})
        self.assertEqual(echoed, token)
        self.assertEqual(image.shape, (20, 24, 3))
        with self.assertRaises(ValueError):
            manager._image_for({'image_token': 'f' * 32})
        manager.controller.model = 'camus'
        with self.assertRaises(ValueError):
            manager._image_for({'image_token': token})

    def test_expected_sha256_guard_keeps_previous_slot(self):
        manager = self.manager()
        original = manager.upload({'model': 'skin', 'generation': 3, 'image': self.encoded()})
        keep = original['image_token']
        with self.assertRaises(ValueError):
            manager.upload({'model': 'skin', 'generation': 3, 'image': self.encoded(shape=(8, 8, 3)),
                            'expected_sha256': original['sha256']})
        image, echoed = manager._image_for({'image_token': keep})
        self.assertEqual(echoed, keep)
        self.assertEqual(image.shape, (20, 24, 3))
        with self.assertRaises(ValueError):
            manager.upload({'model': 'skin', 'generation': 3, 'image': self.encoded(),
                            'expected_sha256': 'z' * 64})

    def test_busy_upload_refused_and_clear_drops_slot(self):
        manager = self.manager()
        uploaded = manager.upload({'model': 'skin', 'generation': 3, 'image': self.encoded()})
        manager.controller.interactive_busy = True
        with self.assertRaises(RuntimeError):
            manager.upload({'model': 'skin', 'generation': 3, 'image': self.encoded()})
        manager.controller.interactive_busy = True
        manager.controller.interactive_busy = False
        manager.controller.stopping = True
        with self.assertRaises(RuntimeError):
            manager.upload({'model': 'skin', 'generation': 3, 'image': self.encoded()})
        manager.controller.stopping = False
        manager.controller.interactive_busy = True
        manager.clear()
        self.assertIsNone(manager.image_slot)
        with self.assertRaises(ValueError):
            manager._image_for({'image_token': uploaded['image_token']})


class ReplayTests(unittest.TestCase):
    def manager(self):
        manager = Manager.__new__(Manager)
        manager.controller = SimpleNamespace(generation=5, model='skin', lock=threading.RLock(),
                                             interactive_busy=False, phase='ready', stopping=False)
        manager.guard = threading.Lock()
        manager.job = None
        manager.image_slot = None
        manager.display_cache_key = None
        manager.display_cache = None
        return manager

    def pixels(self):
        return np.full((24, 32, 3), 40, np.uint8)

    def record(self, **over):
        pixels = self.pixels()
        record = {'schema': REPLAY_SCHEMA, 'model': 'skin', 'generation': 5,
                  'spec_sha256': SPECS['skin'], 'coordinate_space': 'processed_image_pixels',
                  'image': {'width': 32, 'height': 24, 'sha256': pixel_sha256(pixels)},
                  'image_png': png_data(pixels),
                  'original': {'width': 64, 'height': 48,
                               'transform': {'mode': 'fit_long_side_1024',
                                             'processed_width': 32, 'processed_height': 24}},
                  'parameters': {'variant': 'delivery', 'backend': 'int8', 'target_class': 1,
                                 'threshold': .45, 'roi': None, 'box': None, 'points': []},
                  'results': {'baseline_labels_sha256': 'b' * 64, 'final_labels_sha256': 'c' * 64}}
        record.update(over)
        return record

    def test_accepts_valid_replay(self):
        manager = self.manager()
        check = manager.replay_check({'replay': self.record()})
        self.assertTrue(check['replay_ready'])
        self.assertEqual(check['image']['sha256'], pixel_sha256(self.pixels()))
        self.assertEqual(check['expected']['final_labels_sha256'], 'c' * 64)
        self.assertEqual((check['original']['width'], check['original']['height']), (64, 48))

    def test_rejects_wrong_model_spec_schema_and_bad_parameters(self):
        manager = self.manager()
        with self.assertRaises(ValueError):
            manager.replay_check({'replay': self.record(schema='other')})
        with self.assertRaises(ValueError):
            manager.replay_check({'replay': self.record(model='camus')})
        with self.assertRaises(ValueError):
            manager.replay_check({'replay': self.record(spec_sha256='0' * 64)})
        with self.assertRaises(ValueError):
            manager.replay_check({'replay': self.record(parameters={'points': [dict(x=999, y=999, label='foreground')]})})
        with self.assertRaises(ValueError):
            manager.replay_check({'replay': self.record(coordinate_space='native')})
        tampered = self.record()
        tampered['image_png'] = png_data(np.full((24, 32, 3), 90, np.uint8))
        with self.assertRaises(ValueError):
            manager.replay_check({'replay': tampered})
        wrong_size = self.record()
        wrong_size['image_png'] = png_data(np.full((12, 16, 3), 40, np.uint8))
        with self.assertRaises(ValueError):
            manager.replay_check({'replay': wrong_size})
        with self.assertRaises(ValueError):
            manager.replay_check({'replay': self.record(image_png=None)})


class TimingAndIsolationTests(unittest.TestCase):
    def test_timing_phases_are_separate_and_truthful(self):
        svc = SegmentationService(FakePredictor())
        image = np.zeros((150, 200, 3), np.uint8)
        prompted = svc.process('skin', image, {'points': [dict(x=40, y=40, label='foreground')]})
        timings = prompted['timings_ms']
        for key in ('input_validation', 'prediction', 'decision_postprocess', 'baseline_compose',
                    'baseline_png', 'prompt_refinement', 'render_statistics', 'result_png'):
            self.assertIn(key, timings)
            self.assertGreaterEqual(timings[key], 0)
        self.assertLessEqual(timings['prompt_refinement'], prompted['elapsed_ms'] + .05)
        self.assertLessEqual(sum(timings.values()), prompted['elapsed_ms'] + .05)
        cached = svc.process('skin', image, {'points': [dict(x=60, y=50, label='foreground')]})
        self.assertEqual(cached['timings_ms']['baseline_compose'], 0)
        self.assertEqual(cached['timings_ms']['baseline_png'], 0)

    def test_baseline_cache_not_polluted_by_display_outputs(self):
        manager = Manager.__new__(Manager)
        manager.controller = SimpleNamespace(generation=1, lock=threading.RLock(),
                                             interactive_busy=True, phase='ready', stopping=False)
        manager.guard = threading.Lock()
        manager.changed = threading.Condition(manager.guard)
        manager.job = None
        manager.image_slot = None
        manager.display_cache_key = None
        manager.display_cache = None
        manager.service = SegmentationService(FakePredictor())
        manager.predictor = Predictor(manager)
        image = np.zeros((24, 32, 3), np.uint8)
        manager.run('a' * 32, 'skin', image, {'points': [dict(x=6, y=6, label='foreground')]})
        first = manager.snapshot('a' * 32)['result']
        manager.run('b' * 32, 'skin', image, {'points': [dict(x=8, y=8, label='foreground')]})
        second = manager.snapshot('b' * 32)['result']
        self.assertEqual(set(first['baseline']['outputs']), {'labels_png', 'target_mask_png', 'overlay_png'})
        self.assertEqual(set(second['baseline']['outputs']), {'labels_png', 'target_mask_png', 'overlay_png'})
        self.assertEqual(set(second['outputs']), {'labels_png','target_mask_png','overlay_png','original_png'})
        self.assertEqual(second['baseline']['outputs']['labels_png'], first['baseline']['outputs']['labels_png'])
        self.assertTrue(second['baseline']['cache_hit'])


class CutContextTests(unittest.TestCase):
    def setUp(self):
        import prompt_seg.refinement as refinement
        self.refinement = refinement
        rng = np.random.RandomState(7)
        self.image = rng.randint(0, 256, (96, 128, 3), dtype=np.uint8)
        self.prior = np.clip(rng.rand(96, 128) * .6 + .2, .05, .95)
        self.points = [(30, 30, True), (70, 60, False)]
        self.box = (10, 8, 110, 84)

    def test_context_matches_cold_path_exactly(self):
        ref = self.refinement
        cold_mask, cold_info = ref.seeded_cut(self.image, self.prior, self.points, self.box)
        ctx = ref.build_context(self.image, self.prior.shape)
        warm_mask, warm_info = ref.seeded_cut(self.image, self.prior, self.points, self.box, context=ctx)
        np.testing.assert_array_equal(cold_mask, warm_mask)
        self.assertEqual(cold_info['flow_value'], warm_info['flow_value'])
        self.assertFalse(cold_info['context_cache_hit'])
        self.assertTrue(warm_info['context_cache_hit'])

    def test_context_rejects_mismatched_crop(self):
        ref = self.refinement
        ctx = ref.build_context(self.image, self.prior.shape)
        other = self.image.copy()
        other[0, 0, 0] = (int(other[0, 0, 0]) + 7) % 256
        with self.assertRaises(ref.InputError):
            ref.seeded_cut(other, self.prior, self.points, None, context=ctx)
        with self.assertRaises(ref.InputError):
            ref.seeded_cut(self.image[:, :64], self.prior[:, :64], [(10, 10, True)], None, context=ctx)

    def test_service_reuses_context_only_for_same_crop(self):
        svc = SegmentationService(FakePredictor())
        image = self.image
        first = svc.process('skin', image, {'points': [dict(x=30, y=30, label='foreground')]})
        second = svc.process('skin', image, {'points': [dict(x=40, y=40, label='foreground')]})
        self.assertFalse(first['refinement']['context_cache_hit'])
        self.assertTrue(second['refinement']['context_cache_hit'])
        third = svc.process('skin', image, {'roi': dict(x0=4, y0=4, x1=64, y1=64),
                                            'points': [dict(x=20, y=20, label='foreground')]})
        self.assertFalse(third['refinement']['context_cache_hit'])
        self.assertEqual(first['timings_ms']['cut_context'] >= 0, True)
        self.assertEqual(second['timings_ms']['cut_context'], 0)
        svc.clear()
        self.assertIsNone(svc._cut)


if __name__ == '__main__':
    unittest.main()
