"""Saved evidence, measurement scope and RGB immutability; no physical hardware."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from telemetry_hud import load_telemetry, telemetry_lines, annotate
from touch_display import QuadTouch


class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, data):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding='utf-8')
        return path

    def skin(self, mode='infer'):
        self.write('job.json', dict(job=dict(state='done', mode=mode,
            statistics=dict(request_to_job_complete_seconds=2.3, area_percent=12.5))))
        self.write('inference/frame_000001.json', dict(status='success', dtype='int8', model_name='skin-core',
            model_spec_sha256='a'*64, bit_sha256='b'*64, inference_seconds=.12, threshold_q=-1))
        return load_telemetry(self.root, 'skin')

    def test_missing_or_malformed_metadata_never_invents_a_measurement(self):
        for model in ('skin','camus','acdc'):
            info = load_telemetry(self.root, model)
            self.assertIsNone(info['network_seconds'])
            self.assertIn('FPS: unmeasured (single result)', '\n'.join(telemetry_lines(info)))
        (self.root/'job.json').write_text('{broken', encoding='utf-8')
        self.assertIsNone(load_telemetry(self.root, 'skin')['task_seconds'])
        self.write('job.json', dict(job=['invalid']))
        self.assertIsNone(load_telemetry(self.root, 'skin')['task_seconds'])

    def test_current_skin_task_and_network_have_separate_scopes(self):
        info = self.skin()
        lines = '\n'.join(telemetry_lines(info))
        self.assertIn('120.0 ms', lines)
        self.assertIn('Upload to ZIP: 2300.0 ms', lines)
        self.assertIn('q >= -1', lines)
        self.assertNotIn('8.33 FPS', lines)
        self.assertEqual(info['area_percent'], 12.5)

    def test_replay_does_not_relabel_historical_network_time_as_new_inference(self):
        info = self.skin('demo')
        self.assertIsNone(info['network_seconds'])
        self.assertIn('no new PL', info['mode'])
        self.assertIsNone(self.skin('mask')['network_seconds'])

    def test_failed_jobs_and_nonfinite_timing_are_not_measurements(self):
        self.skin()
        self.write('job.json', dict(job=dict(state='error', mode='infer')))
        self.assertIsNone(load_telemetry(self.root, 'skin')['network_seconds'])
        self.skin()
        frame = json.loads((self.root/'inference/frame_000001.json').read_text())
        frame['inference_seconds'] = float('nan')
        self.write('inference/frame_000001.json', frame)
        self.assertIsNone(load_telemetry(self.root, 'skin')['network_seconds'])

    def test_build_resources_require_both_exact_identities_and_provenance(self):
        self.skin()
        row = dict(model_sha256='a'*64, runtime_sha256='c'*64, utilization_report_sha256='d'*64,
                   core='Skin8', clock_mhz=125, dsp_used=210, dsp_total=220, bram_used=106.5, bram_total=140)
        path = self.write('builds.json', dict(schema='pld-hud-build-info-v1', builds=[row]))
        self.assertIsNone(load_telemetry(self.root, 'skin', path)['resources'])
        row['runtime_sha256'] = 'b'*64
        self.write('builds.json', dict(schema='pld-hud-build-info-v1', builds=[row]))
        text = '\n'.join(telemetry_lines(load_telemetry(self.root, 'skin', path)))
        self.assertIn('106.5/140 (76.1%)', text)
        self.assertIn('@ 125 MHz', text)
        row.pop('utilization_report_sha256')
        self.write('builds.json', dict(schema='pld-hud-build-info-v1', builds=[row]))
        self.assertIsNone(load_telemetry(self.root, 'skin', path)['resources'])

    def test_camus_host_result_does_not_claim_pl_network_latency(self):
        report = dict(schema='pld-camus-result-v1', backend='reference',
                      model=dict(head=dict(dtype='int8')), timings_seconds=dict(network=.2), seconds_to_metadata_finalize=3)
        self.write('report.json', report)
        self.assertIsNone(load_telemetry(self.root, 'camus')['network_seconds'])
        report['backend'] = 'pl'; self.write('report.json', report)
        info = load_telemetry(self.root, 'camus')
        self.assertEqual(info['network_seconds'], .2)
        self.assertEqual(info['task_scope'], 'To metadata')
        self.assertIn('Argmax', info['threshold'])
        report['model']['spec_sha256']='a'*64;report['runtime_sha256']='b'*64
        self.write('report.json',report)
        info=load_telemetry(self.root,'camus')
        self.assertEqual(info['model_sha256'],'a'*64)
        self.assertEqual(info['runtime_sha256'],'b'*64)

    def test_acdc_volume_uses_slice_median_and_separate_volume_time(self):
        report = dict(schema='pld-acdc-appliance-numerics-v2', status='numerics_completed', backend='pl',
                      synthetic=False, records=[dict(seconds=v) for v in (.1,.2,3)], model_sha256='a'*64, runtime_sha256='b'*64)
        self.write('numerics.json', report)
        self.write('completion.json', dict(server_upload_to_assets_seconds=9))
        info = load_telemetry(self.root, 'acdc')
        self.assertEqual(info['network_seconds'], .2)
        self.assertEqual(info['task_seconds'], 9)
        self.assertIn('Median slice', info['network_scope'])
        report['synthetic'] = True; self.write('numerics.json', report)
        self.assertIsNone(load_telemetry(self.root, 'acdc')['network_seconds'])

    def test_slice_throughput_uses_the_full_measured_window(self):
        self.write('numerics.json',dict(schema='pld-acdc-appliance-numerics-v2',status='numerics_completed',
            backend='pl',synthetic=False,records=[dict(seconds=.1,inference_begin_seconds=1,inference_end_seconds=1.1),
            dict(seconds=.1,inference_begin_seconds=2,inference_end_seconds=2.1)]))
        info=load_telemetry(self.root,'acdc')
        self.assertAlmostEqual(info['slice_rate'],2/1.1)
        self.assertIn('Slice/s (job window): 1.82','\n'.join(telemetry_lines(info)))

    def test_annotation_copies_pixels_preserves_controls_and_does_not_write_files(self):
        info = self.skin()
        before = {p.relative_to(self.root):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        source = np.full((480,640,3), 90, np.uint8)
        rendered = annotate(source, info, y=48)
        self.assertTrue((source == 90).all())
        self.assertTrue(np.array_equal(rendered[:48], source[:48]))
        self.assertTrue(np.array_equal(rendered[160:400], source[160:400]))
        self.assertFalse(np.shares_memory(rendered, source))
        self.assertEqual(rendered.dtype, np.uint8)
        after = {p.relative_to(self.root):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_touch_enlargement_retains_same_hud_and_disclaimer(self):
        info = self.skin(); view = QuadTouch(np.full((480,640,3), 55, np.uint8), info)
        quad = view.frame(); enlarged = view.tap(100,100)
        self.assertTrue(np.array_equal(quad[:68], enlarged[:68]))
        self.assertTrue(np.array_equal(quad[-16:], enlarged[-16:]))
        self.assertTrue(np.array_equal(view.tap(100,100), quad))

    def test_comparison_overlay_can_close_without_losing_enlarged_panel(self):
        view=QuadTouch(np.full((480,640,3),55,np.uint8),self.skin())
        enlarged=view.tap(400,100);selected=view.selected
        view.tap(50,20);self.assertTrue(view.metrics_open)
        view.tap(400,200);self.assertFalse(view.metrics_open)
        self.assertEqual(view.selected,selected)
        self.assertTrue(np.array_equal(view.frame(),enlarged))


if __name__ == '__main__':
    unittest.main()
