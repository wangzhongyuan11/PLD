import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from perf_eval.evidence import utilization,imported_timing
from perf_eval.core import digest,memory_snapshot
from perf_eval.report import bundle,compare
from perf_eval.benchmark import load_dataset
from perf_eval.engines import Lease,SPECS

class EvidenceTests(unittest.TestCase):
    def test_resource_report_without_external_release_assets(self):
        # Synthetic parser fixture, never reported as measured hardware usage.
        with tempfile.TemporaryDirectory() as temp:
            file=Path(temp)/'utilization.rpt'
            file.write_text('| Design State : Physopt postRoute |\n'
                '| Slice LUTs | 1234 | 0 | 0 | 53200 | 2.32 |\n'
                '| Register as Flip Flop | 2345 | 0 | 0 | 106400 | 2.20 |\n'
                '| Block RAM Tile | 36.5 | 0 | 0 | 140 | 26.07 |\n'
                '| DSPs | 17 | 0 | 0 | 220 | 7.73 |\n')
            row=utilization(file)
        self.assertEqual(row['resources']['FF']['used'],2345)
        self.assertEqual(row['resources']['BRAM36_equivalent']['used'],36.5)
        self.assertEqual(row['resources']['DSP']['used'],17)
        self.assertFalse(row['bitstream_association_verified'])
    def test_import_overlap_and_pid_not_duration(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'timing.json';p.write_text(json.dumps(dict(server_stages=dict(request_to_preview_ready=5.,preview_build=2.,persistent_worker_pid=7787,png_workers=2),name='private image',client_seconds=dict(click_to_preview=6.))))
            row=imported_timing(p)
            self.assertNotIn('persistent_worker_pid',row['server_seconds']);self.assertNotIn('png_workers',row['server_seconds']);self.assertNotIn('private image',json.dumps(row))
            self.assertEqual(row['server_seconds']['preview_build'],2.)
    def test_html_escapes_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'e.json';p.write_text(json.dumps({'kind':'<script>bad()</script>'}))
            page=bundle([], [p],Path(temp)/'report')
            self.assertNotIn('<script>',page.read_text(encoding='utf-8'));self.assertTrue(page.with_name('comparison.csv').is_file())
    def test_bad_dataset_stops_before_hardware(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'dataset.json';p.write_text(json.dumps(dict(schema='pld-prepared-input-v1',model='skin',spec_sha256=SPECS['skin'],cases=[dict(id='a',int8='../escape.npy',fp32='fp.npy',valid='mask.npy',sha256=dict(int8='0'))])))
            with self.assertRaises(ValueError):load_dataset(p,'skin')
    def test_pl_refuses_live_service(self):
        import types
        with patch('perf_eval.engines.sys.platform','linux'),patch('perf_eval.engines.subprocess.run',return_value=types.SimpleNamespace(stdout='ActiveState=active\nMainPID=10\n')):
            with self.assertRaisesRegex(RuntimeError,'normally stopped'):Lease()
    def test_memory_has_explicit_unavailable_fields(self):
        row=memory_snapshot();self.assertIn('cma_free_bytes',row)
        if row['rss_bytes'] is not None:self.assertGreater(row['rss_bytes'],0)

class ComparisonTests(unittest.TestCase):
    def test_compare_rejects_wrong_scope_and_blocks_cross_platform_ratio(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);paths=[]
            for index,backend in enumerate(['cpu-int8','pl']):
                out=root/str(index);out.mkdir();np.save(out/'head.npy',np.zeros((1,),np.int8));np.save(out/'labels.npy',np.zeros((1,1),np.uint8))
                row=dict(schema='pld-benchmark-v1',complete=True,model='skin',dataset_sha256='same',label_scope='grid',backend=backend,
                    provenance=dict(spec_sha256='same'),environment=dict(machine='x86' if index==0 else 'arm',os='linux'),
                    artifacts={'a':dict(head='head.npy',labels='labels.npy',head_file_sha256=digest(out/'head.npy'),labels_file_sha256=digest(out/'labels.npy'),valid_sha256='same')},
                    frames=[dict(case_id='a',repetition=0,input_sha256='same',repeated_output_consistent=True)])
                p=out/'benchmark.json';p.write_text(json.dumps(row));paths.append(p)
            report=compare(*paths);self.assertIsNone(report['inference_acceleration_cpu_int8_over_pl']);self.assertTrue(report['integer_head_all_exact'])
            row['label_scope']='native';paths[1].write_text(json.dumps(row))
            with self.assertRaises(ValueError):compare(*paths)

if __name__=='__main__':unittest.main()

class AccuracyChangeTests(unittest.TestCase):
    def test_same_gt_required_and_delta_in_percentage_points(self):
        from perf_eval.report import score_pairs,precision_change
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);gt=np.array([[1,0],[0,0]],np.uint8);np.save(root/'gt.npy',gt);np.save(root/'a.npy',gt);np.save(root/'b.npy',np.zeros((2,2),np.uint8))
            reports=[]
            for name in ('a','b'):
                pairs=root/(name+'_pairs.json');pairs.write_text(json.dumps(dict(classes=2,dataset_split='synthetic_demo',label_scope='grid',pairs=[dict(id='demo',prediction=name+'.npy',truth='gt.npy')])))
                result=score_pairs(pairs);report=root/(name+'_scores.json');report.write_text(json.dumps(result));reports.append(report)
            delta=precision_change(*reports)
            self.assertEqual(delta['mean_per_image']['foreground_macro_dice']['delta_percentage_points'],-100)
            row=json.loads(reports[1].read_text());row['reference_set_sha256']='different';reports[1].write_text(json.dumps(row))
            with self.assertRaises(ValueError):precision_change(*reports)
