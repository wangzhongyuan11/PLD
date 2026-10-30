"""Paired synthetic fixtures exercise import contracts, not board performance."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from perf_eval.core import digest,summarize
from perf_eval.demo import prepare_catalog
from perf_eval.report import score_pairs
from display_web.demo_metrics import catalog_status,for_task,record_digest,validate_catalog,draw_comparison


class DemoMetricTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        for backend,times,wall in [('cpu-int8',[2.,4.],7.),('pl',[.1,.2],.8)]:
            folder=self.root/backend;folder.mkdir()
            for name,data in [('head',np.zeros((1,1,2,2),np.int8)),('labels',np.zeros((2,2),np.uint8))]:
                np.save(folder/(name+'.npy'),data)
            report=dict(schema='pld-benchmark-v1',model='skin',backend=backend,complete=True,hardware_execution=backend=='pl',
                provenance=dict(spec_sha256='a'*64,bit_sha256='b'*64),dataset_sha256='c'*64,label_scope='raw-grid',
                baseline='NumPy numerical reference, not optimized CPU library' if backend=='cpu-int8' else 'pl',
                synthetic=True,environment=dict(machine='fixture',os='fixture',host_fingerprint='same-test-host'),
                frames=[dict(case_id='case',repetition=i,input_sha256='d'*64,inference_seconds=t,processing_seconds=t+.1,
                    repeated_output_consistent=True,dma=dict(total_packet_bytes=300000)) for i,t in enumerate(times)],
                artifacts=dict(case=dict(head='head.npy',labels='labels.npy',head_file_sha256=digest(folder/'head.npy'),
                    labels_file_sha256=digest(folder/'labels.npy'),valid_sha256='e'*64)),
                summary=dict(inference=summarize(times),processing=summarize([t+.1 for t in times],wall)))
            self.write(backend+'/benchmark.json',report)
        self.config=dict(schema='pld-demo-metrics-input-v1',records=[dict(cpu='cpu-int8/benchmark.json',fpga='pl/benchmark.json')])

    def write(self,name,data):
        p=self.root/name;p.write_text(json.dumps(data),encoding='utf-8');return p

    def prepare(self):
        self.write('input.json',self.config)
        return prepare_catalog(self.root/'input.json',self.root/'metrics.json')

    def test_aggregates_scopes_and_keeps_missing_evidence_unmeasured(self):
        doc=self.prepare();r=doc['records'][0]
        self.assertAlmostEqual(r['fpga']['metrics']['median_ms'],150)
        self.assertAlmostEqual(r['fpga']['metrics']['processing_fps'],4)
        self.assertAlmostEqual(r['fpga']['metrics']['throughput_fps'],2.5)
        self.assertAlmostEqual(r['fpga']['metrics']['ddr_MB_s'],2)
        for name in ('dice','iou','power_w','lut','gmac_s'):
            self.assertIsNone(r['fpga']['metrics'][name])
        self.assertTrue(r['synthetic'])
        self.assertEqual(r['record_sha256'],record_digest(r))
        self.assertEqual(catalog_status(self.root/'metrics.json')['status'],'ready')
        with self.assertRaises(FileExistsError):prepare_catalog(self.root/'input.json',self.root/'metrics.json')

    def test_evidence_cannot_be_reused_for_another_runtime(self):
        p=self.root/'utilization.rpt';p.write_text('| Design State : Routed |\n| Slice LUTs | 100 | 0 | 0 | 53200 | 0.19 |\n')
        self.config['records'][0]['utilization']=dict(path=p.name,sha256=digest(p),model_sha256='a'*64,runtime_sha256='f'*64)
        with self.assertRaisesRegex(ValueError,'model/runtime'):self.prepare()
        self.assertFalse((self.root/'metrics.json').exists())

    def test_bound_resource_power_and_workload_data(self):
        p=self.root/'utilization.rpt';p.write_text('| Design State : Routed |\n| Slice LUTs | 100 | 0 | 0 | 53200 | 0.19 |\n| Register as Flip Flop | 200 | 0 | 0 | 106400 | 0.19 |\n| Block RAM Tile | 2.5 | 0 | 0 | 140 | 1.79 |\n| DSPs | 8 | 0 | 0 | 220 | 3.64 |\n')
        def bound(p):return dict(path=p.name,sha256=digest(p),model_sha256='a'*64,runtime_sha256='b'*64)
        config=self.config['records'][0];config['utilization']=bound(p)
        p=self.root/'power.rpt';p.write_text('| Total On-Chip Power (W) | 1.25 |\n')
        config['power']=dict(fpga=dict(bound(p),kind='chip_estimate'))
        p=self.write('board_power.json',dict(scope='board_input',frames=2,instrument='fixture meter',samples=[dict(time_s=0,power_w=2),dict(time_s=2,power_w=4)]))
        config['power']['cpu']=dict(path=p.name,sha256=digest(p),kind='board_samples',benchmark_sha256=digest(self.root/'cpu-int8/benchmark.json'))
        p=self.write('macs.json',dict(model_sha256='a'*64,macs_per_frame=1e9,method='synthetic unit-test count'))
        config['workload']=dict(path=p.name,sha256=digest(p))
        r=self.prepare()['records'][0]
        self.assertEqual(r['fpga']['metrics']['bram'],2.5)
        self.assertEqual(r['fpga']['metrics']['power_w'],1.25)
        self.assertEqual(r['cpu']['metrics']['power_w'],3)
        self.assertAlmostEqual(r['fpga']['metrics']['gmac_s'],2/.3)
        self.assertIn('chip estimate',r['fpga']['power_method'])
        self.assertIn('measured board',r['cpu']['power_method'])

    def test_labeled_accuracy_is_measured_separately_from_head_agreement(self):
        truth=np.array([[0,1],[0,1]],np.uint8);np.save(self.root/'truth.npy',truth)
        np.save(self.root/'pl/labels.npy',truth);np.save(self.root/'pl/head.npy',truth.astype(np.int8)[None,None])
        p=self.root/'pl/benchmark.json';report=json.loads(p.read_text())
        report['artifacts']['case'].update(head_file_sha256=digest(self.root/'pl/head.npy'),labels_file_sha256=digest(self.root/'pl/labels.npy'))
        self.write('pl/benchmark.json',report)
        for backend in ('cpu-int8','pl'):
            p=self.write('pairs.json',dict(classes=2,dataset_split='synthetic developer labels',label_scope='raw-grid',
                pairs=[dict(id='case',prediction=backend+'/labels.npy',truth='truth.npy')]))
            self.write(backend+'_score.json',score_pairs(p))
        self.config['records'][0]['accuracy']=dict(cpu='cpu-int8_score.json',fpga='pl_score.json',
            cpu_benchmark_sha256=digest(self.root/'cpu-int8/benchmark.json'),fpga_benchmark_sha256=digest(self.root/'pl/benchmark.json'))
        r=self.prepare()['records'][0]
        self.assertEqual(r['cpu']['metrics']['dice'],0)
        self.assertEqual(r['fpga']['metrics']['dice'],1)
        self.assertEqual(r['fpga']['metrics']['dice_delta_pp'],100)
        self.assertFalse(r['checks']['integer_head_all_exact'])
        self.assertIsNone(r['checks']['inference_acceleration_cpu_int8_over_pl'])

    def test_corrupted_original_array_is_rejected_before_import(self):
        (self.root/'pl/head.npy').write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError,'hash mismatch'):self.prepare()

    def test_modified_latency_summary_cannot_change_speedup_claim(self):
        p=self.root/'pl/benchmark.json';report=json.loads(p.read_text())
        report['summary']['inference']['mean_seconds']=.00001
        self.write('pl/benchmark.json',report)
        with self.assertRaisesRegex(ValueError,'summary differs'):self.prepare()

    def test_missing_runtime_identity_is_not_assumed(self):
        p=self.root/'pl/benchmark.json';r=json.loads(p.read_text());r['provenance'].pop('bit_sha256')
        self.write('pl/benchmark.json',r)
        with self.assertRaises(ValueError):self.prepare()

    def test_snapshot_tampering_nan_and_duplicate_identity_are_hidden(self):
        doc=self.prepare();r=doc['records'][0]
        r['fpga']['metrics']['median_ms']=1
        self.write('metrics.json',doc)
        self.assertEqual(catalog_status(self.root/'metrics.json')['status'],'invalid_evidence')
        r['record_sha256']=record_digest(r);validate_catalog(doc)
        r['fpga']['metrics']['power_w']=float('nan')
        with self.assertRaises(ValueError):record_digest(r)
        r['fpga']['metrics']['power_w']=None;r['record_sha256']=record_digest(r)
        doc['records'].append(r)
        with self.assertRaises(ValueError):validate_catalog(doc)

    def test_exact_identity_and_read_only_hdmi_rendering(self):
        self.prepare();info=dict(model='SKIN',model_sha256='a'*64,runtime_sha256='b'*64)
        selection=for_task(info,self.root/'metrics.json');self.assertEqual(selection['status'],'matched')
        info['runtime_sha256']='f'*64
        self.assertIsNone(for_task(info,self.root/'metrics.json')['record'])
        screen=np.full((480,640,3),100,np.uint8);before=screen.copy()
        result=draw_comparison(screen,selection,y=48)
        self.assertTrue(np.array_equal(screen,before))
        self.assertTrue(np.array_equal(result[:48],screen[:48]))
        self.assertFalse(np.shares_memory(screen,result))


if __name__=='__main__':unittest.main()
