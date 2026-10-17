import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from perf_eval.benchmark import benchmark,write_json
from perf_eval.core import digest
from perf_eval.engines import SPECS

class FakeEngine:
    instances=[]
    def __init__(self,*args,**kwargs):self.calls=0;self.closed=False;self.provenance={'spec_sha256':SPECS['skin']};self.instances.append(self)
    def run(self,image,frame):self.calls+=1;return np.zeros((1,1,256,256),np.int8),[]
    def labels(self,head,valid):return np.zeros((256,256),np.uint8)
    def close(self):self.closed=True

class BenchmarkTests(unittest.TestCase):
    def make(self,root):
        for name,array in [('i',np.zeros((1,3,256,256),np.int8)),('f',np.zeros((1,3,256,256),np.float32)),('v',np.ones((256,256),bool))]:np.save(root/(name+'.npy'),array)
        row=dict(id='case',int8='i.npy',fp32='f.npy',valid='v.npy',synthetic=True,sha256=dict(int8=digest(root/'i.npy'),fp32=digest(root/'f.npy'),valid=digest(root/'v.npy')))
        write_json(root/'dataset.json',dict(schema='pld-prepared-input-v1',model='skin',spec_sha256=SPECS['skin'],cases=[row],label_scope='raw-grid'))
        return argparse.Namespace(dataset=root/'dataset.json',model='skin',backend='cpu-int8',source=root,model_root=root,output=root/'run',checkpoint=None,runtime=None,bit=None,native_copy=None,native_control=None,fuse=False,repeat=2,warmup=1)
    def test_cold_warmup_excluded_and_artifacts_saved(self):
        with tempfile.TemporaryDirectory() as temp:
            args=self.make(Path(temp))
            with patch('perf_eval.benchmark.Engine',FakeEngine):report_path=benchmark(args)
            report=json.loads(report_path.read_text(encoding='utf-8'))
            self.assertEqual(FakeEngine.instances[-1].calls,4);self.assertTrue(FakeEngine.instances[-1].closed)
            self.assertEqual(len(report['frames']),2);self.assertEqual(report['summary']['inference']['frames'],2)
            self.assertEqual(len(report['artifacts']),1);self.assertTrue(report['complete'])
    def test_failed_forward_closes_and_leaves_incomplete_report(self):
        with tempfile.TemporaryDirectory() as temp:
            args=self.make(Path(temp))
            class Failing(FakeEngine):
                def run(self,*args):raise RuntimeError('expected failure')
            with patch('perf_eval.benchmark.Engine',Failing):
                with self.assertRaises(RuntimeError):benchmark(args)
            self.assertTrue(Failing.instances[-1].closed)
            self.assertFalse(json.loads((args.output/'benchmark.json').read_text())['complete'])

if __name__=='__main__':unittest.main()

class CleanupTests(unittest.TestCase):
    make=BenchmarkTests.make
    def test_cleanup_error_never_reports_success(self):
        with tempfile.TemporaryDirectory() as temp:
            args=self.make(Path(temp))
            class FailedClose(FakeEngine):
                def close(self):raise RuntimeError('release unconfirmed')
            with patch('perf_eval.benchmark.Engine',FailedClose):
                with self.assertRaises(RuntimeError):benchmark(args)
            row=json.loads((args.output/'benchmark.json').read_text())
            self.assertFalse(row['complete']);self.assertEqual(row['close_error'],'release unconfirmed')
