import json
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile
from pathlib import Path

import nibabel as nib
import numpy as np

from acdc.backend import DemoBackend,RemoteBackend
from acdc.contract import ARCHITECTURE,PROFILE,PREPROCESSING,ContractError
from acdc.protocol import encode,decode,validate_request
from acdc.pipeline import Pipeline
from acdc.volume import make_phantom
from acdc.worker import make_worker


def request(backend):
    return dict(architecture_id=ARCHITECTURE,execution_profile=PROFILE,preprocessing_id=PREPROCESSING,
        run_id='test',orientation='axial',phase='ED',center_index=0,neighbor_indices=[0,0,1],total_slices=2,
        model_sha256=backend.model_sha,geometry_sha256='a'*64,shape=[1,3,256,256],dtype='int8',layout='NCHW',input_scale=1/127)


class IntegrationTest(unittest.TestCase):
    def test_remote_idle_and_transport_failure_do_not_poison_next_job(self):
        backend=DemoBackend();server=make_worker(backend,0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        remote=None
        try:
            remote=RemoteBackend('http://127.0.0.1:'+str(server.server_port),model_sha=backend.model_sha,
                                 head_scale=backend.head_scale,synthetic=True)
            self.assertIsNone(remote.connection.sock,'health socket must not survive volume preparation')
            q=np.zeros((1,3,256,256),np.int8)
            remote.infer(q,request(backend))
            old_socket=remote.connection.sock
            remote._last_response-=31
            remote.infer(q,dict(request(backend),run_id='after-idle'))
            self.assertIsNot(remote.connection.sock,old_socket)
            with patch.object(remote.connection,'send',side_effect=ConnectionAbortedError('peer closed')):
                with self.assertRaises(ConnectionAbortedError):
                    remote.infer(q,dict(request(backend),run_id='uncertain-delivery'))
            self.assertIsNone(remote.connection.sock)
            actual,_=remote.infer(q,dict(request(backend),run_id='new-user-job'))
            expected,_=backend.infer(q,request(backend))
            np.testing.assert_array_equal(actual,expected)
        finally:
            if remote is not None:remote.close()
            server.shutdown();server.server_close();thread.join(2)

    def test_production_worker_must_have_certified_runtime_identity(self):
        class Uncertified(DemoBackend):synthetic=False;kind='pl'
        backend=Uncertified();server=make_worker(backend,0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with self.assertRaises(ContractError):RemoteBackend('http://127.0.0.1:'+str(server.server_port),
                model_sha=backend.model_sha,head_scale=backend.head_scale,synthetic=False)
        finally:server.shutdown();server.server_close();thread.join(2)

    def test_exchange_integrity_and_profile(self):
        b=DemoBackend();meta=request(b);validate_request(meta)
        raw=bytes(196608);packet=encode(meta,raw);m,p=decode(packet,len(raw))
        self.assertEqual(p,raw)
        with self.assertRaises(ContractError):decode(packet[:-1]+b'\x01',len(raw))
        meta['neighbor_indices']=[0,0,0]
        with self.assertRaises(ContractError):validate_request(meta)

    def test_remote_worker_keeps_provenance(self):
        class VerboseDemo(DemoBackend):
            def infer(self,*args,**kwargs):
                head,meta=super().infer(*args,**kwargs)
                meta['events']=[dict(command=4,compute_cycles=100,bytes=256,padding='x'*100) for _ in range(1000)]
                return head,meta
        backend=VerboseDemo();server=make_worker(backend,0)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            remote=RemoteBackend('http://127.0.0.1:'+str(server.server_port),model_sha=backend.model_sha,
                                 head_scale=backend.head_scale,synthetic=True)
            q=np.zeros((1,3,256,256),np.int8);a,meta=remote.infer(q,request(backend))
            expected,_=backend.infer(q,request(backend))
            np.testing.assert_array_equal(a,expected);self.assertTrue(meta['synthetic']);self.assertEqual(meta['backend'],'demo')
            self.assertEqual(meta['event_summary']['commands'],1000)
            self.assertNotIn('events',meta)
            _,repeated=remote.infer(q,request(backend))
            self.assertTrue(repeated['cached_response'])
            remote.close()
        finally:server.shutdown();server.server_close();thread.join(2)

    def test_complete_pair_exports_native_geometry_and_surfaces(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);make_phantom(root/'input')
            backend=DemoBackend()
            result=Pipeline(backend).run(root/'input/ED.nii.gz',root/'input/ES.nii.gz',root/'result')
            self.assertEqual(result['status'],'completed')
            self.assertEqual(result['completed_slices'],58)
            for phase in ('ED','ES'):
                original=nib.load(root/f'input/{phase}.nii.gz');out=nib.load(root/f'result/labels/{phase}.nii.gz')
                self.assertEqual(out.shape,original.shape);np.testing.assert_allclose(out.affine,original.affine)
                self.assertTrue(set(np.unique(out.get_fdata()))<=set(range(4)))
                meshes=json.loads((root/f'result/meshes/{phase}/meshes.json').read_text())
                self.assertEqual(len(meshes),3);self.assertTrue(all(not m['smoothed'] for m in meshes))
                for mesh in meshes:self.assertTrue((root/f'result/meshes/{phase}'/mesh['ply_file']).is_file())
            report=json.loads((root/'result/performance.json').read_text())
            self.assertFalse(report['target_4fps_passed']);self.assertFalse(report['board_validation'])
            self.assertLessEqual(report['measured_slice_throughput_fps'],report['slice_service_rate_fps'])
            self.assertGreaterEqual(report['fresh_inference_wall_seconds'],sum(r['seconds'] for r in report['records']))
            self.assertEqual(report['phases']['ED']['resized_planes'],29)
            with zipfile.ZipFile(root/'result/result.zip') as z:
                self.assertIsNone(z.testzip());self.assertIn('labels/ED.nii.gz',z.namelist())
                self.assertTrue(json.loads(z.read('manifest.json'))['synthetic'])
                self.assertIn('cardiac_function.csv',z.namelist())
                self.assertFalse(any(n.startswith('checkpoint/') for n in z.namelist()))
            # Resume reads byte- and model-bound heads; reused responses must not count as new inference/FPS.
            resumed=Pipeline(backend).run(root/'input/ED.nii.gz',root/'input/ES.nii.gz',root/'resumed',resume_from=root/'result')
            self.assertEqual(resumed['status'],'completed')
            rr=json.loads((root/'resumed/performance.json').read_text())
            self.assertEqual(rr['resumed_count'],58);self.assertEqual(rr['fresh_inference_count'],0)
            self.assertIsNone(rr['measured_slice_throughput_fps'])

    def test_cancel_does_not_report_completed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);make_phantom(root/'in');cancel=threading.Event();cancel.set()
            with self.assertRaises(InterruptedError):Pipeline(DemoBackend()).run(
                root/'in/ED.nii.gz',root/'in/ES.nii.gz',root/'out',cancel=cancel)
            self.assertEqual(json.loads((root/'out/status.json').read_text())['status'],'cancelled')
