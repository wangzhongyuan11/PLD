import importlib.util
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'display_web/unified'))
HAS_NUMPY=importlib.util.find_spec('numpy') is not None
if HAS_NUMPY:
    import numpy as np
    # These tests mock pipeline.run; no postprocessing or SciPy kernel runs.
    if importlib.util.find_spec('scipy') is None:
        sys.modules['scipy']=SimpleNamespace(ndimage=Mock())
        try:
            from camus import web,preprocess
        finally:
            sys.modules.pop('scipy',None)
    else:
        from camus import web,preprocess
    import camus_input_adapter


@unittest.skipUnless(HAS_NUMPY,'CAMUS adapter tests require NumPy and Pillow')
class InputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.old_load=preprocess.load_gray_frame
        self.old_numeric=preprocess._load_numeric;self.old_execute=web.CamusServer.execute
        camus_input_adapter.install(web)
    def tearDown(self):
        preprocess.load_gray_frame=self.old_load
        preprocess._load_numeric=self.old_numeric;web.CamusServer.execute=self.old_execute
        self.temp.cleanup()
    def cache(self,path,native):
        np.savez(path,native_image=native,image=np.zeros((256,256),np.float32),valid=np.ones((256,256),np.float32),
                 meta=np.array('must not unpickle',dtype=object),quality_source=np.array('private',dtype=object),
                 spacing_mm=np.array([.3,.2]),affine=np.eye(4),native_label=np.ones(native.shape,np.int16),label=np.ones((256,256),np.int16))
    def server(self,path):
        pipeline=Mock(backend='pl');pipeline.run.return_value={'output_dir':'unused','report':{},'backend':'pl'}
        server=SimpleNamespace(data=self.root,pipeline=pipeline,fault=False,released=False,guard=threading.Lock(),busy=threading.BoundedSemaphore(1),stop_hdmi=Mock())
        folder=self.root/'job/input';folder.mkdir(parents=True,exist_ok=True)
        (folder/path.name).write_bytes(path.read_bytes())
        server.busy.acquire();job={'id':'job','state':'queued','_started':time.perf_counter()}
        return server,job
    def test_cache_reads_native_not_canvas_or_labels(self):
        raw=np.arange(256,dtype=np.uint8).reshape(16,16)
        p=self.root/'cache.npz';self.cache(p,raw.astype(np.float32)/255)
        original=self.root/'original.npy';np.save(original,raw)
        frame=preprocess.load_gray_frame(p)
        self.assertEqual(frame.data.shape,raw.shape);self.assertEqual(frame.spacing_yx_mm,(.3,.2))
        self.assertEqual(frame.source_kind,'role_a_cache_native')
        np.testing.assert_array_equal(preprocess.preprocess(p).input_int8,preprocess.preprocess(original).input_int8)
    def test_bad_upload_does_not_fault_or_close_model(self):
        p=self.root/'bad.npz';np.savez(p,a=np.zeros((2,2)),b=np.ones((2,2)))
        s,j=self.server(p);web.CamusServer.execute(s,j,p.name,None)
        self.assertFalse(s.fault);self.assertEqual(j['error_kind'],'input')
        s.pipeline.run.assert_not_called();s.pipeline.close.assert_not_called();s.stop_hdmi.assert_not_called()
        self.assertTrue(s.busy.acquire(blocking=False))
    def test_followup_valid_job_after_rejection(self):
        p=self.root/'bad.npz';np.savez(p,a=np.zeros((2,2)),b=np.ones((2,2)))
        s,j=self.server(p);web.CamusServer.execute(s,j,p.name,None)
        np.save(self.root/'job/input/good.npy',np.zeros((2,2),np.uint8));s.busy.acquire()
        follow={'id':'job','state':'queued','_started':time.perf_counter()}
        web.CamusServer.execute(s,follow,'good.npy',None)
        self.assertEqual(follow['state'],'done');self.assertFalse(s.fault)
    def test_hardware_failure_still_faults(self):
        p=self.root/'good.npy';np.save(p,np.zeros((2,2),np.uint8));s,j=self.server(p)
        s.pipeline.run.side_effect=RuntimeError('DMA timeout')
        web.CamusServer.execute(s,j,p.name,None)
        self.assertTrue(s.fault);s.pipeline.close.assert_called_once()
    def test_decoded_input_reused_once_and_released(self):
        p=self.root/'good.npy';np.save(p,np.zeros((2,2),np.uint8));s,j=self.server(p)
        def run(path,*args,**kwargs):
            preprocess.load_gray_frame(path)
            return {'output_dir':'unused','report':{},'backend':'pl'}
        s.pipeline.run.side_effect=run
        with patch.object(preprocess,'_load_numeric',wraps=preprocess._load_numeric) as reader:
            web.CamusServer.execute(s,j,p.name,None)
            self.assertEqual(reader.call_count,1)
            preprocess.load_gray_frame(self.root/'job/input/good.npy')
            self.assertEqual(reader.call_count,2)
        self.assertEqual(j['state'],'done')

    def test_invalid_normalization_and_spacing_rejected(self):
        p=self.root/'cache.npz';self.cache(p,np.full((2,2),255,np.float32))
        with self.assertRaises(preprocess.PreprocessError):preprocess.load_gray_frame(p)
        self.cache(p,np.zeros((2,2),np.float32));s,j=self.server(p)
        web.CamusServer.execute(s,j,p.name,[1,1]);self.assertEqual(j['error_kind'],'input');self.assertFalse(s.fault)


if __name__=='__main__':unittest.main()
