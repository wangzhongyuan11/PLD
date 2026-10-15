import contextlib, importlib.util, io, json, sys, threading, types, unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace as NS

ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def inference_module():
    mods={}
    for name in ['reference','reference.tinyunet','reference.preprocess','reference.parallel_resize','tensor_accel','pl_graph']:
        mods[name]=types.ModuleType(name)
    for name,attrs in {
      'reference.tinyunet':{'Model':Mock(),'DEFAULT_MODEL_ROOT':Path('model'),'digest':Mock()},
      'reference.preprocess':dict.fromkeys(['decode_rgb','letterbox_rgb','letterbox_int8','quantize_input'],Mock()),
      'reference.parallel_resize':{'ParallelBilinear':Mock()},
      'tensor_accel':{'TileAccelerator':Mock()},'pl_graph':{'PLGraph':Mock()}}.items():
        for key,value in attrs.items():setattr(mods[name],key,value)
    with patch.dict(sys.modules,mods):return load('reuse_inference',ROOT/'board/infer_images.py')

class Device:
    def __init__(self):
        self.closed=False;self.events=[]
        self.timing={'init_stages':{'cold':.1},'close_stages':{},'elapsed':{'tile_accelerator_init':.1}}
    def __enter__(self):return self
    def __exit__(self,*args):self.close()
    def close(self):self.closed=True

class SessionTests(unittest.TestCase):
    def setUp(self):
        self.m=inference_module();self.m.Model=Mock(side_effect=lambda p:object())
        self.m.TileAccelerator=Mock(side_effect=lambda *a,**kw:Device())
        self.s=self.m.InferenceSession()
        self.args=NS(model=Path('model'),bit=Path('a.bit'),native_copy=None,native_control=None,tx_range_sync=True,native_resize=None,parallel_resize=False)
    def acquire(self):return self.s.resources(self.args,self.m.StageTimer())
    def tearDown(self):self.s.close()
    def test_repeat_reuses_model_device_and_resets_timing(self):
        with self.acquire() as first:pass
        with self.acquire() as second:
            self.assertEqual(first,second);self.assertEqual(second[2].timing['init_stages'],{})
        self.assertEqual(self.m.Model.call_count,1);self.assertEqual(self.m.TileAccelerator.call_count,1)
        self.assertEqual(self.s.last_reuse,{'model_reused':True,'hardware_reused':True})
    def test_hdmi_release_keeps_model_and_rebuilds_device(self):
        with self.acquire() as first:pass
        self.s.release_hardware();self.assertTrue(first[2].closed)
        with self.acquire() as second:
            self.assertIs(first[0],second[0]);self.assertIsNot(first[2],second[2])
        self.assertEqual(self.m.Model.call_count,1);self.assertEqual(self.m.TileAccelerator.call_count,2)
        self.assertEqual(self.s.last_reuse,{'model_reused':True,'hardware_reused':False})
    def test_inference_failure_poison_and_cleanup(self):
        with self.assertRaises(ValueError):
            with self.acquire() as resources:raise ValueError('failed')
        self.assertTrue(resources[2].closed);self.assertIsNone(self.s.model)
        with self.assertRaises(RuntimeError):
            with self.acquire():pass
    def test_partial_initialization_closes_resizer(self):
        self.args.parallel_resize=True;resizer=Device();self.m.ParallelBilinear=Mock(return_value=resizer)
        self.m.TileAccelerator.side_effect=ValueError('init failed')
        with self.assertRaises(ValueError):
            with self.acquire():pass
        self.assertTrue(resizer.closed);self.assertTrue(self.s.failed)
    def test_release_failure_poison(self):
        with self.acquire() as resources:pass
        resources[2].close=Mock(side_effect=RuntimeError('reset failed'))
        with self.assertRaises(RuntimeError):self.s.release_hardware()
        self.assertTrue(self.s.failed);self.assertIsNone(self.s.model)
    def test_active_release_rejected(self):
        with self.acquire():
            with self.assertRaises(RuntimeError):self.s.release_hardware()
    def test_options_change_recreates_hardware(self):
        with self.acquire() as first:pass
        self.args.bit=Path('b.bit')
        with self.acquire() as second:self.assertIsNot(first[2],second[2])
        self.assertTrue(first[2].closed);self.assertEqual(self.m.Model.call_count,1)
    def test_model_path_change_recreates_both(self):
        with self.acquire() as first:pass
        self.args.model=Path('model2')
        with self.acquire() as second:self.assertIsNot(first[0],second[0])
        self.assertTrue(first[2].closed)

class MainFlowTests(unittest.TestCase):
    def test_cli_warm_and_after_hdmi_outputs_and_lifetimes(self):
        import tempfile, numpy as np
        m=inference_module();devices=[]
        def device_factory(*a,**kw):
            d=Device();d.abi_version=1;d.copy_info={};d.control_info={};d.tx_sync_info={};devices.append(d);return d
        m.TileAccelerator=device_factory
        m.Model=Mock(side_effect=lambda p:NS(layers=[{'input_scale':1}],spec_hash='model'))
        m.digest=lambda p:'hash'
        m.decode_rgb=lambda p:NS(close=lambda:None)
        geometry=dict(left=0,top=0,resized_width=2,resized_height=2)
        m.letterbox_int8=lambda *a,**kw:(np.ones((1,3,2,2),np.int8),np.ones((2,2),bool),geometry)
        class Graph:
            def __init__(self,*a,**kw):pass
            def run(self,data,**kw):return data.copy(),{'result':{},'seconds':.01}
        m.PLGraph=Graph
        with tempfile.TemporaryDirectory() as td,contextlib.redirect_stdout(io.StringIO()):
            root=Path(td);session=m.InferenceSession()
            def run(n,cache):
                dest=root/str(n)
                m.main(['input.jpg','--model',str(root/'model'),'--bit',str(root/'a.bit'),'--output',str(dest),'--fast-quantize'],session=cache)
                return (dest/'frame_000001.bin').read_bytes(),json.loads((dest/'frame_000001.json').read_text())
            baseline=run(0,None);self.assertTrue(devices[0].closed)
            unused=root/'warmup'
            m.main(['no-image.jpg','--model',str(root/'model'),'--bit',str(root/'a.bit'),'--output',str(unused),'--fast-quantize','--warmup-only'],session=session)
            self.assertFalse(unused.exists())
            self.assertFalse(devices[-1].closed)
            first=run(1,session);second=run(2,session)
            self.assertEqual(baseline[0],first[0]);self.assertEqual(first[0],second[0])
            self.assertEqual(second[1]['session_reuse'],{'model_reused':True,'hardware_reused':True})
            self.assertEqual(second[1]['b_timing']['accelerator_init_stage_seconds'],{})
            self.assertFalse(devices[-1].closed)
            session.release_hardware();self.assertTrue(devices[-1].closed)
            third=run(3,session)
            self.assertEqual(third[0],first[0]);self.assertEqual(third[1]['session_reuse'],{'model_reused':True,'hardware_reused':False})
            session.close();self.assertTrue(devices[-1].closed)
            self.assertEqual(m.Model.call_count,2) # one CLI model + one cached worker model


class ProtocolTests(unittest.TestCase):
    def test_release_ack_only_after_close_and_no_inference(self):
        import tempfile
        w=load('reuse_worker',ROOT/'board/infer_worker.py');s=NS(device=object(),last_reuse={})
        def release():s.device=None
        s.release_hardware=release;s.close=Mock();infer=Mock()
        with tempfile.TemporaryDirectory() as d,patch.object(w,'emit') as emit:
            self.assertEqual(w.serve_request(infer,{'request_id':'abc','action':'release','argv':['release'],'log_path':str(Path(d)/'r.log')},s),0)
            infer.assert_not_called();self.assertTrue(emit.call_args.kwargs['device_released'])
    def test_release_failure_never_success(self):
        import tempfile
        w=load('reuse_worker',ROOT/'board/infer_worker.py');s=NS(release_hardware=Mock(side_effect=RuntimeError('fail')),close=Mock(),last_reuse={})
        with tempfile.TemporaryDirectory() as d,patch.object(w,'emit') as emit:
            self.assertEqual(w.serve_request(Mock(),{'request_id':'abc','action':'release','argv':['release'],'log_path':str(Path(d)/'r.log')},s),1)
            self.assertEqual(emit.call_args.args[0],'FATAL')
    def supervisor(self,ack):
        import persistent_worker as p
        s=p.PersistentBWorker.__new__(p.PersistentBWorker);s._lock=threading.RLock();s.close_reason=None
        s.process=NS(stdin=io.BytesIO(),poll=lambda:None,pid=12);s.check_manifest=Mock();s._ensure_started=Mock(return_value=False)
        s._wait_event=Mock(return_value=('RESULT',{'status':'success','device_released':ack}));s.request_timeout=180
        s._stop=Mock();s._write_log_failure=Mock();return s
    def test_missing_release_ack_blocks(self):
        from persistent_worker import PersistentWorkerError
        s=self.supervisor(False)
        with self.assertRaises(PersistentWorkerError):s.release_hardware('release.log')
        s._stop.assert_called_once()
    def test_release_sent_as_protocol_action(self):
        s=self.supervisor(True);s.release_hardware('release.log')
        self.assertEqual(json.loads(s.process.stdin.getvalue().split(b' ',1)[1])['action'],'release')
    def test_dead_cached_worker_never_restarted(self):
        from persistent_worker import PersistentWorkerError,PersistentBWorker
        s=self.supervisor(True);s.process.poll=lambda:1
        with self.assertRaises(PersistentWorkerError):PersistentBWorker._ensure_started(s)

if __name__=='__main__':unittest.main()
