import hashlib,json,os,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch,Mock
import numpy as np
from PIL import Image
from server import Console,save_lossless_png
from infer_then_display import RuntimeVerifier

class FastPngTests(unittest.TestCase):
 def test_exact_pixels_random_masks_gradients_and_strided_arrays(self):
  rng=np.random.default_rng(42)
  arrays=[rng.integers(0,256,(93,107,3),dtype=np.uint8),rng.integers(0,256,(37,71),dtype=np.uint8),np.ones((1,1,3),np.uint8)*255,np.zeros((19,1),np.uint8),rng.random((35,21))>.7]
  arrays.append(arrays[0][::2,::2,:])
  with tempfile.TemporaryDirectory() as d,patch('server.PNG_BACKEND','fast_sub'):
   for i,a in enumerate(arrays):
    path=Path(d)/str(i);save_lossless_png(a,path)
    with Image.open(path) as im:
     self.assertEqual(im.format,'PNG');np.testing.assert_array_equal(np.array(im),a.astype(np.uint8)*255 if a.dtype==bool else a)
 def test_original_encoder_fallback(self):
  a=np.arange(75,dtype=np.uint8).reshape(5,5,3)
  with tempfile.TemporaryDirectory() as d,patch('server.PNG_BACKEND','pillow'):
   p=Path(d)/'a.png';save_lossless_png(a,p)
   with Image.open(p) as im:np.testing.assert_array_equal(np.array(im),a)

class CachedVerifierTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
  self.file=self.root/'code.py';self.file.write_bytes(b'abc')
  self.manifest=self.root/'deployment_manifest.json'
  self.manifest.write_text(json.dumps({'files':{'code.py':hashlib.sha256(b'abc').hexdigest()}}))
  self.v=RuntimeVerifier(self.root)
 def tearDown(self):self.tmp.cleanup()
 def test_warm_check_does_not_hash_payload(self):
  self.assertEqual(self.v.check(),'full_sha256')
  with patch('infer_then_display.platform.system',return_value='Linux'), patch('infer_then_display.verify_runtime',side_effect=AssertionError('unexpected full hash')):
   self.assertEqual(self.v.check(),'metadata_unchanged')
 def test_touch_forces_full_check(self):
  self.v.check();s=self.file.stat();os.utime(self.file,ns=(s.st_atime_ns,s.st_mtime_ns+1000000000))
  self.assertEqual(self.v.check(),'full_sha256');self.assertEqual(self.v.full_checks,2)
 def test_changed_same_size_with_restored_mtime_rejected(self):
  self.v.check();s=self.file.stat();self.file.write_bytes(b'xyz');os.utime(self.file,ns=(s.st_atime_ns,s.st_mtime_ns))
  with self.assertRaises(ValueError):self.v.check()
 def test_manifest_change_requires_restart(self):
  self.v.check();self.manifest.write_text(self.manifest.read_text()+' ')
  with self.assertRaises(ValueError):self.v.check()
 def test_missing_file_rejected(self):
  self.v.check();self.file.unlink()
  with self.assertRaises(FileNotFoundError):self.v.check()

class WarmupTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.app=Console(self.tmp.name,token='test',boot_id='boot')
  self.app.board=True;self.app.runtime_ok=True;self.app.persistent_b_worker=True;self.app.runtime=Path(self.tmp.name)
 def tearDown(self):self.app.close();self.tmp.cleanup()
 def test_warmup_gates_requests_and_reuses_single_worker(self):
  entered=threading.Event();proceed=threading.Event();worker=Mock()
  def run(*a,**kw):entered.set();proceed.wait(3)
  worker.run.side_effect=run
  with patch.object(self.app,'verify_deployment'),patch('server.inference_command',return_value=['python','infer.py','unused']),patch('server.PersistentBWorker',return_value=worker) as make:
   self.app.begin_warmup();self.assertTrue(entered.wait(2))
   try:
    self.assertFalse(self.app.status()['inference_available']);self.assertFalse(self.app.status()['hdmi_available'])
    with self.assertRaises(RuntimeError):self.app.submit({'mode':'demo'})
    with self.assertRaises(RuntimeError):self.app.start_hdmi('invalid','mask')
    self.app.begin_warmup();make.assert_called_once()
   finally:proceed.set();self.app.warmup_thread.join(3)
  self.assertEqual(self.app.warmup_state,'ready');self.assertTrue(self.app.status()['inference_available'])
  self.assertIn('--warmup-only',worker.run.call_args.args[0]);self.assertEqual(self.app.jobs,{})
 def test_failure_blocks_hardware(self):
  with patch.object(self.app,'verify_deployment',side_effect=ValueError('bad runtime')):
   self.app.begin_warmup();self.app.warmup_thread.join(2)
  self.assertEqual(self.app.warmup_state,'failed');self.assertFalse(self.app.status()['inference_available'])
  self.assertIsNotNone(self.app.fault)
 def test_local_mode_never_warms(self):
  self.app.board=False;self.app.begin_warmup();self.assertIsNone(self.app.warmup_thread)
