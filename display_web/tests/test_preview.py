import io,json,threading,unittest,zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
import server
import test_web

class PreviewTests(unittest.TestCase):
 setUp=test_web.WebTests.setUp
 tearDown=test_web.WebTests.tearDown
 request=test_web.WebTests.request
 wait=test_web.WebTests.wait

 def test_first_screen_does_not_wait_for_remaining_previews(self):
  entered=threading.Event();release=threading.Event();original=server.save_lossless_png
  def save(array,path):
   if path.name=='mask.png':
    entered.set()
    if not release.wait(6):raise TimeoutError('blocked')
   original(array,path)
  rgb=np.full((30,40,3),100,np.uint8);mask=np.ones((30,40),np.uint8)
  with patch('server.save_lossless_png',side_effect=save):
   j=self.app.submit({'mode':'mask','image':test_web.encoded(rgb),'mask':test_web.encoded(mask)})
   try:
    self.assertTrue(entered.wait(5));j=self.app.job(j['id'])
    self.assertEqual(j['preview_views'],['original','outlined'])
    self.assertEqual(self.request('/files/'+j['id']+'/preview_outlined.png')[0],200)
    self.assertEqual(self.request('/files/'+j['id']+'/preview_mask.png')[0],404)
    self.assertEqual(j['state'],'running')
   finally:release.set()
   j=self.wait(j['id'])
  self.assertEqual(j['state'],'done');self.assertIn('mask',j['preview_views'])

 def test_preview_access_before_export_and_exact_full_outputs(self):
  rgb=np.random.default_rng(4).integers(0,256,(600,1200,3),dtype=np.uint8)
  mask=np.zeros((600,1200),np.uint8);mask[100:500,200:900]=255
  entered=threading.Event();release=threading.Event();original=server.save_lossless_png
  def save(array,path):
   if path.name=='comparison.png':
    entered.set()
    if not release.wait(6):raise TimeoutError('test blocked too long')
   original(array,path)
  with patch('server.save_lossless_png',side_effect=save):
   j=self.app.submit({'mode':'mask','image':test_web.encoded(rgb),'mask':test_web.encoded(mask)})
   try:
    self.assertTrue(entered.wait(5));ident=j['id'];j=self.app.job(ident)
    self.assertEqual(j['state'],'running');self.assertTrue(j['preview_ready']);self.assertFalse(j['package_ready'])
    code,raw=self.request('/files/'+ident+'/preview_outlined.png');self.assertEqual(code,200)
    self.assertEqual(Image.open(io.BytesIO(raw)).size,(1200,600))
    self.assertEqual(raw,(Path(self.temp.name)/ident/'outlined.png').read_bytes())
    self.assertEqual(self.request('/files/'+ident+'/preview_outlined.png',token=False)[0],401)
    for name in ['original.png','results.zip','report.json','preview_comparison.png','input.jpg']:
     self.assertEqual(self.request('/files/'+ident+'/'+name)[0],404)
    self.assertEqual(self.app.history()['items'],[])
    self.assertEqual(self.app.status()['active'],ident)
    with self.assertRaises(RuntimeError):self.app.submit({'mode':'demo'})
    self.app.board=True
    with self.assertRaises(RuntimeError):self.app.start_hdmi(ident,'outlined')
   finally:release.set()
   j=self.wait(j['id'])
  self.assertEqual(j['state'],'done');self.assertTrue(j['package_ready'])
  self.assertGreater(j['statistics']['request_to_job_complete_seconds'],j['statistics']['preview_ready_seconds'])
  folder=Path(self.temp.name)/j['id']
  expected=server.process(mask,rgb)
  for key in ['mask','contour','outlined','cutout']:
   a=expected[key];a=a.astype(np.uint8)*255 if a.dtype==bool else a
   np.testing.assert_array_equal(np.asarray(Image.open(folder/(key+'.png'))),a)
  np.testing.assert_array_equal(np.asarray(Image.open(folder/'original.png')),rgb)
  with zipfile.ZipFile(folder/'results.zip') as z:
   self.assertIsNone(z.testzip());self.assertFalse(any(n.startswith('preview_') for n in z.namelist()))
  restored=server.Console(self.temp.name,token='test',boot_id=self.app.boot_id)
  self.assertTrue(restored.job(j['id'])['preview_ready'])

 def test_export_failure_keeps_preview_but_never_offers_zip(self):
  original=server.save_lossless_png
  def save(array,path):
   if path.name=='comparison.png':raise OSError('simulated disk full')
   original(array,path)
  with patch('server.save_lossless_png',side_effect=save):
   j=self.wait(self.app.submit({'mode':'demo'})['id'])
  self.assertEqual(j['state'],'error');self.assertTrue(j['preview_ready']);self.assertFalse(j['package_ready'])
  self.assertEqual(self.request('/files/'+j['id']+'/preview_mask.png')[0],200)
  self.assertEqual(self.request('/files/'+j['id']+'/results.zip')[0],404)
  self.assertEqual(self.app.history()['items'],[]);self.assertIsNone(self.app.active)
  restored=server.Console(self.temp.name,token='test',boot_id=self.app.boot_id)
  self.assertEqual(restored.history()['items'],[])

 def test_preview_failure_never_publishes_partial_previews(self):
  original=server.save_lossless_png
  def save(array,path):
   if path.name=='outlined.png':raise OSError('preview failed')
   original(array,path)
  with patch('server.save_lossless_png',side_effect=save):
   j=self.wait(self.app.submit({'mode':'demo'})['id'])
  self.assertEqual(j['state'],'error');self.assertFalse(j.get('preview_ready'))
  self.assertEqual(self.request('/files/'+j['id']+'/preview_mask.png')[0],404)
