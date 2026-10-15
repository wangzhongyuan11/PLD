import hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import contour
from infer_then_display import RuntimeVerifier

class ContourParityTests(unittest.TestCase):
 @unittest.skipIf(contour.cv2 is None,'OpenCV optional; tested on board')
 def test_optional_backend_exact_for_edges_holes_and_strided_input(self):
  rng=np.random.default_rng(18)
  for shape in [(1,1),(1,11),(11,1),(17,23),(51,39)]:
   for typ in ['random','full','empty']:
    m=rng.random(shape)>.25 if typ=='random' else np.full(shape,typ=='full',dtype=bool)
    rgb=rng.integers(0,256,shape+(3,),dtype=np.uint8)
    for conn in (4,8):
     for width in (1,2,3,8,32):
      a=contour.process(m,rgb,conn,width)
      with patch.object(contour,'cv2',None):b=contour.process(m,rgb,conn,width)
      for key in ['mask','contour','outlined','cutout']:np.testing.assert_array_equal(a[key],b[key])
  a=contour.process(m[::2,::2],rgb[::2,::2])
  with patch.object(contour,'cv2',None):b=contour.process(m[::2,::2],rgb[::2,::2])
  np.testing.assert_array_equal(a['cutout'],b['cutout'])

class SymlinkVerifierTests(unittest.TestCase):
 @unittest.skipUnless(os.name=='posix','Linux symlink semantics tested on board')
 def test_parent_and_leaf_links_rechecked_each_call(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d)/'runtime';root.mkdir()
   for name in ('a','b'):
    (root/name).mkdir();(root/name/'x').write_bytes(b'abc')
   (root/'link').symlink_to('a',target_is_directory=True)
   (root/'deployment_manifest.json').write_text(json.dumps({'files':{'link/x':hashlib.sha256(b'abc').hexdigest()}}))
   v=RuntimeVerifier(root);self.assertEqual(v.check(),'full_sha256')
   self.assertEqual(v.check(),'metadata_unchanged')
   (root/'link').unlink();(root/'link').symlink_to('b',target_is_directory=True)
   self.assertEqual(v.check(),'full_sha256')
   outside=Path(d)/'outside';outside.write_bytes(b'abc')
   (root/'b/x').unlink();(root/'b/x').symlink_to(outside)
   with self.assertRaises(ValueError):v.check()
