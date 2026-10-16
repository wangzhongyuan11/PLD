import base64,io,json,tempfile,unittest,hashlib,shutil
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageOps
import server
from pld_result import load_result

class UploadDecodeTests(unittest.TestCase):
 def test_exact_old_pixels_for_formats_modes_and_exif(self):
  rng=np.random.default_rng(9)
  for mode in ['RGB','L','RGBA']:
   shape=(21,37,3) if mode=='RGB' else (21,37,4) if mode=='RGBA' else (21,37)
   im=Image.fromarray(rng.integers(0,256,shape,dtype=np.uint8))
   for orientation in range(1,9):
    for fmt in (['PNG','JPEG'] if mode!='RGBA' else ['PNG']):
     exif=Image.Exif();exif[274]=orientation
     out=io.BytesIO();im.save(out,format=fmt,exif=exif)
     raw=out.getvalue()
     with Image.open(io.BytesIO(raw)) as original:expected=np.asarray(ImageOps.exif_transpose(original).convert('RGB')).copy()
     _,actual=server.decode_upload(base64.b64encode(raw).decode())
     np.testing.assert_array_equal(actual,expected)
 def test_cached_result_skips_decode_but_checks_file_hash_and_dimensions(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);samples=Path(server.ROOT)/'samples'
   meta=json.loads((samples/'frame_000001.json').read_text())
   g=meta['input_metadata'];w,h=g['original_width'],g['original_height']
   rgb=np.full((h,w,3),100,np.uint8);image=root/'input.jpg';Image.fromarray(rgb).save(image)
   g['image_sha256']=hashlib.sha256(image.read_bytes()).hexdigest()
   (root/'frame.json').write_text(json.dumps(meta));shutil.copyfile(samples/'frame_000001.bin',root/'frame.bin')
   _,decoded=server.decode_upload(base64.b64encode(image.read_bytes()).decode())
   expected=load_result(root/'frame.json',image)
   with patch('pld_result.read_rgb',side_effect=AssertionError('must not decode twice')):
    result=load_result(root/'frame.json',image,decoded_rgb=decoded)
   np.testing.assert_array_equal(result.rgb,expected.rgb)
   with self.assertRaises(ValueError):load_result(root/'frame.json',image,decoded_rgb=decoded[:1,:1])
   with self.assertRaises(ValueError):load_result(root/'frame.json',decoded_rgb=decoded)
   image.write_bytes(b'wrong source')
   with self.assertRaises(ValueError):load_result(root/'frame.json',image,decoded_rgb=decoded)
