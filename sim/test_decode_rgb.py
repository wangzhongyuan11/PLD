from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image,ImageOps

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference.preprocess import decode_rgb


class DecodeTests(unittest.TestCase):
    def test_owned_rgb_matches_original_path_for_all_exif_orientations(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'test.jpg'
            pixels=np.arange(7*9*3,dtype=np.uint8).reshape(7,9,3)
            for mode in ('RGB','L'):
                for orientation in range(1,9):
                    source=Image.fromarray(pixels).convert(mode)
                    exif=Image.Exif();exif[274]=orientation
                    source.save(path,exif=exif)
                    with Image.open(path) as original:
                        expected=ImageOps.exif_transpose(original).convert('RGB')
                    actual=decode_rgb(path)
                    self.assertEqual(actual.mode,'RGB')
                    self.assertEqual(actual.size,expected.size)
                    self.assertEqual(actual.tobytes(),expected.tobytes())
                    actual.close();expected.close()


if __name__=='__main__': unittest.main()
