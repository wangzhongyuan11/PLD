# -*- coding: utf-8 -*-
import gzip
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from nifti import write_nii_gz, cardiac_phantom  # noqa: E402

try:
    import nibabel as nib
    NIBABEL = True
except ImportError:
    NIBABEL = False


class NiftiWriterTests(unittest.TestCase):
    def test_header_and_data_roundtrip(self):
        vol = cardiac_phantom((12, 10, 6), scale=1.0, seed=7)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ED.nii.gz"
            write_nii_gz(path, vol, spacing=(2.0, 2.0, 2.0))
            raw = gzip.decompress(path.read_bytes())
        self.assertEqual(struct.unpack_from("<i", raw, 0)[0], 348)
        self.assertEqual(raw[344:348], b"n+1\x00")
        dims = struct.unpack_from("<8h", raw, 40)
        self.assertEqual(dims[:4], (3, 12, 10, 6))
        self.assertEqual(raw[70], 16)  # float32
        pixdim = struct.unpack_from("<8f", raw, 76)
        self.assertEqual(pixdim[1:4], (2.0, 2.0, 2.0))
        self.assertEqual(struct.unpack_from("<f", raw, 108)[0], 352.0)
        self.assertEqual(len(raw), 352 + 12 * 10 * 6 * 4)
        # center voxel = LV cavity (bright), first corner = background (dark)
        def voxel(x, y, z):
            offset = 352 + (x + y * 12 + z * 12 * 10) * 4  # x fastest per spec
            return struct.unpack_from("<f", raw, offset)[0]
        self.assertGreater(voxel(6, 5, 3), 0.8)
        self.assertLess(voxel(0, 0, 0), 0.2)

    def test_ed_larger_than_es(self):
        ed = cardiac_phantom((40, 40, 8), scale=1.0, seed=7)
        es = cardiac_phantom((40, 40, 8), scale=0.6, seed=7)
        bright = lambda v: sum(1 for x in range(40) for y in range(40)
                               for z in range(8) if v[x][y][z] > 0.8)
        self.assertGreater(bright(ed), bright(es))

    @unittest.skipUnless(NIBABEL, "nibabel 不可用时跳过交叉验证")
    def test_readable_by_nibabel(self):
        vol = cardiac_phantom((20, 18, 8), scale=1.0, seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.nii.gz"
            write_nii_gz(path, vol, spacing=(2.0, 2.0, 2.0))
            img = nib.load(str(path))
            self.assertEqual(img.shape, (20, 18, 8))
            self.assertTrue((img.header.get_zooms()[:3] ==
                             (2.0, 2.0, 2.0)))
            data = img.get_fdata()
        self.assertEqual(data.shape, (20, 18, 8))
        self.assertGreater(data[10, 9, 4], 0.8)


if __name__ == "__main__":
    unittest.main()
