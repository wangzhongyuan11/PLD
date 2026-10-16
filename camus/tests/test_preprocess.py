import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from camus.preprocess import PreprocessError, load_gray_frame, preprocess


class PreprocessTests(unittest.TestCase):
    def test_image_and_uint16_clipping_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            png = root / "gray.png"
            Image.fromarray(np.arange(16, dtype=np.uint8).reshape(4, 4), mode="L").save(png)
            loaded = load_gray_frame(png)
            self.assertEqual(loaded.data.shape, (4, 4))
            npy = root / "gray.npy"
            np.save(npy, np.asarray([0, 255, 256, 65535, 300, 400], dtype=np.uint16).reshape(2, 3))
            result = preprocess(npy)
            self.assertEqual((result.clipped_low, result.clipped_high), (0, 4))
            self.assertEqual(result.input_int8.dtype, np.dtype(np.int8))

    def test_mhd_2d_msb_and_spacing(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            display = np.asarray([[0, 256, 7], [512, 65535, 8]], dtype=">u2")
            (root / "frame.raw").write_bytes(display.tobytes())
            mhd = root / "frame.mhd"
            mhd.write_text(
                "NDims = 2\nDimSize = 3 2\nElementType = MET_USHORT\n"
                "BinaryDataByteOrderMSB = True\nElementSpacing = 0.5 0.25\n"
                "CompressedData = False\nElementDataFile = frame.raw\n",
                encoding="ascii", newline="\n",
            )
            loaded = load_gray_frame(mhd)
            self.assertEqual(loaded.data.tolist(), [[0, 256, 7], [512, 65535, 8]])
            self.assertEqual(loaded.spacing_yx_mm, (0.25, 0.5))
            result = preprocess(mhd)
            self.assertEqual((result.clipped_low, result.clipped_high), (0, 3))

    def test_mhd_3d_z1_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "frame.raw").write_bytes(np.zeros((2, 3, 1), dtype="u1").tobytes())
            mhd = root / "frame.mhd"
            mhd.write_text(
                "NDims = 3\nDimSize = 3 2 1\nElementType = MET_UCHAR\n"
                "BinaryDataByteOrderMSB = False\nElementDataFile = frame.raw\n",
                encoding="ascii", newline="\n",
            )
            loaded = load_gray_frame(mhd)
            self.assertEqual(loaded.data.shape, (2, 3))

    def test_mhd_rejects_compression_multiframe_channels_traversal_and_length(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "frame.raw").write_bytes(np.zeros(6, dtype="u1").tobytes())
            base = {
                "NDims": "2", "DimSize": "3 2", "ElementType": "MET_UCHAR",
                "BinaryDataByteOrderMSB": "False", "ElementDataFile": "frame.raw",
            }
            cases = [
                {"CompressedData": "True"},
                {"DimSize": "3 2 2", "NDims": "3"},
                {"ElementNumberOfChannels": "3"},
                {"ElementDataFile": "../frame.raw"},
                {"ElementType": "MET_DOUBLE"},
                {"BinaryDataByteOrderMSB": None},
            ]
            for changes in cases:
                values = dict(base)
                for key, value in changes.items():
                    if value is None:
                        values.pop(key)
                    else:
                        values[key] = value
                path = root / ("bad" + str(len(values)) + changes.get("ElementType", "") + ".mhd")
                path.write_text("".join(f"{k} = {v}\n" for k, v in values.items()), encoding="ascii")
                with self.assertRaises(PreprocessError, msg=repr(changes)):
                    load_gray_frame(path)

            short = root / "short.mhd"
            values["ElementDataFile"] = "frame.raw"
            short.write_text(
                "NDims = 2\nDimSize = 4 4\nElementType = MET_UCHAR\n"
                "BinaryDataByteOrderMSB = False\nElementDataFile = frame.raw\n",
                encoding="ascii",
            )
            with self.assertRaises(PreprocessError):
                load_gray_frame(short)

    def test_letterbox_geometry_and_quantized_corners(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "a.npy"
            frame = np.zeros((100, 300), dtype=np.uint8)
            frame[0, 0] = 255
            np.save(path, frame)
            result = preprocess(path)
            meta = result.letterbox_meta
            self.assertEqual((meta["new_height"], meta["new_width"]), (85, 256))
            self.assertEqual((meta["top"], meta["left"]), (85, 0))
            self.assertEqual(result.input_int8.shape, (1, 1, 256, 256))
            self.assertGreaterEqual(int(result.input_int8.min()), -127)
            self.assertLessEqual(int(result.input_int8.max()), 127)


if __name__ == "__main__":
    unittest.main()
