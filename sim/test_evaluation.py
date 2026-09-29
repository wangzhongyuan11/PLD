import tempfile
from pathlib import Path
import unittest

import numpy as np
from PIL import Image

from reference.evaluation import segmentation_metrics, target_letterbox
from reference.preprocess import letterbox_rgb, letterbox_int8, quantize_input


class EvaluationTests(unittest.TestCase):
    def test_valid_region_and_macro_counts(self):
        valid = np.array([[0, 0], [1, 1]], dtype=bool)
        result = segmentation_metrics([[1, 1], [1, 1]], [[0, 0], [1, 0]], valid)
        self.assertEqual(result["dice"], 2/3)
        self.assertEqual(result["iou"], 1/2)
        self.assertEqual(result["valid_pixels"], 2)
        self.assertEqual(segmentation_metrics([[0]], [[0]], [[1]])["dice"], 1)
        with self.assertRaises(ValueError):
            segmentation_metrics([[0]], [[0]], [[0]])

    def test_non_square_geometry_and_label_resize(self):
        image = Image.new("RGB", (4, 2), (255, 0, 0))
        real, valid, meta = letterbox_rgb(image, 8)
        self.assertEqual((meta["left"], meta["top"], meta["resized_width"], meta["resized_height"]), (0, 2, 8, 4))
        self.assertEqual(int(valid.sum()), 32)
        self.assertTrue(np.all(real[0, 0, 2:6] == 1))
        self.assertTrue(np.all(real[:, :, :2] == 0))
        q = quantize_input(real, 1/127)
        self.assertEqual(q.dtype, np.int8)
        self.assertTrue(np.all(q[0, 0, 2:6] == 127))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"target.png"
            Image.fromarray(np.array([[255, 255, 0, 0]]*2, dtype=np.uint8)).save(path)
            target = target_letterbox(path, meta, 8)
            expected = np.zeros((8, 8), dtype=bool)
            expected[2:6, :4] = True
            np.testing.assert_array_equal(target, expected)
            meta["original_width"] = 5
            with self.assertRaises(ValueError):
                target_letterbox(path, meta, 8)

    def test_rgb8_lookup_matches_float32_for_every_byte_and_letterbox_edge(self):
        values=np.arange(256,dtype=np.uint8)
        pixels=np.broadcast_to(values[None,:,None],(53,256,3)).copy()
        pixels[:,:,1]=pixels[:,:,1][:,::-1]
        pixels[:,:,2]=np.uint8(255)-pixels[:,:,0]
        for size in ((256,53),(53,256),(17,9)):
            image=Image.fromarray(pixels[:size[1],:size[0]],"RGB")
            for target in (64,256):
                real,valid,geometry=letterbox_rgb(image,target)
                expected=quantize_input(real,1/127)
                actual,actual_valid,actual_geometry=letterbox_int8(image,1/127,target)
                np.testing.assert_array_equal(actual,expected)
                np.testing.assert_array_equal(actual_valid,valid)
                self.assertEqual(actual_geometry,geometry)


if __name__ == "__main__":
    unittest.main()
