"""Check separable resize at odd boundaries and degenerate source dimensions."""
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.parallel_resize import ParallelBilinear
from reference.preprocess import letterbox_rgb, quantize_input


class Tests(unittest.TestCase):
    def test_odd_rows_edges_and_identity_preserve_pixels_and_quantization(self):
        rng = np.random.default_rng(2709)
        with ParallelBilinear() as resizer:
            for height, width in ((1, 39), (37, 1), (39, 53), (256, 256), (411, 617)):
                image = Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8))
                expected, valid, geometry = letterbox_rgb(image)
                actual, parallel_valid, parallel_geometry = letterbox_rgb(image, resizer=resizer)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(quantize_input(actual, .00781), quantize_input(expected, .00781))
                np.testing.assert_array_equal(parallel_valid, valid)
                self.assertEqual(parallel_geometry, geometry)


if __name__ == '__main__':
    unittest.main()
