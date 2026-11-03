import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "software"))
from baseline import grayscale, gaussian, morphology, overlap, read_pnm, write_pnm


class BaselineTests(unittest.TestCase):
    def test_gray_known_colors(self):
        self.assertEqual(grayscale([(0, 0, 0), (255, 255, 255), (255, 0, 0),
                                    (0, 255, 0), (0, 0, 255)]), [0, 255, 77, 149, 29])

    def test_constant_filter_and_impulse(self):
        self.assertEqual(gaussian([73] * 9, 3, 3), [73] * 9)
        self.assertEqual(gaussian([0, 0, 0, 0, 255, 0, 0, 0, 0], 3, 3),
                         [16, 32, 16, 32, 64, 32, 16, 32, 16])

    def test_opening_removes_isolated_pixel_preserves_square(self):
        single = [0] * 25
        single[12] = 1
        self.assertEqual(morphology(morphology(single, 5, 5), 5, 5, True), [0] * 25)
        square = [int(1 <= x <= 3 and 1 <= y <= 3) for y in range(5) for x in range(5)]
        self.assertEqual(morphology(morphology(square, 5, 5), 5, 5, True), square)

    def test_scores_exclude_padding(self):
        self.assertEqual(overlap([1, 1, 0], [1, 0, 1], [1, 1, 0]),
                         {"dice": 2 / 3, "iou": .5})
        self.assertEqual(overlap([0], [0], [1]), {"dice": 1.0, "iou": 1.0})
        self.assertEqual(overlap([1], [0], [1]), {"dice": 0.0, "iou": 0.0})
        with self.assertRaises(ValueError):
            overlap([0], [0], [0])

    def test_pnm_roundtrip_and_invalid_size(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "sample.ppm"
            write_pnm(path, 2, 1, [(0, 0, 0), (255, 127, 3)], rgb=True)
            self.assertEqual(read_pnm(path, "P3"), (2, 1, [(0, 0, 0), (255, 127, 3)]))
            path.write_text("P3\n2 2\n255\n0 0 0", encoding="ascii")
            with self.assertRaises(ValueError):
                read_pnm(path, "P3")


if __name__ == "__main__":
    unittest.main()
