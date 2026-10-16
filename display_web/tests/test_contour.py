import unittest
import numpy as np
from contour import extract_contour, draw_contour, process
from pld_result import load_result
from pathlib import Path


def oracle(mask, connectivity, thickness):
    # Independent scalar definition: a foreground pixel is boundary when a
    # background/out-of-image pixel is within the specified lattice distance.
    h, w = mask.shape
    out = np.zeros_like(mask)
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            for dy in range(-thickness, thickness+1):
                for dx in range(-thickness, thickness+1):
                    if connectivity == 4 and abs(dx)+abs(dy) > thickness:
                        continue
                    yy, xx = y+dy, x+dx
                    if not (0 <= yy < h and 0 <= xx < w) or not mask[yy, xx]:
                        out[y, x] = True
    return out


class ContourTests(unittest.TestCase):
    def test_rectangle(self):
        m = np.zeros((9, 11), bool); m[2:7, 3:9] = True
        self.assertEqual(int(extract_contour(m).sum()), 18)

    def test_empty_full_single_and_thin(self):
        for m in (np.zeros((4, 5), bool), np.ones((4, 5), bool), np.ones((1, 1), bool), np.eye(6, dtype=bool)):
            np.testing.assert_array_equal(extract_contour(m), oracle(m, 8, 1))

    def test_holes_and_disconnected_components(self):
        m = np.zeros((14, 15), bool); m[1:10, 1:10] = True; m[4:7, 4:7] = False; m[11:13, 12:14] = True
        np.testing.assert_array_equal(extract_contour(m), oracle(m, 8, 1))
        self.assertTrue(extract_contour(m)[3, 5])

    def test_random_oracle(self):
        rng = np.random.default_rng(42)
        for _ in range(40):
            m = rng.random((7, 9)) > .25
            for c in (4, 8):
                for t in (1, 2, 3):
                    np.testing.assert_array_equal(extract_contour(m, c, t), oracle(m, c, t))

    def test_input_formats(self):
        m = np.eye(5, dtype=bool)
        for a in (m, m.astype(np.uint8), m.astype(np.uint8)*255):
            np.testing.assert_array_equal(extract_contour(a), m)

    def test_invalid_inputs(self):
        for m in (np.array([[0, 2]]), np.array([[0., 1.]]), np.array([[0, 1, 255]]), np.zeros((0, 3)), np.zeros((2, 2, 3))):
            with self.assertRaises(ValueError): extract_contour(m)
        for c, t in ((6, 1), (8, 0), (8, 33), (8, 1.5), (True, 1)):
            with self.assertRaises(ValueError): extract_contour(np.ones((2, 2), bool), c, t)

    def test_overlay_preserves_inputs_and_nonboundary(self):
        rgb = np.full((12, 13, 3), 70, np.uint8)
        m = np.zeros((12, 13), bool); m[2:10, 2:11] = True
        before = m.copy(); edge = extract_contour(m, thickness=2)
        out = draw_contour(rgb, m)
        self.assertTrue((out[edge] == [255, 220, 0]).all())
        self.assertTrue((out[~edge] == 70).all())
        self.assertTrue((rgb == 70).all()); np.testing.assert_array_equal(m, before)

    def test_rgb_validation(self):
        with self.assertRaises(ValueError): draw_contour(np.zeros((3, 3, 3), np.uint8), np.ones((2, 2), bool))
        with self.assertRaises(ValueError): draw_contour(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2), bool), color=(-1, 2, 3))

    def test_real_board_result(self):
        r = load_result(Path(__file__).resolve().parents[1]/"samples/frame_000001.json")
        mask = r.original_mask()
        result = process(mask)
        self.assertEqual(mask.shape, (767, 1022))
        self.assertGreater(result["statistics"]["boundary_pixels"], 0)
        self.assertTrue((~result["contour"] | mask).all())
        self.assertEqual(result["statistics"]["foreground_pixels"], int(mask.sum()))
        self.assertFalse(result["statistics"]["original_rgb_supplied"])


if __name__ == "__main__": unittest.main()
