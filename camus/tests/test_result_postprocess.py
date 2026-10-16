import unittest

import numpy as np

from camus.postprocess import PostprocessError, apply_island5_hole1
from camus.result import native_labels_from_head


class ResultTests(unittest.TestCase):
    def test_native_argmax_tie_keeps_lower_class(self):
        head = np.zeros((1, 4, 256, 256), dtype=np.int8)
        head[0, 1] = 1
        head[0, 2] = 1
        labels = native_labels_from_head(head, 256, 256, 0.19834566491795338)
        self.assertTrue(np.all(labels == 1))
        head[0, 1] = 0
        head[0, 2] = 0
        labels = native_labels_from_head(head, 256, 256, 0.19834566491795338)
        self.assertTrue(np.all(labels == 0))

    def test_native_shape_difference_uses_valid_letterbox_crop(self):
        head = np.zeros((1, 4, 256, 256), dtype=np.int8)
        head[0, 3, 128:, 128:] = 127
        labels = native_labels_from_head(head, 100, 300, 0.19834566491795338)
        self.assertEqual(labels.shape, (100, 300))
        self.assertEqual(labels.dtype, np.uint8)
        self.assertGreaterEqual(int(labels.max()), 0)


class PostprocessTests(unittest.TestCase):
    def test_small_island_removed_and_large_island_retained(self):
        labels = np.zeros((20, 20), dtype=np.uint8)
        labels[2:12, 2:12] = 1
        labels[17:19, 17:19] = 1
        outcome = apply_island5_hole1(labels)
        self.assertTrue(outcome.island_applied)
        self.assertTrue(np.all(outcome.labels[17:19, 17:19] == 0))
        big = labels.copy()
        big[17:20, 17:20] = 1
        self.assertFalse(apply_island5_hole1(big).island_applied)

    def test_small_background_hole_filled_and_other_labels_preserved(self):
        labels = np.ones((25, 20), dtype=np.uint8)
        labels[11:13, 9:11] = 0
        labels[0, 0] = 2
        outcome = apply_island5_hole1(labels)
        self.assertTrue(outcome.hole_applied)
        self.assertTrue(np.all(outcome.labels[9:11, 9:11] == 1))
        self.assertEqual(outcome.labels[0, 0], 2)

    def test_invalid_labels_rejected(self):
        with self.assertRaises(PostprocessError):
            apply_island5_hole1(np.full((2, 2), 4, dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
