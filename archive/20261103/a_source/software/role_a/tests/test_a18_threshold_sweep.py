"""A18 阈值扫描的针对性测试（合成小样本，无网络无训练）。"""
from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from software.role_a import a18_threshold_sweep as a18


class ThresholdTests(unittest.TestCase):
    def test_boundary_is_inclusive(self) -> None:
        prob = np.array([[0.30]], dtype=np.float32)
        truth = np.array([[True]])
        valid = np.array([[True]])
        res = a18.per_image_for_grid(prob, truth, valid, [0.30, 0.3000001])
        self.assertEqual(res[0.30][0], 1.0)          # prob >= t 取等号
        self.assertEqual(res[0.3000001][0], 0.0)     # 严格小于则不取

    def test_invalid_region_is_always_background(self) -> None:
        prob = np.array([[0.9, 0.9]], dtype=np.float32)
        truth = np.array([[True, True]])
        valid = np.array([[True, False]])            # 第 2 像素无效
        res = a18.per_image_for_grid(prob, truth, valid, [0.15])
        # 无效区被强制为背景 => 预测 1px、真值 1px、交集 1 => Dice 1.0
        self.assertAlmostEqual(res[0.15][0], 1.0, places=12)

    def test_empty_truth_conventions(self) -> None:
        prob = np.zeros((2, 2), dtype=np.float32)
        truth_empty = np.zeros((2, 2), dtype=bool)
        valid = np.ones((2, 2), dtype=bool)
        d, i = a18.binary_metrics(np.zeros((2, 2), dtype=bool), truth_empty)
        self.assertEqual((d, i), (1.0, 1.0))                       # 都空 => 1.0
        d, i = a18.binary_metrics(np.zeros((2, 2), dtype=bool), np.ones((2, 2), dtype=bool))
        self.assertEqual((d, i), (0.0, 0.0))                       # 只有真值非空 => 0.0
        res = a18.per_image_for_grid(prob, truth_empty, valid, [0.45])
        self.assertEqual(res[0.45][0], 1.0)                        # 空预测 vs 空真值

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            a18.per_image_for_grid(np.zeros((2, 2)), np.zeros((2, 3), dtype=bool),
                                   np.ones((2, 2), dtype=bool), [0.45])

    def test_all_thresholds_from_one_probability_map(self) -> None:
        prob = np.array([[0.10, 0.42], [0.47, 0.99]], dtype=np.float32)
        truth = np.ones((2, 2), dtype=bool)
        valid = np.ones((2, 2), dtype=bool)
        grid = [0.15, 0.45, 0.60]
        res = a18.per_image_for_grid(prob, truth, valid, grid)
        counts = {t: int((prob >= t).sum()) for t in grid}
        self.assertEqual(counts[0.15], 3)   # 0.42/0.47/0.99
        self.assertEqual(counts[0.45], 2)   # 0.47/0.99
        self.assertEqual(counts[0.60], 1)   # 0.99
        self.assertAlmostEqual(res[0.15][0], 2 * 3 / (3 + 4), places=12)

    def test_jaccard_official_rule(self) -> None:
        # IoU < 0.65 记 0，否则记 IoU：手工构造两例
        truth = np.zeros((1, 10), dtype=bool); truth[0, :4] = True
        pred_bad = np.zeros((1, 10), dtype=bool); pred_bad[0, 0] = True      # IoU = 1/(1+4-1)=0.25 -> 0
        pred_good = np.zeros((1, 10), dtype=bool); pred_good[0, :4] = True   # IoU = 1.0
        self.assertLess(a18.binary_metrics(pred_bad, truth)[1], a18.JACCARD_CUT)
        self.assertAlmostEqual(a18.binary_metrics(pred_good, truth)[1], 1.0, places=12)


if __name__ == "__main__":
    unittest.main()