"""A17 针对性测试：比例多分量保留 + 填洞（合成样本；含 A11/A15 接口核对与 256x256 基准）。"""
from __future__ import annotations
import sys
import time
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from software.role_a import a17_component_ratio as a17
from software.role_a import postprocess_int8 as pp


def _blank(n: int = 12) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((n, n), dtype=bool), np.ones((n, n), dtype=bool)


class RatioRuleTests(unittest.TestCase):
    def test_empty_prediction_stays_empty(self) -> None:
        pred, valid = _blank()
        for p in a17.RATIOS:
            out = a17.apply_ratio_rule(pred, valid, p)
            self.assertFalse(bool(out.any()), f"p={p}")

    def test_boundary_equality_is_inclusive(self) -> None:
        # 最大块 20 px、小块 1 px：p=5 时 100*1 == 5*20 恰等 -> 保留；p=10 时不等 -> 丢弃
        self.assertEqual(a17.keep_labels_by_ratio([20, 1], 5), {1, 2})
        self.assertEqual(a17.keep_labels_by_ratio([20, 1], 10), {1})
        pred, valid = _blank(24)
        pred[1:5, 1:6] = True          # 4x5 = 20 px
        pred[20, 20] = True            # 1 px
        out5 = a17.apply_ratio_rule(pred, valid, 5)
        out10 = a17.apply_ratio_rule(pred, valid, 10)
        self.assertTrue(out5[20, 20])   # 恰等 => 保留
        self.assertFalse(out10[20, 20])

    def test_tied_largest_components_all_kept(self) -> None:
        labels, count, sizes, _ = pp._label_components(
            np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 1, 0, 0]], dtype=bool), connectivity=8)
        del labels
        self.assertEqual(count, 3)
        self.assertEqual(sorted(sizes), [1, 1, 1])
        self.assertEqual(a17.keep_labels_by_ratio(sizes, 20), {1, 2, 3})   # 并列全部保留
        pred, valid = _blank(10)
        pred[1, 1] = pred[1, 5] = True     # 两个等面积 1px 分量（距离 4，不连通）
        out = a17.apply_ratio_rule(pred, valid, 20)
        self.assertTrue(out[1, 1] and out[1, 5])

    def test_invalid_foreground_is_cleared(self) -> None:
        pred, valid = _blank(8)
        valid[0:2, 0:2] = False            # 左上角 2x2 为无效区
        pred[0, 0] = True                  # 完全落在无效区
        pred[4, 4] = True                  # 有效区内（孤立像素）
        for p in a17.RATIOS:
            out = a17.apply_ratio_rule(pred, valid, p)
            self.assertFalse(out[0, 0], f"p={p}")
            self.assertTrue(out[4, 4], f"p={p}")
            self.assertFalse(bool((out & ~valid).any()), f"p={p}")

    def test_diagonal_pair_is_one_eight_connected_component(self) -> None:
        diag = np.zeros((4, 4), dtype=bool)
        diag[0, 0] = diag[1, 1] = True
        labels, count, sizes, _ = pp._label_components(diag, connectivity=8)
        del labels
        self.assertEqual((count, sizes), (1, [2]))       # 8 连通：对角算一块
        self.assertEqual(pp.label_components(diag, connectivity=4)[1], 2)
        out = a17.apply_ratio_rule(diag, np.ones((4, 4), dtype=bool), 20)
        self.assertTrue(out[0, 0] and out[1, 1])         # 唯一分量 => 必留

    def test_hole_filling_is_four_connected_background(self) -> None:
        pred, valid = _blank(7)
        pred[1:6, 1:6] = True
        pred[3, 3] = False                                # 封闭空洞
        out = a17.apply_ratio_rule(pred, valid, 20)
        self.assertTrue(out[3, 3])                        # 填洞
        open_pred, _ = _blank(7)
        open_pred[1:6, 1] = True
        open_pred[1:6, 5] = True
        open_pred[1, 1:6] = True                          # 缺下边 -> 开口
        out2 = a17.apply_ratio_rule(open_pred, np.ones((7, 7), dtype=bool), 20)
        self.assertFalse(out2[3, 3])                       # 与外部连通，不填

    def test_fragmented_large_lesion_ratios(self) -> None:
        # 最大分量 40 px；另有 8 / 4 / 2 px：p=5 留 40/8/4/2；p=10 留 40/8/4；p=20 留 40/8
        sizes = [40, 8, 4, 2]
        self.assertEqual(a17.keep_labels_by_ratio(sizes, 5), {1, 2, 3, 4})
        self.assertEqual(a17.keep_labels_by_ratio(sizes, 10), {1, 2, 3})
        self.assertEqual(a17.keep_labels_by_ratio(sizes, 20), {1, 2})
        pred, valid = _blank(40)
        pred[1:9, 1:6] = True            # 8x5 = 40
        pred[20:24, 1:3] = True          # 4x2 = 8
        pred[30:32, 1:3] = True          # 2x2 = 4
        pred[36, 1:3] = True             # 1x2 = 2
        counts = {}
        for p in a17.RATIOS:
            labels, count, sz, _ = pp._label_components(pred & valid, connectivity=8)
            del labels
            counts[p] = len(a17.keep_labels_by_ratio(sz, p))
        self.assertEqual(counts, {5: 4, 10: 3, 20: 2})

    def test_dense_noise_runs_and_respects_valid(self) -> None:
        rng = np.random.default_rng(7)
        pred = rng.random((256, 256)) < 0.5
        valid = np.ones((256, 256), dtype=bool)
        for p in a17.RATIOS:
            t0 = time.perf_counter()
            out = a17.apply_ratio_rule(pred, valid, p)
            dt = time.perf_counter() - t0
            self.assertEqual(out.shape, pred.shape)
            self.assertFalse(bool((out & ~valid).any()))
            self.assertLess(dt, 2.0, f"p={p} 用时 {dt:.3f}s")

    def test_interface_matches_a11_a15(self) -> None:
        pred, valid = _blank(16)
        pred[2:7, 2:7] = True            # 25 px 最大块
        pred[2:4, 2:4] = False           # 制造空洞
        pred[12, 12] = True              # 1 px 小块
        # A15 combined 的等价写法（A11 原语组合）
        expect = pp.fill_holes(pp.largest_component(pred, valid), valid)
        self.assertTrue(np.array_equal(pp.apply_candidate("combined", pred, valid), expect))
        # 本题 p=100 只保留最大分量（唯一最大时与 A15 combined 相同）
        self.assertTrue(np.array_equal(a17.apply_ratio_rule(pred, valid, 100), expect))

    def test_apply_all_ratios_matches_three_independent_calls(self) -> None:
        pred, valid = _blank(40)
        pred[1:9, 1:6] = True            # 40 px
        pred[20:24, 1:3] = True          # 8 px
        pred[30:32, 1:3] = True          # 4 px
        pred[36, 1:3] = True             # 2 px
        pred[3, 3] = False               # 制造空洞
        all_out = a17.apply_all_ratios(pred, valid)
        for p in a17.RATIOS:
            single = a17.apply_ratio_rule(pred, valid, p)
            self.assertTrue(np.array_equal(all_out[f"keep_ratio_{p}"], single), f"p={p}")

    def test_apply_all_ratios_calls_labeling_exactly_once(self) -> None:
        from unittest import mock
        pred, valid = _blank(24)
        pred[1:5, 1:6] = True
        pred[20, 20] = True
        original = pp._label_components
        calls = {"n": 0}

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        with mock.patch.object(pp, "_label_components", counting):
            a17.apply_all_ratios(pred, valid)
        self.assertEqual(calls["n"], 1, "连通域标记必须只做一次")

    def test_apply_all_ratios_empty_and_fragmented(self) -> None:
        pred, valid = _blank(16)
        out = a17.apply_all_ratios(pred, valid)                 # 空图：不应触发标记
        for p in a17.RATIOS:
            self.assertFalse(bool(out[f"keep_ratio_{p}"].any()))
        pred[2:6, 2:4] = True
        pred[10:11, 10:12] = True
        out2 = a17.apply_all_ratios(pred, valid)                # 碎裂例
        self.assertEqual(set(out2), {f"keep_ratio_{p}" for p in a17.RATIOS})
        for p in a17.RATIOS:
            self.assertFalse(bool((out2[f"keep_ratio_{p}"] & ~valid).any()))

    def test_stats_counts_components_and_kept(self) -> None:
        pred, valid = _blank(40)
        pred[1:9, 1:6] = True            # 40 px
        pred[20:24, 1:3] = True          # 8 px
        pred[30:32, 1:3] = True          # 4 px
        pred[36, 1:3] = True             # 2 px
        stats = {}
        a17.apply_all_ratios(pred, valid, stats=stats)
        self.assertEqual(stats["components_total"], 4)
        self.assertEqual(stats["largest_component_px"], 40)
        self.assertEqual(stats["sizes"], [40, 8, 4, 2])
        self.assertEqual(stats["kept"], {5: 4, 10: 3, 20: 2})
        self.assertGreaterEqual(stats["kept"][5], stats["kept"][10])
        self.assertGreaterEqual(stats["kept"][10], stats["kept"][20])

    def test_stats_empty_prediction(self) -> None:
        pred, valid = _blank(8)
        stats = {}
        out = a17.apply_all_ratios(pred, valid, stats=stats)
        self.assertEqual(stats["components_total"], 0)
        self.assertEqual(stats["largest_component_px"], 0)
        self.assertEqual(stats["kept"], {5: 0, 10: 0, 20: 0})
        for p in a17.RATIOS:
            self.assertFalse(bool(out[f"keep_ratio_{p}"].any()))

    def test_stats_still_labels_only_once(self) -> None:
        from unittest import mock
        pred, valid = _blank(24)
        pred[1:5, 1:6] = True
        pred[20, 20] = True
        original = pp._label_components
        calls = {"n": 0}

        def counting(*args, **kwargs):
            calls["n"] += 1
            return original(*args, **kwargs)

        with mock.patch.object(pp, "_label_components", counting):
            a17.apply_all_ratios(pred, valid, stats={})
        self.assertEqual(calls["n"], 1)

    def test_256_benchmark_all_ratios(self) -> None:
        h = w = 256
        gy, gx = np.mgrid[0:h, 0:w]
        blob = ((gy - 128) ** 2 + (gx - 128) ** 2) < 90 ** 2
        holes = (((gy - 100) ** 2 + (gx - 100) ** 2) < 25 ** 2) | \
                (((gy - 160) ** 2 + (gx - 110) ** 2) < 18 ** 2)
        frag = (((gy - 40) ** 2 + (gx - 200) ** 2) < 14 ** 2) | \
               (((gy - 210) ** 2 + (gx - 60) ** 2) < 10 ** 2)
        pred = (blob & ~holes) | frag
        valid = np.ones((h, w), dtype=bool)
        total = 0.0
        outs = {}
        for p in a17.RATIOS:
            t0 = time.perf_counter()
            outs[p] = a17.apply_ratio_rule(pred, valid, p)
            total += time.perf_counter() - t0
        self.assertLess(total, 1.0, f"三比例合计 {total:.3f}s")
        for p in a17.RATIOS:
            self.assertFalse(bool((outs[p] & ~valid).any()))
        self.assertLessEqual(int(outs[20].sum()), int(outs[10].sum()))
        self.assertLessEqual(int(outs[10].sum()), int(outs[5].sum()))
        self.assertTrue(outs[5][100, 100])       # 内部空洞被填


if __name__ == "__main__":
    unittest.main()