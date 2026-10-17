"""``postprocess_int8`` 的针对性测试（合成小样本，人工真值，不读真实数据）。"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import numpy as np

    from software.role_a import postprocess_int8 as pp

    HAVE_DEPENDENCIES = True
except ImportError as error:  # pragma: no cover
    IMPORT_ERROR = str(error)
    HAVE_DEPENDENCIES = False
else:
    IMPORT_ERROR = ""

SKIP_REASON = ("缺少 numpy 或本地模块，跳过（不算通过）: " + IMPORT_ERROR
               if not HAVE_DEPENDENCIES else "")


def _b(rows: list[str]) -> "np.ndarray":
    """把 '#/.' 字符串画布转成 bool 数组。"""
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


@unittest.skipUnless(HAVE_DEPENDENCIES, SKIP_REASON)
class ConnectivityTests(unittest.TestCase):
    def test_four_and_eight_connectivity_differ_on_diagonal(self) -> None:
        chain = np.zeros((3, 3), dtype=bool)
        chain[0, 0] = chain[1, 1] = chain[2, 2] = True
        # 4 连通：三个像素各成一体；8 连通：连成一条
        self.assertEqual(pp.label_components(chain, connectivity=4)[1], 3)
        self.assertEqual(pp.label_components(chain, connectivity=8)[1], 1)
        # 背景侧同样只走 4 连通：对角泄漏不算连通
        free = np.zeros((3, 3), dtype=bool)
        free[0, 0] = free[1, 1] = True
        reached = pp._flood_from_edge(free, np.ones((3, 3), dtype=bool))
        self.assertTrue(reached[0, 0])                  # 边缘上的背景可达
        self.assertFalse(reached[1, 1])                 # 只对角相邻 => 不可达

    def test_component_labels_respect_connectivity(self) -> None:
        diagonal = np.zeros((3, 3), dtype=bool)
        diagonal[0, 0] = diagonal[1, 1] = True
        self.assertEqual(pp.label_components(diagonal, connectivity=4)[1], 2)
        self.assertEqual(pp.label_components(diagonal, connectivity=8)[1], 1)

    def test_invalid_connectivity_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pp.label_components(np.ones((2, 2), dtype=bool), connectivity=6)


@unittest.skipUnless(HAVE_DEPENDENCIES, SKIP_REASON)
class FillHolesTests(unittest.TestCase):
    def test_closed_ring_hole_is_filled(self) -> None:
        pred = _b([".....", ".###.", ".#.#.", ".###.", "....."])
        valid = np.ones((5, 5), dtype=bool)
        out = pp.fill_holes(pred, valid)
        self.assertTrue(out[2, 2])                       # 内部空洞被填
        self.assertTrue(np.array_equal(out[1:4, 1:4], np.ones((3, 3), dtype=bool)))
        self.assertFalse(out[0, 0])
        self.assertEqual(int(out.sum()) - int(pred.sum()), 1)

    def test_open_notch_is_not_filled(self) -> None:
        pred = _b([".....", ".#.#.", ".#.#.", ".#.#.", "....."])
        valid = np.ones((5, 5), dtype=bool)
        out = pp.fill_holes(pred, valid)
        self.assertFalse(out[2, 2])                      # 开口处与外部连通，不填
        self.assertTrue(np.array_equal(out, pred))

    def test_background_flood_is_four_connected(self) -> None:
        # (1,1) 是内部背景，(2,0) 在边界上是外部背景；两者只对角相邻。
        # 4 连通背景判据下 (1,1) 不与外部连通 => 视为内部空洞并被填。
        pred = _b(["#####", "#.###", ".####", "#####", "#####"])
        valid = np.ones((5, 5), dtype=bool)
        self.assertFalse(pred[1, 1] and pred[2, 0])
        out = pp.fill_holes(pred, valid)
        self.assertTrue(out[1, 1])
        # 同时证明差异真实存在：自由背景若按 8 连通标记，(1,1) 会与 (2,0) 连成一体
        free = valid & ~pred
        self.assertEqual(pp.label_components(free, connectivity=4)[1], 2)
        self.assertEqual(pp.label_components(free, connectivity=8)[1], 1)
        reached = pp._flood_from_edge(free, valid)
        self.assertFalse(reached[1, 1])                  # 4 连通洪泛到不了
        self.assertTrue(reached[2, 0])

    def test_valid_padding_is_respected(self) -> None:
        valid = np.zeros((7, 7), dtype=bool)
        valid[1:6, 1:6] = True
        pred = np.zeros((7, 7), dtype=bool)
        pred[2:5, 2:5] = True
        pred[3, 3] = False
        out = pp.fill_holes(pred, valid)
        self.assertTrue(out[3, 3])                       # 有效区域内的空洞
        self.assertFalse(out[0, 0])                      # 补边区永远是背景
        self.assertFalse(out[6, 6])
        self.assertFalse(bool((out & ~valid).any()))

    def test_background_touching_valid_edge_is_not_filled(self) -> None:
        valid = np.zeros((5, 5), dtype=bool)
        valid[1:4, 1:4] = True                           # 有效区域 3x3
        pred = np.zeros((5, 5), dtype=bool)
        pred[1:3, 1:3] = True                            # 只占左上 2x2
        out = pp.fill_holes(pred, valid)
        # 有效区域其余背景都接触有效区域边缘 => 不填
        self.assertFalse(out[3, 3])
        self.assertFalse(out[1, 3])
        self.assertTrue(out[1, 1] and out[2, 2])
        self.assertFalse(bool((out & ~valid).any()))

    def test_empty_prediction_stays_empty(self) -> None:
        valid = np.zeros((6, 6), dtype=bool)
        valid[1:5, 1:5] = True
        pred = np.zeros((6, 6), dtype=bool)
        out = pp.fill_holes(pred, valid)
        self.assertFalse(bool(out.any()))

    def test_foreground_outside_valid_is_always_cleared(self) -> None:
        valid = np.zeros((5, 5), dtype=bool)
        valid[1:4, 1:4] = True
        pred = np.zeros((5, 5), dtype=bool)
        pred[0, 0] = True                     # 补边区前景（至少要有一个）
        pred[4, 4] = True
        pred[2, 2] = True                     # 有效区前景
        out = pp.fill_holes(pred, valid)      # 底层函数也必须自洽
        self.assertFalse(bool((out & ~valid).any()))
        self.assertFalse(out[0, 0])
        self.assertFalse(out[4, 4])
        self.assertTrue(out[2, 2])
        # 前景完全落在有效区之外 => 结果全空
        outside_only = np.zeros((5, 5), dtype=bool)
        outside_only[0, :] = True
        outside_only[4, :] = True
        self.assertFalse(bool(pp.fill_holes(outside_only, valid).any()))
        # valid 全部被前景覆盖的极端情形也只返回有效区
        covered = np.ones((5, 5), dtype=bool)
        self.assertTrue(np.array_equal(pp.fill_holes(covered, valid), valid))

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            pp.fill_holes(np.zeros((3, 3), dtype=bool), np.zeros((3, 4), dtype=bool))


@unittest.skipUnless(HAVE_DEPENDENCIES, SKIP_REASON)
class LargestComponentTests(unittest.TestCase):
    def test_diagonal_blocks_are_one_eight_connected_component(self) -> None:
        pred = np.zeros((5, 5), dtype=bool)
        pred[0, 0] = pred[1, 1] = True                   # 对角相连，8 连通算一个分量
        pred[4, 4] = True
        valid = np.ones((5, 5), dtype=bool)
        out = pp.largest_component(pred, valid)
        self.assertTrue(out[0, 0] and out[1, 1])
        self.assertFalse(out[4, 4])

    def test_larger_component_wins(self) -> None:
        pred = np.zeros((6, 6), dtype=bool)
        pred[0, 0:3] = True                              # 3 像素
        pred[5, 5] = True                                # 1 像素
        valid = np.ones((6, 6), dtype=bool)
        out = pp.largest_component(pred, valid)
        self.assertEqual(int(out.sum()), 3)
        self.assertTrue(out[0, 0] and out[0, 2] and not out[5, 5])

    def test_tie_break_prefers_topmost_then_leftmost(self) -> None:
        # 三个等面积分量：上方右侧 / 下方左侧 / 同一行更靠左
        pred = np.zeros((6, 6), dtype=bool)
        pred[1, 4] = True
        pred[4, 0] = True
        pred[1, 1] = True
        valid = np.ones((6, 6), dtype=bool)
        out = pp.largest_component(pred, valid)
        self.assertTrue(out[1, 1])                       # 最靠上，其次最靠左
        self.assertFalse(out[1, 4])
        self.assertFalse(out[4, 0])
        self.assertEqual(int(out.sum()), 1)

    def test_single_component_is_unchanged(self) -> None:
        pred = _b([".....", ".###.", ".###.", ".....", "....."])
        valid = np.ones((5, 5), dtype=bool)
        out = pp.largest_component(pred, valid)
        self.assertTrue(np.array_equal(out, pred))

    def test_outside_valid_is_always_background(self) -> None:
        valid = np.zeros((5, 5), dtype=bool)
        valid[1:4, 1:4] = True
        pred = np.zeros((5, 5), dtype=bool)
        pred[0, 0] = True                                # 完全在补边区
        pred[2, 2] = True                                # 在有效区内
        out = pp.largest_component(pred, valid)
        self.assertFalse(out[0, 0])
        self.assertTrue(out[2, 2])
        self.assertFalse(bool((out & ~valid).any()))

    def test_empty_prediction_stays_empty(self) -> None:
        pred = np.zeros((4, 4), dtype=bool)
        valid = np.ones((4, 4), dtype=bool)
        self.assertFalse(bool(pp.largest_component(pred, valid).any()))


@unittest.skipUnless(HAVE_DEPENDENCIES, SKIP_REASON)
class CandidateTests(unittest.TestCase):
    def test_identity_is_exact_and_never_creates_pixels(self) -> None:
        pred = _b([".....", ".#.#.", ".#.#.", ".....", "....."])
        valid = np.ones((5, 5), dtype=bool)
        out = pp.apply_candidate("identity", pred, valid)
        self.assertTrue(np.array_equal(out, pred))

    def test_combined_is_fill_after_largest(self) -> None:
        pred = _b([".###.", ".#.#.", ".###.", ".....", "..#.."])
        valid = np.ones((5, 5), dtype=bool)
        manual = pp.fill_holes(pp.largest_component(pred, valid), valid)
        self.assertTrue(np.array_equal(pp.apply_candidate("combined", pred, valid), manual))
        combined = pp.apply_candidate("combined", pred, valid)
        self.assertTrue(combined[2, 2])                  # 最大分量里的空洞被填
        self.assertFalse(combined[4, 2])                 # 小分量被丢弃

    def test_noop_image_is_unchanged_by_every_candidate(self) -> None:
        pred = _b([".....", ".##..", ".##..", ".....", "....."])   # 单分量、无空洞
        valid = np.ones((5, 5), dtype=bool)
        for name in pp.CANDIDATES:
            out = pp.apply_candidate(name, pred, valid)
            self.assertTrue(np.array_equal(out, pred), name)

    def test_all_candidates_keep_output_inside_valid(self) -> None:
        valid = np.zeros((6, 6), dtype=bool)
        valid[1:5, 1:5] = True
        pred = np.zeros((6, 6), dtype=bool)
        pred[0, 0] = pred[2, 2] = pred[2, 3] = pred[4, 4] = True
        pred[1, 1] = pred[1, 2] = pred[2, 1] = True      # 有效区外也要能丢弃
        for name in pp.CANDIDATES:
            out = pp.apply_candidate(name, pred, valid)
            self.assertFalse(bool((out & ~valid).any()), name)
        # 底层函数本身也必须清除 valid 外前景（不依赖 apply_candidate 的裁边）
        for name in ("fill_holes", "largest_component"):
            raw = pp.fill_holes(pred, valid) if name == "fill_holes" \
                else pp.largest_component(pred, valid)
            self.assertFalse(bool((raw & ~valid).any()), name)

    def test_unknown_candidate_and_shape_mismatch_are_rejected(self) -> None:
        pred = np.zeros((3, 3), dtype=bool)
        with self.assertRaises(ValueError):
            pp.apply_candidate("median_filter", pred, pred)
        with self.assertRaises(ValueError):
            pp.largest_component(np.zeros((3, 3), dtype=bool), np.zeros((3, 4), dtype=bool))


@unittest.skipUnless(HAVE_DEPENDENCIES, SKIP_REASON)
class MetricTests(unittest.TestCase):
    def test_per_image_metrics_match_manual_truth(self) -> None:
        truth = np.zeros((4, 4), dtype=bool)
        truth[1:3, 1:3] = True                           # 4 个前景像素
        same = truth.copy()
        dice, iou, empty, pred_sum = pp._per_image_metrics(same, truth)
        self.assertAlmostEqual(dice, 1.0, places=12)
        self.assertAlmostEqual(iou, 1.0, places=12)
        self.assertFalse(empty)
        self.assertEqual(pred_sum, 4)

        partial = np.zeros((4, 4), dtype=bool)
        partial[1, 1] = True                             # 交集 1
        dice, iou, empty, pred_sum = pp._per_image_metrics(partial, truth)
        self.assertAlmostEqual(dice, 2 * 1 / (1 + 4), places=12)   # 0.4
        self.assertAlmostEqual(iou, 1 / (1 + 4 - 1), places=12)    # 0.25
        self.assertFalse(empty)

        disjoint = np.zeros((4, 4), dtype=bool)
        disjoint[0, 0] = True
        dice, iou, empty, pred_sum = pp._per_image_metrics(disjoint, truth)
        self.assertAlmostEqual(dice, 0.0, places=12)
        self.assertAlmostEqual(iou, 0.0, places=12)
        self.assertFalse(empty)

    def test_empty_conventions_follow_metrics_module(self) -> None:
        empty = np.zeros((3, 3), dtype=bool)
        filled = np.zeros((3, 3), dtype=bool)
        filled[0, 0] = True
        dice, iou, is_empty, _ = pp._per_image_metrics(empty, empty)
        self.assertAlmostEqual(dice, 1.0, places=12)     # 都空 => 1.0
        self.assertAlmostEqual(iou, 1.0, places=12)
        self.assertTrue(is_empty)
        dice, iou, is_empty, _ = pp._per_image_metrics(empty, filled)
        self.assertAlmostEqual(dice, 0.0, places=12)     # 只有一方空 => 0.0
        self.assertAlmostEqual(iou, 0.0, places=12)
        self.assertTrue(is_empty)

    def test_ground_truth_synthetic_case_end_to_end(self) -> None:
        # 人工真值：真值 = 3x3 实心块；预测 = 外环且中心为空洞
        truth = np.zeros((7, 7), dtype=bool)
        truth[2:5, 2:5] = True
        pred = np.zeros((7, 7), dtype=bool)
        pred[2:5, 2:5] = True
        pred[3, 3] = False
        valid = np.ones((7, 7), dtype=bool)
        before = pp._per_image_metrics(pred, truth)[0]
        after = pp._per_image_metrics(pp.apply_candidate("fill_holes", pred, valid), truth)[0]
        self.assertAlmostEqual(before, 2 * 8 / (8 + 9), places=12)
        self.assertAlmostEqual(after, 1.0, places=12)
        self.assertGreater(after, before)

    def test_noop_candidates_produce_identical_metrics(self) -> None:
        truth = np.zeros((5, 5), dtype=bool)
        truth[1:4, 1:4] = True
        pred = truth.copy()
        valid = np.ones((5, 5), dtype=bool)
        baseline = pp._per_image_metrics(pred, truth)
        for name in pp.CANDIDATES:
            self.assertEqual(pp._per_image_metrics(pp.apply_candidate(name, pred, valid), truth),
                             baseline)


# ------------------------------------------------------------------ 性能守卫
def _vertical_comb(height: int = 256, width: int = 256) -> "np.ndarray":
    """1 像素宽的竖向梳状长蛇通道：测地距离约 H*W/2，是迭代式传播的最坏情形。"""
    path = np.zeros((height, width), dtype=bool)
    for col in range(0, width, 2):
        path[:, col] = True
        if col + 2 < width:
            link = 0 if (col // 2) % 2 else height - 1
            path[link, col:col + 3] = True
    return path


@unittest.skipUnless(HAVE_DEPENDENCIES, SKIP_REASON)
class PerformanceTests(unittest.TestCase):
    """稠密噪点 / 蛇形最坏情形 / 256x256 短基准：保证单次遍历不退化。"""

    def test_dense_noise_masks_stay_within_time_budget(self) -> None:
        rng = np.random.default_rng(20260925)
        valid = np.ones((256, 256), dtype=bool)
        components: dict[float, int] = {}
        for density in (0.30, 0.50, 0.70):
            pred = rng.random((256, 256)) < density
            for name in pp.CANDIDATES:
                started = time.perf_counter()
                out = pp.apply_candidate(name, pred, valid)
                elapsed = time.perf_counter() - started
                self.assertEqual(out.shape, pred.shape)
                self.assertFalse(bool((out & ~valid).any()), name)
                self.assertLess(elapsed, 1.0, f"{name} density={density} 用时 {elapsed:.3f}s")
            components[density] = pp.label_components(pred, connectivity=8)[1]
        # 8 连通点渗流阈值约 0.407：0.30 会碎成大量小块，0.70 应几乎连成一片
        self.assertGreater(components[0.30], 1000)
        self.assertLess(components[0.70], 50)
        # 最大分量候选必须只留一块，且不超过原前景总量
        merged = pp.largest_component(pred, valid)
        self.assertEqual(pp.label_components(merged, connectivity=8)[1], 1)
        self.assertGreater(int(merged.sum()), 0)
        self.assertLessEqual(int(merged.sum()), int(pred.sum()))

    def test_snake_background_is_not_filled_and_completes_quickly(self) -> None:
        comb = _vertical_comb()
        pred = ~comb                                   # 背景就是那条长蛇
        valid = np.ones(comb.shape, dtype=bool)
        started = time.perf_counter()
        out = pp.fill_holes(pred, valid)
        elapsed = time.perf_counter() - started
        self.assertTrue(np.array_equal(out, pred))     # 整条通道都与边缘连通 => 不填
        self.assertLess(elapsed, 1.0, f"蛇形填洞用时 {elapsed:.3f}s")

    def test_snake_foreground_is_one_component_and_fast(self) -> None:
        comb = _vertical_comb()
        valid = np.ones(comb.shape, dtype=bool)
        started = time.perf_counter()
        keep = pp.largest_component(comb, valid)
        elapsed = time.perf_counter() - started
        self.assertTrue(np.array_equal(keep, comb))    # 连通路径不应被截断
        self.assertEqual(pp.label_components(comb, connectivity=8)[1], 1)
        self.assertLess(elapsed, 1.0, f"蛇形最大分量用时 {elapsed:.3f}s")

    def test_256_benchmark_for_all_candidates(self) -> None:
        height = width = 256
        grid_y, grid_x = np.mgrid[0:height, 0:width]
        blob = ((grid_y - 128) ** 2 + (grid_x - 128) ** 2) < 90 ** 2
        holes = (((grid_y - 100) ** 2 + (grid_x - 100) ** 2) < 25 ** 2) | \
                (((grid_y - 160) ** 2 + (grid_x - 110) ** 2) < 18 ** 2)
        island = (((grid_y - 30) ** 2 + (grid_x - 220) ** 2) < 12 ** 2)   # 与团块完全分离
        pred = (blob & ~holes) | island
        valid = np.ones((height, width), dtype=bool)
        total = 0.0
        results = {}
        for name in pp.CANDIDATES:
            started = time.perf_counter()
            results[name] = pp.apply_candidate(name, pred, valid)
            total += time.perf_counter() - started
        self.assertLess(total, 1.0, f"四候选合计 {total:.3f}s")
        self.assertTrue(results["fill_holes"][100, 100])      # 内部空洞被填
        self.assertTrue(results["fill_holes"][160, 110])
        self.assertTrue(results["largest_component"][128, 128])
        self.assertFalse(results["largest_component"][30, 220])   # 小岛被丢弃
        self.assertFalse(results["combined"][30, 220])
        self.assertTrue(results["combined"][100, 100])


if __name__ == "__main__":
    unittest.main()