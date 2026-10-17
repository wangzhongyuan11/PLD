"""指标与损失测试：Dice/IoU 边界情形、valid 掩膜的作用、损失对补边不敏感。"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _support import MISSING_DEPENDENCY, ensure_role_a_on_path  # noqa: E402

ensure_role_a_on_path()

import torch  # noqa: E402

from metrics import (  # noqa: E402
    MetricAccumulator,
    binary_scores,
    dice_coefficient,
    dice_masked,
    iou_masked,
    iou_score,
    masked_bce_with_logits,
    segmentation_loss,
    soft_dice_loss,
)


@unittest.skipIf(MISSING_DEPENDENCY is not None, MISSING_DEPENDENCY or "")
class DiceIouBoundaryTests(unittest.TestCase):
    def test_perfect_and_disjoint(self) -> None:
        truth = torch.tensor([[1.0, 1.0], [0.0, 0.0]])
        self.assertAlmostEqual(dice_coefficient(truth, truth), 1.0)
        self.assertAlmostEqual(iou_score(truth, truth), 1.0)
        disjoint = 1.0 - truth
        self.assertAlmostEqual(dice_coefficient(disjoint, truth), 0.0)
        self.assertAlmostEqual(iou_score(disjoint, truth), 0.0)

    def test_both_empty_is_one(self) -> None:
        """空预测 + 空真值 = 完全一致，按约定为 1.0（README 中明确写出）。"""
        empty = torch.zeros(1, 4, 4)
        self.assertAlmostEqual(dice_coefficient(empty, empty), 1.0)
        self.assertAlmostEqual(iou_score(empty, empty), 1.0)

    def test_one_side_empty_is_zero(self) -> None:
        empty = torch.zeros(4)
        full = torch.ones(4)
        self.assertAlmostEqual(dice_coefficient(full, empty), 0.0)
        self.assertAlmostEqual(dice_coefficient(empty, full), 0.0)
        self.assertAlmostEqual(iou_score(full, empty), 0.0)
        self.assertAlmostEqual(iou_score(empty, full), 0.0)

    def test_half_overlap_matches_closed_form(self) -> None:
        prediction = torch.tensor([1.0, 1.0, 0.0, 0.0])
        truth = torch.tensor([1.0, 0.0, 1.0, 0.0])
        # |P|=2, |T|=2, |P∩T|=1 -> Dice=2/4=0.5, IoU=1/3
        self.assertAlmostEqual(dice_coefficient(prediction, truth), 0.5)
        self.assertAlmostEqual(iou_score(prediction, truth), 1.0 / 3.0)

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            dice_coefficient(torch.zeros(2, 2), torch.zeros(4))
        with self.assertRaises(ValueError):
            iou_score(torch.zeros(2, 2), torch.zeros(1, 4))

    def test_nan_rejected(self) -> None:
        with self.assertRaises(ValueError):
            dice_coefficient(torch.tensor([float("nan")]), torch.tensor([1.0]))

    def test_empty_tensor_rejected(self) -> None:
        with self.assertRaises(ValueError):
            dice_coefficient(torch.zeros(0), torch.zeros(0))

    def test_bool_input_supported(self) -> None:
        truth = torch.tensor([[True, False], [True, False]])
        self.assertAlmostEqual(dice_coefficient(truth, truth), 1.0)


@unittest.skipIf(MISSING_DEPENDENCY is not None, MISSING_DEPENDENCY or "")
class MaskedMetricTests(unittest.TestCase):
    def test_padding_is_ignored(self) -> None:
        """补边区域即使真值/预测完全错误，也不应影响指标。"""
        truth = torch.zeros(1, 1, 4, 4)
        truth[..., :2, :2] = 1.0
        valid = torch.zeros(1, 1, 4, 4)
        valid[..., :2, :2] = 1.0
        prediction = torch.zeros_like(truth)
        prediction[..., :2, :2] = 1.0
        # 把补边区域填成"全错"：在 valid 外预测 1、真值 0。
        prediction[..., 2:, 2:] = 1.0
        self.assertAlmostEqual(dice_masked(prediction, truth, valid), 1.0)
        self.assertAlmostEqual(iou_masked(prediction, truth, valid), 1.0)
        # 不加 valid 时会掉分，说明掩膜确实在起作用。
        self.assertLess(dice_coefficient(prediction, truth), 1.0)

    def test_empty_valid_region_raises(self) -> None:
        zeros = torch.zeros(1, 1, 4, 4)
        with self.assertRaises(ValueError):
            dice_masked(zeros, zeros, zeros)

    def test_valid_shape_mismatch_raises(self) -> None:
        zeros = torch.zeros(1, 1, 4, 4)
        with self.assertRaises(ValueError):
            iou_masked(zeros, zeros, torch.ones(1, 1, 3, 3))


@unittest.skipIf(MISSING_DEPENDENCY is not None, MISSING_DEPENDENCY or "")
class BinaryScoresTests(unittest.TestCase):
    def test_threshold_boundary(self) -> None:
        logits = torch.tensor([[[[0.0, 10.0], [-10.0, 0.0]]]])  # sigmoid -> 0.5, ~1, ~0, 0.5
        truth = torch.ones_like(logits)
        at_half = binary_scores(logits, truth, threshold=0.5)  # >= 0.5，除 -10 外三个像素判正
        self.assertAlmostEqual(at_half["dice"], 6.0 / 7.0)

        threshold = 0.5000001  # 严格大于之后只剩 sigmoid(10) 判正
        above_half = binary_scores(logits, truth, threshold=threshold)
        predicted = (torch.sigmoid(logits) >= threshold).sum().item()
        self.assertEqual(predicted, 1)
        # |P|=1, |T|=4, |P∩T|=1 -> Dice = 2/(1+4), IoU = 1/4
        self.assertAlmostEqual(above_half["dice"], 2.0 / (predicted + 4))
        self.assertAlmostEqual(above_half["iou"], 1.0 / 4.0)

    def test_empty_flags(self) -> None:
        empty = torch.full((1, 1, 2, 2), -20.0)
        scores = binary_scores(empty, torch.zeros_like(empty))
        self.assertTrue(scores["empty_target"])
        self.assertTrue(scores["empty_prediction"])
        self.assertAlmostEqual(scores["dice"], 1.0)
        self.assertEqual(scores["valid_pixels"], 4)

    def test_valid_regions_shrink(self) -> None:
        logits = torch.full((1, 1, 4, 4), 10.0)
        truth = torch.ones_like(logits)
        valid = torch.zeros_like(logits)
        valid[..., :2, :] = 1.0
        scores = binary_scores(logits, truth, valid)
        self.assertEqual(scores["valid_pixels"], 8)

    def test_bad_threshold_raises(self) -> None:
        logits = torch.zeros(1, 1, 2, 2)
        for bad in (0.0, 1.0, -0.5, 2.0):
            with self.subTest(threshold=bad):
                with self.assertRaises(ValueError):
                    binary_scores(logits, logits, threshold=bad)

    def test_empty_valid_raises(self) -> None:
        logits = torch.zeros(1, 1, 2, 2)
        with self.assertRaises(ValueError):
            binary_scores(logits, logits, torch.zeros_like(logits))


@unittest.skipIf(MISSING_DEPENDENCY is not None, MISSING_DEPENDENCY or "")
class LossTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.valid = torch.zeros(2, 1, 8, 8)
        self.valid[..., 2:6, 2:6] = 1.0
        self.target = torch.zeros(2, 1, 8, 8)
        self.target[..., 2:6, 2:6] = 1.0

    def test_perfect_prediction_has_low_loss(self) -> None:
        logits = torch.where(self.target > 0, 20.0, -20.0)
        total, parts = segmentation_loss(logits, self.target, self.valid)
        self.assertLess(parts["bce"], 1e-4)
        self.assertLess(parts["soft_dice"], 1e-4)
        self.assertLess(float(total), 1e-4)

    def test_inverted_prediction_has_high_loss(self) -> None:
        logits = torch.where(self.target > 0, -20.0, 20.0)
        total, parts = segmentation_loss(logits, self.target, self.valid)
        self.assertGreater(parts["bce"], 1.0)
        self.assertGreater(parts["soft_dice"], 0.5)
        self.assertGreater(float(total), 1.5)

    def test_loss_ignores_padding(self) -> None:
        """只改变补边区域的 logits，损失必须完全不变。"""
        base = torch.where(self.target > 0, 1.0, -1.0)
        perturbed = base.clone()
        perturbed[self.valid == 0] = 50.0
        first, _ = segmentation_loss(base, self.target, self.valid)
        second, _ = segmentation_loss(perturbed, self.target, self.valid)
        self.assertAlmostEqual(float(first), float(second), places=6)

    def test_loss_is_differentiable(self) -> None:
        logits = torch.zeros(2, 1, 8, 8, requires_grad=True)
        total, _ = segmentation_loss(logits, self.target, self.valid)
        total.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(float(logits.grad[self.valid > 0].abs().sum()), 0.0)

    def test_bce_matches_manual_mean_over_valid(self) -> None:
        logits = torch.linspace(-2, 2, steps=self.target.numel()).reshape(self.target.shape)
        expected_values = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[self.valid > 0], self.target[self.valid > 0], reduction="mean")
        actual = masked_bce_with_logits(logits, self.target, self.valid)
        self.assertAlmostEqual(float(actual), float(expected_values), places=6)

    def test_soft_dice_range(self) -> None:
        logits = torch.zeros(2, 1, 8, 8)
        value = float(soft_dice_loss(logits, self.target, self.valid))
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 1.0)

    def test_invalid_arguments_raise(self) -> None:
        logits = torch.zeros(2, 1, 8, 8)
        with self.assertRaises(ValueError):
            masked_bce_with_logits(logits, self.target, torch.zeros_like(self.valid))
        with self.assertRaises(ValueError):
            segmentation_loss(logits, self.target, self.valid, bce_weight=0, dice_weight=0)
        with self.assertRaises(ValueError):
            soft_dice_loss(logits, self.target, self.valid, epsilon=0.0)
        with self.assertRaises(ValueError):
            masked_bce_with_logits(logits, torch.zeros(2, 1, 4, 4), self.valid)


@unittest.skipIf(MISSING_DEPENDENCY is not None, MISSING_DEPENDENCY or "")
class AccumulatorTests(unittest.TestCase):
    def test_means_and_counts(self) -> None:
        accumulator = MetricAccumulator()
        accumulator.update({"dice": 1.0, "iou": 1.0, "empty_target": False,
                            "empty_prediction": False, "valid_pixels": 10})
        accumulator.update({"dice": 0.0, "iou": 0.0, "empty_target": False,
                            "empty_prediction": True, "valid_pixels": 10})
        accumulator.update({"dice": 1.0, "iou": 1.0, "empty_target": True,
                            "empty_prediction": True, "valid_pixels": 10})
        summary = accumulator.summary()
        self.assertEqual(summary["images"], 3)
        self.assertAlmostEqual(summary["dice_mean"], 2.0 / 3.0)
        self.assertAlmostEqual(summary["dice_median"], 1.0)
        self.assertAlmostEqual(summary["dice_mean_non_empty_target"], 0.5)
        self.assertEqual(summary["non_empty_target_images"], 2)
        self.assertEqual(summary["empty_target_images"], 1)
        self.assertEqual(summary["empty_prediction_images"], 2)
        self.assertEqual(summary["valid_pixels_total"], 30)

    def test_median_of_even_count(self) -> None:
        accumulator = MetricAccumulator()
        for dice in (0.0, 0.2, 0.6, 1.0):
            accumulator.update({"dice": dice, "iou": dice, "empty_target": False,
                                "empty_prediction": False})
        self.assertAlmostEqual(accumulator.summary()["dice_median"], 0.4)

    def test_missing_field_raises(self) -> None:
        accumulator = MetricAccumulator()
        with self.assertRaises(ValueError):
            accumulator.update({"dice": 1.0})

    def test_empty_summary_raises(self) -> None:
        with self.assertRaises(ValueError):
            MetricAccumulator().summary()

    def test_nan_free_summary(self) -> None:
        accumulator = MetricAccumulator()
        accumulator.update({"dice": 0.5, "iou": 0.25, "empty_target": False,
                            "empty_prediction": False})
        summary = accumulator.summary()
        for key in ("dice_mean", "iou_mean", "dice_median", "iou_median"):
            self.assertFalse(math.isnan(summary[key]), key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
