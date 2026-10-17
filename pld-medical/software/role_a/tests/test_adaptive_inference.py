"""Check area threshold selection and padding behavior on small synthetic masks."""
from __future__ import annotations

import unittest

import torch

from software.role_a.adaptive_inference import score_adaptive


class AdaptiveInferenceTests(unittest.TestCase):
    def test_small_prediction_uses_small_threshold(self) -> None:
        probability = torch.tensor([[[0.52, 0.7], [0.1, 0.1]]])
        result = score_adaptive(torch.logit(probability), torch.zeros_like(probability),
                                torch.ones_like(probability), area_cut=0.6,
                                small_threshold=0.53, large_threshold=0.35)
        self.assertEqual(result["predicted_area_at_0_5"], 0.5)
        self.assertEqual(result["chosen_threshold"], 0.53)

    def test_large_prediction_uses_large_threshold_and_ignores_padding(self) -> None:
        probability = torch.tensor([[[0.52, 0.7], [0.1, 0.1]]])
        valid = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
        result = score_adaptive(torch.logit(probability), torch.zeros_like(probability),
                                valid, area_cut=0.6,
                                small_threshold=0.53, large_threshold=0.35)
        self.assertEqual(result["predicted_area_at_0_5"], 1.0)
        self.assertEqual(result["chosen_threshold"], 0.35)

    def test_empty_valid_mask_is_rejected(self) -> None:
        image = torch.zeros((1, 2, 2))
        with self.assertRaises(ValueError):
            score_adaptive(image, image, image, area_cut=0.4,
                           small_threshold=0.53, large_threshold=0.35)


if __name__ == "__main__":
    unittest.main()
