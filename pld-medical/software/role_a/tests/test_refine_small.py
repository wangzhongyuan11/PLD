"""Critical acceptance and pixel-score checks for the refinement protocol."""
import unittest

import torch

from software.role_a.refine_small import acceptance, paired_score, _loss
from software.role_a.refine_conservative import set_blend
from software.role_a.refine_full2017 import full_loss
from software.role_a.refine_preserve_large import loss_with_retention


class RefineSmallTests(unittest.TestCase):
    def test_acceptance_requires_tiny_gain_and_no_other_group_loss(self) -> None:
        base = {key: {"dice": 0.7} for key in ("tiny", "medium", "large", "all")}
        better = {key: dict(value) for key, value in base.items()}
        better["tiny"]["dice"] = 0.71
        self.assertTrue(acceptance(better, base))
        better["medium"]["dice"] = 0.6999
        self.assertFalse(acceptance(better, base))

    def test_scores_respect_valid_area(self) -> None:
        logits = torch.tensor([[[10.0, -10.0], [10.0, 10.0]]])
        target = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
        valid = torch.tensor([[[1.0, 1.0], [0.0, 0.0]]])
        result = paired_score(logits, target, valid)
        self.assertAlmostEqual(result["dice"], 2 / 3)
        self.assertAlmostEqual(result["recall"], 1 / 2)
        self.assertAlmostEqual(result["precision"], 1.0)

    def test_small_loss_has_gradient(self) -> None:
        logits = torch.zeros((2, 1, 4, 4), requires_grad=True)
        target = torch.zeros_like(logits)
        target.data[0, 0, 1, 1] = 1
        target.data[1, 0, 1:3, 1:3] = 1
        valid = torch.ones_like(logits)
        loss, _ = _loss(logits, target, valid, torch.tensor([True, False]), logits.detach())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_threshold_and_weight_blend_change_candidate(self) -> None:
        logit = torch.tensor([[[0.1]]])
        target = torch.ones_like(logit)
        valid = torch.ones_like(logit)
        self.assertEqual(paired_score(logit, target, valid, 0.5)["dice"], 1.0)
        self.assertEqual(paired_score(logit, target, valid, 0.6)["dice"], 0.0)
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(0.0)
        original = {"weight": model.weight.detach().clone()}
        trained = {"weight": torch.ones_like(model.weight)}
        set_blend(model, trained, original, 0.25)
        self.assertAlmostEqual(float(model.weight.item()), 0.25)

    def test_full_data_loss_keeps_core_and_micro_gradient(self) -> None:
        logits = torch.zeros((2, 1, 4, 4), requires_grad=True)
        target = torch.zeros_like(logits)
        target.data[0, 0, 1, 1] = 1
        target.data[1, 0, 1:3, 1:3] = 1
        loss = full_loss(logits, target, torch.ones_like(logits),
                         torch.tensor([True, False]), logits.detach())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

    def test_retention_loss_updates_micro_and_original_large(self) -> None:
        logits = torch.zeros((3, 1, 4, 4), requires_grad=True)
        target = torch.zeros_like(logits)
        target.data[0, 0, 1, 1] = 1
        target.data[1, 0, 1:4, 1:4] = 1
        target.data[2, 0, 1:3, 1:3] = 1
        loss = loss_with_retention(logits, target, torch.ones_like(logits),
                                   torch.tensor([True, False, False]),
                                   torch.tensor([False, True, False]), logits.detach())
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad.abs().sum(dim=(1, 2, 3)) > 0).all())


if __name__ == "__main__":
    unittest.main()
