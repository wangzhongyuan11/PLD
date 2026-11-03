"""Single-network area-conditioned output threshold for the A-role candidate."""
from __future__ import annotations

from typing import Any

import torch

from software.role_a.refine_small import paired_score


def score_adaptive(logit: torch.Tensor, target: torch.Tensor, valid: torch.Tensor,
                   *, area_cut: float, small_threshold: float,
                   large_threshold: float) -> dict[str, Any]:
    """Determine threshold from the model's own 0.5 mask, then score one image."""
    if not 0 < area_cut < 1 or not all(0 < t < 1 for t in
                                      (small_threshold, large_threshold)):
        raise ValueError("area cut and thresholds must be inside (0, 1)")
    if logit.shape != target.shape or logit.shape != valid.shape:
        raise ValueError("logit, target, and valid masks need matching shapes")
    keep = valid > 0
    denominator = int(keep.sum())
    if denominator == 0:
        raise ValueError("valid mask has no pixels")
    probability = torch.sigmoid(logit)
    predicted_area = float(((probability >= 0.5) & keep).sum()) / denominator
    threshold = large_threshold if predicted_area >= area_cut else small_threshold
    return {**paired_score(logit, target, valid, threshold),
            "predicted_area_at_0_5": predicted_area,
            "chosen_threshold": threshold}
