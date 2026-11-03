"""只统计有效区域的 Dice / IoU，以及训练用的掩膜 BCE + soft Dice 损失。

边界约定（与 ``docs/06_数据与实验规范.md`` 的"空预测与空真值的处理规则需明确"一致）:

* 有效区域内预测与真值**都为空** ⇒ Dice = 1.0，IoU = 1.0（视为完全一致）；
* 只有一方为空 ⇒ Dice = 0.0，IoU = 0.0；
* 有效区域本身为空 ⇒ ``ValueError``（不猜、不返回 1.0）。

所有函数只接受 ``float``/``bool`` 张量，拒绝 ``NaN``；形状不一致直接报错，
避免广播把错误静默变成"更好的分数"。
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "dice_coefficient",
    "iou_score",
    "dice_masked",
    "iou_masked",
    "binary_scores",
    "masked_bce_with_logits",
    "soft_dice_loss",
    "segmentation_loss",
    "MetricAccumulator",
]


def _as_binary(tensor: Tensor, name: str) -> Tensor:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} 必须是 torch.Tensor，收到 {type(tensor).__name__}")
    if tensor.is_complex():
        raise ValueError(f"{name} 不能是复数张量")
    if tensor.is_floating_point() and bool(torch.isnan(tensor).any()):
        raise ValueError(f"{name} 含 NaN，无法计算指标")
    return tensor


def _broadcast_same_shape(prediction: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
    if prediction.shape != target.shape:
        raise ValueError(
            f"预测与真值 shape 必须一致，收到 {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    return prediction, target


def dice_coefficient(prediction: Tensor, target: Tensor) -> float:
    """二值 Dice。输入按 ``> 0`` 视为前景（可为 bool / 0-1 / logits 阈值化后的结果）。"""
    prediction, target = _broadcast_same_shape(
        _as_binary(prediction, "prediction"), _as_binary(target, "target")
    )
    pred = prediction.reshape(-1) > 0
    true = target.reshape(-1) > 0
    if pred.numel() == 0:
        raise ValueError("输入为空张量，无法计算 Dice")
    pred_sum = int(pred.sum())
    true_sum = int(true.sum())
    denominator = pred_sum + true_sum
    if denominator == 0:
        return 1.0
    intersection = int((pred & true).sum())
    return 2.0 * intersection / denominator


def iou_score(prediction: Tensor, target: Tensor) -> float:
    """二值 IoU（Jaccard）。空/空约定同 :func:`dice_coefficient`。"""
    prediction, target = _broadcast_same_shape(
        _as_binary(prediction, "prediction"), _as_binary(target, "target")
    )
    pred = prediction.reshape(-1) > 0
    true = target.reshape(-1) > 0
    if pred.numel() == 0:
        raise ValueError("输入为空张量，无法计算 IoU")
    union = int((pred | true).sum())
    if union == 0:
        return 1.0
    intersection = int((pred & true).sum())
    return intersection / union


def _masked_pair(prediction: Tensor, truth: Tensor, valid: Tensor | None) -> tuple[Tensor, Tensor]:
    """把预测与真值同时裁到有效区域，保证二者仍逐像素对应。"""
    if valid is None:
        return prediction, truth
    valid = _as_binary(valid, "valid")
    if valid.shape != truth.shape:
        raise ValueError(
            f"valid shape 必须与 target 一致，收到 {tuple(valid.shape)} vs {tuple(truth.shape)}"
        )
    if int((valid > 0).sum()) == 0:
        raise ValueError("有效区域为空，无法计算指标")
    keep = valid.reshape(-1) > 0
    return prediction.reshape(-1)[keep], truth.reshape(-1)[keep]


def dice_masked(prediction: Tensor, target: Tensor, valid: Tensor | None = None) -> float:
    """只在 ``valid > 0`` 的像素上计算 Dice。"""
    pred, true = _masked_pair(prediction, target, valid)
    return dice_coefficient(pred, true)


def iou_masked(prediction: Tensor, target: Tensor, valid: Tensor | None = None) -> float:
    """只在 ``valid > 0`` 的像素上计算 IoU。"""
    pred, true = _masked_pair(prediction, target, valid)
    return iou_score(pred, true)


def binary_scores(
    logits: Tensor,
    target: Tensor,
    valid: Tensor | None = None,
    threshold: float = 0.5,
) -> dict[str, float | bool]:
    """由 logits 计算单张图的 Dice/IoU，并返回是否空真值/空预测。

    返回::

        {"dice", "iou", "empty_target", "empty_prediction", "valid_pixels"}

    异常:
        ValueError: shape 不一致、``threshold`` 不在 ``(0, 1)``、有效区域为空。
    """
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold 必须在 (0, 1) 内，收到 {threshold}")
    logits = _as_binary(logits, "logits")
    target = _as_binary(target, "target")
    if logits.shape != target.shape:
        raise ValueError(
            f"logits 与 target shape 必须一致，收到 {tuple(logits.shape)} vs {tuple(target.shape)}"
        )
    if valid is None:
        valid = torch.ones_like(target)
    valid = _as_binary(valid, "valid")
    if valid.shape != target.shape:
        raise ValueError(
            f"valid shape 必须与 target 一致，收到 {tuple(valid.shape)} vs {tuple(target.shape)}"
        )
    if int((valid > 0).sum()) == 0:
        raise ValueError("有效区域为空，无法计算指标")
    prediction = (torch.sigmoid(logits) >= threshold).to(target.dtype)
    truth = (target > 0).to(target.dtype)
    return {
        "dice": dice_masked(prediction, truth, valid),
        "iou": iou_masked(prediction, truth, valid),
        "empty_target": int(((truth > 0) & (valid > 0)).sum()) == 0,
        "empty_prediction": int(((prediction > 0) & (valid > 0)).sum()) == 0,
        "valid_pixels": int((valid > 0).sum()),
    }


# ------------------------------------------------------------------- 损失
def _flatten_with_valid(
    logits: Tensor, target: Tensor, valid: Tensor
) -> tuple[Tensor, Tensor]:
    if logits.shape != target.shape:
        raise ValueError(
            f"logits 与 target shape 必须一致，收到 {tuple(logits.shape)} vs {tuple(target.shape)}"
        )
    if valid.shape != target.shape:
        raise ValueError(
            f"valid shape 必须与 target 一致，收到 {tuple(valid.shape)} vs {tuple(target.shape)}"
        )
    keep = (valid.reshape(-1) > 0)
    if int(keep.sum()) == 0:
        raise ValueError("batch 的有效区域为空，无法计算损失")
    return logits.reshape(-1)[keep], target.reshape(-1)[keep]


def masked_bce_with_logits(logits: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """只在有效像素上求 ``binary_cross_entropy_with_logits``。

    实现方式是先用 ``valid`` 把 logits/target 压平成一维再按像素取均值，等价于
    "对有效像素求平均 BCE"。不使用 ``pos_weight`` 或类别重加权，保持与后续 INT8
    实现一致的简单形式。
    """
    logits, target = _flatten_with_valid(logits, target, valid)
    return F.binary_cross_entropy_with_logits(logits, target.float(), reduction="mean")


def soft_dice_loss(logits: Tensor, target: Tensor, valid: Tensor, epsilon: float = 1.0) -> Tensor:
    """soft Dice 损失 ``1 - (2*I + eps) / (P + T + eps)``，同样只在有效像素上统计。

    ``epsilon=1.0`` 避免空掩膜时的 0/0；注意它同时意味着"全空且预测全空"时损失
    为 0（与 :func:`dice_coefficient` 的 1.0 约定一致）。
    """
    if epsilon <= 0:
        raise ValueError(f"epsilon 必须为正，收到 {epsilon}")
    logits, target = _flatten_with_valid(logits, target, valid)
    probability = torch.sigmoid(logits)
    target = target.float()
    intersection = (probability * target).sum()
    denominator = probability.sum() + target.sum()
    return 1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)


def segmentation_loss(
    logits: Tensor,
    target: Tensor,
    valid: Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    epsilon: float = 1.0,
) -> tuple[Tensor, dict[str, float]]:
    """``bce_weight * BCE + dice_weight * soft Dice``，返回 ``(loss, 分项)``。"""
    if bce_weight < 0 or dice_weight < 0:
        raise ValueError(f"权重必须非负，收到 bce={bce_weight}, dice={dice_weight}")
    if bce_weight == 0 and dice_weight == 0:
        raise ValueError("bce_weight 与 dice_weight 不能同时为 0")
    bce = masked_bce_with_logits(logits, target, valid)
    dice = soft_dice_loss(logits, target, valid, epsilon=epsilon)
    total = bce_weight * bce + dice_weight * dice
    return total, {"bce": float(bce.detach()), "soft_dice": float(dice.detach())}


# ---------------------------------------------------------------- 统计累加
class MetricAccumulator:
    """按图累计 Dice/IoU，并在最后计算均值/中位数/失败计数。

    空真值（病灶面积为 0）与空预测的图会被单独计数，且可以要求只在非空真值上
    统计，避免"空对空得 1.0"把均值抬高。
    """

    def __init__(self) -> None:
        self._scores: list[tuple[float, float, bool, bool]] = []
        self._empty_target = 0
        self._empty_prediction = 0
        self._valid_pixels = 0

    def __len__(self) -> int:
        return len(self._scores)

    def update(self, scores: Mapping[str, object]) -> None:
        required = {"dice", "iou", "empty_target", "empty_prediction"}
        missing = required - set(scores)
        if missing:
            raise ValueError(f"scores 缺少字段: {sorted(missing)}")
        empty_target = bool(scores["empty_target"])
        empty_prediction = bool(scores["empty_prediction"])
        self._scores.append((float(scores["dice"]),   # type: ignore[arg-type]
                             float(scores["iou"]),    # type: ignore[arg-type]
                             empty_target, empty_prediction))
        self._empty_target += int(empty_target)
        self._empty_prediction += int(empty_prediction)
        self._valid_pixels += int(scores.get("valid_pixels", 0))  # type: ignore[arg-type]

    @staticmethod
    def _mean(values: list[float]) -> float:
        return float(sum(values) / len(values)) if values else float("nan")

    @staticmethod
    def _median(values: list[float]) -> float:
        if not values:
            return float("nan")
        ordered = sorted(values)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return float(ordered[middle])
        return float((ordered[middle - 1] + ordered[middle]) / 2.0)

    def summary(self) -> dict[str, Any]:
        """汇总。同时给出全部图与"非空真值图"两种 Dice 均值，避免空对空抬分。"""
        if not self._scores:
            raise ValueError("没有任何已累计的样本，无法汇总")
        dice = [score[0] for score in self._scores]
        iou = [score[1] for score in self._scores]
        non_empty_dice = [score[0] for score in self._scores if not score[2]]
        return {
            "images": len(self._scores),
            "dice_mean": self._mean(dice),
            "dice_median": self._median(dice),
            "iou_mean": self._mean(iou),
            "iou_median": self._median(iou),
            "dice_mean_non_empty_target": self._mean(non_empty_dice),
            "non_empty_target_images": len(non_empty_dice),
            "empty_target_images": self._empty_target,
            "empty_prediction_images": self._empty_prediction,
            "valid_pixels_total": self._valid_pixels,
        }
