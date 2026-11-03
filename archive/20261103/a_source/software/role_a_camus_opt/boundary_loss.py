"""三前景 signed-distance boundary loss（与 `PROTOCOL.json` 定义一致）。

定义：

* 对三前景结构（LV 腔 / 心肌 / 左心房）分别计算**基于当前（已成对增强）标签**的带符号距离：
  结构内部为负、外部为正，距离按 **256 canvas 对角线**归一化；
* 损失 = 三结构的 ``mean( softmax(logits)[class] * signed_distance )`` 之和的均值
  （``boundary_weight=0`` 时该路径必须与原 ``multiclass_loss`` **逐值一致**）；
* 只统计 **valid** 像素；补边/``ignore_index=255`` 不参与；**目标中不存在的类别贡献 0**；
* 不使用任何预先计算的、未增强的距离图；
* 保持 FP32：softmax/CE/Dice/求和均在 FP32 下计算，卷积可继续 AMP。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

LV = 1
MYO = 2
LA = 3
FOREGROUND: tuple[int, int, int] = (LV, MYO, LA)
IGNORE_INDEX = 255
KLASS_NAMES = {0: "background", 1: "lv_cavity", 2: "myocardium", 3: "left_atrium"}


class BoundaryLossError(ValueError):
    """输入不满足前置条件。"""


def canvas_diagonal(size: int) -> float:
    """256 canvas 的对角线长度（用于距离归一化）。"""
    return float(np.hypot(size, size))


def signed_distance_map(target: np.ndarray, class_index: int, *, valid: np.ndarray | None = None,
                        size: int | None = None) -> np.ndarray:
    """带符号距离（内部负、外部正），按 canvas 对角线归一化。

    ``valid`` 为 0 的像素（补边）在返回图中标记为 ``np.nan``，调用方必须排除。
    目标中不存在该类时返回**全 NaN**（贡献 0）。
    """
    from scipy import ndimage

    label = np.asarray(target)
    if label.ndim != 2:
        raise BoundaryLossError(f"期望二维标签，收到 shape={label.shape}")
    model_size = int(size or label.shape[0])
    structure = np.ones((3, 3), dtype=bool)
    if not np.issubdtype(label.dtype, np.integer):
        raise BoundaryLossError("标签必须为整数类型")
    if class_index not in FOREGROUND or model_size <= 0:
        raise BoundaryLossError("非法前景类别或canvas大小")
    if valid is not None and np.asarray(valid).shape != label.shape:
        raise BoundaryLossError("valid与标签形状不符")
    mask = (label == class_index) & (label != IGNORE_INDEX)
    if valid is not None:
        mask = mask & (np.asarray(valid) > 0)
    if not mask.any():
        return np.full(label.shape, np.nan, dtype=np.float32)
    inside = ndimage.distance_transform_edt(mask)            # 内部到边界
    outside = ndimage.distance_transform_edt(~mask)          # 外部到边界
    signed = np.where(mask, -(inside - 1.0), outside)
    diagonal = canvas_diagonal(model_size)
    normalized = (signed / diagonal).astype(np.float32)
    if valid is not None:
        normalized = np.where(np.asarray(valid) > 0, normalized, np.nan).astype(np.float32)
    _ = structure
    return normalized


def boundary_loss_from_maps(logits: Tensor, distance_maps: Tensor, valid: Tensor,
                            *, foreground: Sequence[int] = FOREGROUND) -> Tensor:
    """由预先算好的（当前批次的）归一化带符号距离计算边界损失。

    ``distance_maps``: ``(N, C, H, W)`` 中只填前景通道；``NaN`` 表示该像素/类别不参与。
    """
    if logits.ndim != 4 or distance_maps.shape != logits.shape:
        raise BoundaryLossError(f"logits/distance 形状不匹配: {tuple(logits.shape)} vs "
                                f"{tuple(distance_maps.shape)}")
    probabilities = torch.softmax(logits.float(), dim=1)
    if valid.shape != (logits.shape[0], 1, *logits.shape[2:]):
        raise BoundaryLossError("valid必须为(N,1,H,W)")
    keep = valid.to(logits.device).bool().squeeze(1)
    distance_maps = distance_maps.to(logits.device)
    terms = []
    for class_index in foreground:
        distance = distance_maps[:, class_index]
        usable = keep & torch.isfinite(distance)
        if not bool(usable.any()):
            terms.append(probabilities[:, class_index].sum() * 0.0)
            continue
        terms.append((probabilities[:, class_index][usable] * distance[usable]).mean())
    return torch.stack(terms).mean() if terms else torch.zeros((), dtype=torch.float32,
                                                              device=logits.device)


def boundary_maps_for_batch(target: Tensor, valid: Tensor, *, size: int,
                            foreground: Sequence[int] = FOREGROUND) -> Tensor:
    """对当前批次（已成对增强后的标签）生成距离图张量（CPU，供训练循环调用）。"""
    labels = target.detach().cpu().numpy()
    valids = valid.detach().cpu().numpy()
    batch, height, width = labels.shape
    maps = np.zeros((batch, 4, height, width), dtype=np.float32)
    for index in range(batch):
        for class_index in foreground:
            maps[index, class_index] = signed_distance_map(labels[index], class_index,
                                                           valid=valids[index, 0], size=size)
    return torch.from_numpy(maps)


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    parts: dict[str, float]

    def as_dict(self) -> dict[str, float]:
        return dict(self.parts)


def multiclass_loss_with_boundary(logits: Tensor, target: Tensor, valid: Tensor | None = None,
                                  distance_maps: Tensor | None = None, *,
                                  ce_weight: float = 1.0, dice_weight: float = 1.0,
                                  boundary_weight: float = 0.0, size: int = 256,
                                  foreground: Sequence[int] = FOREGROUND
                                  ) -> tuple[Tensor, dict[str, float]]:
    """CE + 三前景 soft Dice + 可选的 boundary 项；``boundary_weight=0`` 时与原实现逐值一致。"""
    from software.role_a_camus import train as base_train

    if target.shape != (logits.shape[0], *logits.shape[2:]):
        raise BoundaryLossError("标签与logits形状不符")
    effective = target != IGNORE_INDEX
    if valid is not None:
        if valid.shape != (logits.shape[0], 1, *logits.shape[2:]):
            raise BoundaryLossError("valid必须为(N,1,H,W)")
        effective &= valid.bool().squeeze(1)
    if not bool(effective.any()):
        raise BoundaryLossError("批次没有有效标签像素")
    if not np.isfinite(boundary_weight) or boundary_weight < 0:
        raise BoundaryLossError("边界损失权重必须非负且finite")
    base_loss, breakdown = base_train.multiclass_loss(logits, target, valid,
                                                      ce_weight=ce_weight, dice_weight=dice_weight,
                                                      foreground=foreground)
    if boundary_weight == 0.0:
        return base_loss, {"loss": float(base_loss.detach()), "ce": breakdown["ce"],
                           "foreground_dice": breakdown["foreground_dice"],
                           "boundary": 0.0}
    if distance_maps is None:
        raise BoundaryLossError("boundary_weight != 0 时必须提供当前批次的 distance_maps")
    boundary = boundary_loss_from_maps(logits, distance_maps,
                                       valid if valid is not None
                                       else torch.ones_like(target, dtype=torch.float32
                                                            ).unsqueeze(1))
    total = base_loss.float() + float(boundary_weight) * boundary
    return total, {"loss": float(total.detach()), "ce": breakdown["ce"],
                   "foreground_dice": breakdown["foreground_dice"],
                   "boundary": float(boundary.detach())}


def multiclass_loss_for_arm(logits: Tensor, target: Tensor, valid: Tensor, *,
                            boundary_weight: float, size: int, ce_weight: float = 1.0,
                            dice_weight: float = 1.0) -> tuple[Tensor, dict[str, float]]:
    """训练循环入口：按权重决定是否构造当前批次距离图（λ=0 时完全不构造）。"""
    maps = None
    if boundary_weight != 0.0:
        maps = boundary_maps_for_batch(target, valid, size=size)
    return multiclass_loss_with_boundary(logits, target, valid, maps,
                                         ce_weight=ce_weight, dice_weight=dice_weight,
                                         boundary_weight=boundary_weight, size=size)


def assert_parameters_finite(module: nn.Module) -> None:
    """所有参数必须 finite（CPU 检查用）。"""
    for name, parameter in module.named_parameters():
        if not bool(torch.isfinite(parameter).all()):
            raise BoundaryLossError(f"参数非有限: {name}")


def describe(lambda_value: float, size: int) -> dict[str, Any]:
    return {"boundary_weight": float(lambda_value), "size": size,
            "normalization": "signed distance / canvas diagonal",
            "foreground": list(FOREGROUND), "amp": "convs AMP, losses FP32"}
