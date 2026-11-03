"""CAMUS 连通性后处理（LV-only，纯预测几何规则，**绝不使用 GT**）。

规则（与 `PROTOCOL.json` 的 `postprocessing` 完全一致）：

* 只处理 **LV 标签 1**，前景连通性为 **8 连通**；
* ``island5``：仅当「非最大 8 连通分量的 LV 像素合计 <= 总 LV 的 5%」时，把这些 secondary LV
  像素**改判为背景 0**；超过 5% 则**全部保留**（视为真正断裂，不做修补）；
* ``island5_hole1``：在 ``island5`` 之后，**只填**「完全封闭（不接触图像边界）且整体为背景 0」
  的孔洞，且孔洞合计 <= 保留 LV 的 **1%**；**不覆盖 MYO(2)/LA(3)**，打开到图边的区域不填；
* **绝不修改输入数组**；不改变 LV 之外任何标签；不使用任何 GT 信息。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

LV = 1
MYO = 2
LA = 3
BACKGROUND = 0
ISLAND_RATIO = 0.05
HOLE_RATIO = 0.01
VARIANTS = ("raw_reference", "island5", "island5_hole1")
_CONNECTIVITY8 = np.ones((3, 3), dtype=int)


class PostprocessError(ValueError):
    """输入标签不满足前置条件。"""


@dataclass(frozen=True)
class PostprocessOutcome:
    """单个变体的结果与可审计统计。"""

    variant: str
    labels: np.ndarray
    lv_pixels_before: int
    lv_pixels_after: int
    components: int
    largest_component_pixels: int
    secondary_pixels: int
    secondary_ratio: float
    island_applied: bool
    hole_pixels_filled: int
    hole_ratio: float
    hole_applied: bool
    changed_pixels: int

    def as_dict(self) -> dict[str, Any]:
        return {"variant": self.variant, "lv_pixels_before": self.lv_pixels_before,
                "lv_pixels_after": self.lv_pixels_after, "components": self.components,
                "largest_component_pixels": self.largest_component_pixels,
                "secondary_pixels": self.secondary_pixels,
                "secondary_ratio": self.secondary_ratio, "island_applied": self.island_applied,
                "hole_pixels_filled": self.hole_pixels_filled, "hole_ratio": self.hole_ratio,
                "hole_applied": self.hole_applied, "changed_pixels": self.changed_pixels}


def _check(labels: np.ndarray) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 2:
        raise PostprocessError(f"期望二维标签 (H, W)，收到 shape={array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise PostprocessError("标签必须为整数类型")
    if array.size == 0:
        raise PostprocessError("标签为空数组")
    values = set(np.unique(array).tolist())
    if not values <= {0, 1, 2, 3}:
        raise PostprocessError(f"标签取值超出 0..3: {sorted(values)}")
    return array


def _label8(mask: np.ndarray) -> tuple[np.ndarray, int]:
    from scipy import ndimage

    return ndimage.label(mask, structure=_CONNECTIVITY8)


def apply_island5(labels: np.ndarray) -> PostprocessOutcome:
    """仅当 secondary LV 合计 <= 5% 时删除 secondary（改背景），否则原样保留。"""
    array = _check(labels)
    working = array.copy()
    lv = working == LV
    lv_pixels = int(lv.sum())
    if lv_pixels == 0:
        return PostprocessOutcome("island5", working, 0, 0, 0, 0, 0, 0.0, False, 0, 0.0, False, 0)
    component_map, components = _label8(lv)
    sizes = np.bincount(component_map.ravel())
    largest_index = int(np.argmax(sizes[1:]) + 1) if components else 0
    largest_pixels = int(sizes[largest_index]) if components else 0
    secondary_pixels = lv_pixels - largest_pixels
    ratio = secondary_pixels / lv_pixels
    applied = components > 1 and ratio <= ISLAND_RATIO
    changed = 0
    if applied:
        secondary_mask = lv & (component_map != largest_index)
        changed = int(secondary_mask.sum())
        working[secondary_mask] = BACKGROUND
    return PostprocessOutcome("island5", working, lv_pixels, int((working == LV).sum()),
                              int(components), largest_pixels, secondary_pixels, ratio, applied,
                              0, 0.0, False, changed)


def apply_island5_hole1(labels: np.ndarray, *, island_outcome: PostprocessOutcome | None = None
                        ) -> PostprocessOutcome:
    """先执行 island5，再填「完全封闭且全背景」的孔洞（合计 <= 1% 保留 LV）。"""
    base = island_outcome if island_outcome is not None else apply_island5(labels)
    working = base.labels.copy()
    retained_lv = int((working == LV).sum())
    filled = 0
    hole_ratio = 0.0
    applied = False
    if retained_lv > 0:
        from scipy import ndimage
        # A hole must be enclosed by LV itself, not by a MYO or LA ring.
        enclosed = ndimage.binary_fill_holes(working == LV, structure=_CONNECTIVITY8) & (working != LV)
        hole_map, hole_count = ndimage.label(enclosed, structure=_CONNECTIVITY8)
        closed = []
        for index in range(1, hole_count + 1):
            mask = hole_map == index
            if np.all(working[mask] == BACKGROUND):
                closed.append((int(mask.sum()), index))
        total_closed = sum(count for count, _index in closed)
        hole_ratio = total_closed / retained_lv
        if closed and hole_ratio <= HOLE_RATIO:
            applied = True
            for _count, index in closed:
                mask = hole_map == index
                working[mask] = LV
                filled += int(mask.sum())
    return PostprocessOutcome("island5_hole1", working, base.lv_pixels_before,
                              int((working == LV).sum()), base.components,
                              base.largest_component_pixels, base.secondary_pixels,
                              base.secondary_ratio, base.island_applied, filled,
                              hole_ratio if retained_lv else 0.0, applied,
                              base.changed_pixels + filled)


def process(labels: np.ndarray, variant: str) -> PostprocessOutcome:
    """按变体执行后处理；``raw_reference`` 原样返回（不修改输入）。"""
    array = _check(labels)
    if variant == "raw_reference":
        lv = int((array == LV).sum())
        component_map, components = _label8(array == LV) if lv else (None, 0)
        largest = 0
        if components:
            sizes = np.bincount(component_map.ravel())
            largest = int(sizes[1:].max())
        return PostprocessOutcome("raw_reference", array.copy(), lv, lv, int(components), largest,
                                  lv - largest, (lv - largest) / lv if lv else 0.0, False,
                                  0, 0.0, False, 0)
    if variant == "island5":
        return apply_island5(array)
    if variant == "island5_hole1":
        return apply_island5_hole1(array)
    raise PostprocessError(f"未知变体: {variant!r}（可选 {VARIANTS}）")


def process_all(labels: np.ndarray) -> dict[str, PostprocessOutcome]:
    """一次得到三个固定版本（同一份输入，互不影响）。"""
    island = apply_island5(labels)
    return {"raw_reference": process(labels, "raw_reference"), "island5": island,
            "island5_hole1": apply_island5_hole1(labels, island_outcome=island)}
