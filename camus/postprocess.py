"""Ported fixed CAMUS island/hole postprocessing (LV-only, no GT)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

LV = 1
BACKGROUND = 0
ISLAND_RATIO = 0.05
HOLE_RATIO = 0.01
CONNECTIVITY8 = np.ones((3, 3), dtype=int)


class PostprocessError(ValueError):
    pass


@dataclass(frozen=True)
class PostprocessOutcome:
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


def _check(labels: np.ndarray) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 2 or array.size == 0 or not np.issubdtype(array.dtype, np.integer):
        raise PostprocessError("labels must be a nonempty 2-D integer array")
    if not set(np.unique(array).tolist()) <= {0, 1, 2, 3}:
        raise PostprocessError("labels must be in 0..3")
    return array


def _label8(mask: np.ndarray):
    return ndimage.label(mask, structure=CONNECTIVITY8)


def apply_island5(labels: np.ndarray) -> PostprocessOutcome:
    """Delete secondary LV only when its total is <=5% of all LV pixels."""
    array = _check(labels)
    working = array.copy()
    lv = working == LV
    lv_pixels = int(lv.sum())
    if lv_pixels == 0:
        return PostprocessOutcome("island5", working, 0, 0, 0, 0, 0, 0.0, False, 0, 0.0, False, 0)
    component_map, components = _label8(lv)
    sizes = np.bincount(component_map.ravel())
    largest_index = int(np.argmax(sizes[1:]) + 1) if components else 0
    largest = int(sizes[largest_index]) if components else 0
    secondary = lv_pixels - largest
    ratio = secondary / lv_pixels
    applied = components > 1 and ratio <= ISLAND_RATIO
    changed = 0
    if applied:
        mask = lv & (component_map != largest_index)
        changed = int(mask.sum())
        working[mask] = BACKGROUND
    return PostprocessOutcome(
        "island5", working, lv_pixels, int((working == LV).sum()), int(components),
        largest, secondary, ratio, applied, 0, 0.0, False, changed,
    )


def apply_island5_hole1(labels: np.ndarray) -> PostprocessOutcome:
    """Apply island5, then fill only small all-background holes enclosed by LV."""
    base = apply_island5(labels)
    working = base.labels.copy()
    retained = int((working == LV).sum())
    filled = 0
    hole_ratio = 0.0
    applied = False
    if retained:
        enclosed = ndimage.binary_fill_holes(working == LV, structure=CONNECTIVITY8) & (working != LV)
        hole_map, hole_count = _label8(enclosed)
        closed = []
        for index in range(1, hole_count + 1):
            mask = hole_map == index
            if np.all(working[mask] == BACKGROUND):
                closed.append((int(mask.sum()), index))
        total_closed = sum(count for count, _index in closed)
        hole_ratio = total_closed / retained
        if closed and hole_ratio <= HOLE_RATIO:
            applied = True
            for _count, index in closed:
                mask = hole_map == index
                working[mask] = LV
                filled += int(mask.sum())
    return PostprocessOutcome(
        "island5_hole1", working, base.lv_pixels_before, int((working == LV).sum()),
        base.components, base.largest_component_pixels, base.secondary_pixels,
        base.secondary_ratio, base.island_applied, filled,
        hole_ratio if retained else 0.0, applied, base.changed_pixels + filled,
    )
