"""A17：soup INT8 的固定比例多分量保留 + 填洞（复用 A11 的一次遍历实现，不改旧模块）。

规则（整数比较，避免浮点取整歧义）：保留所有满足
    ``100 * 分量面积 >= p * 最大分量面积``  其中 p in {5, 10, 20}
的 8 连通前景分量；随后按 A11 的四连通背景规则填洞。空预测保持空；相等时全部保留；
无效区始终为背景。三个比例固定，不按单图真值自适应。

复杂度（如实）：
* ``apply_ratio_rule``（单规则）= **1 次 8 连通标记 + 1 次填洞扫图 = O(H*W)**；
* ``apply_all_ratios``（三比例）= **共享 1 次 8 连通标记 + 3 次输出各自一次填洞扫图 = O(3*H*W)**。

两者都复用 A11 的 ``postprocess_int8._label_components`` / ``fill_holes``，不新增算子；
但三比例总耗时**不是**单次整图遍历，请按 O(3*H*W) 理解。
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from software.role_a import postprocess_int8 as pp

RATIOS: tuple[int, ...] = (5, 10, 20)
RULE_NAMES: tuple[str, ...] = tuple(f"keep_ratio_{p}" for p in RATIOS)


def keep_labels_by_ratio(sizes: Sequence[int], percent: int) -> set[int]:
    """返回应保留的标签集合（标签从 1 开始，与 A11 ``_label_components`` 一致）。

    判据为整数比较 ``100 * size >= percent * max_size``；边界相等时保留（``>=``）。
    """
    percent = int(percent)
    if percent < 0:
        raise ValueError(f"percent 必须非负，收到 {percent}")
    sizes = [int(s) for s in sizes]
    if not sizes:
        return set()
    largest = max(sizes)
    return {index + 1 for index, size in enumerate(sizes) if 100 * size >= percent * largest}


def apply_ratio_rule(pred: np.ndarray, valid: np.ndarray, percent: int) -> np.ndarray:
    """按比例保留 8 连通分量，再填洞；只在 valid 内处理。"""
    pred = np.asarray(pred, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if pred.shape != valid.shape:
        raise ValueError("pred 与 valid 形状不一致")
    inside = pred & valid                      # 无效区恒为背景
    if not inside.any():
        return np.zeros_like(inside)
    labels, count, sizes, _positions = pp._label_components(inside, connectivity=8)
    kept = keep_labels_by_ratio(sizes, percent)
    if not kept:
        return np.zeros_like(inside)
    mask = np.isin(labels, np.fromiter(kept, dtype=np.int32, count=len(kept)))
    return pp.fill_holes(mask, valid)


def apply_all_ratios(pred: np.ndarray, valid: np.ndarray,
                     stats: dict | None = None) -> dict[str, np.ndarray]:
    """一次性返回三个固定比例的规则输出：**共享同一次 8 连通标记**。

    复杂度（如实）：连通域标记 1 次 O(H*W)；三个比例各自构造掩膜并填洞，各需扫图一次，
    合计 O(3*H*W)。整体仍是线性阶，但不是单次遍历。
    """
    pred = np.asarray(pred, dtype=bool)
    valid = np.asarray(valid, dtype=bool)
    if pred.shape != valid.shape:
        raise ValueError("pred 与 valid 形状不一致")
    inside = pred & valid
    outs = {f"keep_ratio_{p}": np.zeros_like(inside) for p in RATIOS}
    if not inside.any():
        if stats is not None:
            stats.update({"components_total": 0, "largest_component_px": 0,
                          "kept": {p: 0 for p in RATIOS}, "sizes": []})
        return outs
    labels, count, sizes, _positions = pp._label_components(inside, connectivity=8)  # 仅此一次
    if stats is not None:
        stats.update({"components_total": int(count),
                      "largest_component_px": int(max(sizes)) if sizes else 0,
                      "kept": {}, "sizes": [int(s) for s in sizes]})
    for p in RATIOS:
        kept = keep_labels_by_ratio(sizes, p)
        if stats is not None:
            stats["kept"][int(p)] = int(len(kept))
        if not kept:
            continue
        mask = np.isin(labels, np.fromiter(sorted(kept), dtype=np.int32, count=len(kept)))
        outs[f"keep_ratio_{p}"] = pp.fill_holes(mask, valid)      # 每个比例各一次扫图
    return outs


def complexity_note() -> dict[str, object]:
    return {
        "labels": "A11 postprocess_int8._label_components（8 连通，单次遍历）",
        "fill_holes": "A11 postprocess_int8.fill_holes（4 连通背景，单次遍历）",
        "time": "apply_ratio_rule 单规则 = O(H*W)；apply_all_ratios 三比例 = O(3*H*W)（共享 1 次标记 + 3 次填洞扫图）",
        "space": "O(H*W)（bool 标签画布 + int32 标签 + 少量临时）",
        "backend_calls": "不依赖 scipy；不新增算子",
        "ratios": list(RATIOS),
        "included_rule": "100 * size >= percent * max_size（相等保留）",
    }