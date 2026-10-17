"""CAMUS 可视化：ED/ES 成对展示、真实 prediction 与 GT 轮廓叠加（含图例）。

* 轮廓叠加必须同时包含 **原生灰度图 + GT 轮廓（虚线）+ prediction 轮廓（实线）+ 图例**；
  禁止把 GT 叠加到 GT 上冒充预测。
* ED/ES 成对输出；LV 面积对比只汇总已计算的物理面积。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import geometry as geo

CLASS_STYLES: dict[int, dict[str, Any]] = {
    geo.LV_CAVITY: {"color": (1.0, 0.30, 0.30), "label": "LV cavity"},
    geo.MYOCARDIUM: {"color": (0.30, 0.85, 0.40), "label": "myocardium"},
    geo.LEFT_ATRIUM: {"color": (0.35, 0.55, 1.0), "label": "left atrium"},
}


def _figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def contour_overlay(image: np.ndarray, truth: np.ndarray | None, prediction: np.ndarray | None,
                    *, axis: Any = None, title: str | None = None, linewidth: float = 1.2) -> Any:
    """原生灰度图 + GT（虚线）+ 预测（实线）轮廓；带类别图例。"""
    plt = _figure()
    if axis is None:
        _, axis = plt.subplots(figsize=(4.4, 4.4))
    axis.imshow(np.asarray(image), cmap="gray")
    legend_added: set[str] = set()
    if truth is not None:
        for class_index, style in CLASS_STYLES.items():
            mask = np.asarray(truth) == class_index
            if mask.any():
                axis.contour(mask.astype(float), levels=[0.5], colors=[style["color"]],
                             linewidths=linewidth, linestyles="dashed")
                legend_added.add(style["label"] + " (GT)")
    if prediction is not None:
        for class_index, style in CLASS_STYLES.items():
            mask = np.asarray(prediction) == class_index
            if mask.any():
                axis.contour(mask.astype(float), levels=[0.5], colors=[style["color"]],
                             linewidths=linewidth, linestyles="solid")
                legend_added.add(style["label"] + " (pred)")
    handles = [plt.Line2D([], [], color=style["color"], linestyle="--", label=style["label"] + " (GT)")
               for style in CLASS_STYLES.values()]
    handles += [plt.Line2D([], [], color=style["color"], linestyle="-", label=style["label"] + " (pred)")
                for style in CLASS_STYLES.values()]
    axis.legend(handles=handles, fontsize=6, loc="lower right", framealpha=0.6)
    axis.set_xticks([])
    axis.set_yticks([])
    if title:
        axis.set_title(title, fontsize=8)
    return axis


def ed_es_pair(ed_image: np.ndarray, ed_truth: np.ndarray, ed_prediction: np.ndarray,
               es_image: np.ndarray, es_truth: np.ndarray, es_prediction: np.ndarray,
               *, title: str = "") -> Any:
    """ED/ES 成对：左 ED、右 ES，均为 原生图 + GT + prediction。"""
    plt = _figure()
    figure, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    contour_overlay(ed_image, ed_truth, ed_prediction, axis=axes[0], title="ED")
    contour_overlay(es_image, es_truth, es_prediction, axis=axes[1], title="ES")
    if title:
        figure.suptitle(title, fontsize=9)
    return figure


def lv_area_comparison(records: Sequence[Mapping[str, Any]], *, title: str = "LV area (mm^2)") -> Any:
    """按帧比较 GT 与预测的 LV 物理面积（mm²）。"""
    plt = _figure()
    figure, axis = plt.subplots(figsize=(6, 3))
    labels = [str(record.get("frame_id")) for record in records]
    truth = [record.get("gt_area_mm2", {}).get("lv_cavity") for record in records]
    predicted = [record.get("pred_area_mm2", {}).get("lv_cavity") for record in records]
    positions = np.arange(len(records))
    axis.bar(positions - 0.2, [v if v is not None else 0.0 for v in truth], width=0.4, label="GT")
    axis.bar(positions + 0.2, [v if v is not None else 0.0 for v in predicted], width=0.4, label="pred")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    axis.set_ylabel("mm^2")
    axis.set_title(title)
    axis.legend()
    return figure


def save_figure(figure: Any, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    return path
