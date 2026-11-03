"""CAMUS 指标：逐结构 Dice/召回/覆盖、按帧再按患者等权、物理面积与 HD95。

口径：

* 主指标 = 三前景结构（LV 腔 / 心肌 / 左心房）Dice 的按帧等权 macro，再按患者等权。
* ``ignore_index``（letterbox 补边）不参与统计；``valid`` 掩膜亦可屏蔽。
* **GT 空、预测也空** → ``None``（无法定义 Dice）并计数；
  **GT 空但预测非空** → Dice 数学定义为 **0**（假阳性不得被忽略）；
  **预测空但 GT 非空** → Dice = 0、召回 = 0。
* 面积单位 mm²；HD95 单位 mm。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from . import geometry as geo

NUM_CLASSES = 4
IGNORE_INDEX = geo.IGNORE_INDEX


@dataclass
class ClassResult:
    class_index: int
    name: str
    dice: float | None
    recall: float | None
    precision: float | None
    gt_pixels: int
    pred_pixels: int
    intersection: int
    reason: str | None = None


def _keep_mask(truth: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    mask = np.asarray(truth) != IGNORE_INDEX
    if valid is not None:
        mask &= np.asarray(valid) > 0
    return mask


def class_scores(prediction: np.ndarray, truth: np.ndarray, class_index: int, *,
                 valid: np.ndarray | None = None) -> ClassResult:
    """单类 Dice / 召回 / 精确率（严格按数学定义，不掩饰假阳性）。"""
    pred = np.asarray(prediction)
    real = np.asarray(truth)
    if pred.shape != real.shape:
        raise geo.LabelError(f"pred/truth 形状不一致: {pred.shape} vs {real.shape}")
    mask = _keep_mask(real, valid)
    pred_mask = (pred == class_index) & mask
    real_mask = (real == class_index) & mask
    intersection = int(np.logical_and(pred_mask, real_mask).sum())
    pred_pixels = int(pred_mask.sum())
    gt_pixels = int(real_mask.sum())
    name = geo.CLASS_NAMES[class_index]
    if gt_pixels == 0 and pred_pixels == 0:
        return ClassResult(class_index, name, None, None, None, 0, 0, 0,
                           reason="空 GT 且空预测：Dice 无定义，记为 null")
    if gt_pixels == 0:
        return ClassResult(class_index, name, 0.0, None, 0.0, 0, pred_pixels, 0,
                           reason="空 GT 但存在预测：Dice 定义为 0（假阳性计入）")
    if pred_pixels == 0:
        return ClassResult(class_index, name, 0.0, 0.0, None, gt_pixels, 0, 0,
                           reason="空预测：Dice=0，召回=0")
    dice = 2.0 * intersection / (pred_pixels + gt_pixels)
    return ClassResult(class_index, name, float(dice), float(intersection / gt_pixels),
                       float(intersection / pred_pixels), gt_pixels, pred_pixels, intersection)


def frame_scores(prediction: np.ndarray, truth: np.ndarray, *, valid: np.ndarray | None = None,
                 foreground: Sequence[int] = geo.FOREGROUND_CLASSES) -> dict[str, Any]:
    """一帧的多结构指标 + 前景 macro（``None`` 类不参与均值但被列出）。"""
    results = {index: class_scores(prediction, truth, index, valid=valid) for index in foreground}
    dice_values = [r.dice for r in results.values() if r.dice is not None]
    recall_values = [r.recall for r in results.values() if r.recall is not None]
    return {"per_class": {r.name: {"dice": r.dice, "recall": r.recall, "precision": r.precision,
                                   "gt_pixels": r.gt_pixels, "pred_pixels": r.pred_pixels,
                                   "reason": r.reason} for r in results.values()},
            "foreground_macro_dice": (float(np.mean(dice_values)) if dice_values else None),
            "foreground_macro_recall": (float(np.mean(recall_values)) if recall_values else None),
            "null_classes": [r.name for r in results.values() if r.dice is None]}


def native_frame_scores(prediction_native: np.ndarray, truth_native: np.ndarray,
                        spacing_mm: Sequence[float], *, quality: str = "unknown",
                        patient: str = "unknown", view: str | None = None, phase: str | None = None,
                        frame_id: str | None = None, valid: np.ndarray | None = None
                        ) -> dict[str, Any]:
    """原生网格（native）逐帧记录：Dice/召回/覆盖、物理面积、spacing-aware HD95。"""
    truth = np.asarray(truth_native)
    record: dict[str, Any] = {"frame_id": frame_id, "patient": patient, "view": view,
                              "phase": phase, "quality": quality,
                              "pixel_area_mm2": float(spacing_mm[0]) * float(spacing_mm[1])}
    record.update(frame_scores(prediction_native, truth, valid=valid))
    record["hd95_mm"] = {name: geo.hd95_mm(np.asarray(prediction_native) == index, truth == index,
                                           float(spacing_mm[0]), float(spacing_mm[1]))
                         for index, name in geo.CLASS_NAMES.items() if index != geo.BACKGROUND}
    record["gt_area_mm2"] = {
        name: float((truth == index).sum()) * record["pixel_area_mm2"]
        for index, name in geo.CLASS_NAMES.items() if index != geo.BACKGROUND}
    record["pred_area_mm2"] = {
        name: float((np.asarray(prediction_native) == index).sum()) * record["pixel_area_mm2"]
        for index, name in geo.CLASS_NAMES.items() if index != geo.BACKGROUND}
    return record


def coverage_ratio(prediction: np.ndarray, truth: np.ndarray, class_index: int, *,
                   valid: np.ndarray | None = None) -> float | None:
    """覆盖比 = 交集 / GT 像素（即召回）；空 GT 返回 ``None``。"""
    return class_scores(prediction, truth, class_index, valid=valid).recall


def low_coverage_cases(frames: Sequence[Mapping[str, Any]], threshold: float = 0.5
                       ) -> list[dict[str, Any]]:
    """列出前景结构召回 < 阈值或空预测的帧，供逐例复核。"""
    flagged: list[dict[str, Any]] = []
    for frame in frames:
        for name, values in frame.get("per_class", {}).items():
            recall = values.get("recall")
            if recall is not None and recall < threshold:
                flagged.append({"frame": frame.get("frame_id"), "patient": frame.get("patient"),
                                "class": name, "recall": recall,
                                "gt_pixels": values.get("gt_pixels"),
                                "pred_pixels": values.get("pred_pixels")})
    return flagged


def _mean_or_none(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if values else None


def aggregate_frames(frames: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    """按 view/phase/quality 分组，组内按帧等权平均 macro Dice。"""
    buckets: dict[str, list[float]] = {}
    for frame in frames:
        value = frame.get("foreground_macro_dice")
        if value is None:
            continue
        buckets.setdefault(str(frame.get(key)), []).append(float(value))
    return {group: {"n_frames": len(values), "macro_dice": _mean_or_none(values)}
            for group, values in sorted(buckets.items())}


def patient_macro(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """患者级等权：先患者内按帧平均，再对患者平均。"""
    by_patient: dict[str, list[float]] = {}
    null_frames = 0
    for frame in frames:
        value = frame.get("foreground_macro_dice")
        if value is None:
            null_frames += 1
            continue
        by_patient.setdefault(str(frame.get("patient")), []).append(float(value))
    per_patient = {patient: _mean_or_none(values) for patient, values in sorted(by_patient.items())}
    values = [v for v in per_patient.values() if v is not None]
    return {"per_patient": per_patient, "patient_count": len(values),
            "patient_balanced_macro_dice": _mean_or_none(values),
            "null_frames": null_frames}


def evaluate_frames(frames: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """完整汇总：主指标（帧等权 → 患者等权）、分类别、分视角/时相/质量、低覆盖清单。"""
    macro = [f["foreground_macro_dice"] for f in frames
             if f.get("foreground_macro_dice") is not None]
    by_class: dict[str, list[float]] = {}
    for frame in frames:
        for name, values in frame.get("per_class", {}).items():
            if values.get("dice") is not None:
                by_class.setdefault(name, []).append(float(values["dice"]))
    hd95_values: dict[str, list[float]] = {}
    for frame in frames:
        for name, value in (frame.get("hd95_mm") or {}).items():
            if value is not None:
                hd95_values.setdefault(name, []).append(float(value))
    return {"frame_count": len(frames),
            "macro_dice_frame_balanced": _mean_or_none(macro),
            "primary_metric_patient_balanced": patient_macro(frames)["patient_balanced_macro_dice"],
            "per_class_frame_balanced": {name: _mean_or_none(values)
                                         for name, values in sorted(by_class.items())},
            "hd95_mm_frame_balanced": {name: _mean_or_none(values)
                                       for name, values in sorted(hd95_values.items())},
            "by_view": aggregate_frames(frames, "view"),
            "by_phase": aggregate_frames(frames, "phase"),
            "by_quality": aggregate_frames(frames, "quality"),
            "patient_balanced": patient_macro(frames),
            "low_coverage": low_coverage_cases(frames),
            "note": "null 类不参与均值；空 GT/无效帧单独计数，不伪造满分或 0 误差"}


def dice_macro_for_selection(frames: Sequence[Mapping[str, Any]]) -> float | None:
    """每 epoch 快速 native Dice 选模用：患者等权 macro（不依赖 dev loss）。"""
    return evaluate_frames(frames)["primary_metric_patient_balanced"]
