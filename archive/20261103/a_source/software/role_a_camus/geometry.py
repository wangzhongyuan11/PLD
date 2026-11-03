"""CAMUS 几何：半像素中心的重采样、letterbox、原网格重建、HD95 与物理面积。

关键约定（修正上一版"两次 floor+nearest 自逆"的错误）：

* 缩放采用**半像素中心**约定：目标下标 ``j`` 对应源坐标 ``src = (j + 0.5) * scale - 0.5``，
  其中 ``scale = orig / new``（各轴独立，按实际 ``new_h/orig_h``、``new_w/orig_w``）。
* 灰度图像与 4 通道 logits 用**双线性**；类别标签用**最近邻**（保类别语义）。
* 原网格重建顺序：先裁掉补边 → 逐通道双线性回到原尺寸 → **再 argmax**；
  禁止先 argmax 再放大当作 native 主指标。
* 补边处标签记为 ``ignore_index=255``，不参与损失与指标。
* 灰度归一化固定 ``clip(0,255)/255``（不改动输入数组）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

BACKGROUND = 0
LV_CAVITY = 1
MYOCARDIUM = 2
LEFT_ATRIUM = 3
FOREGROUND_CLASSES: tuple[int, int, int] = (LV_CAVITY, MYOCARDIUM, LEFT_ATRIUM)
CLASS_NAMES: dict[int, str] = {BACKGROUND: "background", LV_CAVITY: "lv_cavity",
                               MYOCARDIUM: "myocardium", LEFT_ATRIUM: "left_atrium"}
PHASES: tuple[str, ...] = ("ED", "ES")
VIEWS: tuple[str, ...] = ("2CH", "4CH")
IGNORE_INDEX = 255
GRAY_MAX = 255.0


class LabelError(ValueError):
    """标签/几何不满足语义约束。"""


@dataclass(frozen=True)
class VolumeGeometry:
    """一帧的物理几何（spacing 单位 mm，顺序为 ``(spacing_y, spacing_x)``）。"""

    height: int
    width: int
    spacing_y_mm: float
    spacing_x_mm: float
    patient: str | None = None
    phase: str | None = None
    view: str | None = None
    source_path: str | None = None

    @property
    def pixel_area_mm2(self) -> float:
        return float(self.spacing_y_mm) * float(self.spacing_x_mm)

    def validate(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise LabelError(f"非法尺寸: {self.height}x{self.width}")
        for name, value in (("spacing_y_mm", self.spacing_y_mm), ("spacing_x_mm", self.spacing_x_mm)):
            if not math.isfinite(value) or value <= 0:
                raise LabelError(f"非法 {name}: {value!r}")


def geometry_from_nifti(header: Any, *, patient: str | None = None, phase: str | None = None,
                        view: str | None = None, source_path: str | None = None) -> VolumeGeometry:
    """从 NIfTI header 取 ``(H, W)`` 与对应 spacing。

    数组轴为 ``(X, Y, Z)``、显示为 ``(H, W) = (Y, X)``，因此 zooms 亦交换为
    ``(spacing_y, spacing_x)``；末尾 singleton 维度允许（二维帧）。
    """
    zooms = tuple(float(z) for z in header.get_zooms()[:3])
    shape = tuple(int(s) for s in header.get_data_shape()[:3])
    if len(zooms) < 2 or len(shape) < 2:
        raise LabelError(f"header 维度不足: shape={shape!r} zooms={zooms!r}")
    geometry = VolumeGeometry(height=shape[1], width=shape[0],
                              spacing_y_mm=abs(zooms[1]), spacing_x_mm=abs(zooms[0]),
                              patient=patient, phase=phase, view=view, source_path=source_path)
    geometry.validate()
    return geometry


def normalize_gray(image: np.ndarray) -> np.ndarray:
    """灰度归一化：``clip(0,255)/255``；输入不被修改，输出为新的 float32 数组。"""
    array = np.asarray(image, dtype=np.float32)
    return (np.clip(array, 0.0, GRAY_MAX) / GRAY_MAX).astype(np.float32)


def validate_label(label: np.ndarray, *, num_classes: int = 4, ignore_index: int = IGNORE_INDEX) -> None:
    """标签语义校验：整数、非负、不超类别数（ignore_index 除外）。"""
    array = np.asarray(label)
    if array.ndim != 2:
        raise LabelError(f"标签必须是二维 (H, W)，收到 shape={array.shape}")
    if array.size == 0:
        raise LabelError("标签为空数组")
    if not np.issubdtype(array.dtype, np.integer):
        if not np.all(np.isfinite(array)):
            raise LabelError("标签含 NaN/Inf")
        if not np.all(array == np.floor(array)):
            raise LabelError("标签含非整数值，类别标签必须为整数")
        array = array.astype(np.int64)
    values = np.unique(array)
    negative = values[values < 0]
    if negative.size:
        raise LabelError(f"标签含负值: {negative.tolist()}")
    illegal = values[(values >= num_classes) & (values != ignore_index)]
    if illegal.size:
        raise LabelError(f"标签值超出类别范围 0..{num_classes - 1}: {illegal.tolist()}")


def _source_coordinates(target: int, scale: float) -> np.ndarray:
    """半像素中心约定的逆向坐标 ``src = (j + 0.5) * scale - 0.5``。"""
    return (np.arange(target, dtype=np.float64) + 0.5) * scale - 0.5


def _bilinear_axis(array: np.ndarray, new_size: int, axis: int, scale: float) -> np.ndarray:
    coordinates = np.clip(_source_coordinates(new_size, scale), 0.0, array.shape[axis] - 1)
    lower = np.floor(coordinates).astype(np.int64)
    upper = np.minimum(lower + 1, array.shape[axis] - 1)
    weight = (coordinates - lower).astype(np.float32)
    shape_lower = [1] * array.ndim
    shape_lower[axis] = new_size
    shape_upper = list(shape_lower)
    view_lower = np.take(array, lower, axis=axis)
    view_upper = np.take(array, upper, axis=axis)
    if axis == 0:
        weight = weight[:, None]
        return view_lower * (1.0 - weight) + view_upper * weight
    weight = weight[None, :]
    return view_lower * (1.0 - weight) + view_upper * weight


def resize_bilinear(array: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """双线性缩放到 ``(new_h, new_w)``（半像素中心，各轴独立）。"""
    source = np.asarray(array, dtype=np.float32)
    if source.ndim != 2:
        raise LabelError(f"resize_bilinear 期望二维数组，收到 shape={source.shape}")
    scale_y = source.shape[0] / float(new_h)
    scale_x = source.shape[1] / float(new_w)
    intermediate = _bilinear_axis(source, new_h, 0, scale_y)
    return _bilinear_axis(intermediate, new_w, 1, scale_x)


def resize_nearest(array: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """最近邻缩放到 ``(new_h, new_w)``（半像素中心取向）。"""
    source = np.asarray(array)
    if source.ndim != 2:
        raise LabelError(f"resize_nearest 期望二维数组，收到 shape={source.shape}")
    scale_y = source.shape[0] / float(new_h)
    scale_x = source.shape[1] / float(new_w)
    rows = np.clip(np.floor(_source_coordinates(new_h, scale_y) + 0.5).astype(np.int64),
                   0, source.shape[0] - 1)
    cols = np.clip(np.floor(_source_coordinates(new_w, scale_x) + 0.5).astype(np.int64),
                   0, source.shape[1] - 1)
    return np.ascontiguousarray(source[rows][:, cols])


def letterbox(image: np.ndarray, size: int = 256, *, pad_value: float = 0.0
              ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """等比缩放 + 居中补边到 ``size×size``（双线性），返回 ``(canvas, valid, meta)``。"""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2:
        raise LabelError(f"letterbox 期望二维数组，收到 shape={array.shape}")
    height, width = array.shape
    scale = min(size / height, size / width)
    new_h = max(1, min(size, int(round(height * scale))))
    new_w = max(1, min(size, int(round(width * scale))))
    resized = resize_bilinear(array, new_h, new_w)
    canvas = np.full((size, size), float(pad_value), dtype=np.float32)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    valid = np.zeros((size, size), dtype=np.float32)
    valid[top:top + new_h, left:left + new_w] = 1.0
    meta = {"original_height": height, "original_width": width, "scale": scale,
            "new_height": new_h, "new_width": new_w, "top": top, "left": left, "size": size}
    return canvas, valid, meta


def letterbox_labels(labels: np.ndarray, meta: Mapping[str, Any], *, num_classes: int = 4,
                     ignore_index: int = IGNORE_INDEX) -> np.ndarray:
    """标签按最近邻放入 letterbox 画布，补边记 ``ignore_index``（不伪造背景）。"""
    validate_label(labels, num_classes=num_classes, ignore_index=ignore_index)
    array = np.asarray(labels)
    resized = resize_nearest(array, int(meta["new_height"]), int(meta["new_width"]))
    size = int(meta["size"])
    canvas = np.full((size, size), ignore_index, dtype=np.int64)
    canvas[int(meta["top"]):int(meta["top"]) + int(meta["new_height"]),
           int(meta["left"]):int(meta["left"]) + int(meta["new_width"])] = resized
    return canvas


def crop_letterbox(canvas: np.ndarray, meta: Mapping[str, Any]) -> np.ndarray:
    """只裁掉补边，得到 ``(new_h, new_w)`` 内容区。"""
    array = np.asarray(canvas)
    if array.ndim == 2:
        return np.ascontiguousarray(array[int(meta["top"]):int(meta["top"]) + int(meta["new_height"]),
                                          int(meta["left"]):int(meta["left"]) + int(meta["new_width"])])
    return np.ascontiguousarray(array[..., int(meta["top"]):int(meta["top"]) + int(meta["new_height"]),
                                      int(meta["left"]):int(meta["left"]) + int(meta["new_width"])])


def restore_to_native(canvas: np.ndarray, meta: Mapping[str, Any], *, mode: str = "bilinear") -> np.ndarray:
    """裁补边后按给定模式重建到原尺寸。

    ``mode="bilinear"`` 用于灰度与 logits；``mode="nearest"`` 用于标签。
    """
    cropped = crop_letterbox(canvas, meta)
    if mode == "bilinear":
        if cropped.ndim == 2:
            return resize_bilinear(cropped, int(meta["original_height"]), int(meta["original_width"]))
        return np.stack([resize_bilinear(cropped[index], int(meta["original_height"]),
                                         int(meta["original_width"]))
                         for index in range(cropped.shape[0])], axis=0)
    if mode == "nearest":
        if cropped.ndim == 2:
            return resize_nearest(cropped, int(meta["original_height"]), int(meta["original_width"]))
        raise LabelError("类别标签只能二维；多通道请逐通道处理")
    raise LabelError(f"未知重建模式: {mode!r}")


def argmax_on_native(logits: np.ndarray, meta: Mapping[str, Any]) -> np.ndarray:
    """**先**逐通道双线性重建到原网格，**再** argmax（native 主指标的唯一入口）。"""
    array = np.asarray(logits)
    if array.ndim != 3:
        raise LabelError(f"期望 (C, H, W) logits，收到 shape={array.shape}")
    native_continuous = restore_to_native(array, meta, mode="bilinear")
    return np.argmax(native_continuous, axis=0).astype(np.int64)


def argmax_on_native_torch(logits: torch.Tensor, meta: Mapping[str, Any]) -> torch.Tensor:
    """torch 版 native 重建 + argmax（训练/评估中直接用 GPU 张量）。"""
    if logits.ndim != 4:
        raise LabelError(f"期望 (N, C, H, W) logits，收到 shape={tuple(logits.shape)}")
    cropped = logits[..., int(meta["top"]):int(meta["top"]) + int(meta["new_height"]),
                     int(meta["left"]):int(meta["left"]) + int(meta["new_width"])]
    restored = torch.nn.functional.interpolate(
        cropped.float(), size=(int(meta["original_height"]), int(meta["original_width"])),
        mode="bilinear", align_corners=False)
    return torch.argmax(restored, dim=1)


def class_pixel_area_mm2(labels: np.ndarray, geometry: VolumeGeometry, class_index: int = LV_CAVITY
                         ) -> float:
    """某一类别的物理面积（mm²）；类别为空时为 0.0（面积本身有定义）。"""
    geometry.validate()
    return float((np.asarray(labels) == int(class_index)).sum()) * geometry.pixel_area_mm2


def hd95_mm(prediction: np.ndarray, truth: np.ndarray, spacing_y_mm: float, spacing_x_mm: float
            ) -> float | None:
    """spacing-aware HD95（mm）；空预测或空真值返回 ``None``（不伪造 0）。"""
    pred = np.asarray(prediction, dtype=bool)
    real = np.asarray(truth, dtype=bool)
    if pred.shape != real.shape:
        raise LabelError(f"pred/truth 形状不一致: {pred.shape} vs {real.shape}")
    if not pred.any() or not real.any():
        return None
    from scipy import ndimage

    structure = np.ones((3, 3), dtype=bool)
    pred_surface = pred & ~ndimage.binary_erosion(pred, structure=structure, border_value=0)
    real_surface = real & ~ndimage.binary_erosion(real, structure=structure, border_value=0)
    sampling = (float(spacing_y_mm), float(spacing_x_mm))
    forward = ndimage.distance_transform_edt(~real_surface, sampling=sampling)[pred_surface]
    backward = ndimage.distance_transform_edt(~pred_surface, sampling=sampling)[real_surface]
    if forward.size == 0 or backward.size == 0:
        return None
    return float(np.percentile(np.concatenate([forward, backward]), 95))


def summarize_cases(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """逐类汇总；``None`` 单独计数，不参与均值（既不当作 0 也不当作满分）。"""
    buckets: dict[str, list[float]] = {name: [] for name in CLASS_NAMES.values()}
    nulls = 0
    for case in cases:
        for name in CLASS_NAMES.values():
            value = case.get(name)
            if value is None:
                nulls += 1
            else:
                buckets[name].append(float(value))
    return {"per_class_mean": {name: (sum(v) / len(v) if v else None) for name, v in buckets.items()},
            "per_class_count": {name: len(v) for name, v in buckets.items()},
            "null_values": nulls,
            "note": "null 表示空/无效情形，不参与均值，也不当作 0 或满分"}

def to_display_array(data_xy: np.ndarray) -> np.ndarray:
    """NIfTI X,Y -> displayed H,W, paired with spacing Y,X."""
    array=np.asarray(data_xy)
    if array.ndim!=2 or not np.isfinite(array).all():
        raise LabelError("expected a finite 2D NIfTI frame")
    return np.ascontiguousarray(array.T)
