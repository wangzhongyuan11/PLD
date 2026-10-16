"""Exact native CAMUS reconstruction and independent result writers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from functools import lru_cache

import numpy as np
from PIL import Image

from .preprocess import INPUT_SIZE, PreprocessError

CLASS_NAMES = ("background", "lv_cavity", "myocardium", "left_atrium")
LABEL_DISPLAY_VALUES = (0, 85, 170, 255)
CLASS_COLORS = np.asarray(
    [[0, 0, 0], [236, 58, 58], [58, 176, 82], [82, 126, 236]], dtype=np.uint8
)


class ResultError(ValueError):
    pass


def letterbox_meta_for_native(height: int, width: int, size: int = INPUT_SIZE):
    if height <= 0 or width <= 0:
        raise ResultError("native dimensions must be positive")
    scale = min(size / height, size / width)
    new_h = max(1, min(size, int(round(height * scale))))
    new_w = max(1, min(size, int(round(width * scale))))
    return {
        "original_height": int(height), "original_width": int(width),
        "scale": scale, "new_height": new_h, "new_width": new_w,
        "top": (size - new_h) // 2, "left": (size - new_w) // 2, "size": size,
    }


@lru_cache(maxsize=32)
def output_axis_plan(input_size: int, output_size: int):
    scale = np.float32(input_size / output_size)
    coordinate = np.maximum(
        scale * (np.arange(output_size, dtype=np.float32) + np.float32(0.5))
        - np.float32(0.5),
        np.float32(0.0),
    )
    lower = np.floor(coordinate).astype(np.int64)
    lower = np.minimum(lower, input_size - 1)
    upper = np.minimum(lower + 1, input_size - 1)
    weight = (coordinate - lower).astype(np.float32)
    for value in (lower, upper, weight): value.flags.writeable = False
    return lower, upper, weight


def _output_bilinear_axis(array: np.ndarray, output_size: int, axis: int) -> np.ndarray:
    lower, upper, weight = output_axis_plan(array.shape[axis], output_size)
    view_lower = np.take(array, lower.astype(np.intp), axis=axis)
    view_upper = np.take(array, upper.astype(np.intp), axis=axis)
    if axis == 0:
        weight = weight[:, None]
    else:
        weight = weight[None, :]
    return view_lower * (np.float32(1.0) - weight) + view_upper * weight


def _resize_output_channel(channel: np.ndarray, height: int, width: int) -> np.ndarray:
    if channel.dtype != np.float32 or channel.ndim != 2:
        raise ResultError("output reconstruction channel must be 2-D float32")
    # Native reconstruction is horizontal-first, distinct from input vertical-first.
    horizontal = _output_bilinear_axis(channel, width, 1)
    return _output_bilinear_axis(horizontal, height, 0)


def native_labels_from_head(
    head_int8: np.ndarray,
    native_height: int,
    native_width: int,
    head_scale: float,
) -> np.ndarray:
    logits = np.asarray(head_int8)
    if logits.shape != (1, 4, INPUT_SIZE, INPUT_SIZE) or logits.dtype != np.int8:
        raise ResultError(f"head must be int8 (1,4,256,256), got {logits.shape}/{logits.dtype}")
    scale32 = np.float32(head_scale)
    if not np.isfinite(scale32) or scale32 <= 0:
        raise ResultError("head scale must be finite positive")
    meta = letterbox_meta_for_native(native_height, native_width)
    top, left = int(meta["top"]), int(meta["left"])
    new_h, new_w = int(meta["new_height"]), int(meta["new_width"])

    def dequant_crop(channel: int) -> np.ndarray:
        values = logits[0, channel].astype(np.float32) * scale32
        return values[top:top + new_h, left:left + new_w]

    best = _resize_output_channel(dequant_crop(0), native_height, native_width)
    labels = np.zeros((native_height, native_width), dtype=np.uint8)
    for class_index in range(1, 4):
        values = _resize_output_channel(dequant_crop(class_index), native_height, native_width)
        update = values > best
        labels[update] = class_index
        np.copyto(best, values, where=update)
    return labels


def class_areas(labels: np.ndarray, spacing_yx_mm: tuple[float, float] | None):
    result = {}
    for index, name in enumerate(CLASS_NAMES):
        pixels = int(np.count_nonzero(labels == index))
        row: dict[str, Any] = {"pixels": pixels}
        if spacing_yx_mm is not None:
            row["area_mm2"] = pixels * float(spacing_yx_mm[0]) * float(spacing_yx_mm[1])
        result[name] = row
    return result


def label_display_png(labels: np.ndarray) -> Image.Image:
    values = np.asarray(labels)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
        raise ResultError("labels must be 2-D integer")
    lut = np.asarray(LABEL_DISPLAY_VALUES, dtype=np.uint8)
    return Image.fromarray(lut[np.clip(values, 0, 3).astype(np.uint8)], mode="L")


def segmentation_rgb(labels: np.ndarray) -> Image.Image:
    values = np.clip(np.asarray(labels), 0, 3).astype(np.uint8)
    return Image.fromarray(CLASS_COLORS[values], mode="RGB")


def overlay_rgb(gray: np.ndarray, labels: np.ndarray, alpha: float = 0.45) -> Image.Image:
    source = np.asarray(gray, dtype=np.float32)
    if source.shape != np.asarray(labels).shape:
        raise ResultError("overlay gray/labels shape mismatch")
    display = np.clip(source, 0.0, 255.0).astype(np.uint8)
    values = np.clip(np.asarray(labels), 0, 3).astype(np.uint8)
    blended = _overlay_lut(alpha)[values, display]
    return Image.fromarray(blended, mode="RGB")


@lru_cache(maxsize=4)
def _overlay_lut(alpha):
    gray = np.arange(256, dtype=np.float32)[None, :, None]
    colors = CLASS_COLORS.astype(np.float32)[:, None, :]
    return np.rint((1.0-alpha)*gray + alpha*colors).astype(np.uint8)


def write_html_preview(path: Path, title: str, entries: list[tuple[str, str]]) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{title}</title></head><body>",
        f"<h1>{title}</h1>",
    ]
    for heading, image in entries:
        parts.append(f"<section><h2>{heading}</h2><img src='{image}'></section>")
    parts.append("</body></html>")
    path.write_text("\n".join(parts), encoding="utf-8", newline="\n")
