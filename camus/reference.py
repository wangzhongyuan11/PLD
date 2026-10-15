"""Independent full-tensor NumPy reference for the frozen CAMUS topology."""

from __future__ import annotations

from typing import Any

import numpy as np

from .integer import (
    Int8Feature,
    apply_relu_int8,
    concat_int8,
    conv2d_int32,
    maxpool2_int8,
    nearest_upsample2_int8,
    requantize_int8,
    rescale_int8,
)
from .model import CamusModel


class ReferenceNetworkError(ValueError):
    """The requested execution is outside the frozen CAMUS reference contract."""


class ReferenceNetwork:
    """Numerical oracle; all 15 layers and nonconv operators execute here."""

    def __init__(self, model: CamusModel):
        self.model = model

    def _conv(self, feature: Int8Feature, index: int) -> Int8Feature:
        layer = self.model.layers[index]
        if feature.scale != layer.input_scale:
            raise ReferenceNetworkError(f"producer scale mismatch at {layer.name}")
        acc = conv2d_int32(feature.data, layer.weight, layer.bias, padding=layer.padding)
        out = requantize_int8(acc, layer.multipliers)
        if layer.relu:
            out = apply_relu_int8(out)
        return Int8Feature(out, layer.output_scale)

    def forward(self, first_input: np.ndarray, *, capture: bool = False) -> dict[str, Any]:
        image = np.asarray(first_input)
        if image.shape != self.model.input_shape or image.dtype != np.int8:
            raise ReferenceNetworkError(
                f"reference accepts only int8 {self.model.input_shape}, got {image.shape}/{image.dtype}"
            )
        feature = Int8Feature(image.copy(), self.model.input_scale)
        captures: dict[str, np.ndarray] = {}
        skips: list[Int8Feature] = []
        cursor = 0
        peak = feature.data.nbytes

        def apply_conv(value: Int8Feature) -> Int8Feature:
            nonlocal cursor, peak
            layer = self.model.layers[cursor]
            if capture:
                captures[f"{cursor:02d}_{layer.name}_input"] = value.data.copy()
            result = self._conv(value, cursor)
            peak = max(peak, result.data.nbytes)
            if capture:
                captures[f"{cursor:02d}_{layer.name}_output"] = result.data.copy()
            cursor += 1
            return result

        for _level in range(4):
            feature = apply_conv(feature)
            feature = apply_conv(feature)
            skips.append(feature)
            if _level < 3:
                feature = Int8Feature(maxpool2_int8(feature.data), feature.scale)
                peak = max(peak, feature.data.nbytes)

        for concat in self.model.concats:
            up = Int8Feature(nearest_upsample2_int8(feature.data), feature.scale)
            skip = skips[concat.level]
            if skip.data.shape[2:] != up.data.shape[2:]:
                raise ReferenceNetworkError(f"concat spatial mismatch at level {concat.level}")
            first = rescale_int8(skip.data, concat.multiplier_q24[0])
            second = rescale_int8(up.data, concat.multiplier_q24[1])
            merged_data = concat_int8((first, second))
            feature = Int8Feature(merged_data, concat.common_scale)
            peak = max(peak, feature.data.nbytes)
            if capture:
                captures[f"concat_level{concat.level}_output"] = feature.data.copy()
            feature = apply_conv(feature)
            feature = apply_conv(feature)

        if capture:
            captures["14_head_input"] = feature.data.copy()
        head = self._conv(feature, 14)
        if capture:
            captures["14_head_output"] = head.data.copy()
        if head.scale != self.model.head_scale:
            raise ReferenceNetworkError("head scale mismatch")
        return {
            "head": head.data.copy(),
            "head_scale": head.scale,
            "captures": captures,
            "peak_int8_tensor_bytes": peak,
        }
