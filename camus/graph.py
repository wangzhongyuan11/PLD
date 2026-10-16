"""Host scheduling graph; every CAMUS operator is issued to a device."""

from __future__ import annotations

from typing import Any

import numpy as np

from .device import EmulatedTileAccelerator
from .model import CamusModel


class CamusGraphError(ValueError):
    """A device execution or topology invariant failed."""


class CamusGraph:
    def __init__(self, model: CamusModel, device: EmulatedTileAccelerator):
        self.model = model
        self.device = device

    def forward(self, first_input: np.ndarray, *, capture: bool = False) -> dict[str, Any]:
        image = np.asarray(first_input)
        if image.shape != self.model.input_shape or image.dtype != np.int8:
            raise CamusGraphError(
                f"graph accepts only first-layer int8 {self.model.input_shape}"
            )
        feature = image.copy()
        self.device.begin_frame()
        feature_scale = self.model.input_scale
        captures: dict[str, np.ndarray] = {}
        skips: list[tuple[np.ndarray, float]] = []
        cursor = 0

        def conv(value: np.ndarray, value_scale: float) -> tuple[np.ndarray, float]:
            nonlocal cursor
            layer = self.model.layers[cursor]
            if value_scale != layer.input_scale:
                raise CamusGraphError(f"producer scale mismatch at {layer.name}")
            if capture:
                captures[f"{cursor:02d}_{layer.name}_input"] = value.copy()
            result = self.device.conv(
                value, layer.weight, layer.bias, layer.multipliers,
                padding=layer.padding, relu=layer.relu,
            )
            if capture:
                captures[f"{cursor:02d}_{layer.name}_output"] = result.copy()
            cursor += 1
            return result, layer.output_scale

        for level in range(4):
            feature, feature_scale = conv(feature, feature_scale)
            feature, feature_scale = conv(feature, feature_scale)
            skips.append((feature, feature_scale))
            if level < 3:
                feature = self.device.max_pool(feature)

        for concat in self.model.concats:
            skip, _skip_scale = skips[concat.level]
            if skip.shape[2:] != tuple(v * 2 for v in feature.shape[2:]):
                raise CamusGraphError(f"concat spatial mismatch at level {concat.level}")
            feature = self.device.concat_upsample_rescale(
                skip, feature, concat.multiplier_q24
            )
            skips[concat.level] = (None, 0.)
            feature_scale = concat.common_scale
            if capture:
                captures[f"concat_level{concat.level}_output"] = feature.copy()
            feature, feature_scale = conv(feature, feature_scale)
            feature, feature_scale = conv(feature, feature_scale)

        if cursor != 14:
            raise CamusGraphError("device graph did not reach the head producer")
        feature, feature_scale = conv(feature, feature_scale)
        if feature.shape != self.model.head_shape or feature_scale != self.model.head_scale:
            raise CamusGraphError("head shape/scale mismatch")
        return {
            "head": feature,
            "head_scale": feature_scale,
            "captures": captures,
            "conv_records": [record.__dict__.copy() for record in self.device.conv_records],
            "events": list(self.device.events),
        }
