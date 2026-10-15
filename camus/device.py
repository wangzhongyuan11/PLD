"""Emulated CAMUS tile device and the independent CAM2 hardware contract.

The real PL driver implements this same tiled Python interface.
The emulator performs all integer arithmetic behind the device boundary while
the host graph only gathers tile bytes and stores owned results.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .integer import (
    apply_relu_int8,
    concat_int8,
    conv2d_int32,
    maxpool2_int8,
    nearest_upsample2_int8,
    requantize_int8,
    rescale_int8,
)
from reference.tiling import Tile, input_tile, tiles
from .contract import shape, tensor_tile_shape

DEVICE_ID = 0x43414D32
ABI_VERSION = 0x00020000
VLNV = "user:pld:axis_tensor_accel_camus:1.0"
SUPPORTED_MODES = frozenset({0, 1, 2, 3, 4})
MAX_CI = 192
MAX_CO_PER_RUN = 64
INPUT_BANK_COUNT = 4
INPUT_BANK_BYTES = 32768
OUTPUT_BANK_COUNT = 8
OUTPUT_BANK_BYTES = 8192
WEIGHT_BANK_COUNT = 32
WEIGHT_BANK_DEPTH = 2048
RELU_REGISTER_OFFSET = 0x05C
RELU_BIT0 = 0x1


class DeviceContractError(ValueError):
    """A request violates the independent CAMUS CAM2 ABI/capacity contract."""


@dataclass(frozen=True)
class DeviceContract:
    device_id: int = DEVICE_ID
    abi_version: int = ABI_VERSION
    vlnv: str = VLNV
    modes: tuple[int, ...] = (0, 1, 2, 3, 4)
    max_ci: int = MAX_CI
    max_co_per_run: int = MAX_CO_PER_RUN
    input_banks: tuple[int, int] = (INPUT_BANK_COUNT, INPUT_BANK_BYTES)
    output_banks: tuple[int, int] = (OUTPUT_BANK_COUNT, OUTPUT_BANK_BYTES)
    weight_banks: tuple[int, int] = (WEIGHT_BANK_COUNT, WEIGHT_BANK_DEPTH)
    relu_register_offset: int = RELU_REGISTER_OFFSET
    relu_bit0: int = RELU_BIT0


def weight_bank_usage(co: int, ci: int, kernel: int) -> int:
    return math.ceil(co / 8) * math.ceil(ci / 4) * kernel * kernel


def input_bank_usage(ci: int, tile_height: int, tile_width: int, kernel: int) -> int:
    return math.ceil(ci / 4) * (tile_height + kernel - 1) * (tile_width + kernel - 1)


def output_bank_usage(co: int, tile_height: int, tile_width: int) -> int:
    return math.ceil(co / 8) * tile_height * tile_width


def output_channel_chunks(ci: int, co: int, kernel: int) -> tuple[int, ...]:
    """Complete-Ci output chunks; each chunk quantizes only after full accumulation."""
    if min(ci, co, kernel) <= 0 or kernel not in (1, 3):
        raise DeviceContractError("invalid convolution dimensions")
    if ci > MAX_CI:
        raise DeviceContractError(f"Ci={ci} exceeds CAM2 maximum {MAX_CI}")
    limit = min(MAX_CO_PER_RUN, (WEIGHT_BANK_DEPTH // weight_bank_usage(8, ci, kernel)) * 8)
    if limit <= 0:
        raise DeviceContractError(f"Ci={ci}, K={kernel} cannot fit one CAM2 weight bank")
    result = []
    remaining = co
    while remaining:
        take = min(limit, remaining)
        if weight_bank_usage(take, ci, kernel) > WEIGHT_BANK_DEPTH:
            raise DeviceContractError("internal weight-bank limit error")
        result.append(take)
        remaining -= take
    return tuple(result)


def choose_camus_tile(ci: int, co: int, height: int, width: int, kernel: int) -> tuple[int, int]:
    """Choose a spatial tile under explicit CAMUS input/output bank occupancy."""
    if min(ci, co, height, width) <= 0 or kernel not in (1, 3):
        raise DeviceContractError("invalid tile dimensions")
    try:
        shape(0, kernel, kernel, ci, co, kernel)
    except ValueError as exc:
        raise DeviceContractError(str(exc)) from exc
    candidates: list[tuple[int, int, int, int, int]] = []
    heights = {height} | {1 << bit for bit in range(height.bit_length())}
    widths = {width} | {1 << bit for bit in range(width.bit_length())}
    for tile_h in heights:
        for tile_w in widths:
            if tile_h > height or tile_w > width:
                continue
            input_usage = input_bank_usage(ci, tile_h, tile_w, kernel)
            output_usage = output_bank_usage(co, tile_h, tile_w)
            if input_usage <= INPUT_BANK_BYTES and output_usage <= OUTPUT_BANK_BYTES:
                count = math.ceil(height / tile_h) * math.ceil(width / tile_w)
                candidates.append((count, -tile_h * tile_w, -tile_w, tile_h, tile_w))
    if not candidates:
        raise DeviceContractError(f"no tile fits CAMUS banks: Ci={ci}, Co={co}, HW={height}x{width}")
    return min(candidates)[-2:]


def _validate_nchw_int8(data: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim != 4 or array.shape[0] != 1 or array.dtype != np.int8 or min(array.shape) < 1:
        raise DeviceContractError(
            f"{name} must be batch=1 NCHW int8, got {array.shape}/{array.dtype}"
        )
    if array.size and int(np.min(array)) < -127:
        raise DeviceContractError(f"{name} contains -128")
    return array


@dataclass
class ConvOperationRecord:
    ci: int
    co_chunk: int
    kernel: int
    tile_shape: tuple[int, int]
    tile_count: int
    input_bank_usage: int
    output_bank_usage: int
    weight_bank_usage: int
    mac_count: int


class EmulatedTileAccelerator:
    """Deterministic software emulator for scheduling and numerical checks."""

    def __init__(self) -> None:
        self.contract = DeviceContract()
        self.current_mode: int | None = None
        self.current_relu: bool = False
        self.conv_records: list[ConvOperationRecord] = []
        self.events: list[dict] = []
        self.frame = 0

    def begin_frame(self):
        self.conv_records.clear()
        self.events.clear()
        self.frame = self.frame % 0xffffffff + 1

    def close(self):
        pass

    def _prepare_conv(self, weight, bias, mult, h, w, relu):
        co, ci, k, _ = weight.shape
        cfg = shape(0, h, w, ci, co, k, relu=relu)
        self.events.append(dict(command=1, mode=0, bytes=cfg['parameter_bytes']))

    def _conv_tile(self, data, weight, bias, mult, relu):
        co, ci, k, _ = weight.shape
        cfg = shape(0, *data.shape[2:], ci, co, k, relu=relu)
        self.configure(0, relu=relu)
        acc = conv2d_int32(data, weight, bias, padding=0)
        out = requantize_int8(acc, mult)
        self.events.extend((dict(command=2, mode=0, bytes=cfg['input_bytes']),
                            dict(command=4, mode=0, bytes=cfg['output_bytes'])))
        return apply_relu_int8(out) if relu else out

    def capabilities(self) -> DeviceContract:
        return self.contract

    def configure(self, mode: int, *, relu: bool = False) -> None:
        if isinstance(mode, (bool, np.bool_)) or not isinstance(mode, (int, np.integer)):
            raise DeviceContractError('mode must be an integer')
        if mode not in SUPPORTED_MODES:
            raise DeviceContractError(f"mode {mode} is not CAM2 ABI 0x00020000")
        if mode != 0 and relu:
            raise DeviceContractError("ReLU bit0 is defined only for Conv mode 0")
        self.current_mode = mode
        self.current_relu = bool(relu)

    def _configured(self, mode: int, *, relu: bool = False) -> None:
        if self.current_mode != mode or self.current_relu != relu:
            raise DeviceContractError(f"device not configured for mode={mode}, relu={relu}")

    def conv(
        self,
        data: np.ndarray,
        weight: np.ndarray,
        bias: np.ndarray,
        multipliers: np.ndarray,
        *,
        padding: int,
        relu: bool,
    ) -> np.ndarray:
        x = _validate_nchw_int8(data, "conv input")
        w = np.asarray(weight)
        b = np.asarray(bias)
        m = np.asarray(multipliers)
        if w.ndim != 4 or w.dtype != np.int8 or b.dtype != np.int32 or m.dtype != np.int32:
            raise DeviceContractError("conv parameters must be int8 OIHW/int32/int32")
        if min(w.shape) < 1 or np.any(w == -128) or np.any(m <= 0):
            raise DeviceContractError("invalid convolution parameters")
        ci = x.shape[1]
        co = w.shape[0]
        if w.shape[1] != ci or b.shape != (co,) or m.shape != (co,):
            raise DeviceContractError("conv parameter channel mismatch")
        kernel = w.shape[2]
        if w.shape[3] != kernel or kernel not in (1, 3) or padding not in (0, 1):
            raise DeviceContractError("CAM2 supports only frozen 1x1/3x3 conv padding")
        expected_h = x.shape[2] + 2 * padding - kernel + 1
        expected_w = x.shape[3] + 2 * padding - kernel + 1
        if min(expected_h, expected_w) <= 0:
            raise DeviceContractError("convolution output has no pixels")
        chunks = output_channel_chunks(ci, co, kernel)
        output = np.empty((1, co, expected_h, expected_w), dtype=np.int8)
        output_offset = 0
        for co_chunk in chunks:
            chunk_w = w[output_offset:output_offset + co_chunk]
            chunk_b = b[output_offset:output_offset + co_chunk]
            chunk_m = m[output_offset:output_offset + co_chunk]
            w_usage = weight_bank_usage(co_chunk, ci, kernel)
            if w_usage > WEIGHT_BANK_DEPTH:
                raise DeviceContractError(f"weight occupancy {w_usage} exceeds 2048")
            tile_h, tile_w = choose_camus_tile(ci, co_chunk, expected_h, expected_w, kernel)
            self._prepare_conv(chunk_w, chunk_b, chunk_m, tile_h + kernel - 1, tile_w + kernel - 1, relu)
            for tile in tiles(expected_h, expected_w, tile_h, tile_w):
                gathered = input_tile(x, tile, kernel, padding)
                chunk_out = self._conv_tile(gathered, chunk_w, chunk_b, chunk_m, relu)
                output[
                    :, output_offset:output_offset + co_chunk,
                    tile.y:tile.y + tile.height, tile.x:tile.x + tile.width,
                ] = chunk_out
            record = ConvOperationRecord(
                ci=ci, co_chunk=co_chunk, kernel=kernel,
                tile_shape=(tile_h, tile_w),
                tile_count=math.ceil(expected_h / tile_h) * math.ceil(expected_w / tile_w),
                input_bank_usage=input_bank_usage(ci, tile_h, tile_w, kernel),
                output_bank_usage=output_bank_usage(co_chunk, tile_h, tile_w),
                weight_bank_usage=w_usage,
                mac_count=expected_h * expected_w * co_chunk * ci * kernel * kernel,
            )
            self.conv_records.append(record)
            output_offset += co_chunk
        return np.ascontiguousarray(output)

    def max_pool(self, data: np.ndarray) -> np.ndarray:
        return self._tensor(2, data)

    def upsample(self, data: np.ndarray) -> np.ndarray:
        return self._tensor(3, data)

    def rescale(self, data: np.ndarray, multiplier_q24: int) -> np.ndarray:
        return self._tensor(1, data, multipliers=(multiplier_q24,))

    def concat_upsample_rescale(self, skip, deep, multipliers_q24):
        return self._tensor(4, skip, deep, multipliers_q24)

    def _tensor_tile(self, cfg, first, second, multipliers, load_lut):
        mode = cfg['mode']
        self.configure(mode)
        if load_lut:
            self.events.append(dict(command=3, mode=mode, bytes=cfg['lut_bytes']))
        if mode == 2:
            out = maxpool2_int8(first)
        elif mode == 3:
            out = nearest_upsample2_int8(first)
        elif mode == 4:
            out = concat_int8((rescale_int8(first, multipliers[0]),
                               rescale_int8(nearest_upsample2_int8(second), multipliers[1])))
        else:
            out = rescale_int8(first, multipliers[0])
        self.events.extend((dict(command=2, mode=mode, bytes=cfg['input_bytes']),
                            dict(command=4, mode=mode, bytes=cfg['output_bytes'])))
        return out.copy()

    def _tensor(self, mode, data, second=None, multipliers=(1 << 24,)):
        first = _validate_nchw_int8(data, 'tensor input')
        multipliers = tuple(int(m) for m in multipliers)
        if len(multipliers) != (2 if mode == 4 else 1) or any(not 0 < m < (1 << 31) for m in multipliers):
            raise DeviceContractError('invalid tensor Q24 multipliers')
        _, c, h, w = first.shape
        split = 0
        if mode == 4:
            second = _validate_nchw_int8(second, 'deep input')
            if h % 2 or w % 2 or second.shape[2:] != (h // 2, w // 2):
                raise DeviceContractError('concat expects skip and half-resolution deep')
            split, c = c, c + second.shape[1]
        if mode == 2 and (h % 2 or w % 2):
            raise DeviceContractError('pool requires even dimensions')
        oh, ow = (h // 2, w // 2) if mode == 2 else (h * 2, w * 2) if mode == 3 else (h, w)
        th, tw = tensor_tile_shape(mode, c, oh, ow, split)
        result = np.empty((1, c, oh, ow), np.int8)
        for index, tile in enumerate(tiles(oh, ow, th, tw)):
            y, x, hh, ww = tile.y, tile.x, tile.height, tile.width
            if mode == 2:
                part = first[:, :, 2*y:2*(y+hh), 2*x:2*(x+ww)]
            elif mode == 3:
                part = first[:, :, y//2:(y+hh)//2, x//2:(x+ww)//2]
            else:
                part = first[:, :, y:y+hh, x:x+ww]
            other = second[:, :, y//2:(y+hh)//2, x//2:(x+ww)//2] if mode == 4 else None
            cfg = shape(mode, *part.shape[2:], c, split=split)
            result[:, :, y:y+hh, x:x+ww] = self._tensor_tile(cfg, part, other, multipliers, index == 0)
        return result


__all__ = [
    "ABI_VERSION", "DEVICE_ID", "VLNV",
    "ConvOperationRecord", "DeviceContract", "DeviceContractError",
    "EmulatedTileAccelerator", "choose_camus_tile", "input_bank_usage",
    "output_bank_usage", "output_channel_chunks", "weight_bank_usage",
]
