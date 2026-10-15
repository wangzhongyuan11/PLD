"""Read-only CAMUS capacity and workload planning."""

from __future__ import annotations

import math
from typing import Any

from .device import (
    INPUT_BANK_BYTES,
    OUTPUT_BANK_BYTES,
    WEIGHT_BANK_DEPTH,
    choose_camus_tile,
    input_bank_usage,
    output_bank_usage,
    output_channel_chunks,
    weight_bank_usage,
)
from .model import CamusModel
from .contract import tensor_tile_shape, shape
from reference.tiling import tiles


def layer_plan(model: CamusModel, layer_index: int) -> dict[str, Any]:
    layer = model.layers[layer_index]
    ci = layer.full_input_shape[1]
    co = layer.full_output_shape[1]
    height = layer.full_output_shape[2]
    width = layer.full_output_shape[3]
    chunks = output_channel_chunks(ci, co, layer.kernel)
    chunk_rows = []
    for co_chunk in chunks:
        tile_h, tile_w = choose_camus_tile(ci, co_chunk, height, width, layer.kernel)
        chunk_rows.append({
            "output_channels": co_chunk,
            "tile_shape": [tile_h, tile_w],
            "tile_count": math.ceil(height / tile_h) * math.ceil(width / tile_w),
            "input_bank_usage": input_bank_usage(ci, tile_h, tile_w, layer.kernel),
            "input_bank_limit": INPUT_BANK_BYTES,
            "output_bank_usage": output_bank_usage(co_chunk, tile_h, tile_w),
            "output_bank_limit": OUTPUT_BANK_BYTES,
            "weight_bank_usage": weight_bank_usage(co_chunk, ci, layer.kernel),
            "weight_bank_limit": WEIGHT_BANK_DEPTH,
        })
    return {
        "index": layer.index,
        "name": layer.name,
        "ci": ci,
        "co": co,
        "kernel": layer.kernel,
        "padding": layer.padding,
        "relu": layer.relu,
        "output_hw": [height, width],
        "output_chunks": list(chunks),
        "chunk_schedule": chunk_rows,
        "run_count": sum(row["tile_count"] for row in chunk_rows),
        "mac_count": height * width * co * ci * layer.kernel * layer.kernel,
    }


def build_plan(model: CamusModel) -> dict[str, Any]:
    layers = [layer_plan(model, index) for index in range(15)]
    pool_elements = sum(
        model.layers[index].full_output_shape[1]
        * model.layers[index].full_output_shape[2]
        * model.layers[index].full_output_shape[3]
        for index in (1, 3, 5)
    )
    concat_elements = 0
    upsample_elements = 0
    for entry_layer_index in (8, 10, 12):
        entry = model.layers[entry_layer_index]
        concat_elements += entry.full_input_shape[1] * entry.full_input_shape[2] * entry.full_input_shape[3]
        upsample_elements += model.layers[entry_layer_index - 1].full_output_shape[1] * entry.full_input_shape[2] * entry.full_input_shape[3]
    operations = []
    for index in (2, 4, 6, 8, 10, 12):
        _, c, h, w = model.layers[index].full_input_shape
        mode = 2 if index < 8 else 4
        split = {8:64, 10:32, 12:16}.get(index, 0)
        th, tw = tensor_tile_shape(mode, c, h, w, split)
        configs = []
        for tile in tiles(h, w, th, tw):
            ih, iw = (2*tile.height, 2*tile.width) if mode == 2 else (tile.height, tile.width)
            configs.append(shape(mode, ih, iw, c, split=split))
        operations.append(dict(before_layer=index, mode=mode, channels=c, split=split,
                               tile_shape=[th,tw], tile_count=len(configs),
                               input_bytes=sum(cfg['input_bytes'] for cfg in configs),
                               output_bytes=sum(cfg['output_bytes'] for cfg in configs),
                               lut_bytes=configs[0]['lut_bytes']))
    conv_input_bytes = conv_output_bytes = parameter_bytes = 0
    projected_compute_cycles = 0
    for row in layers:
        k, ci = row['kernel'], row['ci']
        h, w = row['output_hw']
        for block in row['chunk_schedule']:
            co = block['output_channels']; th, tw = block['tile_shape']
            parameter_bytes += co*ci*k*k + 8*co
            group_count = (co+7)//8
            projected_compute_cycles += 4*block['tile_count'] + h*w*(group_count*((ci+3)//4*k*k+11)+co)
            for tile in tiles(h,w,th,tw):
                conv_input_bytes += ci*(tile.height+k-1)*(tile.width+k-1)
                conv_output_bytes += co*tile.height*tile.width
    tensor_runs = sum(row['tile_count'] for row in operations)
    byte_count = conv_input_bytes+conv_output_bytes+parameter_bytes+sum(row['input_bytes']+row['output_bytes']+row['lut_bytes'] for row in operations)
    commands = sum(len(row['output_chunks']) for row in layers) + 2*sum(row['run_count'] for row in layers) + len(operations)+2*tensor_runs
    return {
        "schema": "pld-camus-plan-v1",
        "device": "CAM2 ABI 0x00020000 software plan (no PYNQ access)",
        "layers": layers,
        "total_conv_runs": sum(row["run_count"] for row in layers),
        "total_mac_count": sum(row["mac_count"] for row in layers),
        "conv_parameter_loads": sum(len(row['output_chunks']) for row in layers),
        "tensor_schedule": operations,
        "total_commands": commands,
        "total_packet_bytes": byte_count,
        "conv_input_bytes": conv_input_bytes,
        "conv_output_bytes": conv_output_bytes,
        "parameter_bytes": parameter_bytes,
        "projected_conv_compute_cycles": projected_compute_cycles,
        "projection_scope": "stable CAM2 FSM model; validate against RTL regression cycles, not measured board performance",
        "nonconv": {
            "maxpool2_runs": 3,
            "maxpool2_input_elements": pool_elements,
            "upsample2_runs": 3,
            "upsample2_output_elements": upsample_elements,
            "q24_rescale_runs": 6,
            "concat_runs": 3,
            "concat_output_elements": concat_elements,
        },
    }
