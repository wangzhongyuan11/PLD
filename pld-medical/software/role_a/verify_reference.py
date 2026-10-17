"""不依赖 torch/numpy/Pillow 的参考计算：参数量、逐层 shape、I/O 元素数、MAC。

用途：在无法运行 :mod:`model` 的环境（例如只想核对硬件规模、或 torch 未安装）下，
用纯 Python 复算 :class:`~model.TinyUNet` 的**参数量与逐层张量 shape**，为 B 组的
INT8 RTL 提供权重规模与缓冲区大小估计。

它**不训练、不推理、不量化**，只做确定性算术；因此可以放心与 `model.describe_graph`
的输出互相核对。

用法（在 `software/role_a` 目录）::

    py verify_reference.py                    # 默认 base=8, pools=3, 256x256
    py verify_reference.py --base-channels 4 --num-pools 2 --size 64
    py verify_reference.py --export runs/role_a_reference.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def conv_parameters(in_channels: int, out_channels: int, kernel: int, bias: bool = True) -> int:
    """一个 2D 卷积的参数量（含 bias）。"""
    if min(in_channels, out_channels, kernel) < 1:
        raise ValueError("卷积维度必须为正")
    return in_channels * out_channels * kernel * kernel + (out_channels if bias else 0)


class ModelReference:
    """TinyUNet 的纯 Python 参考模型（只算 shape / 参数量 / MAC）。

    结构与 ``model.TinyUNet`` 一一对应：每层两层 3x3 卷积 + ReLU，编码器间 2x2
    MaxPool，解码器 nearest Upsample + skip concat + 两层 3x3 卷积，最后 1x1 卷积。
    """

    def __init__(self, base_channels: int = 8, num_pools: int = 3,
                 size: int = 256, in_channels: int = 3) -> None:
        if base_channels < 1:
            raise ValueError(f"base_channels 必须 >= 1，收到 {base_channels}")
        if not 1 <= num_pools <= 3:
            raise ValueError(f"num_pools 必须在 1..3 之间，收到 {num_pools}")
        if size < 1:
            raise ValueError(f"size 必须为正，收到 {size}")
        factor = 2**num_pools
        if size % factor:
            raise ValueError(f"size 必须能被 2**num_pools={factor} 整除，收到 {size}")
        self.base_channels = base_channels
        self.num_pools = num_pools
        self.size = size
        self.in_channels = in_channels
        self.channels: tuple[int, ...] = tuple(base_channels * 2**level
                                              for level in range(num_pools + 1))

    # ------------------------------------------------------------- 参数量
    def parameters(self) -> dict[str, Any]:
        """按模块返回参数量与总计。"""
        breakdown: dict[str, int] = {}
        for level, out_channels in enumerate(self.channels):
            in_ch = self.in_channels if level == 0 else self.channels[level - 1]
            breakdown[f"encoders.{level}"] = (
                conv_parameters(in_ch, out_channels, 3)
                + conv_parameters(out_channels, out_channels, 3)
            )
        for level in range(self.num_pools - 1, -1, -1):
            cat = self.channels[level + 1] + self.channels[level]
            out_channels = self.channels[level]
            breakdown[f"decoder_blocks.{self.num_pools - 1 - level}"] = (
                conv_parameters(cat, out_channels, 3)
                + conv_parameters(out_channels, out_channels, 3)
            )
        breakdown["head"] = conv_parameters(self.channels[0], 1, 1)
        weights = sum(breakdown.values())
        # 权重（不含 bias）字节数：FP32 4 字节 / INT8 1 字节。
        biases = sum(self._bias_counts().values())
        return {
            "by_module": breakdown,
            "total": weights,
            "weight_parameters_excluding_bias": weights - biases,
            "bias_parameters": biases,
            "bytes_fp32": weights * 4,
            "bytes_int8_weights_only": weights - biases,
        }

    def _bias_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for level, out_channels in enumerate(self.channels):
            counts[f"encoders.{level}"] = 2 * out_channels
        for level in range(self.num_pools - 1, -1, -1):
            counts[f"decoder_blocks.{self.num_pools - 1 - level}"] = 2 * self.channels[level]
        counts["head"] = 1
        return counts

    # ---------------------------------------------------------------- shape
    def shapes(self) -> list[dict[str, Any]]:
        """返回逐层 shape / 参数量 / MAC 的记录，顺序即数据流顺序。"""
        records: list[dict[str, Any]] = []
        height = width = self.size
        current = self.in_channels
        skips: list[dict[str, int]] = []

        def add(name: str, kind: str, out_channels: int, out_h: int, out_w: int,
                macs: int, params: int, detail: str = "") -> None:
            records.append({
                "name": name, "type": kind, "in_channels": current, "out_channels": out_channels,
                "in_hw": [height, width], "out_hw": [out_h, out_w],
                "macs": macs, "parameters": params, "detail": detail,
            })

        for level, out_channels in enumerate(self.channels):
            in_ch = current
            params = conv_parameters(in_ch, out_channels, 3)
            add(f"encoders.{level}.0", "Conv3x3", out_channels, height, width,
                in_ch * out_channels * 9 * height * width, params, "stride=1 pad=1")
            add(f"encoders.{level}.1", "ReLU", out_channels, height, width, 0, 0)
            in_ch = out_channels
            params = conv_parameters(in_ch, out_channels, 3)
            add(f"encoders.{level}.3", "Conv3x3", out_channels, height, width,
                in_ch * out_channels * 9 * height * width, params, "stride=1 pad=1")
            add(f"encoders.{level}.4", "ReLU", out_channels, height, width, 0, 0)
            current = out_channels
            if level < self.num_pools:
                skips.append({"channels": out_channels, "hw": [height, width]})
                out_h, out_w = height // 2, width // 2
                add(f"pools.{level}", "MaxPool2x2", out_channels, out_h, out_w, 0, 0,
                    "stride=2")
                height, width = out_h, out_w

        for offset in range(self.num_pools):
            level = self.num_pools - 1 - offset
            skip = skips[level]
            out_h, out_w = height * 2, width * 2
            add(f"upsamples.{level}", "UpsampleNearest", current, out_h, out_w, 0, 0,
                "scale_factor=2")
            height, width = out_h, out_w
            cat_channels = skip["channels"] + current
            records.append({
                "name": f"concat.{level}", "type": "Concat", "in_channels": cat_channels,
                "out_channels": cat_channels, "in_hw": [height, width],
                "out_hw": [height, width], "macs": 0, "parameters": 0,
                "detail": f"skip {skip['channels']}ch {skip['hw'][0]}x{skip['hw'][1]} + {current}ch",
            })
            out_channels = self.channels[level]
            params = conv_parameters(cat_channels, out_channels, 3)
            add(f"decoder_blocks.{offset}.0", "Conv3x3", out_channels, height, width,
                cat_channels * out_channels * 9 * height * width, params, "stride=1 pad=1")
            add(f"decoder_blocks.{offset}.1", "ReLU", out_channels, height, width, 0, 0)
            params = conv_parameters(out_channels, out_channels, 3)
            add(f"decoder_blocks.{offset}.3", "Conv3x3", out_channels, height, width,
                out_channels * out_channels * 9 * height * width, params, "stride=1 pad=1")
            add(f"decoder_blocks.{offset}.4", "ReLU", out_channels, height, width, 0, 0)
            current = out_channels

        add("head", "Conv1x1", 1, height, width,
            current * 1 * 1 * height * width, conv_parameters(current, 1, 1), "stride=1 pad=0")
        return records

    def summary(self) -> dict[str, Any]:
        records = self.shapes()
        parameters = self.parameters()
        total_macs = sum(record["macs"] for record in records)
        activations = [{"name": record["name"], "channels": record["out_channels"],
                        "hw": record["out_hw"],
                        "elements": record["out_channels"] * record["out_hw"][0] * record["out_hw"][1]}
                       for record in records]
        return {
            "base_channels": self.base_channels,
            "num_pools": self.num_pools,
            "channels": list(self.channels),
            "input_shape": [1, self.in_channels, self.size, self.size],
            "output_shape": [1, 1, self.size, self.size],
            "parameters": parameters,
            "total_macs_fp32": total_macs,
            "total_multiply_accumulate_int8": total_macs,
            "peak_activation_elements": max(item["elements"] for item in activations),
            "activation_elements_total": sum(item["elements"] for item in activations),
            "layers": records,
            "notes": [
                "MAC 按每个输出像素 1 次乘加计，未计入 ReLU/MaxPool/Upsample/Concat",
                "bytes_int8_weights_only 只是权重字节数，不是实测资源占用",
                "本脚本不量化、不训练，也不代表 FPGA 上的时序或资源",
            ],
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-channels", type=int, default=8)
    parser.add_argument("--num-pools", type=int, default=3)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--export", type=Path, default=None)
    parser.add_argument("--no-layers", action="store_true", help="不打印逐层明细")
    args = parser.parse_args(argv)

    reference = ModelReference(base_channels=args.base_channels, num_pools=args.num_pools,
                               size=args.size)
    summary = reference.summary()
    layers = summary.pop("layers")
    if args.export:
        payload = dict(summary)
        payload["layers"] = layers
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                               encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not args.no_layers:
        print(f"\n{'layer':<26}{'type':<18}{'out_ch':>7}{'out_hw':>12}{'MACs':>14}{'params':>10}")
        for record in layers:
            print(f"{record['name']:<26}{record['type']:<18}{record['out_channels']:>7}"
                  f"{str(record['out_hw'][0]) + 'x' + str(record['out_hw'][1]):>12}"
                  f"{record['macs']:>14}{record['parameters']:>10}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
