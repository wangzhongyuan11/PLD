"""CAMUS 4 类 U-Net：单通道灰度输入 → 背景/LV腔/心肌/左心房 4 通道 logits。

算子仅 Conv/ReLU/MaxPool/nearest 上采样/concat/1×1 head，便于 INT8 与板端实现；
4 个输出通道**共用同一输出激活 scale**（无 sigmoid/softmax）。默认 base16/3pool/256。
预算函数给出参数量、MAC、INT8 权重字节、实测峰值激活与输出字节。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

NUM_CLASSES = 4
DEFAULT_BASE_CHANNELS = 16
DEFAULT_NUM_POOLS = 3
DEFAULT_INPUT_SIZE = 256


class CamusUNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = NUM_CLASSES,
                 base_channels: int = DEFAULT_BASE_CHANNELS, num_pools: int = DEFAULT_NUM_POOLS,
                 input_shape: Sequence[int] = (1, DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE)) -> None:
        super().__init__()
        if in_channels != 1:
            raise ValueError(f"CAMUS 基线仅支持单通道灰度，收到 in_channels={in_channels}")
        if num_classes != NUM_CLASSES:
            raise ValueError(f"CAMUS 基线输出固定 {NUM_CLASSES} 类，收到 num_classes={num_classes}")
        if base_channels < 1 or num_pools < 1:
            raise ValueError("base_channels 与 num_pools 必须 >= 1")
        self.in_channels = int(in_channels)
        self.num_classes = int(num_classes)
        self.base_channels = int(base_channels)
        self.num_pools = int(num_pools)
        self.input_shape = tuple(int(v) for v in input_shape)
        self.channels: tuple[int, ...] = tuple(base_channels * (2 ** level)
                                               for level in range(num_pools + 1))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoders = nn.ModuleList()
        previous = in_channels
        for width in self.channels:
            self.encoders.append(self._block(previous, width))
            previous = width
        # 解码器：nearest 上采样 + 3×3 Conv（可部署算子集，不使用 ConvTranspose2d）
        self.decoders = nn.ModuleList()
        for level in range(num_pools - 1, -1, -1):
            self.decoders.append(self._block(self.channels[level + 1] + self.channels[level],
                                             self.channels[level]))
        self.head = nn.Conv2d(self.channels[0], num_classes, kernel_size=1, bias=True)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 4 or inputs.shape[1] != self.in_channels:
            raise ValueError(f"输入必须是 (N, {self.in_channels}, H, W)，收到 {tuple(inputs.shape)}")
        features: list[Tensor] = []
        value = inputs
        for level, encoder in enumerate(self.encoders):
            value = encoder(value)
            features.append(value)
            if level < self.num_pools:
                value = self.pool(value)
        for index, decoder in enumerate(self.decoders):
            skip = features[self.num_pools - 1 - index]
            value = nn.functional.interpolate(value, size=skip.shape[-2:], mode="nearest")
            value = decoder(torch.cat([skip, value], dim=1))
        return self.head(value)

    def num_parameters(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only))

    def parameter_breakdown(self) -> dict[str, int]:
        return {"encoder": sum(p.numel() for p in self.encoders.parameters()),
                "decoder": sum(p.numel() for p in self.decoders.parameters()),
                "head": sum(p.numel() for p in self.head.parameters())}

    def assert_deployable_operators(self) -> None:
        """确认前向只使用可部署算子集：Conv/ReLU/MaxPool/nearest 上采样/concat/1×1 head。"""
        allowed = (nn.Conv2d, nn.ReLU, nn.MaxPool2d, nn.Sequential, nn.ModuleList, CamusUNet)
        offenders = sorted({type(module).__name__ for module in self.modules()
                            if not isinstance(module, allowed)})
        if offenders:
            raise AssertionError(f"出现不可部署/未批准算子: {offenders}")
        if any(isinstance(module, nn.ConvTranspose2d) for module in self.modules()):
            raise AssertionError("不得使用 ConvTranspose2d（改用 nearest 上采样 + Conv）")
        self.assert_no_sigmoid()

    def assert_no_sigmoid(self) -> None:
        banned = (nn.Sigmoid, nn.Softmax, nn.LogSoftmax, nn.Softmax2d)
        offenders = [type(m).__name__ for m in self.modules() if isinstance(m, banned)]
        if offenders:
            raise AssertionError(f"输出头不得包含二值/归一化激活: {offenders}")
        if self.head.out_channels != self.num_classes:
            raise AssertionError(f"head 输出通道 {self.head.out_channels} != {self.num_classes}")

    def int8_weight_bytes(self) -> int:
        """INT8 权重存储：每参数 1 字节；4 个输出通道共用同一激活 scale。"""
        return self.num_parameters()

    @torch.no_grad()
    def activation_budget(self, size: int = DEFAULT_INPUT_SIZE, device: str = "cpu"
                          ) -> dict[str, Any]:
        """一次前向的**实测**峰值激活/输出字节与逐层张量表。"""
        self.eval()
        inputs = torch.zeros(1, self.in_channels, size, size, device=device)
        table: list[dict[str, Any]] = []
        handles: list[Any] = []

        def hook(module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
            if isinstance(output, Tensor):
                table.append({"layer": type(module).__name__,
                              "shape": list(output.shape),
                              "bytes": int(output.numel() * output.element_size())})

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.ReLU, nn.MaxPool2d)):
                handles.append(module.register_forward_hook(hook))
        output = self(inputs)
        for handle in handles:
            handle.remove()
        peak = max((entry["bytes"] for entry in table), default=0)
        return {"peak_activation_bytes": peak, "activation_tensors": len(table),
                "output_bytes": int(output.numel() * output.element_size()),
                "output_shape": list(output.shape), "top_layers": sorted(
                    table, key=lambda entry: entry["bytes"], reverse=True)[:5]}

    def estimate_macs(self, size: int = DEFAULT_INPUT_SIZE) -> int:
        """MAC 估算（按实际分辨率逐层累计；nearest 上采样不计 MAC，Conv 计）。"""
        total = 0
        current = size
        for level, _width in enumerate(self.channels):
            for module in self.encoders[level]:
                if isinstance(module, nn.Conv2d):
                    total += (module.in_channels * module.out_channels * module.kernel_size[0]
                              * module.kernel_size[1] * current * current)
            if level < self.num_pools:
                current = current // 2
        for index, decoder in enumerate(self.decoders):
            current = current * 2
            for module in decoder:
                if isinstance(module, nn.Conv2d):
                    total += (module.in_channels * module.out_channels * module.kernel_size[0]
                              * module.kernel_size[1] * current * current)
        total += self.channels[0] * self.num_classes * size * size
        return int(total)


@dataclass(frozen=True)
class BudgetReport:
    parameters: int
    breakdown: dict[str, int]
    macs: int
    int8_weight_bytes: int
    peak_activation_bytes: int
    output_bytes: int
    input_size: int
    shared_output_scale: bool = True
    note: str = "单次前向估算/实测；4 类共用输出激活 scale；未做板端验证"

    def as_dict(self) -> dict[str, Any]:
        return {"parameters": self.parameters, "breakdown": self.breakdown, "macs": self.macs,
                "int8_weight_bytes": self.int8_weight_bytes,
                "peak_activation_bytes": self.peak_activation_bytes,
                "output_bytes": self.output_bytes, "input_size": self.input_size,
                "shared_output_scale": self.shared_output_scale, "note": self.note}


def build_model(base_channels: int = DEFAULT_BASE_CHANNELS, num_pools: int = DEFAULT_NUM_POOLS,
                size: int = DEFAULT_INPUT_SIZE) -> CamusUNet:
    return CamusUNet(in_channels=1, num_classes=NUM_CLASSES, base_channels=base_channels,
                     num_pools=num_pools, input_shape=(1, size, size))


def budget_report(base_channels: int = DEFAULT_BASE_CHANNELS, num_pools: int = DEFAULT_NUM_POOLS,
                  size: int = DEFAULT_INPUT_SIZE) -> dict[str, Any]:
    model = build_model(base_channels=base_channels, num_pools=num_pools, size=size)
    activation = model.activation_budget(size=size)
    return BudgetReport(parameters=model.num_parameters(), breakdown=model.parameter_breakdown(),
                        macs=model.estimate_macs(size=size),
                        int8_weight_bytes=model.int8_weight_bytes(),
                        peak_activation_bytes=activation["peak_activation_bytes"],
                        output_bytes=activation["output_bytes"], input_size=size).as_dict()


def synthetic_batch(batch: int = 2, size: int = DEFAULT_INPUT_SIZE, seed: int = 20261012
                    ) -> tuple[Tensor, Tensor, Tensor]:
    """合成 batch ``(image, label, valid)``，仅用于代码测试。"""
    generator = torch.Generator().manual_seed(seed)
    image = torch.rand(batch, 1, size, size, generator=generator)
    label = torch.zeros(batch, size, size, dtype=torch.long)
    for index in range(batch):
        for class_index in range(1, NUM_CLASSES):
            top = int(torch.randint(0, size // 2, (1,), generator=generator).item())
            left = int(torch.randint(0, size // 2, (1,), generator=generator).item())
            height = int(torch.randint(size // 8, size // 3, (1,), generator=generator).item())
            width = int(torch.randint(size // 8, size // 3, (1,), generator=generator).item())
            label[index, top:top + height, left:left + width] = class_index
    valid = torch.ones(batch, 1, size, size)
    return image, label, valid


def tensor_from_array(array: np.ndarray) -> Tensor:
    """numpy ``(H, W)`` 灰度 → ``(1, 1, H, W)`` float32（不修改输入）。"""
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))[None, None]
