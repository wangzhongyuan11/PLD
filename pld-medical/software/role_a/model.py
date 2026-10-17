"""用于 ISIC 2016 皮肤病灶分割的极简 Tiny U-Net（FP32 软件基线）。

设计约束（为后续 B 组的 INT8 RTL 实现服务）：

* 只使用 Conv 3x3、Conv 1x1、ReLU、MaxPool 2x2、nearest Upsample、Concat；
* 不使用 BatchNorm / LayerNorm / Dropout / 注意力 / 转置卷积 / 复杂激活；
* 模型输出 **logits**，sigmoid 只出现在损失与评估侧；
* 最深下采样 3 次（32x32），便于硬件上的 buffer 复用。

本文件与第三方仓库代码无复制关系，结构为常规 U-Net 的独立实现（见 README 的
"参考与许可" 一节）。本文件不训练、不下数据、不声称任何精度。
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

import torch
from torch import Tensor, nn

__all__ = [
    "TinyUNet",
    "count_parameters",
    "build_model",
    "describe_graph",
    "DEFAULT_BASE_CHANNELS",
    "DEFAULT_NUM_POOLS",
    "DEFAULT_INPUT_SHAPE",
]

DEFAULT_BASE_CHANNELS: int = 8
DEFAULT_NUM_POOLS: int = 3
DEFAULT_INPUT_SHAPE: tuple[int, int, int] = (3, 256, 256)


def _pairwise_double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    """两层 3x3 卷积 + ReLU，是 U-Net 的基本块。"""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
        nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    """基通道可配置的 Tiny U-Net，输入 ``[N, 3, H, W]``，输出 ``[N, 1, H, W]`` 的 logits。

    结构（默认 ``base_channels=8``、``num_pools=3``，以 256x256 输入为例）::

        enc0: 3  -> 8     256      ──skip────────────────────────┐
        pool -> 128                                               │
        enc1: 8  -> 16    128      ──skip───────────────┐         │
        pool -> 64                                        │         │
        enc2: 16 -> 32    64       ──skip───────┐         │         │
        pool -> 32                               │         │         │
        enc3: 32 -> 64    32  (瓶颈)              │         │         │
        up -> concat(enc2) -> 2x Conv3x3 -> 32   │         │         │
        up -> concat(enc1) -> 2x Conv3x3 -> 16   │         │         │
        up -> concat(enc0) -> 2x Conv3x3 -> 8    │         │         │
        head: Conv 1x1 -> 1 logit channel        256               │
        (256 = 原始分辨率，1 个前景 logit 通道)

    参数:
        in_channels: 输入通道数，默认 3（RGB）。
        base_channels: 第一层通道数，默认 8。
        num_pools: 2x2 MaxPool 的次数，默认 3；必须在 ``1..3`` 之间。
        input_shape: 该 shape 仅用于 :meth:`assert_shapes` 的来源检查，
            不改变网络行为。

    异常:
        ValueError: 参数越界，或 ``input_shape`` 的 H/W 不能被 ``2**num_pools`` 整除。
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = DEFAULT_BASE_CHANNELS,
        num_pools: int = DEFAULT_NUM_POOLS,
        input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
    ) -> None:
        super().__init__()
        if in_channels < 1:
            raise ValueError(f"in_channels 必须 >= 1，收到 {in_channels}")
        if base_channels < 1:
            raise ValueError(f"base_channels 必须 >= 1，收到 {base_channels}")
        if not 1 <= num_pools <= 3:
            raise ValueError(f"num_pools 必须在 1..3 之间，收到 {num_pools}")
        if len(tuple(input_shape)) != 3:
            raise ValueError(f"input_shape 必须是 (C, H, W)，收到 {tuple(input_shape)}")
        height, width = int(input_shape[1]), int(input_shape[2])
        factor = 2**num_pools
        if height % factor or width % factor:
            raise ValueError(
                f"input_shape 的 H/W 必须能被 2**num_pools={factor} 整除，"
                f"收到 {(height, width)}；否则 Upsample 与 skip 无法对齐"
            )

        self.in_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.num_pools = int(num_pools)
        self.input_shape = (int(input_shape[0]), height, width)

        channels: list[int] = [base_channels * (2**level) for level in range(num_pools + 1)]
        self.channels: tuple[int, ...] = tuple(channels)

        # 编码器：encoders[0] 是浅层，encoders[-1] 是瓶颈（不再池化）。
        self.encoders = nn.ModuleList(
            _pairwise_double_conv(in_channels if level == 0 else channels[level - 1], channels[level])
            for level in range(len(channels))
        )
        self.pools = nn.ModuleList(
            nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(num_pools)
        )
        self.upsamples = nn.ModuleList(nn.Upsample(scale_factor=2, mode="nearest") for _ in range(num_pools))
        # decoder_blocks[0] 处理最深的一级，decoder_blocks[-1] 处理最浅的一级。
        self.decoder_blocks = nn.ModuleList(
            _pairwise_double_conv(channels[level + 1] + channels[level], channels[level])
            for level in range(num_pools - 1, -1, -1)
        )
        # 1x1 卷积输出单通道 logits，不做 sigmoid。
        self.head = nn.Conv2d(channels[0], 1, kernel_size=1, bias=True)

    # ------------------------------------------------------------------ 前向
    def forward(self, inputs: Tensor) -> Tensor:
        """``[N, 3, H, W]`` -> ``[N, 1, H, W]`` 的 logits（无 sigmoid）。"""
        if inputs.dim() != 4:
            raise ValueError(f"输入必须是 4 维 [N, C, H, W]，收到 shape={tuple(inputs.shape)}")
        if inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"输入通道数必须是 {self.in_channels}，收到 {inputs.shape[1]}"
            )
        factor = 2**self.num_pools
        if inputs.shape[-2] % factor or inputs.shape[-1] % factor:
            raise ValueError(
                f"输入 H/W 必须能被 {factor} 整除，收到 {(inputs.shape[-2], inputs.shape[-1])}"
            )
        return self.forward_with_features(inputs)[0]

    def forward_with_features(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """返回 ``(logits, features)``。

        ``features`` 是最深一层编码器的输出（即最后一次 MaxPool **之前**的激活），
        shape ``[N, channels[-1], H/2**num_pools, W/2**num_pools]``。它保留完整空间
        分辨率，便于后续板卡端验证特征搬运与量化，不参与训练损失。
        """
        skips: list[Tensor] = []
        x = inputs
        for level, encoder in enumerate(self.encoders):
            x = encoder(x)
            if level < self.num_pools:
                skips.append(x)
                x = self.pools[level](x)
        features = x

        for offset, block in enumerate(self.decoder_blocks):
            level = self.num_pools - 1 - offset
            x = self.upsamples[level](x)
            x = block(torch.cat((skips[level], x), dim=1))
        return self.head(x), features

    # -------------------------------------------------------------- 工具方法
    def num_parameters(self, trainable_only: bool = True) -> int:
        """可训练（或全部）参数数量。"""
        parameters: Iterable[nn.Parameter] = self.parameters()
        return int(sum(p.numel() for p in parameters if (p.requires_grad or not trainable_only)))

    def parameter_breakdown(self) -> dict[str, int]:
        """逐子模块参数数量，便于核查硬件所需权重规模。"""
        return {name: int(sum(p.numel() for p in module.parameters()))
                for name, module in self.named_children()}

    @torch.no_grad()
    def assert_shapes(self, batch_size: int = 1, device: torch.device | str | None = None) -> tuple[int, ...]:
        """用随机输入跑一次前向，校验输出 shape 与参数量，返回输出 shape。

        异常:
            RuntimeError: 输出 shape 与 ``input_shape`` 不一致（说明结构被改坏）。
        """
        if batch_size < 1:
            raise ValueError(f"batch_size 必须 >= 1，收到 {batch_size}")
        target = torch.device(device) if device is not None else next(self.parameters()).device
        self.eval()
        dummy = torch.zeros((batch_size, *self.input_shape), dtype=torch.float32, device=target)
        logits = self(dummy)
        expected = (batch_size, 1, self.input_shape[1], self.input_shape[2])
        if tuple(logits.shape) != expected:
            raise RuntimeError(f"输出 shape 应为 {expected}，实际为 {tuple(logits.shape)}")
        return tuple(logits.shape)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """便捷函数：统计 ``model`` 的参数数量（默认只数可训练参数）。"""
    return int(sum(parameter.numel() for parameter in model.parameters()
                   if parameter.requires_grad or not trainable_only))


def build_model(
    base_channels: int = DEFAULT_BASE_CHANNELS,
    in_channels: int = 3,
    num_pools: int = DEFAULT_NUM_POOLS,
    input_shape: Sequence[int] = DEFAULT_INPUT_SHAPE,
) -> TinyUNet:
    """``TinyUNet`` 的显式构造入口，参数与 :class:`TinyUNet` 一致。"""
    return TinyUNet(
        in_channels=in_channels,
        base_channels=base_channels,
        num_pools=num_pools,
        input_shape=input_shape,
    )


# 硬件可映射性白名单：只允许这些算子出现在网络里。
ALLOWED_LAYER_TYPES: tuple[type[nn.Module], ...] = (
    nn.Conv2d,
    nn.ReLU,
    nn.MaxPool2d,
    nn.Upsample,
    nn.Sequential,
    TinyUNet,
    nn.ModuleList,
)
_ALLOWED_MODULE_PATTERN = re.compile(
    r"^(TinyUNet|Sequential|Conv2d|ReLU|MaxPool2d|Upsample|ModuleList|Module)$"
)


def describe_graph(model: TinyUNet) -> dict[str, object]:
    """给出可读的算子序列，供 B 组评估 INT8 可映射性。

    返回字段::

        {"base_channels", "num_pools", "channels", "input_shape",
         "layers": [{"name", "type", "detail"}, ...],
         "allowed_only": bool, "forbidden": [类型名, ...],
         "parameters": {"total", "by_module"}}
    """
    if not isinstance(model, TinyUNet):
        raise TypeError(f"describe_graph 需要 TinyUNet，收到 {type(model).__name__}")
    layers: list[dict[str, str]] = []
    forbidden: list[str] = []
    for name, module in model.named_modules():
        if name == "":
            continue
        type_name = type(module).__name__
        if not _ALLOWED_MODULE_PATTERN.match(type_name):
            forbidden.append(type_name)
        detail = ""
        if isinstance(module, nn.Conv2d):
            detail = (f"in={module.in_channels} out={module.out_channels} "
                      f"k={module.kernel_size[0]} stride={module.stride[0]} "
                      f"pad={module.padding[0]} bias={module.bias is not None}")
        elif isinstance(module, nn.MaxPool2d):
            detail = f"k={module.kernel_size} stride={module.stride}"
        elif isinstance(module, nn.Upsample):
            detail = f"mode={module.mode} scale_factor={module.scale_factor}"
        layers.append({"name": name, "type": type_name, "detail": detail})
    return {
        "base_channels": model.base_channels,
        "num_pools": model.num_pools,
        "channels": list(model.channels),
        "input_shape": list(model.input_shape),
        "layers": layers,
        "allowed_only": not forbidden,
        "forbidden": sorted(set(forbidden)),
        "parameters": {
            "total": model.num_parameters(),
            "by_module": model.parameter_breakdown(),
        },
    }


if __name__ == "__main__":  # 便于人工快速查看，不训练、不读数据。
    import json

    network = build_model()
    print(json.dumps(describe_graph(network), indent=2, ensure_ascii=False))
    shape = network.assert_shapes(batch_size=1)
    print(f"forward ok: input (1, {network.input_shape[0]}, {network.input_shape[1]}, "
          f"{network.input_shape[2]}) -> output {shape}")
