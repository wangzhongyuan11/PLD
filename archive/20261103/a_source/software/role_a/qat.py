"""A 组最终训练的量化感知（QAT）原语：不修改参数存储、不打 hook 的可微假量化。

设计要点（对应 Codex 阶段 0 审查与放行裁定）：
* **三处假量化全覆盖**：每个 Conv 的**输入、权重、输出**都按部署同规则取整/饱和/反量化，
  并用 STE 回传梯度。输入与输出用该层的 ``input_scale``/``output_scale``。
* 不用 forward pre/post hook 改写 ``conv.weight``（会抹掉优化器更新，并使反向的输入
  梯度与权重不一致）；继承 ``nn.Conv2d`` **只覆写 forward**，参数仍由模块自身持有，
  因此 ``state_dict`` 的键名、形状、顺序与原始模型完全一致（``torch.nn.utils.parametrize``
  会把键改成 ``parametrizations.weight.original``，故不采用）。
* 尺度来源有二：
  - ``set_scales_from_calibration(...)``：用既有导出同源的 96 张训练校准图统计
    ``input_max_abs``/``output_max_abs``（**固定尺度**，训练中不自适应漂移）；
  - ``refresh_scales_from_calibration(...)``：第 4 轮后用同一批 96 张**无增强**重算一次，
    5-8 轮冻结（由驱动按预设时点调用）。
* 权重口径与导出器一致：per-output-channel 对称、``scale = max_abs/127``、半偶舍入、
  clamp 到 [-127,127]（对齐 ``quant_math.quantize_weights``）。
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from software.role_a.quant_math import QMAX

__all__ = ["quantized_weight", "fake_quant_weight", "fake_quant_activation",
           "FakeQuantConv2d", "enable_weight_qat", "disable_weight_qat", "unwrap",
           "qat_layer_names", "calibration_scales", "set_scales_from_calibration",
           "scales_snapshot"]


def _symmetric_scale(max_abs: float) -> float:
    return max(float(max_abs) / QMAX, 1e-8)


def quantized_weight(weight: Tensor) -> tuple[Tensor, Tensor]:
    """返回 (按输出通道反量化后的权重, 逐输出通道 scale)，与导出器同口径（无可微性）。"""
    scales = torch.clamp(weight.detach().abs().amax(dim=(1, 2, 3)) / QMAX, min=1e-8)
    with torch.no_grad():
        coded = torch.clamp(torch.round(weight.detach() / scales[:, None, None, None]),
                            -QMAX, QMAX)
        dequantized = coded * scales[:, None, None, None]
    return dequantized, scales


def fake_quant_weight(weight: Tensor) -> Tensor:
    """可微假量化：前向等于量化-反量化值，反向对 ``weight`` 按 STE 传梯度。"""
    dequantized, _scales = quantized_weight(weight)
    return weight + (dequantized - weight).detach()


def fake_quant_activation(value: Tensor, scale: float) -> Tensor:
    """可微激活假量化：``x_fake = q*scale``，``q = clamp(round(x/scale), -127, 127)``。

    前向与部署一致（取整 + 饱和 + 反量化），反向对 ``value`` 按 STE 恒等回传。
    """
    scale = float(scale)
    if scale <= 0:
        raise ValueError(f"activation scale 必须为正，收到 {scale}")
    with torch.no_grad():
        coded = torch.clamp(torch.round(value.detach() / scale), -QMAX, QMAX) * scale
    return value + (coded - value).detach()


class FakeQuantConv2d(nn.Conv2d):
    """与 ``nn.Conv2d`` 同键同形状；前向对输入、权重、输出做部署同口径假量化。

    刻意保持**简洁的 float 假量化代理**（Q24 整数路径只在导出/部署侧），
    最终效果以每轮真实 INT8 的 638 开发集指标裁定。
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.enabled = True
        self.input_scale: float | None = None
        self.output_scale: float | None = None

    def forward(self, input_tensor: Tensor) -> Tensor:  # type: ignore[override]
        if not bool(getattr(self, "enabled", True)):
            return F.conv2d(input_tensor, self.weight, self.bias, self.stride, self.padding,
                            self.dilation, self.groups)
        if self.input_scale is None or self.output_scale is None:
            raise RuntimeError(
                "FakeQuantConv2d 需要先设置 input_scale/output_scale"
                "（见 set_scales_from_calibration）")
        quantized_input = fake_quant_activation(input_tensor, self.input_scale)
        weight = fake_quant_weight(self.weight)
        output = F.conv2d(quantized_input, weight, self.bias, self.stride, self.padding,
                          self.dilation, self.groups)
        return fake_quant_activation(output, self.output_scale)


def _as_pair(value: Any) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    return int(value), int(value)


def _replace(model: nn.Module, name: str, new_module: nn.Module) -> None:
    parent_name, _, leaf = name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    if isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def _build_from(conv: nn.Conv2d) -> FakeQuantConv2d:
    replacement = FakeQuantConv2d(
        conv.in_channels, conv.out_channels, _as_pair(conv.kernel_size),
        stride=_as_pair(conv.stride), padding=_as_pair(conv.padding),
        dilation=_as_pair(conv.dilation), groups=int(conv.groups),
        bias=conv.bias is not None, padding_mode=conv.padding_mode,
        device=conv.weight.device, dtype=conv.weight.dtype)
    with torch.no_grad():
        replacement.weight.copy_(conv.weight)
        if conv.bias is not None and replacement.bias is not None:
            replacement.bias.copy_(conv.bias)
    return replacement


def enable_weight_qat(model: nn.Module) -> list[str]:
    """把所有 ``nn.Conv2d`` 原地替换为 ``FakeQuantConv2d``（权重值逐位继承）。"""
    names = [name for name, module in model.named_modules() if type(module) is nn.Conv2d]
    for name in names:
        _replace(model, name, _build_from(model.get_submodule(name)))
    return names


def disable_weight_qat(model: nn.Module) -> list[str]:
    """把 ``FakeQuantConv2d`` 还原为普通 ``nn.Conv2d``（键/顺序保持一致）。"""
    names = [name for name, module in model.named_modules()
             if isinstance(module, FakeQuantConv2d)]
    for name in names:
        module = model.get_submodule(name)
        plain = nn.Conv2d(
            module.in_channels, module.out_channels, _as_pair(module.kernel_size),
            stride=_as_pair(module.stride), padding=_as_pair(module.padding),
            dilation=_as_pair(module.dilation), groups=int(module.groups),
            bias=module.bias is not None, padding_mode=module.padding_mode,
            device=module.weight.device, dtype=module.weight.dtype)
        with torch.no_grad():
            plain.weight.copy_(module.weight)
            if module.bias is not None and plain.bias is not None:
                plain.bias.copy_(module.bias)
        _replace(model, name, plain)
    return names


def unwrap(model: nn.Module) -> nn.Module:
    """返回已还原为普通 Conv2d 的深拷贝（保存/导出前使用）。"""
    import copy

    clone = copy.deepcopy(model)
    disable_weight_qat(clone)
    return clone


def qat_layer_names(model: nn.Module) -> list[str]:
    return [name for name, module in model.named_modules()
            if isinstance(module, FakeQuantConv2d)]


def calibration_scales(model: nn.Module, dataset: Any, limit: int | None = None,
                       hooks_output: bool = True) -> dict[str, dict[str, float]]:
    """用校准数据统计每层 ``input_max_abs``/``output_max_abs``（与既有导出同源口径）。

    调用前应把模型置于 ``eval()``；前向按 ``dataset[index].image`` 逐张进行（无增强）。
    返回 ``{layer_name: {"input_max_abs": float, "output_max_abs": float}}``。
    """
    total = len(dataset) if limit is None else min(int(limit), len(dataset))
    stats: dict[str, dict[str, float]] = {}
    handles = []
    for name, module in model.named_modules():
        if not isinstance(module, FakeQuantConv2d):
            continue
        stats[name] = {"input_max_abs": 0.0, "output_max_abs": 0.0}

        def pre_hook(_module, arguments, key=name):
            value = arguments[0].detach()
            stats[key]["input_max_abs"] = max(stats[key]["input_max_abs"],
                                              float(value.abs().amax()))

        def post_hook(_module, _arguments, output, key=name):
            if not hooks_output:
                return
            value = output.detach()
            stats[key]["output_max_abs"] = max(stats[key]["output_max_abs"],
                                               float(value.abs().amax()))

        handles.append(module.register_forward_pre_hook(pre_hook))
        handles.append(module.register_forward_hook(post_hook))
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            for index in range(total):
                item = dataset[index]
                image = item.image if hasattr(item, "image") else item["image"]
                # 校准必须看到 FP32 直通（禁用假量化），否则统计的是已量化后的分布
                for module in model.modules():
                    if isinstance(module, FakeQuantConv2d):
                        module.enabled = False
                try:
                    model(image.unsqueeze(0).to(device))
                finally:
                    for module in model.modules():
                        if isinstance(module, FakeQuantConv2d):
                            module.enabled = True
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return stats


def set_scales_from_calibration(model: nn.Module,
                                stats: Mapping[str, Mapping[str, float]]) -> dict[str, dict[str, float]]:
    """把统计结果换算为尺度并写入各层；返回实际写入的 ``{layer: {input_scale, output_scale}}``。

    两段式：先对**全部**层校验统计齐全，再统一写入，避免中途失败留下"部分层已改"的
    非原子状态。
    """
    layers: list[tuple[str, FakeQuantConv2d]] = [
        (name, module) for name, module in model.named_modules()
        if isinstance(module, FakeQuantConv2d)]
    if not layers:
        raise ValueError("模型中没有 FakeQuantConv2d 层")
    missing = [name for name, _module in layers if name not in stats]
    if missing:
        raise KeyError(f"缺少校准统计的层: {missing[:5]}")
    prepared: dict[str, dict[str, float]] = {}
    for name, _module in layers:
        prepared[name] = {"input_scale": _symmetric_scale(stats[name]["input_max_abs"]),
                          "output_scale": _symmetric_scale(stats[name]["output_max_abs"]),
                          "input_max_abs": float(stats[name]["input_max_abs"]),
                          "output_max_abs": float(stats[name]["output_max_abs"])}
    for name, module in layers:      # 校验通过后才写入
        module.input_scale = prepared[name]["input_scale"]
        module.output_scale = prepared[name]["output_scale"]
    return prepared


def scales_snapshot(model: nn.Module) -> dict[str, dict[str, float]]:
    """导出当前各层尺度，供断点记录与 resume 逐位一致性校验。"""
    snapshot: dict[str, dict[str, float]] = {}
    for name, module in model.named_modules():
        if isinstance(module, FakeQuantConv2d):
            snapshot[name] = {"input_scale": float(module.input_scale),
                              "output_scale": float(module.output_scale)}
    return snapshot
