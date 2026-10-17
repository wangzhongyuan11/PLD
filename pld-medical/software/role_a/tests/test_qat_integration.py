"""A 组 QAT 集成测试：三处假量化、固定尺度、尺度刷新、导出兼容与量化一致性。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from software.role_a import qat
from software.role_a import quant
from software.role_a.model import build_model
from software.role_a.quant_math import QMAX

SOUP = ROOT / "runs" / "role_a_a14_soup" / "candidates" / "soup_v5_a050.pt"


class _Item:
    def __init__(self, image: torch.Tensor) -> None:
        self.image = image


class _CalibrationSet:
    """固定合成校准集（确定性、无增强）。"""

    def __init__(self, count: int = 4, size: int = 32, seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.images = [torch.rand(3, size, size, generator=generator) for _ in range(count)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> _Item:
        return _Item(self.images[index])


class ActivationQuantizationTests(unittest.TestCase):
    def test_matches_deployment_rule_and_ste(self) -> None:
        value = torch.tensor([[-2.0, -0.3, 0.0, 0.4, 1.4]], dtype=torch.float32,
                             requires_grad=True)
        scale = 1.0 / 127
        result = qat.fake_quant_activation(value, scale)
        expected = torch.clamp(torch.round(value.detach() / scale), -QMAX, QMAX) * scale
        self.assertTrue(torch.equal(result.detach(), expected))
        # 饱和到 ±1.0
        self.assertEqual(float(result.detach()[0, 4]), 1.0)
        self.assertEqual(float(result.detach()[0, 0]), -1.0)
        result.sum().backward()
        self.assertTrue(torch.equal(value.grad, torch.ones_like(value)))  # STE 恒等

    def test_zero_scale_rejected(self) -> None:
        with self.assertRaises(ValueError):
            qat.fake_quant_activation(torch.zeros(1), 0.0)


class FakeQuantConv2dTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=True)
        self.wrapped = qat.FakeQuantConv2d(3, 4, 3, padding=1, bias=True)
        with torch.no_grad():
            self.wrapped.weight.copy_(self.conv.weight)
            self.wrapped.bias.copy_(self.conv.bias)
        self.wrapped.input_scale = 1.0 / 127
        self.wrapped.output_scale = 2.0 / 127

    def test_forward_is_input_weight_output_quantized(self) -> None:
        images = torch.randn(2, 3, 8, 8)
        with torch.no_grad():
            actual = self.wrapped(images)
            xq = torch.clamp(torch.round(images / self.wrapped.input_scale), -QMAX, QMAX) \
                * self.wrapped.input_scale
            weight = qat.fake_quant_weight(self.conv.weight)
            manual = F.conv2d(xq, weight, self.conv.bias, self.conv.stride, self.conv.padding)
            manual = torch.clamp(torch.round(manual / self.wrapped.output_scale), -QMAX, QMAX) \
                * self.wrapped.output_scale
        self.assertTrue(torch.equal(actual, manual))

    def test_missing_scales_raise(self) -> None:
        plain = qat.FakeQuantConv2d(3, 4, 3, padding=1)
        with self.assertRaises(RuntimeError):
            plain(torch.randn(1, 3, 8, 8))

    def test_keys_unchanged_and_weights_inherited(self) -> None:
        model = build_model(base_channels=8, num_pools=3, input_shape=(3, 32, 32))
        before = {k: v.clone() for k, v in model.state_dict().items()}
        order = list(model.state_dict())
        qat.enable_weight_qat(model)
        after = model.state_dict()
        self.assertEqual(list(after), order)
        for key in order:
            self.assertTrue(torch.equal(after[key], before[key]))

    def test_output_quantization_is_ste_and_gradients_finite(self) -> None:
        partial = build_model(base_channels=8, num_pools=3, input_shape=(3, 32, 32))
        qat.enable_weight_qat(partial)
        for module in partial.modules():
            if isinstance(module, qat.FakeQuantConv2d):
                module.input_scale = 1.0 / 127
                module.output_scale = 1.0 / 127
        output = partial(torch.rand(1, 3, 32, 32))
        output.square().mean().backward()
        self.assertTrue(all(bool(torch.isfinite(p.grad).all())
                            for p in partial.parameters() if p.grad is not None))


class CalibrationScaleTests(unittest.TestCase):
    def test_calibration_scales_match_fp32_max_abs(self) -> None:
        torch.manual_seed(0)
        model = build_model(base_channels=8, num_pools=3, input_shape=(3, 32, 32)).eval()
        qat.enable_weight_qat(model)
        dataset = _CalibrationSet(count=4, size=32)
        stats = qat.calibration_scales(model, dataset, len(dataset))
        self.assertEqual(len(stats), 15)
        # 校准统计必须来自 FP32 直通：与手工 FP32 前向的 max_abs 一致
        reference = build_model(base_channels=8, num_pools=3, input_shape=(3, 32, 32)).eval()
        reference.load_state_dict({k: v for k, v in model.state_dict().items()}, strict=True)
        captured: dict[str, float] = {}
        handles = []
        for name, conv in quant._conv_modules(reference):
            def hook(_module, _arguments, output, key=name):
                captured[key] = max(captured.get(key, 0.0), float(output.abs().amax()))
            handles.append(conv.register_forward_hook(hook))
        with torch.no_grad():
            for index in range(len(dataset)):
                reference(dataset[index].image.unsqueeze(0))
        for handle in handles:
            handle.remove()
        for name, entry in stats.items():
            self.assertAlmostEqual(entry["output_max_abs"], captured[name], places=6, msg=name)

    def test_scales_are_fixed_during_training_while_weights_change(self) -> None:
        torch.manual_seed(0)
        model = build_model(base_channels=8, num_pools=3, input_shape=(3, 32, 32))
        qat.enable_weight_qat(model)
        dataset = _CalibrationSet(count=2, size=32)
        stats = qat.calibration_scales(model, dataset, len(dataset))
        qat.set_scales_from_calibration(model, stats)
        snapshot = qat.scales_snapshot(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        conv = model.encoders[0][0]
        weight_before = conv.weight.detach().clone()
        for _step in range(2):
            optimizer.zero_grad(set_to_none=True)
            model(torch.rand(1, 3, 32, 32)).square().mean().backward()
            optimizer.step()
        self.assertEqual(qat.scales_snapshot(model), snapshot)          # 尺度不漂移
        self.assertFalse(torch.equal(conv.weight.detach(), weight_before))  # 权重确实更新

    def test_recompute_on_same_set_is_deterministic(self) -> None:
        torch.manual_seed(0)
        model = build_model(base_channels=8, num_pools=3, input_shape=(3, 32, 32)).eval()
        qat.enable_weight_qat(model)
        dataset = _CalibrationSet(count=3, size=32)
        first = qat.calibration_scales(model, dataset, len(dataset))
        qat.set_scales_from_calibration(model, first)
        second = qat.calibration_scales(model, dataset, len(dataset))
        self.assertEqual(first, second)


class ExportCompatibilityTests(unittest.TestCase):
    @unittest.skipUnless(SOUP.is_file(), "soup 权重不存在，跳过导出兼容检查")
    def test_export_loader_accepts_unwrapped_candidate(self) -> None:
        model, config = quant._load_fp32(SOUP)
        qat.enable_weight_qat(model)
        for module in model.modules():
            if isinstance(module, qat.FakeQuantConv2d):
                module.input_scale = 1.0 / 127
                module.output_scale = 1.0 / 127
        with torch.no_grad():
            model(torch.rand(1, 3, int(config["size"]), int(config["size"])))
        plain = qat.unwrap(model)
        with tempfile.TemporaryDirectory(prefix="qat_export_") as tmp:
            path = Path(tmp) / "candidate.pt"
            torch.save({"schema": "a-final-qat-candidate-v1", "config": config,
                        "model_state": plain.state_dict()}, path)
            reloaded, reloaded_config = quant._load_fp32(path)
            self.assertEqual(int(reloaded_config["base_channels"]), 8)
            self.assertEqual(len(quant._conv_modules(reloaded)), 15)
            self.assertEqual({name for name, _ in quant._conv_modules(reloaded)},
                             {name for name, _ in quant._conv_modules(quant._load_fp32(SOUP)[0])})


if __name__ == "__main__":
    unittest.main()
