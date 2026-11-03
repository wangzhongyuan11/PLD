"""A 组 QAT 原语的针对性测试：不改键、梯度有限、权重确实更新、导出兼容。

无训练、无网络、不读数据集；只做小张量前向/反向与一次真实 checkpoint 兼容检查。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from software.role_a import qat
from software.role_a import quant
from software.role_a.model import build_model
from software.role_a.quant_math import QMAX

SOUP = ROOT / "runs" / "role_a_a14_soup" / "candidates" / "soup_v5_a050.pt"


def _set_unit_scales(model) -> None:
    """给所有假量化层设置统一尺度（单元测试用的最小合法配置）。"""
    for module in model.modules():
        if isinstance(module, qat.FakeQuantConv2d):
            module.input_scale = 1.0 / QMAX
            module.output_scale = 2.0 / QMAX


class PrimitiveTests(unittest.TestCase):
    def test_quantized_weight_round_trip(self) -> None:
        weight = torch.linspace(-1.0, 1.0, 3 * 4 * 3 * 3).reshape(3, 4, 3, 3)
        dequantized, scales = qat.quantized_weight(weight)
        coded = (dequantized / scales[:, None, None, None]).round()
        self.assertLessEqual(float(coded.abs().max()), QMAX)
        self.assertGreaterEqual(float(coded.abs().max()), QMAX - 1)
        self.assertLess(float((dequantized - weight).norm() / weight.norm()), 0.01)
        self.assertTrue(bool((scales > 0).all()))

    def test_fake_quant_is_quantized_value_with_ste(self) -> None:
        weight = torch.randn(4, 3, 3, 3, requires_grad=True)
        fake = qat.fake_quant_weight(weight)
        dequantized, _scales = qat.quantized_weight(weight)
        self.assertTrue(torch.equal(fake.detach(), dequantized))
        self.assertFalse(torch.equal(fake.detach(), weight.detach()))
        fake.sum().backward()
        self.assertIsNotNone(weight.grad)
        self.assertTrue(torch.equal(weight.grad, torch.ones_like(weight)))  # STE 恒等传梯度


class WrappingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = build_model(base_channels=8, num_pools=3, input_shape=(3, 32, 32))
        self.before = {key: value.clone() for key, value in self.model.state_dict().items()}
        self.before_order = list(self.model.state_dict())

    def test_keys_shapes_and_order_unchanged(self) -> None:
        names = qat.enable_weight_qat(self.model)
        self.assertEqual(len(names), 15)
        after = self.model.state_dict()
        self.assertEqual(list(after), self.before_order)
        for key in self.before_order:
            self.assertEqual(tuple(after[key].shape), tuple(self.before[key].shape))
            self.assertEqual(after[key].dtype, self.before[key].dtype)
            self.assertTrue(torch.equal(after[key], self.before[key]))  # 权重值逐位继承

    def test_forward_equals_manual_quantized_weight_forward(self) -> None:
        qat.enable_weight_qat(self.model)
        _set_unit_scales(self.model)
        images = torch.rand(2, 3, 32, 32)
        with torch.no_grad():
            wrapped = self.model(images)
            for _name, conv in quant._conv_modules(self.model):
                dequantized, _scales = qat.quantized_weight(conv.weight)
                conv.weight.data.copy_(dequantized)
            manual = self.model(images)
        self.assertTrue(torch.equal(wrapped, manual))

    def test_disable_restores_plain_conv_and_original_output(self) -> None:
        qat.enable_weight_qat(self.model)
        _set_unit_scales(self.model)
        images = torch.rand(2, 3, 32, 32)
        names = qat.disable_weight_qat(self.model)
        self.assertEqual(len(names), 15)
        self.assertEqual(qat.qat_layer_names(self.model), [])
        with torch.no_grad():
            output = self.model(images)
        model2 = build_model(base_channels=8, num_pools=3, input_shape=(3, 32, 32))
        model2.load_state_dict(self.before, strict=True)
        with torch.no_grad():
            reference = model2(images)
        self.assertTrue(torch.allclose(output, reference, atol=1e-6))

    def test_backward_finite_and_ste_gradient_nonzero(self) -> None:
        qat.enable_weight_qat(self.model)
        _set_unit_scales(self.model)
        images = torch.rand(1, 3, 32, 32)
        self.model(images).square().mean().backward()
        gradient = self.model.encoders[0][0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(bool(torch.isfinite(gradient).all()))
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_two_optimizer_steps_change_master_and_quantized_weight(self) -> None:
        """审查重点：两次优化步骤后权重必须真的变化（旧 hook 方案会抹掉更新）。"""
        qat.enable_weight_qat(self.model)
        _set_unit_scales(self.model)
        conv = self.model.encoders[0][0]
        master_before = conv.weight.detach().clone()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2)
        quantized_history = []
        for _step in range(2):
            images = torch.rand(1, 3, 32, 32)
            optimizer.zero_grad(set_to_none=True)
            self.model(images).square().mean().backward()
            optimizer.step()
            dequantized, _scales = qat.quantized_weight(conv.weight)
            quantized_history.append(dequantized.detach().clone())
        self.assertFalse(torch.equal(conv.weight.detach(), master_before),
                         "主权重两次优化后未变化")
        self.assertFalse(torch.equal(quantized_history[0], quantized_history[1]),
                         "量化权重两次优化后未变化：更新被抹掉")

    def test_optimizer_sees_same_parameter_count(self) -> None:
        before = sum(p.numel() for p in self.model.parameters())
        qat.enable_weight_qat(self.model)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()), before)
        self.assertEqual(len(list(self.model.parameters())), len(self.before_order))


class RealCheckpointCompatibilityTests(unittest.TestCase):
    @unittest.skipUnless(SOUP.is_file(), "soup 权重不存在，跳过真实兼容检查")
    def test_strict_load_after_wrapping_and_unwrap_keys(self) -> None:
        model, config = quant._load_fp32(SOUP)
        state = {key: value.clone() for key, value in model.state_dict().items()}
        qat.enable_weight_qat(model)
        _set_unit_scales(model)
        model.load_state_dict(state, strict=True)      # 键名/形状未变 => strict 仍成立
        self.assertEqual(len(qat.qat_layer_names(model)), 15)

        model.train()
        logits = model(torch.rand(1, 3, int(config["size"]), int(config["size"])))
        logits.square().mean().backward()
        self.assertTrue(all(bool(torch.isfinite(p.grad).all())
                            for p in model.parameters() if p.grad is not None))

        plain = qat.unwrap(model)
        self.assertEqual(qat.qat_layer_names(plain), [])
        self.assertEqual(list(plain.state_dict()), list(state))

    @unittest.skipUnless(SOUP.is_file(), "soup 权重不存在，跳过真实兼容检查")
    def test_unwrapped_checkpoint_is_readable_by_export_loader(self) -> None:
        import tempfile

        model, config = quant._load_fp32(SOUP)
        qat.enable_weight_qat(model)
        _set_unit_scales(model)
        with torch.no_grad():
            model(torch.rand(1, 3, int(config["size"]), int(config["size"])))
        plain = qat.unwrap(model)
        with tempfile.TemporaryDirectory(prefix="qat_export_") as tmp:
            path = Path(tmp) / "candidate.pt"
            torch.save({"schema": "qat-selftest-v1", "config": config,
                        "model_state": plain.state_dict()}, path)
            reloaded, reloaded_config = quant._load_fp32(path)   # 导出器必须能读
            self.assertEqual(int(reloaded_config["base_channels"]), 8)
            self.assertEqual(len(quant._conv_modules(reloaded)), 15)
            self.assertEqual({name for name, _ in quant._conv_modules(reloaded)},
                             {name for name, _ in quant._conv_modules(quant._load_fp32(SOUP)[0])})


if __name__ == "__main__":
    unittest.main()
