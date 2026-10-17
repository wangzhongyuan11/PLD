"""模型测试：I/O shape、参数量、算子白名单、logits 语义。

这些测试故意使用小尺寸输入（64/32）以保持快速，同时也覆盖 shape 校验。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _support import MISSING_DEPENDENCY, ensure_role_a_on_path  # noqa: E402

ensure_role_a_on_path()

import torch  # noqa: E402
from torch import nn  # noqa: E402

from model import (  # noqa: E402
    DEFAULT_BASE_CHANNELS,
    TinyUNet,
    build_model,
    count_parameters,
    describe_graph,
)


@unittest.skipIf(MISSING_DEPENDENCY is not None, MISSING_DEPENDENCY or "")
class TinyUNetShapeTests(unittest.TestCase):
    def test_default_construction_matches_spec(self) -> None:
        network = TinyUNet()
        self.assertEqual(network.base_channels, DEFAULT_BASE_CHANNELS)
        self.assertEqual(network.input_shape, (3, 256, 256))
        self.assertEqual(network.channels, (8, 16, 32, 64))

    def test_forward_output_shape(self) -> None:
        network = build_model(base_channels=8, num_pools=3, input_shape=(3, 64, 64))
        network.eval()
        with torch.no_grad():
            output = network(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 1, 64, 64))

    def test_256_input_output_shape(self) -> None:
        network = build_model(input_shape=(3, 256, 256))
        self.assertEqual(network.assert_shapes(batch_size=1), (1, 1, 256, 256))

    def test_output_is_logits_not_probabilities(self) -> None:
        """未训练的 logits 会落在 [0,1] 之外；若输出被 sigmoid 过就会露馅。"""
        torch.manual_seed(0)
        network = build_model(input_shape=(3, 64, 64))
        with torch.no_grad():
            output = network(torch.randn(4, 3, 64, 64) * 4.0)
        self.assertTrue(bool((output < 0).any()) or bool((output > 1).any()),
                        "输出全在 [0,1] 内，怀疑模型内部做了 sigmoid")
        self.assertEqual(output.shape[1], 1)

    def test_wrong_channel_count_raises(self) -> None:
        network = build_model(input_shape=(3, 64, 64))
        with self.assertRaises(ValueError):
            network(torch.zeros(1, 1, 64, 64))

    def test_wrong_rank_raises(self) -> None:
        network = build_model(input_shape=(3, 64, 64))
        with self.assertRaises(ValueError):
            network(torch.zeros(3, 64, 64))

    def test_odd_spatial_size_raises(self) -> None:
        network = build_model(input_shape=(3, 64, 64))
        with self.assertRaises(ValueError) as ctx:
            network(torch.zeros(1, 3, 60, 64))
        self.assertIn("整除", str(ctx.exception))

    def test_invalid_constructor_arguments_raise(self) -> None:
        for kwargs in ({"base_channels": 0}, {"num_pools": 0}, {"num_pools": 4},
                       {"input_shape": (3, 60, 60)}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    TinyUNet(**kwargs)  # type: ignore[arg-type]

    def test_num_pools_variants_round_trip(self) -> None:
        for num_pools in (1, 2, 3):
            with self.subTest(num_pools=num_pools):
                network = build_model(base_channels=4, num_pools=num_pools,
                                      input_shape=(3, 64, 64))
                size = 2**num_pools
                with torch.no_grad():
                    output = network(torch.zeros(1, 3, size * 4, size * 4))
                self.assertEqual(tuple(output.shape), (1, 1, size * 4, size * 4))


@unittest.skipIf(MISSING_DEPENDENCY is not None, MISSING_DEPENDENCY or "")
class TinyUNetParameterTests(unittest.TestCase):
    def test_parameter_count_is_small_and_consistent(self) -> None:
        network = build_model()
        total = network.num_parameters()
        breakdown = network.parameter_breakdown()
        self.assertEqual(total, sum(breakdown.values()))
        self.assertEqual(total, count_parameters(network))
        self.assertLess(total, 250_000, "基通道 8 的 Tiny U-Net 参数量不应超过 25 万")
        self.assertGreater(total, 1_000)

    def test_training_parameter_count_excludes_frozen(self) -> None:
        network = build_model(input_shape=(3, 64, 64))
        before = network.num_parameters()
        for name, parameter in network.named_parameters():
            if name.startswith("encoders.0"):
                parameter.requires_grad_(False)
        self.assertLess(network.num_parameters(), before)
        self.assertEqual(network.num_parameters(trainable_only=False), before)

    def test_gradients_flow_to_every_parameter(self) -> None:
        torch.manual_seed(0)
        network = build_model(base_channels=4, input_shape=(3, 32, 32))
        logits = network(torch.randn(2, 3, 32, 32))
        logits.mean().backward()
        for name, parameter in network.named_parameters():
            self.assertIsNotNone(parameter.grad, f"{name} 没有梯度")
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), f"{name} 梯度非有限")


@unittest.skipIf(MISSING_DEPENDENCY is not None, MISSING_DEPENDENCY or "")
class TinyUNetGraphTests(unittest.TestCase):
    def test_only_hardware_friendly_layers(self) -> None:
        graph = describe_graph(build_model())
        self.assertTrue(graph["allowed_only"], f"出现不允许的算子: {graph['forbidden']}")
        for layer in graph["layers"]:
            self.assertIn(layer["type"],
                          {"Conv2d", "ReLU", "MaxPool2d", "Upsample", "Sequential", "ModuleList"})

    def test_graph_reports_no_normalization_or_attention(self) -> None:
        network = build_model()
        banned = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm,
                  nn.GroupNorm, nn.Dropout, nn.ConvTranspose2d, nn.Sigmoid, nn.Softmax)
        modules = list(network.modules())
        for kind in banned:
            self.assertFalse(any(isinstance(module, kind) for module in modules),
                             f"模型中不应出现 {kind.__name__}")

    def test_graph_detail_fields(self) -> None:
        graph = describe_graph(build_model(base_channels=8, input_shape=(3, 64, 64)))
        self.assertEqual(graph["base_channels"], 8)
        self.assertEqual(graph["channels"], [8, 16, 32, 64])
        convolutions = [layer for layer in graph["layers"] if layer["type"] == "Conv2d"]
        kernels = {"k=3", "k=1"}
        for layer in convolutions:
            self.assertTrue(any(token in layer["detail"] for token in kernels))
            self.assertIn("bias=True", layer["detail"])
        self.assertEqual(graph["parameters"]["total"], build_model(
            base_channels=8, input_shape=(3, 64, 64)).num_parameters())

    def test_describe_graph_rejects_other_modules(self) -> None:
        with self.assertRaises(TypeError):
            describe_graph(nn.Conv2d(3, 3, 3))  # type: ignore[arg-type]

    def test_features_keep_full_spatial_resolution(self) -> None:
        network = build_model(base_channels=4, num_pools=3, input_shape=(3, 64, 64))
        with torch.no_grad():
            logits, features = network.forward_with_features(torch.zeros(2, 3, 64, 64))
        self.assertEqual(tuple(logits.shape), (2, 1, 64, 64))
        self.assertEqual(tuple(features.shape), (2, 4 * 2**3, 8, 8))


if __name__ == "__main__":
    unittest.main(verbosity=2)
