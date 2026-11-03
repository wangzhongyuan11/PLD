"""参考计算器与真实 torch 模型的交叉验证。

`verify_reference.py` 是不依赖 torch 的纯算术实现；一旦它与 `model.py` 的实际结构
不一致（例如有人改了卷积层数或通道数），参数量会立刻对不上，这里就会失败。
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

from model import build_model  # noqa: E402
from verify_reference import ModelReference, conv_parameters  # noqa: E402


class ReferenceArithmeticTests(unittest.TestCase):
    """纯算术部分不依赖 torch，任何时候都必须能跑。"""

    def test_conv_parameters(self) -> None:
        self.assertEqual(conv_parameters(3, 8, 3), 3 * 8 * 9 + 8)
        self.assertEqual(conv_parameters(8, 1, 1), 8 * 1 + 1)
        self.assertEqual(conv_parameters(4, 4, 3, bias=False), 4 * 4 * 9)

    def test_invalid_arguments_raise(self) -> None:
        with self.assertRaises(ValueError):
            conv_parameters(0, 8, 3)
        with self.assertRaises(ValueError):
            ModelReference(base_channels=0)
        with self.assertRaises(ValueError):
            ModelReference(num_pools=4)
        with self.assertRaises(ValueError):
            ModelReference(size=60, num_pools=3)

    def test_reference_shapes(self) -> None:
        reference = ModelReference(base_channels=8, num_pools=3, size=256)
        summary = reference.summary()
        self.assertEqual(summary["input_shape"], [1, 3, 256, 256])
        self.assertEqual(summary["output_shape"], [1, 1, 256, 256])
        self.assertEqual(summary["channels"], [8, 16, 32, 64])
        # 最深一级是 32x32；最宽的临时张量是浅层 skip 与上采样输出拼接。
        deepest = [layer for layer in summary["layers"] if layer["name"] == "encoders.3.4"]
        self.assertEqual(deepest[0]["out_hw"], [32, 32])
        self.assertEqual(summary["peak_activation_elements"], 24 * 256 * 256)

    def test_parameter_breakdown_is_consistent(self) -> None:
        parameters = ModelReference(base_channels=8, num_pools=3).parameters()
        self.assertEqual(parameters["total"], sum(parameters["by_module"].values()))
        self.assertGreater(parameters["total"], 0)
        self.assertEqual(parameters["total"],
                         parameters["weight_parameters_excluding_bias"]
                         + parameters["bias_parameters"])
        self.assertEqual(parameters["bytes_fp32"], parameters["total"] * 4)
        self.assertEqual(parameters["bytes_int8_weights_only"],
                         parameters["weight_parameters_excluding_bias"])

    def test_macs_count_only_convolutions(self) -> None:
        summary = ModelReference(base_channels=8, num_pools=3, size=256).summary()
        for layer in summary["layers"]:
            if layer["type"] in ("ReLU", "MaxPool2x2", "UpsampleNearest", "Concat"):
                self.assertEqual(layer["macs"], 0, layer["name"])
        self.assertGreater(summary["total_macs_fp32"], 0)


@unittest.skipIf(MISSING_DEPENDENCY is not None, MISSING_DEPENDENCY or "")
class ReferenceMatchesModelTests(unittest.TestCase):
    def test_parameter_count_matches_torch_model(self) -> None:
        for base_channels, num_pools in ((8, 3), (4, 2), (8, 1), (16, 3)):
            with self.subTest(base_channels=base_channels, num_pools=num_pools):
                size = 2**num_pools * 8
                network = build_model(base_channels=base_channels, num_pools=num_pools,
                                      input_shape=(3, size, size))
                reference = ModelReference(base_channels=base_channels, num_pools=num_pools,
                                           size=size)
                self.assertEqual(network.num_parameters(), reference.parameters()["total"])

    def test_activation_shapes_match_real_forward(self) -> None:
        size = 32
        network = build_model(base_channels=8, num_pools=3, input_shape=(3, size, size))
        network.eval()
        with torch.no_grad():
            logits = network(torch.zeros(1, 3, size, size))
        summary = ModelReference(base_channels=8, num_pools=3, size=size).summary()
        self.assertEqual(list(logits.shape), summary["output_shape"])
        # 逐层核对编码器输出 shape（Sequential 内索引 0/3 是卷积，1/4 是 ReLU）。
        with torch.no_grad():
            x = torch.zeros(1, 3, size, size)
            for level in range(4):
                x = network.encoders[level](x)
                expected = next(layer for layer in summary["layers"]
                                if layer["name"] == f"encoders.{level}.4")
                self.assertEqual([x.shape[1], x.shape[2], x.shape[3]],
                                 [expected["out_channels"], *expected["out_hw"]])
                if level < 3:
                    x = network.pools[level](x)
                    pooled = next(layer for layer in summary["layers"]
                                  if layer["name"] == f"pools.{level}")
                    self.assertEqual([x.shape[2], x.shape[3]], pooled["out_hw"])

    def test_peak_activation_matches_largest_tensor(self) -> None:
        size = 32
        network = build_model(base_channels=8, num_pools=3, input_shape=(3, size, size))
        network.eval()
        with torch.no_grad():
            _, features = network.forward_with_features(torch.zeros(1, 3, size, size))
        summary = ModelReference(base_channels=8, num_pools=3, size=size).summary()
        # 浅层 Concat 实际物化为 24 通道；瓶颈特征远小于它。
        self.assertEqual(summary["peak_activation_elements"], 24 * size * size)
        self.assertLess(features.numel(), summary["peak_activation_elements"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
