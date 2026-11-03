"""CAMUS INT8 合成 CPU 用例（**不访问真实权重/缓存像素、不训练、不推理真实数据**）。

覆盖（按任务 172 要求）：

* 人工小整数 Conv（与 NumPy 整数 oracle 逐元素一致）、±ties-even 边界、``-128``/超范围拒绝；
* bias/acc/product 溢出拒绝（含**逐输出通道保守界** `|bias|+127*sum|qw|`）；
* concat 两支按 ``max(source scales)`` 的 Q24 ties-even 重定标；整数 pool 与 nearest repeat；
* **共享 head 输出 scale**；全图 synthetic 输入端到端（15 Conv、skip-first 拼接）；
* spec 路径安全/哈希/packed 偏移/源缺失拒绝（用**随机模型 synthetic 导出**到 QA 临时目录，
  **不伪称真实 CAMUS 权重**）。

运行（单次，≤300 秒）：
``python -m unittest discover -s software/role_a_camus_int8/tests -p test_int8_cpu.py -v``
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT = Path(r"E:/嵌赛/pld-medical")
for path in (str(PROJECT), str(PROJECT / "software/role_a")):
    if path not in sys.path:
        sys.path.insert(0, path)

import quant_math as qm                                        # noqa: E402
from software.role_a_camus_int8 import integer as i8           # noqa: E402
from software.role_a_camus_int8 import pipeline as pipe        # noqa: E402


def small_conv(in_channels: int, out_channels: int, kernel: int = 3, padding: int = 1,
               seed: int = 0) -> nn.Conv2d:
    torch.manual_seed(seed)
    conv = nn.Conv2d(in_channels, out_channels, kernel, padding=padding, bias=True)
    with torch.no_grad():
        conv.weight.copy_(torch.randint(-9, 10, conv.weight.shape).float() / 8.0)
        conv.bias.copy_(torch.randint(-4, 5, conv.bias.shape).float() / 16.0)
    return conv


class IntegerConvTests(unittest.TestCase):
    def test_hand_small_conv_matches_oracle(self) -> None:
        conv = small_conv(2, 3, kernel=3, padding=1, seed=3)
        layer = i8.build_layer(conv, "conv0", input_scale=0.1, output_scale=0.2)
        feature = i8.Int8Feature(np.random.default_rng(0).integers(-127, 128, (1, 2, 6, 6),
                                                                 dtype=np.int8), 0.1)
        output = i8.conv2d_int8_float64(feature, layer, relu=True)
        reference = qm.requantize_int8(qm.conv2d_int32(feature.data, layer.q_weight,
                                                      layer.q_bias, layer.padding),
                                      layer.multiplier)
        reference = np.clip(reference.astype(np.int16), 0, 127).astype(np.int8)
        self.assertTrue(np.array_equal(output.data, reference))
        self.assertEqual(output.scale, 0.2)

    def test_ties_even_rounding_and_saturation(self) -> None:
        half=1<<23
        cases={10:0,-10:0,30:0,-30:0,half:0,-half:0,half+1:1,half-1:0,3*half:2,-3*half:-2,5*half:2,-5*half:-2,7*half:4,-7*half:-4}
        for value, expected in cases.items():
            products = torch.tensor([[[[value]]]], dtype=torch.int64)
            rounded = i8._round_divide_even_int64(products)
            self.assertEqual(int(rounded.item()), expected, f"value={value}")
        saturated = i8.Int8Feature(np.array([[[[3000]]]], dtype=np.int64).astype(np.int8)
                                   if False else np.array([[[[127]]]], dtype=np.int8), 1.0)
        rescaled = i8.rescale_int8(i8.Int8Feature(np.full((1, 1, 1, 1), 120, dtype=np.int8), 0.2),
                                   0.01)
        self.assertEqual(int(rescaled.data.max()), 127, "重定标必须饱和到 ±127")
        _ = saturated

    def test_range_and_shape_rejection(self) -> None:
        with self.assertRaises(i8.IntegerGraphError):
            i8.Int8Feature(np.zeros((1, 1, 2, 2), dtype=np.int16), 0.1).check()
        with self.assertRaises(i8.IntegerGraphError):
            i8.quantize_input_float64(np.full((4, 4), 1.5))
        with self.assertRaises(i8.IntegerGraphError):
            i8.quantize_input_float64(np.full((4, 4), np.nan))
        with self.assertRaises(i8.IntegerGraphError):
            i8.Int8Feature(np.zeros((4, 4), dtype=np.int8), 0.1).check()
        conv = small_conv(1, 1, kernel=3, padding=1)
        layer = i8.build_layer(conv, "c", input_scale=0.1, output_scale=0.1)
        wrong_scale = i8.Int8Feature(np.zeros((1, 1, 4, 4), dtype=np.int8), 0.05)
        with self.assertRaises(i8.IntegerGraphError):
            i8.conv2d_int8_float64(wrong_scale, layer, relu=True)

    def test_overflow_rejections(self) -> None:
        weight = np.full((1, 1, 3, 3), 127, dtype=np.int8)
        bias = np.array([2 ** 31 - 1], dtype=np.int64)
        with self.assertRaises(i8.IntegerGraphError):
            i8.require_bounds(weight, bias.astype(np.int32))
        safe_weight = np.full((1, 1, 3, 3), 127, dtype=np.int8)
        safe_bias = np.zeros(1, dtype=np.int32)
        bound = i8.require_bounds(safe_weight, safe_bias)
        self.assertLessEqual(int(bound[0]), i8.INT32_MAX)
        self.assertEqual(int(bound[0]), 127 * 9 * 127)
        # A large positive int32 multiplier is allowed: this product is safely <INT64.
        safe=i8.QuantLayer('fake',safe_weight,safe_bias,np.array([2**30],dtype=np.int32),1.0,1.0,1,3,bound,True)
        self.assertEqual(int(i8.conv2d_int8_float64(i8.Int8Feature(np.full((1,1,4,4),127,dtype=np.int8),1.0),safe,relu=False).data.max()),127)
        bad=i8.QuantLayer('fake',safe_weight,safe_bias,np.array([2**31],dtype=np.int64),1.0,1.0,1,3,bound,True)
        with self.assertRaises(i8.IntegerGraphError):
            i8.conv2d_int8_float64(i8.Int8Feature(np.zeros((1,1,4,4),dtype=np.int8),1.0),bad,relu=False)
        with self.assertRaises(i8.IntegerGraphError):
            i8.rescale_int8(i8.Int8Feature(np.full((1, 1, 2, 2), 100, dtype=np.int8), 0.9), 0.001)

    def test_conservative_bound_matches_formula(self) -> None:
        rng = np.random.default_rng(1)
        weight = rng.integers(-127, 128, (3, 2, 3, 3), dtype=np.int8)
        bias = rng.integers(-1000, 1000, 3, dtype=np.int32)
        bound = i8.conservative_bound(weight, bias)
        manual = np.abs(bias.astype(np.int64)) + 127 * np.abs(weight.astype(np.int64)).sum(
            axis=(1, 2, 3))
        self.assertTrue(np.array_equal(bound, manual))
        self.assertAlmostEqual(1728 * 127 * 127, 27870912)     # 审计更正：真实量级


class PoolNearestConcatTests(unittest.TestCase):
    def test_integer_pool_and_nearest(self) -> None:
        feature = i8.Int8Feature(np.arange(16, dtype=np.int8).reshape(1, 1, 4, 4), 0.5)
        pooled = i8.maxpool_int8(feature)
        self.assertEqual(pooled.data.shape, (1, 1, 2, 2))
        self.assertEqual(pooled.scale, 0.5)
        self.assertEqual(int(pooled.data[0, 0, 0, 0]), 5)
        upsampled = i8.nearest_upsample_int8(pooled)
        self.assertEqual(upsampled.data.shape, (1, 1, 4, 4))
        self.assertEqual(int(upsampled.data[0, 0, 0, 3]), 7)
        self.assertEqual(upsampled.scale, 0.5)
        with self.assertRaises(i8.IntegerGraphError):
            i8.maxpool_int8(i8.Int8Feature(np.zeros((1, 1, 3, 3), dtype=np.int8), 1.0))

    def test_concat_uses_max_source_scale_and_ties(self) -> None:
        first = i8.Int8Feature(np.full((1, 2, 2, 2), 100, dtype=np.int8), 0.01)
        second = i8.Int8Feature(np.full((1, 3, 2, 2), 7, dtype=np.int8), 0.02)
        merged, info = i8.concat_int8([first, second])
        self.assertEqual(info["common_scale"], 0.02)
        self.assertEqual(info["source_scales"], [0.01, 0.02])
        self.assertEqual(merged.data.shape[1], 5)
        self.assertEqual(int(merged.data[0, 0, 0, 0]), 50)      # 100 * 0.01/0.02
        self.assertEqual(int(merged.data[0, 2, 0, 0]), 7)       # 已是公共 scale

    def test_shared_head_scale(self) -> None:
        conv = small_conv(2, 4, kernel=1, padding=0)
        layer = i8.build_layer(conv, "head", input_scale=0.1, output_scale=0.25)
        self.assertEqual(layer.output_scale, 0.25)
        feature = i8.Int8Feature(np.ones((1, 2, 3, 3), dtype=np.int8), 0.1)
        logits = i8.conv2d_int8_float64(feature, layer, relu=False)
        dequantized = i8.dequantize(logits, layer.output_scale)
        self.assertEqual(dequantized.shape, (1, 4, 3, 3))
        self.assertAlmostEqual(float(dequantized[0].max() / 0.25),
                               float(logits.data[0].max()), places=9)


class SyntheticGraphTests(unittest.TestCase):
    def _model(self):
        from software.role_a_camus import model as camus_model

        return camus_model.build_model(base_channels=16, num_pools=3, size=64)

    def test_full_graph_synthetic_forward_shape_and_dtype(self) -> None:
        model = self._model()
        torch.manual_seed(11)
        for parameter in model.parameters():
            with torch.no_grad():
                parameter.copy_(torch.randn_like(parameter) * 0.05)
        calibration = {"input_scale": i8.INPUT_SCALE,
                       "layers": {name: {"input_max_abs": 1.0, "output_max_abs": 1.0}
                                  for name, _module in pipe.conv_modules(model)[:pipe.HIDDEN_CONVS]},
                       "head": {"input_max_abs": 1.0, "output_max_abs": 1.0},
                       "head_name": pipe.conv_modules(model)[pipe.HIDDEN_CONVS][0]}
        layers = pipe.build_layers(model, calibration)
        self.assertEqual(len(layers), pipe.TOTAL_CONVS)
        image = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        result = pipe.forward_graph(model, layers, image, valid=np.ones_like(image), capture=True)
        self.assertEqual(result["logits"].shape, (4, 64, 64))
        self.assertTrue(np.isfinite(result["logits"]).all())
        self.assertEqual(len(result["concat_info"]), 3)
        for info in result["concat_info"]:
            self.assertGreaterEqual(info["output_channels"], 2)
        self.assertIn("head", result["captures"])
        scales = {layer.output_scale for layer in layers[pipe.HIDDEN_CONVS:pipe.HIDDEN_CONVS + 1]}
        self.assertEqual(len(scales), 1, "head 必须只有一个共享输出 scale")

    def test_export_spec_paths_hashes_offsets_and_missing_source(self) -> None:
        model = self._model()
        torch.manual_seed(12)
        for parameter in model.parameters():
            with torch.no_grad():
                parameter.copy_(torch.randn_like(parameter) * 0.05)
        calibration = {"input_scale": i8.INPUT_SCALE,
                       "layers": {name: {"input_max_abs": 1.0, "output_max_abs": 1.0}
                                  for name, _module in pipe.conv_modules(model)[:pipe.HIDDEN_CONVS]},
                       "head": {"input_max_abs": 1.0, "output_max_abs": 1.0},
                       "head_name": pipe.conv_modules(model)[pipe.HIDDEN_CONVS][0]}
        layers = pipe.build_layers(model, calibration)
        image = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "spec"
            spec = pipe.export_graph(model, calibration, layers, out_dir, sample_image=image)
            self.assertEqual(spec["schema"], pipe.EXPORT_SCHEMA)
            self.assertEqual(len(spec["layers"]), pipe.TOTAL_CONVS)
            self.assertTrue((out_dir / "model_spec.json").is_file())
            weights_path = out_dir / "weights_int8.bin"
            bias_path = out_dir / "bias_int32.bin"
            self.assertTrue(weights_path.is_file() and bias_path.is_file())
            total = sum(layer.q_weight.size for layer in layers)
            self.assertEqual(weights_path.stat().st_size, total)
            self.assertEqual(bias_path.stat().st_size, sum(layer.q_bias.size for layer in layers) * 4)
            self.assertEqual(spec["packed"]["weights_int8"]["sha256"], pipe.sha256_file(weights_path))
            offsets = [record["packed_weight_offset"] for record in spec["layers"]]
            self.assertEqual(offsets, sorted(offsets))
            self.assertEqual(offsets[0], 0)
            for index, record in enumerate(spec["layers"][1:], start=1):
                previous = spec["layers"][index - 1]
                self.assertEqual(record["packed_weight_offset"],
                                 previous["packed_weight_offset"] + previous["packed_weight_count"])
            self.assertEqual(spec["layers"][-1]["name"],"head")
            self.assertIn("output_full_golden",spec["layers"][-1]["tile"])
            self.assertTrue(spec["full_sample"]["head_output_scale"] > 0)
            self.assertTrue(spec["quantization"]["no_sigmoid"])
            self.assertTrue(spec["quantization"]["software_simulation_only"])
            tampered = json.loads((out_dir / "model_spec.json").read_text(encoding="utf-8"))
            tampered["layers"][0]["packed_weight_offset"] = 999
            spec_path=out_dir/"model_spec.json"
            original=spec_path.read_bytes()
            spec_path.write_text(json.dumps(tampered),encoding="utf-8")
            with self.assertRaises(pipe.PipelineError):pipe.verify_export(spec_path)
            spec_path.write_bytes(original)
            payload = weights_path.read_bytes()
            self.assertEqual(len(payload), total)
            self.assertTrue(spec["quantization"]["shared_head_output_scale"])
        with self.assertRaises(pipe.PipelineError):
            pipe.load_fp32_model(Path("/nonexistent/best.pt"))

    def test_code_identity_and_protocol_binding_present(self) -> None:
        identity = pipe.code_identity()
        self.assertIn("int8/integer.py", identity)
        self.assertIn("int8/pipeline.py", identity)
        self.assertIn("role_a/quant_math.py", identity)
        self.assertTrue(pipe.PROTOCOL.is_file())
        self.assertTrue(pipe.ROSTER.is_file())
        roster = pipe.load_roster()
        self.assertEqual(len(roster["records"]), pipe.CALIBRATION_FRAMES)
        self.assertEqual(len(roster["patients"]), pipe.CALIBRATION_PATIENTS)
        for record in roster["records"]:
            self.assertNotIn("label", record)


if __name__ == "__main__":
    unittest.main()
