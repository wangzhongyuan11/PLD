"""INT8 算术与 A→B 二进制交付的合成样本验证。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant_math import conv2d_int32, requantize_int8, round_divide_even


class IntegerMathTests(unittest.TestCase):
    def test_nearest_even_for_positive_and_negative_ties(self) -> None:
        half = 1 << 23
        products = np.asarray([half, 3 * half, 5 * half, -half, -3 * half, -5 * half])
        self.assertEqual(round_divide_even(products).tolist(), [0, 2, 2, 0, -2, -2])

    def test_conv_padding_bias_and_requant(self) -> None:
        x = np.ones((1, 1, 3, 3), dtype=np.int8)
        w = np.ones((1, 1, 3, 3), dtype=np.int8)
        acc = conv2d_int32(x, w, np.asarray([1], dtype=np.int32), padding=1)
        self.assertEqual(acc[0, 0].tolist(), [[5, 7, 5], [7, 10, 7], [5, 7, 5]])
        out = requantize_int8(acc, np.asarray([1 << 24], dtype=np.int32))
        np.testing.assert_array_equal(out.astype(np.int32), acc)

    def test_torch_quant_conv_agrees_with_integer_reference(self) -> None:
        import torch
        from quant import QuantConv2d

        rng = np.random.default_rng(123)
        x = rng.integers(-8, 9, size=(1, 3, 5, 5), dtype=np.int8)
        w = rng.integers(-4, 5, size=(2, 3, 3, 3), dtype=np.int8)
        b = np.asarray([17, -23], dtype=np.int32)
        multiplier = np.asarray([1 << 23, 3 << 23], dtype=np.int32)
        expected = requantize_int8(conv2d_int32(x, w, b, padding=1), multiplier)
        module = QuantConv2d(w, b, multiplier, input_scale=1.0,
                             output_scale=1.0, padding=1)
        actual = module(torch.from_numpy(x.astype(np.float32))).detach().numpy().astype(np.int8)
        np.testing.assert_array_equal(actual, expected)


class QuantizedExportTests(unittest.TestCase):
    def test_exported_vectors_are_self_consistent(self) -> None:
        import torch
        from _support import TemporaryRoot, make_dataset
        import train
        import quant

        with TemporaryRoot() as root:
            make_dataset(root, count=4)
            split_dir = root / "split"
            self.assertEqual(train.main([
                "prepare-split", "--image-dir", str(root / "images"),
                "--mask-dir", str(root / "masks"), "--output-dir", str(split_dir),
                "--no-stratify", "--seed", "7", "--val-fraction", "0.25",
            ]), 0)
            run_dir = root / "run"
            self.assertEqual(train.main([
                "train", "--image-dir", str(root / "images"),
                "--mask-dir", str(root / "masks"), "--split", str(split_dir / "split.json"),
                "--output-dir", str(run_dir), "--epochs", "1", "--size", "32",
                "--base-channels", "4", "--num-pools", "2", "--device", "cpu",
            ]), 0)
            export_dir = root / "export"
            self.assertEqual(quant.main([
                "export", "--checkpoint", str(run_dir / "best.pt"),
                "--split", str(split_dir / "split.json"),
                "--image-dir", str(root / "images"), "--mask-dir", str(root / "masks"),
                "--output-dir", str(export_dir), "--calibration-images", "2",
                "--tile-size", "8",
            ]), 0)
            spec = json.loads((export_dir / "model_spec.json").read_text(encoding="utf-8"))
            self.assertGreater(len(spec["layers"]), 1)
            self.assertEqual(spec["calibration_images"], 2)
            self.assertEqual(spec["calibration_mode"], quant.LEGACY_CALIBRATION_MODE)
            self.assertNotIn("calibration_sources", spec)
            self.assertNotIn("training_split", spec)
            selected_train = json.loads((split_dir / "split.json").read_text(encoding="utf-8"))["splits"]["train"]
            self.assertEqual(spec["calibration_stems"], [selected_train[0], selected_train[-1]])
            self.assertEqual(spec["layers"][0]["input_shape"], [1, 3, 32, 32])
            self.assertEqual(spec["layers"][0]["output_shape"], [1, 4, 32, 32])
            packed_weights = (export_dir / spec["weights_int8_file"]).read_bytes()
            packed_biases = (export_dir / spec["bias_int32_file"]).read_bytes()
            for layer in spec["layers"]:
                directory = export_dir / layer["directory"]
                weight_start = layer["weight_offset_bytes"]
                bias_start = layer["bias_offset_bytes"]
                self.assertEqual(packed_weights[weight_start:weight_start + layer["weight_nbytes"]],
                                 (directory / "weight.bin").read_bytes())
                self.assertEqual(packed_biases[bias_start:bias_start + layer["bias_nbytes"]],
                                 (directory / "bias.bin").read_bytes())
                config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
                x = np.fromfile(directory / "input.bin", dtype=np.int8).reshape(config["input_shape"])
                w = np.fromfile(directory / "weight.bin", dtype=np.int8).reshape(layer["weight_shape"])
                b = np.fromfile(directory / "bias.bin", dtype="<i4")
                expected_acc = conv2d_int32(x, w, b, layer["padding"])
                saved_acc = np.fromfile(directory / "acc_golden.bin", dtype="<i4").reshape(config["output_shape"])
                np.testing.assert_array_equal(saved_acc, expected_acc)
                expected_output = requantize_int8(saved_acc, np.asarray(layer["multiplier_q24"], dtype=np.int32))
                saved_output = np.fromfile(directory / "output_golden.bin", dtype=np.int8).reshape(config["output_shape"])
                np.testing.assert_array_equal(saved_output, expected_output)
                full_input = np.fromfile(directory / config["full_input_file"], dtype=np.int8)
                full_output = np.fromfile(directory / config["full_output_file"], dtype=np.int8)
                self.assertEqual(full_input.size, np.prod(config["full_input_shape"]))
                self.assertEqual(full_output.size, np.prod(config["full_output_shape"]))
                if layer is spec["layers"][0]:
                    full_acc = conv2d_int32(full_input.reshape(config["full_input_shape"]),
                                            w, b, layer["padding"])
                    full_expected = requantize_int8(
                        full_acc, np.asarray(layer["multiplier_q24"], dtype=np.int32))
                    np.testing.assert_array_equal(full_output.reshape(config["full_output_shape"]),
                                                  full_expected)
            int8_model, _ = quant._load_quantized(export_dir / "model_spec.json")
            self.assertEqual(tuple(int8_model(torch.zeros(1, 3, 32, 32)).shape), (1, 1, 32, 32))
            self.assertEqual(quant.main([
                "compare", "--checkpoint", str(run_dir / "best.pt"),
                "--spec", str(export_dir / "model_spec.json"),
                "--split", str(split_dir / "split.json"), "--role", "val",
                "--image-dir", str(root / "images"), "--mask-dir", str(root / "masks"),
                "--output-json", str(root / "comparison.json"),
            ]), 0)
            comparison = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(comparison["images"], 1)
            self.assertIn("dice_delta_fp32_minus_int8", comparison)
            self.assertAlmostEqual(comparison["dice_abs_difference"],
                                   abs(comparison["dice_delta_fp32_minus_int8"]))
            first_weight = export_dir / spec["layers"][0]["directory"] / "weight.bin"
            damaged = bytearray(first_weight.read_bytes())
            damaged[0] ^= 1
            first_weight.write_bytes(damaged)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                quant._load_quantized(export_dir / "model_spec.json")


class OverallCalibrationTests(unittest.TestCase):
    """``--overall-train`` 的跨来源均匀取样（纯函数，不读磁盘、不下模型）。"""

    SOURCE_SIZES = (897, 946, 466)

    def test_allocation_is_even_and_deterministic(self) -> None:
        import quant

        self.assertEqual(quant.allocate_calibration_counts(96, self.SOURCE_SIZES),
                         [32, 32, 32])
        self.assertEqual(quant.allocate_calibration_counts(100, self.SOURCE_SIZES),
                         [34, 33, 33])
        self.assertEqual(quant.allocate_calibration_counts(1, self.SOURCE_SIZES), [1, 0, 0])
        for total in (2, 3, 4, 5, 32, 96):
            counts = quant.allocate_calibration_counts(total, self.SOURCE_SIZES)
            self.assertEqual(sum(counts), total)
            self.assertLessEqual(max(counts) - min(counts), 1)
            self.assertTrue(all(0 <= c <= cap for c, cap in zip(counts, self.SOURCE_SIZES)))
        # 全部取满时按来源容量精确分配，不重复也不遗漏。
        self.assertEqual(quant.allocate_calibration_counts(sum(self.SOURCE_SIZES),
                                                           self.SOURCE_SIZES),
                         list(self.SOURCE_SIZES))
        self.assertEqual(quant.allocate_calibration_counts(96, self.SOURCE_SIZES),
                         quant.allocate_calibration_counts(96, self.SOURCE_SIZES))

    def test_allocation_respects_capacity_and_rejects_overflow(self) -> None:
        import quant

        self.assertEqual(quant.allocate_calibration_counts(4, [1, 10]), [1, 3])
        self.assertEqual(quant.allocate_calibration_counts(2, [0, 3, 3]), [0, 1, 1])
        with self.assertRaisesRegex(ValueError, "超过全部训练图"):
            quant.allocate_calibration_counts(5, [1, 1])
        with self.assertRaises(ValueError):
            quant.allocate_calibration_counts(0, [1])

    def test_indices_are_unique_sorted_and_cover_the_range(self) -> None:
        import quant

        indices = quant.evenly_spaced_indices(897, 32)
        self.assertEqual(len(set(indices)), 32)
        self.assertEqual(indices, sorted(indices))
        self.assertEqual((indices[0], indices[-1]), (0, 896))
        self.assertEqual(quant.evenly_spaced_indices(5, 5), [0, 1, 2, 3, 4])
        self.assertEqual(quant.evenly_spaced_indices(5, 1), [0])
        self.assertEqual(quant.evenly_spaced_indices(0, 0), [])
        with self.assertRaises(ValueError):
            quant.evenly_spaced_indices(3, 4)


class OverallTrainExportTests(unittest.TestCase):
    """用合成三来源项目（14 训练 / 9 开发）跑通 ``--overall-train`` 导出。"""

    SIZE = 32

    def _synthetic_checkpoint(self, root: Path) -> Path:
        import torch
        from model import build_model

        model = build_model(base_channels=4, num_pools=2,
                            input_shape=(3, self.SIZE, self.SIZE))
        path = root / "synthetic_fp32.pt"
        torch.save({"model_state": model.state_dict(),
                    "config": {"size": self.SIZE, "base_channels": 4, "num_pools": 2}}, path)
        return path

    def test_overall_train_export_records_sources_and_train_only_ids(self) -> None:
        import torch
        from unittest import mock

        from _support import TemporaryRoot
        from test_train_overall_v1 import SyntheticProject, overall_temp
        import quant
        import train_overall_v1 as overall

        with overall_temp() as tmp:
            project = SyntheticProject(tmp / "proj")
            assembly = overall.assemble(project.spec())
            checkpoint = self._synthetic_checkpoint(tmp)
            export_dir = tmp / "export"
            with mock.patch.object(quant, "load_overall_assembly", return_value=assembly):
                self.assertEqual(quant.main([
                    "export", "--overall-train", "--checkpoint", str(checkpoint),
                    "--output-dir", str(export_dir), "--calibration-images", "6",
                    "--tile-size", "4",
                ]), 0)
            spec = json.loads((export_dir / "model_spec.json").read_text(encoding="utf-8"))
            self.assertEqual(spec["calibration_mode"], quant.OVERALL_CALIBRATION_MODE)
            self.assertEqual(spec["calibration_split"], quant.OVERALL_CALIBRATION_SPLIT)
            self.assertEqual(spec["calibration_images"], 6)
            sources = spec["calibration_sources"]
            self.assertEqual([row["name"] for row in sources],
                             [source.name for source in project.spec().train_sources])
            self.assertEqual([row["selected"] for row in sources], [2, 2, 2])
            self.assertEqual(sum(row["train_count"] for row in sources), 14)
            stems = spec["calibration_stems"]
            self.assertEqual(len(stems), 6)
            self.assertEqual(len(set(stems)), 6)
            self.assertEqual([stem for row in sources for stem in row["stems"]], stems)
            self.assertTrue(set(stems) <= set(assembly.train_ids))
            dev_ids = {stem for ids in assembly.dev_ids.values() for stem in ids}
            self.assertFalse(set(stems) & dev_ids)
            summary = spec["training_split"]
            self.assertEqual(summary["train_total"], 14)
            self.assertEqual(summary["development_total"], 9)
            self.assertEqual(summary["train_counts_by_source"]["isic2016_train"], 5)
            self.assertEqual(summary["train_ids_sha256"], assembly.split_digest["train_ids"])
            self.assertTrue(summary["calibration_ids_are_train_only"])
            int8_model, _ = quant._load_quantized(export_dir / "model_spec.json")
            self.assertEqual(tuple(int8_model(torch.zeros(1, 3, self.SIZE, self.SIZE)).shape),
                             (1, 1, self.SIZE, self.SIZE))
            # 同一装配下的取样必须完全可复现。
            with TemporaryRoot() as other:
                second = Path(other) / "export"
                with mock.patch.object(quant, "load_overall_assembly", return_value=assembly):
                    quant.main(["export", "--overall-train", "--checkpoint", str(checkpoint),
                                "--output-dir", str(second), "--calibration-images", "6",
                                "--tile-size", "4"])
                again = json.loads((second / "model_spec.json").read_text(encoding="utf-8"))
                self.assertEqual(again["calibration_stems"], stems)

    def test_overall_train_rejects_legacy_data_options(self) -> None:
        from _support import TemporaryRoot
        import quant

        with TemporaryRoot() as root:
            with self.assertRaisesRegex(ValueError, "--overall-train 只使用冻结划分"):
                quant.main([
                    "export", "--overall-train", "--checkpoint", str(root / "missing.pt"),
                    "--split", str(root / "split.json"), "--image-dir", str(root / "images"),
                    "--mask-dir", str(root / "masks"), "--output-dir", str(root / "out"),
                ])
            with self.assertRaisesRegex(ValueError, "或改用 --overall-train"):
                quant.main(["export", "--checkpoint", str(root / "missing.pt"),
                            "--output-dir", str(root / "out")])


if __name__ == "__main__":
    unittest.main()
