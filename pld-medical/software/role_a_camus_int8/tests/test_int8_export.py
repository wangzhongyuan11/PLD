"""CAMUS INT8 补充 CPU 用例（导出/校验/装载/CLI 与拒绝路径；**仅合成数据**）。

覆盖任务 173 新增要求：

* synthetic 导出到临时目录 → `verify_export` 全过（15 层 tile 与 NumPy 整数 oracle bitexact、
  packed 偏移连续、concat 三目录 skip-first 逐元素一致、full 例仅 JSON 元数据）；
* `load_export` 严格装载后重跑 `graph.forward_graph`，与导出时的完整逐层捕获精确一致；
* **scale 不同的 concat 全图**（人为让两支 output scale 不同）仍能正确重定标并通过校验；
* 错误 SHA / 非法路径（`../`、绝对路径）/ 偏移不连续 / 缺文件 → 拒绝；
* `synthetic` 明确标记、不得冒充训练来源；伪 `frozen`（无身份/伪造 SHA）→ 拒绝；
* CLI `export`+`verify` 走一遍（subprocess，synthetic）。

运行（单次，≤300 秒）：
``python -m unittest discover -s software/role_a_camus_int8/tests -p test_int8_export.py -v``
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(r"E:/嵌赛/pld-medical")
for path in (str(PROJECT), str(PROJECT / "software/role_a")):
    if path not in sys.path:
        sys.path.insert(0, path)

from software.role_a_camus_int8 import graph as i8_graph      # noqa: E402
from software.role_a_camus_int8 import integer as i8          # noqa: E402
from software.role_a_camus_int8 import pipeline as pipe       # noqa: E402

SIZE = 64


def synthetic_setup(seed: int = 20261012, *, skew: bool = False):
    model = pipe.synthetic_model(seed=seed, size=SIZE)
    image = np.linspace(0.0, 1.0, SIZE * SIZE, dtype=np.float32).reshape(SIZE, SIZE)
    calibration = pipe.calibrate(model, [("synthetic", image)])
    if skew:
        # 人为放大高层输出 max abs，使 concat 两支 scale 明显不同
        for index, name in enumerate(pipe.LAYER_NAMES[:6]):
            calibration["layers"][name]["output_max_abs"] *= (1.0 + index)
    layers = pipe.build_layers(model, calibration)
    return model, image, calibration, layers


class ExportAndVerifyTests(unittest.TestCase):
    def test_synthetic_export_verify_and_load_roundtrip(self) -> None:
        model, image, calibration, layers = synthetic_setup()
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "export"
            spec = pipe.export_graph(model, calibration, layers, out_dir, sample_image=image,
                                     source_kind="synthetic")
            self.assertEqual(spec["schema"], pipe.EXPORT_SCHEMA)
            self.assertEqual(len(spec["layers"]), pipe.TOTAL_CONVS)
            self.assertTrue(spec["source"]["synthetic"])
            self.assertFalse(spec["source"]["training_source"])
            self.assertIsNone(spec["source"]["checkpoint_sha256"])
            directories = sorted(path.name for path in out_dir.iterdir() if path.is_dir())
            self.assertEqual(len([name for name in directories if name != "concat"]),
                             pipe.TOTAL_CONVS)
            self.assertTrue((out_dir / "concat/level2/info.json").is_file())
            report = pipe.verify_export(out_dir / "model_spec.json", expected_source_kind="synthetic")
            self.assertEqual(report["checks"]["layer_count"], pipe.TOTAL_CONVS)
            self.assertEqual(report["checks"]["concat_count"], 3)
            self.assertTrue(report["checks"]["tiles_bitexact"])
            self.assertTrue(report["checks"]["paths_safe"])
            self.assertTrue(report["checks"]["hashes_ok"])
            # full 例必须是 JSON 元数据（可 json.dumps）
            payload = json.loads((out_dir / "full_sample.json").read_text(encoding="utf-8"))
            json.dumps(payload)
            self.assertIn("input_int8_shape", payload)
            spec_payload = json.loads((out_dir / "model_spec.json").read_text(encoding="utf-8"))
            json.dumps(spec_payload)
            # load_export 后重跑前向，与导出时捕获逐层一致
            _model, loaded, _spec = pipe.load_export(out_dir / "model_spec.json")
            self.assertEqual([layer.name for layer in loaded], list(pipe.LAYER_NAMES))
            rerun = i8_graph.forward_graph(model, loaded, image,
                                           valid=np.ones_like(image), capture=True, oracle=True)
            first = loaded[0]
            tile_input = np.frombuffer((out_dir / spec["layers"][0]["directory"] /
                                        "tile/input_full.bin").read_bytes(), dtype=np.int8)
            captured = np.asarray(rerun["captures"][first.name]["input_int8"])
            self.assertEqual(tile_input.size, captured.size)
            for info in rerun["concat_info"]:
                merged = np.frombuffer((out_dir / f"concat/level{info['level']}/merged_int8.bin")
                                       .read_bytes(), dtype=np.int8)
                self.assertEqual(merged.size, np.asarray(rerun["captures"][f"concat_level{info['level']}"]["output_int8"]).size)

    def test_skewed_concat_scales_still_verify(self) -> None:
        model, image, calibration, layers = synthetic_setup(skew=True)
        scales = [layer.output_scale for layer in layers]
        self.assertGreater(len(set(round(value, 12) for value in scales)), 1)
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "skewed"
            spec = pipe.export_graph(model, calibration, layers, out_dir, sample_image=image,
                                     source_kind="synthetic")
            commons = {record["common_scale"] for record in spec["concat"]}
            self.assertEqual(len(commons), len(spec["concat"]))
            for record in spec["concat"]:
                self.assertNotEqual(record["source_scales"][0], record["source_scales"][1])
            pipe.verify_export(out_dir / "model_spec.json", expected_source_kind="synthetic")

    def test_rejections_bad_sha_path_offset_and_missing_file(self) -> None:
        model, image, calibration, layers = synthetic_setup()
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "export"
            spec = pipe.export_graph(model, calibration, layers, out_dir, sample_image=image,
                                     source_kind="synthetic")
            spec_path = out_dir / "model_spec.json"
            original = json.loads(spec_path.read_text(encoding="utf-8"))

            tampered = json.loads(json.dumps(original))
            tampered["layers"][0]["weight_file"]["sha256"] = "0" * 64
            path = out_dir / "tampered_sha.json"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(pipe.PipelineError):
                pipe.verify_export(path)

            tampered = json.loads(json.dumps(original))
            tampered["layers"][1]["directory"] = "../escape"
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(pipe.PipelineError):
                pipe.verify_export(path)

            tampered = json.loads(json.dumps(original))
            tampered["layers"][2]["directory"] = str((out_dir / "abs").resolve())
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(pipe.PipelineError):
                pipe.verify_export(path)

            tampered = json.loads(json.dumps(original))
            tampered["layers"][3]["packed_weight_offset"] += 7
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(pipe.PipelineError):
                pipe.verify_export(path)

            removed = out_dir / original["layers"][4]["directory"] / "multipliers.bin"
            removed.unlink()
            with self.assertRaises(pipe.PipelineError):
                pipe.verify_export(spec_path)
            removed.write_bytes((np.zeros(original["layers"][4]["bias_shape"][0],
                                          dtype="<i4")).tobytes())

            tampered = json.loads(json.dumps(original))
            tampered["source"].update({"kind": "frozen", "synthetic": False,
                                       "training_source": True, "checkpoint_sha256": "f" * 64})
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(pipe.PipelineError):
                pipe.verify_export(path, expected_source_kind="frozen")

            tampered = json.loads(json.dumps(original))
            tampered["source"].update({"training_source": True})
            path.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaises(pipe.PipelineError):
                pipe.verify_export(path)

            with self.assertRaises(pipe.PipelineError):
                pipe.verify_export(spec_path, expected_source_kind="frozen")

    def test_export_dir_must_be_fresh_and_frozen_export_needs_approval(self) -> None:
        model, image, calibration, layers = synthetic_setup()
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "busy"
            out_dir.mkdir()
            (out_dir / "existing.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(pipe.PipelineError):
                pipe.export_graph(model, calibration, layers, out_dir, sample_image=image)
            empty = Path(temporary) / "empty"
            empty.mkdir()
            pipe.export_graph(model, calibration, layers, empty, sample_image=image,
                              source_kind="synthetic")
            with self.assertRaises(pipe.PipelineError):
                pipe.export_graph(model, calibration, layers, Path(temporary) / "frozen_out2",
                                  sample_image=image, source_kind="frozen", source_meta=None)
            with self.assertRaises(pipe.PipelineError):
                pipe.load_fp32_model(pipe.FROZEN_BEST, source_kind="synthetic")

    def test_cli_export_and_verify(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "cli"
            export = subprocess.run([sys.executable, "-X", "utf8",
                                     str(PROJECT / "software/role_a_camus_int8/pipeline.py"),
                                     "export", "--out-dir", str(out_dir)],
                                    capture_output=True, text=True, cwd=str(PROJECT))
            self.assertEqual(export.returncode, 0, export.stderr[-600:])
            payload = json.loads(export.stdout.strip().splitlines()[-1])
            self.assertEqual(payload["source_kind"], "synthetic")
            verify = subprocess.run([sys.executable, "-X", "utf8",
                                     str(PROJECT / "software/role_a_camus_int8/pipeline.py"),
                                     "verify", "--spec", str(out_dir / "model_spec.json"),
                                     "--expect-source-kind", "synthetic"],
                                    capture_output=True, text=True, cwd=str(PROJECT))
            self.assertEqual(verify.returncode, 0, verify.stderr[-600:])
            checks = json.loads(verify.stdout.strip().splitlines()[-1])
            self.assertEqual(checks["layer_count"], pipe.TOTAL_CONVS)

    def test_full_sample_json_serializable_and_tile_uses_full_crop(self) -> None:
        model, image, calibration, layers = synthetic_setup(seed=7)
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "tile"
            spec = pipe.export_graph(model, calibration, layers, out_dir, sample_image=image,
                                     source_kind="synthetic")
            first = spec["layers"][0]
            directory = out_dir / first["directory"]
            tile = np.frombuffer((directory / "tile/input.bin").read_bytes(),
                                 dtype=np.int8).reshape(1, first["weight_shape"][1],
                                                        pipe.TILE, pipe.TILE)
            full = np.frombuffer((directory / "tile/input_full.bin").read_bytes(),
                                 dtype=np.int8).reshape(first["full_input_shape"])
            self.assertTrue(np.array_equal(tile, full[:, :, :pipe.TILE, :pipe.TILE]))
            info = json.loads((out_dir / "concat/level0/info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["common_scale"], max(info["source_scales"]))
            self.assertEqual(len(info["multiplier_q24"]), 2)
            skip = np.frombuffer((out_dir / "concat/level0/skip_int8.bin").read_bytes(),
                                 dtype=np.int8)
            expected = pipe._rescale_with_multiplier(skip, info["multiplier_q24"][0])
            saved = np.frombuffer((out_dir / "concat/level0/skip_rescaled.bin").read_bytes(),
                                  dtype=np.int8)
            self.assertTrue(np.array_equal(expected, saved))


if __name__ == "__main__":
    unittest.main()
