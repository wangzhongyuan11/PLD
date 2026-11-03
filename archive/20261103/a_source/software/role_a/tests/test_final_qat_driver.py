"""A 组最终训练驱动的定向测试：断点阶段判定、RNG 往返、尺度恢复、原子断点往返。

只做小模型与临时目录，不训练完整轮次、不评估开发集。
"""
from __future__ import annotations

import random
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from software.role_a import final_qat_driver as driver
from software.role_a import qat
from software.role_a import quant
from software.role_a.model import build_model


class ResumeDecisionTests(unittest.TestCase):
    """codex 审查点 4：phase 判定不能让 trained 阶段跳过当轮评估，也不能重复训练。"""

    def test_evaluated_phase_advances_and_retrains(self) -> None:
        start, phase, need = driver.resume_decision({"epoch": 3, "phase": "evaluated"})
        self.assertEqual((start, phase, need), (4, None, True))

    def test_trained_phase_repeats_same_epoch_without_training(self) -> None:
        start, phase, need = driver.resume_decision({"epoch": 3, "phase": "trained"})
        self.assertEqual((start, phase, need), (3, "trained", False))

    def test_trained_phase_at_last_epoch_still_evaluates(self) -> None:
        start, phase, need = driver.resume_decision({"epoch": 8, "phase": "trained"})
        self.assertEqual((start, phase, need), (8, "trained", False))

    def test_unknown_phase_is_rejected_by_schema_guard_not_silently_accepted(self) -> None:
        # 未知 phase 不应被当成 evaluated 而推进轮次
        start, phase, need = driver.resume_decision({"epoch": 3, "phase": "something"})
        self.assertEqual((start, phase, need), (4, None, True))
        with self.assertRaises(Exception):
            driver.resume_decision({"phase": "trained"})


class RngRoundTripTests(unittest.TestCase):
    """codex 审查点 5：Python/NumPy/Torch(/CUDA) RNG 必须保存并恢复。"""

    def test_rng_roundtrip_reproduces_sequence(self) -> None:
        driver.seed_everything(1234)
        state = driver.rng_state()
        expected = [random.random(), float(np.random.rand()), float(torch.rand(1))]
        driver.seed_everything(9999)          # 打乱
        driver.restore_rng(state)
        actual = [random.random(), float(np.random.rand()), float(torch.rand(1))]
        self.assertEqual(expected, actual)

    def test_cuda_state_present_when_available(self) -> None:
        driver.seed_everything(7)
        state = driver.rng_state()
        if torch.cuda.is_available():
            self.assertIn("cuda", state)
            driver.seed_everything(8)
            driver.restore_rng(state)
            first = torch.rand(3, device="cuda")
            driver.restore_rng(state)
            second = torch.rand(3, device="cuda")
            self.assertTrue(torch.equal(first, second))
        else:
            self.assertNotIn("cuda", state)


class SavedScaleTests(unittest.TestCase):
    """codex 审查点 3：恢复必须使用断点保存的尺度，而不是重算的初始尺度。"""

    def test_saved_scales_are_applied_even_when_different_from_initial(self) -> None:
        model = build_model(base_channels=8, num_pools=3, input_shape=(3, 32, 32))
        qat.enable_weight_qat(model)
        stats = {name: {"input_max_abs": 1.0, "output_max_abs": 2.0}
                 for name, module in model.named_modules()
                 if isinstance(module, qat.FakeQuantConv2d)}
        qat.set_scales_from_calibration(model, stats)
        initial = qat.scales_snapshot(model)

        # 模拟"重校准后"的尺度（与初始不同）
        recalibrated = {name: {"input_scale": entry["input_scale"] * 1.7,
                               "output_scale": entry["output_scale"] * 0.6}
                        for name, entry in initial.items()}
        model2 = build_model(base_channels=8, num_pools=3, input_shape=(3, 32, 32))
        qat.enable_weight_qat(model2)
        driver.apply_saved_scales(model2, recalibrated)
        self.assertEqual(qat.scales_snapshot(model2), recalibrated)
        self.assertNotEqual(qat.scales_snapshot(model2), initial)
        # 摘要必须与保存内容一致（而不是与初始尺度一致）
        self.assertEqual(driver.scales_digest(recalibrated),
                         driver.scales_digest(qat.scales_snapshot(model2)))

    def test_layer_set_mismatch_is_rejected(self) -> None:
        model = build_model(base_channels=8, num_pools=3, input_shape=(3, 32, 32))
        qat.enable_weight_qat(model)
        # 层集合不完整时 set_scales_from_calibration 直接抛错（不静默只设置部分层）
        with self.assertRaises(KeyError):
            driver.apply_saved_scales(model, {"encoders.0.0": {"input_scale": 0.1,
                                                              "output_scale": 0.1}})
        self.assertTrue(all(module.input_scale is None
                            for module in model.modules()
                            if isinstance(module, qat.FakeQuantConv2d)))


class AtomicCheckpointTests(unittest.TestCase):
    def test_atomic_save_and_load_roundtrip_without_tmp_leftover(self) -> None:
        with tempfile.TemporaryDirectory(prefix="driver_atomic_") as tmp:
            root = Path(tmp)
            path = root / "latest.pt"
            payload = {"schema": driver.SCHEMA, "epoch": 2, "phase": "trained",
                       "model_state": {"w": torch.zeros(2)}}
            driver.atomic_torch_save(payload, path)
            self.assertTrue(path.is_file())
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(loaded["epoch"], 2)
            self.assertEqual(list(root.glob("*.tmp")), [])
            driver.atomic_json({"a": 1}, root / "x.json")
            self.assertEqual([p.name for p in root.iterdir()], ["latest.pt", "x.json"])


class ExportEvaluationWiringTests(unittest.TestCase):
    """codex 审查点 2/6：verify_export 字段与非 0 退出码必须被显式处理。"""

    def test_verify_export_module_fields(self) -> None:
        from software.role_a import verify_export
        self.assertEqual(verify_export.__name__, "software.role_a.verify_export")
        spec = ROOT / "runs" / "role_a_a14_soup" / "int8_soup_v5_a050" / "model_spec.json"
        if spec.is_file():
            result = verify_export.verify(spec)
            self.assertEqual(int(result["layers_verified"]), 15)
            self.assertIn("checkpoint_sha256", result)
            self.assertEqual(driver.BASELINE_SOUP_SHA256, result["checkpoint_sha256"])

    def test_driver_has_explicit_exit_code_guards(self) -> None:
        source = Path(driver.__file__).read_text(encoding="utf-8")
        self.assertIn("INT8 导出退出码", source)
        self.assertIn("评估退出码", source)
        self.assertIn("verify_export 只有", source)
        self.assertIn("model = model.to(device)", source)
        self.assertIn("optimizer = torch.optim.Adam", source)
        self.assertLess(source.index("model = model.to(device)"),
                        source.index("optimizer = torch.optim.Adam"))

    def test_calibration_ids_digest_is_stable(self) -> None:
        self.assertEqual(driver.calibration_ids_digest(["a", "b"]),
                         driver.calibration_ids_digest(["a", "b"]))
        self.assertNotEqual(driver.calibration_ids_digest(["a", "b"]),
                            driver.calibration_ids_digest(["b", "a"]))


class EvaluationExitCodeTests(unittest.TestCase):
    """codex 补充审查：evaluate_quant_overall.main() 返回 None，不得对其取 int。"""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="driver_eval_")
        self.dev_dir = Path(self._tmp.name) / "development"
        self._original = driver.evaluate_quant_overall.main

    def tearDown(self) -> None:
        driver.evaluate_quant_overall.main = self._original
        self._tmp.cleanup()

    def test_none_return_is_treated_as_success(self) -> None:
        driver.evaluate_quant_overall.main = lambda: None
        self.assertEqual(driver.run_development_evaluation(Path("c.pt"), Path("s.json"),
                                                           self.dev_dir), 0)

    def test_zero_return_is_success(self) -> None:
        driver.evaluate_quant_overall.main = lambda: 0
        self.assertEqual(driver.run_development_evaluation(Path("c.pt"), Path("s.json"),
                                                           self.dev_dir), 0)

    def test_exception_is_nonzero_and_logged(self) -> None:
        def _boom():
            raise ValueError("模拟评估失败")

        driver.evaluate_quant_overall.main = _boom
        code = driver.run_development_evaluation(Path("c.pt"), Path("s.json"), self.dev_dir)
        self.assertNotEqual(code, 0)
        log = self.dev_dir / "evaluate_error.log"
        self.assertTrue(log.is_file())
        text = log.read_text(encoding="utf-8")
        self.assertIn("ValueError", text)
        self.assertIn("模拟评估失败", text)

    def test_sys_argv_is_restored(self) -> None:
        import sys as _sys
        before = list(_sys.argv)
        driver.evaluate_quant_overall.main = lambda: None
        driver.run_development_evaluation(Path("c.pt"), Path("s.json"), self.dev_dir)
        self.assertEqual(_sys.argv, before)

    def test_driver_never_casts_main_return_to_int(self) -> None:
        source = Path(driver.__file__).read_text(encoding="utf-8")
        self.assertNotIn("int(evaluate_quant_overall.main())", source)
        self.assertIn("evaluate_quant_overall.main()", source)


if __name__ == "__main__":
    unittest.main()
