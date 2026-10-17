"""A20 阶段 1 插值的针对性测试：合成张量为主，真实端点测试在权重存在时执行。

无网络、无训练、不读开发集、不导出 INT8。
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from software.role_a import a20_interp as a20


class BlendSyntheticTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = {"w": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
                     "b": torch.tensor([0.5, -0.5], dtype=torch.float32)}
        self.other = {"w": torch.tensor([[3.0, 4.0], [5.0, 6.0]], dtype=torch.float32),
                      "b": torch.tensor([1.5, 0.5], dtype=torch.float32)}

    def test_alpha_zero_is_bitwise_base(self) -> None:
        blended = a20.blend(self.base, self.other, 0.0)
        for key in self.base:
            self.assertTrue(torch.equal(blended[key], self.base[key]), key)

    def test_alpha_one_is_bitwise_other(self) -> None:
        blended = a20.blend(self.base, self.other, 1.0)
        for key in self.other:
            self.assertTrue(torch.equal(blended[key], self.other[key]), key)

    def test_alpha_half_is_float64_round_trip(self) -> None:
        blended = a20.blend(self.base, self.other, 0.5)
        expected = (0.5 * self.base["w"].to(torch.float64)
                    + 0.5 * self.other["w"].to(torch.float64)).to(torch.float32)
        self.assertTrue(torch.equal(blended["w"], expected))
        self.assertEqual(blended["w"].dtype, self.base["w"].dtype)

    def test_alpha_is_a19_share(self) -> None:
        """alpha 是 other(A19) 的占比：alpha=0.25 时 base 权重占 0.75。"""
        blended = a20.blend(self.base, self.other, 0.25)
        expected = (0.75 * self.base["w"].to(torch.float64)
                    + 0.25 * self.other["w"].to(torch.float64)).to(torch.float32)
        self.assertTrue(torch.equal(blended["w"], expected))
        self.assertAlmostEqual(float(blended["w"][0, 0]), 1.5, places=6)  # 0.75*1 + 0.25*3

    def test_float64_intermediate_avoids_float32_rounding_loss(self) -> None:
        """在极端尺度差下，float32 中间精度会吞掉小量，float64 中间精度不会。

        base=1.0、other=1e-9、alpha=0.75：float32 逐步计算得 0.25（小量被舍掉），
        float64 中间精度保留 7.5e-10 这一增量。本模块用 float64 中间精度，
        结果必须与 float64 期望在 1e-7 相对量级内一致，而不是退化成 float32 的 0.25。
        """
        base = {"w": torch.tensor([1.0], dtype=torch.float32)}
        other = {"w": torch.tensor([1e-9], dtype=torch.float32)}
        blended = a20.blend(base, other, 0.75)
        exact64 = 0.25 * 1.0 + 0.75 * float(other["w"].item())
        naive = (torch.tensor([0.25], dtype=torch.float32) * base["w"]
                 + torch.tensor([0.75], dtype=torch.float32) * other["w"])
        self.assertEqual(float(naive[0]), 0.25)                       # float32 中间精度吞掉小量
        self.assertAlmostEqual(float(blended["w"][0]), exact64, delta=0.25 * 1e-7)

    def test_blend_on_float64_input_is_exact(self) -> None:
        """float64 输入下结果必须逐位等于 float64 公式（证明中间精度是 float64）。"""
        base = {"w": torch.tensor([0.1, 1.0, 1e-9], dtype=torch.float64)}
        other = {"w": torch.tensor([0.3, 1.0, 3e-9], dtype=torch.float64)}
        blended = a20.blend(base, other, 0.75)
        expected = 0.25 * base["w"] + 0.75 * other["w"]
        self.assertTrue(torch.equal(blended["w"], expected))
        self.assertEqual(float(blended["w"][2]), 0.25 * 1e-9 + 0.75 * 3e-9)

    def test_blend_with_a19_as_only_source_matches_a19(self) -> None:
        """单源退化：other==base 时任意 alpha 都应逐位等于 base（合成张量）。"""
        for alpha in a20.ALPHAS:
            blended = a20.blend(self.base, self.base, alpha)
            for key in self.base:
                self.assertTrue(torch.equal(blended[key], self.base[key]), (alpha, key))

    def test_shape_mismatch_raises(self) -> None:
        other = {"w": torch.zeros(3, dtype=torch.float32), "b": self.other["b"]}
        with self.assertRaises(ValueError):
            a20.blend(self.base, other, 0.5)

    def test_key_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            a20.blend(self.base, {"w": self.other["w"]}, 0.5)

    def test_dtype_preserved_for_float16_base(self) -> None:
        base = {"w": torch.tensor([1.0], dtype=torch.float16)}
        other = {"w": torch.tensor([3.0], dtype=torch.float16)}
        blended = a20.blend(base, other, 0.5)
        self.assertEqual(blended["w"].dtype, torch.float16)

    def test_integer_dtype_would_truncate(self) -> None:
        """说明不做整型插值：int 输入会被 cast 回整型（本模块只用于浮点权重）。"""
        base = {"w": torch.tensor([1], dtype=torch.int64)}
        other = {"w": torch.tensor([2], dtype=torch.int64)}
        blended = a20.blend(base, other, 0.5)
        self.assertEqual(blended["w"].dtype, torch.int64)
        self.assertEqual(int(blended["w"][0]), 1)  # 1.5 -> 1，提醒不要对整型张量用它


    def test_endpoints_use_both_sources_not_base_twice(self) -> None:
        """端点校验必须拿真实双源：other 含 NaN 时，用 base 冒充两源会漏掉它。"""
        base = {"w": torch.ones(2, dtype=torch.float32)}
        other = {"w": torch.tensor([float("nan"), 1.0], dtype=torch.float32)}
        # 真实双源下 alpha=0 端点仍是 base 的位模式，alpha=1 端点则是 other（含 NaN）
        zero = a20.blend(base, other, 0.0)
        one = a20.blend(base, other, 1.0)
        self.assertTrue(torch.equal(zero["w"][1], base["w"][1]))
        self.assertFalse(torch.equal(one["w"], base["w"]))
        # 若 check_endpoints 错误地写成 blend(base, base, 0)，NaN 永远不会被发现；
        # 真实双源下 NaN 会经 0*NaN 传播，端点校验必须报错（拒绝该来源）。
        with self.assertRaises(SystemExit) as context:
            a20.check_endpoints(base, other)
        self.assertIn("[A20]", str(context.exception))
        self.assertIn("端点", str(context.exception))


class StructureFiniteTests(unittest.TestCase):
    """结构校验必须逐源检查有限性，不能只查 base。"""

    @staticmethod
    def _state(shape=(2,), value: float = 1.0) -> dict[str, torch.Tensor]:
        return {"w": torch.full(shape, value, dtype=torch.float32)}

    def test_detects_nan_in_a19_instead_of_only_checking_base(self) -> None:
        base = self._state()
        other = self._state()
        other["w"][0] = float("nan")
        with self.assertRaises(SystemExit) as context:
            a20.check_sources_finite({"soup": {"state": base}, "a19": {"state": other}})
        self.assertIn("非有限", str(context.exception))
        self.assertIn("a19", str(context.exception))

    def test_detects_nan_in_base(self) -> None:
        base = self._state()
        base["w"][1] = float("nan")
        with self.assertRaises(SystemExit) as context:
            a20.check_sources_finite({"soup": {"state": base}, "a19": {"state": self._state()}})
        self.assertIn("soup", str(context.exception))

    def test_detects_infinity_too(self) -> None:
        other = self._state()
        other["w"][0] = float("inf")
        with self.assertRaises(SystemExit):
            a20.check_sources_finite({"soup": {"state": self._state()}, "a19": {"state": other}})

    def test_reports_per_source_finite_counts_when_clean(self) -> None:
        counts = a20.check_sources_finite({"soup": {"state": self._state()},
                                           "a19": {"state": self._state()}})
        self.assertEqual(counts, {"soup": 1, "a19": 1})


class ConstantsTests(unittest.TestCase):
    def test_frozen_protocol_constants(self) -> None:
        self.assertEqual(a20.ALPHAS, (0.25, 0.50, 0.75))
        self.assertEqual(a20.SELECTION_THRESHOLDS, (0.45, 0.50, 0.53, 0.56, 0.60))
        self.assertEqual(a20.PRIMARY_METRIC, "development_638_mean_per_image_dice")
        self.assertEqual(a20.EXPECTED_STRUCTURE, (8, 3, 256))

    def test_candidate_labels_are_three_fixed_points(self) -> None:
        self.assertEqual(a20.CANDIDATE_NAMES,
                         ("interp_a19_a025", "interp_a19_a050", "interp_a19_a075"))

    def test_expected_source_hashes_are_frozen(self) -> None:
        self.assertEqual(a20.SOURCES["soup"][1],
                         "f488719fe6d4d09c9b86be84d06b19d34241d6c14060b87da7bab25d1367eb77")
        self.assertEqual(a20.SOURCES["a19"][1],
                         "57b78cfc6df14b6f2fb53aef42c617836fe6b22c394960a0bcaf95e5e1bf4b53")


class SourceHashTests(unittest.TestCase):
    def test_missing_source_raises_instead_of_silently_skipping(self) -> None:
        original = a20.SOURCES["soup"]
        try:
            a20.SOURCES["soup"] = (Path("does/not/exist.pt"), original[1])
            with self.assertRaises(FileNotFoundError):
                a20.sha256(a20.SOURCES["soup"][0])
        finally:
            a20.SOURCES["soup"] = original

    def test_wrong_expected_hash_raises_systemexit(self) -> None:
        original = a20.SOURCES["soup"]
        try:
            a20.SOURCES["soup"] = (original[0], "0" * 64)
            with self.assertRaises(SystemExit):
                a20.load_source("soup")
        finally:
            a20.SOURCES["soup"] = original


@unittest.skipUnless(all(path.is_file() for path, _ in a20.SOURCES.values()),
                     "真实源权重不存在，跳过端到端硬校验")
class RealEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.loaded = {name: a20.load_source(name) for name in a20.SOURCES}

    def test_source_hashes_match_frozen_values(self) -> None:
        for name, (_, expected) in a20.SOURCES.items():
            self.assertEqual(self.loaded[name]["sha256"], expected, name)

    def test_structure_is_identical_and_batchnorm_free(self) -> None:
        info = a20.check_structure(self.loaded)
        self.assertTrue(info["batchnorm_absent"])
        self.assertTrue(info["key_sets_identical"])
        self.assertTrue(info["shapes_identical"])
        self.assertTrue(info["strict_load_ok"])
        self.assertEqual(info["base_channels"], 8)

    def test_endpoints_bitwise_identical(self) -> None:
        info = a20.check_endpoints(self.loaded[a20.BASE]["state"], self.loaded[a20.OTHER]["state"])
        self.assertTrue(info["alpha_0_bitwise_equals_base"])
        self.assertTrue(info["alpha_1_bitwise_equals_other"])
        self.assertGreater(info["checked_tensors"], 0)

    def test_alpha_half_is_exact_midpoint_of_float64(self) -> None:
        state = a20.blend(self.loaded[a20.BASE]["state"], self.loaded[a20.OTHER]["state"], 0.5)
        for key, value in state.items():
            expected = (0.5 * self.loaded[a20.BASE]["state"][key].to(torch.float64)
                        + 0.5 * self.loaded[a20.OTHER]["state"][key].to(torch.float64)
                        ).to(self.loaded[a20.BASE]["state"][key].dtype)
            if not torch.equal(value, expected):
                self.fail(f"alpha=0.5 中点与 float64 期望不一致: {key}")
        # 抽样核对可读数值
        head = state['head.weight']
        base_head = self.loaded[a20.BASE]["state"]['head.weight']
        other_head = self.loaded[a20.OTHER]["state"]['head.weight']
        self.assertFalse(torch.equal(head, base_head))
        self.assertFalse(torch.equal(head, other_head))

    def test_candidates_are_distinct_and_between_endpoints(self) -> None:
        """三个候选写出后：互相不同、都不等于两个端点、可被严格加载。"""
        import tempfile

        from software.role_a.model import build_model

        with tempfile.TemporaryDirectory(prefix="a20_test_") as tmp:
            built = [a20.build_candidate(self.loaded, alpha, Path(tmp)) for alpha in a20.ALPHAS]
            self.assertEqual([b["label"] for b in built], list(a20.CANDIDATE_NAMES))
            states = []
            for entry in built:
                payload = torch.load(entry["checkpoint"], map_location="cpu", weights_only=True)
                self.assertEqual(payload["schema"], a20.CANDIDATE_SCHEMA)
                self.assertEqual(payload["interp"]["alpha_a19_share"], entry["alpha_a19_share"])
                self.assertEqual(payload["interp"]["base_sha256"], self.loaded[a20.BASE]["sha256"])
                self.assertEqual(payload["interp"]["other_sha256"], self.loaded[a20.OTHER]["sha256"])
                model = build_model(base_channels=8, num_pools=3, input_shape=(3, 256, 256))
                model.load_state_dict(payload["model_state"], strict=True)  # 不抛错即结构一致
                states.append(payload["model_state"])
            for index in range(len(states)):
                self.assertFalse(torch.equal(states[index]["head.weight"],
                                             self.loaded[a20.BASE]["state"]["head.weight"]))
                self.assertFalse(torch.equal(states[index]["head.weight"],
                                             self.loaded[a20.OTHER]["state"]["head.weight"]))
                for other_index in range(index + 1, len(states)):
                    self.assertFalse(torch.equal(states[index]["head.weight"],
                                                 states[other_index]["head.weight"]))
            self.assertEqual(len({entry["checkpoint_sha256"] for entry in built}), len(built))

    def test_candidates_pass_quant_load_fp32_compatibility_smoke(self) -> None:
        """下一阶段现成导出器 quant._load_fp32 必须能直接读入候选（只读，不校准）。"""
        import tempfile

        with tempfile.TemporaryDirectory(prefix="a20_compat_") as tmp:
            for alpha in a20.ALPHAS:
                entry = a20.build_candidate(self.loaded, alpha, Path(tmp))
                report = a20.check_candidate_compatibility(entry)
                self.assertTrue(report["readable_by_quant_load_fp32"])
                self.assertTrue(report["strict_load_ok"])
                self.assertEqual(report["config_triple"], [8, 3, 256])
                self.assertEqual(report["state_keys"], 30)
                self.assertFalse(report["calibration_or_forward_run"])

    def test_compatibility_smoke_does_not_write_anything(self) -> None:
        import tempfile

        from software.role_a import quant

        with tempfile.TemporaryDirectory(prefix="a20_compat_readonly_") as tmp:
            entry = a20.build_candidate(self.loaded, 0.5, Path(tmp))
            before = sorted(p.name for p in Path(tmp).rglob("*"))
            calls: list[object] = []
            original = quant.calibrate
            try:
                quant.calibrate = lambda *args, **kwargs: calls.append(args)  # 若被调用即失败
                a20.check_candidate_compatibility(entry)
            finally:
                quant.calibrate = original
            after = sorted(p.name for p in Path(tmp).rglob("*"))
            self.assertEqual(calls, [])
            self.assertEqual(before, after)


class CandidateFormulaTests(unittest.TestCase):
    """候选权重必须逐位等于 (1-alpha)*soup + alpha*a19 的 float64 公式结果。"""

    @classmethod
    def setUpClass(cls) -> None:
        import tempfile

        cls._tmp = tempfile.TemporaryDirectory(prefix="a20_formula_")
        cls.loaded = {name: a20.load_source(name) for name in a20.SOURCES}
        cls.tmp = Path(cls._tmp.name)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def test_alpha_0_25_matches_explicit_formula(self) -> None:
        entry = a20.build_candidate(self.loaded, 0.25, self.tmp)
        state = torch.load(entry["checkpoint"], map_location="cpu", weights_only=True)["model_state"]
        for key, value in self.loaded[a20.BASE]["state"].items():
            expected = (0.75 * value.to(torch.float64)
                        + 0.25 * self.loaded[a20.OTHER]["state"][key].to(torch.float64)
                        ).to(value.dtype)
            if not torch.equal(state[key], expected):
                self.fail(f"alpha=0.25 与显式公式不一致: {key}")

    def test_cli_verify_only_exits_zero(self) -> None:
        import subprocess

        good = subprocess.run([sys.executable, "-m", "software.role_a.a20_interp", "--verify-only"],
                              cwd=ROOT, capture_output=True, text=True, timeout=900)
        self.assertEqual(good.returncode, 0, good.stderr[-500:])
        self.assertIn("verify-only", good.stdout)

    def test_gate_failure_exits_nonzero(self) -> None:
        """把源指向一个哈希不符的真实权重，门禁必须非零退出（不是只打印布尔）。"""
        import subprocess

        script = (
            "import sys; sys.path.insert(0, r'{root}');\n"
            "from pathlib import Path\n"
            "from software.role_a import a20_interp as a20\n"
            "a20.SOURCES['soup'] = (Path(a20.SOURCES['a19'][0]), 'f'*64)\n"
            "raise SystemExit(a20.main(['--verify-only']))\n"
        ).format(root=str(ROOT))
        result = subprocess.run([sys.executable, "-c", script], cwd=ROOT,
                                capture_output=True, text=True, timeout=900)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA256", result.stderr + result.stdout)


if __name__ == "__main__":
    unittest.main()
