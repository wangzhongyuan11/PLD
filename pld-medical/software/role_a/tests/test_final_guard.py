"""断点门禁（硬拒绝）的针对性测试：源码/配置/输入任一变化都必须拒绝 resume。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from software.role_a import final_guard as guard


class GuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="final_guard_")
        self.tmp = Path(self._tmp.name)
        self.driver = self.tmp / "driver.py"
        self.dependency = self.tmp / "dependency.py"
        self.driver.write_text("print('v1')\n", encoding="utf-8")
        self.dependency.write_text("VALUE = 1\n", encoding="utf-8")
        self.code_paths = {"driver": self.driver, "dependency": self.dependency}
        self.inputs = {"split_train": "a" * 64, "split_dev": "b" * 64,
                       "init_checkpoint": "c" * 64, "calibration_ids": "d" * 64}
        self.config_hash = "e" * 64

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _record(self):
        return guard.build_guard(code_paths=self.code_paths, config_hash=self.config_hash,
                                 inputs=self.inputs)

    def test_matching_state_passes(self) -> None:
        guard.verify_guard(self._record(), code_paths=self.code_paths,
                           config_hash=self.config_hash, inputs=self.inputs)

    def test_source_change_is_rejected(self) -> None:
        record = self._record()
        self.dependency.write_text("VALUE = 2\n", encoding="utf-8")
        with self.assertRaises(guard.GuardMismatch) as context:
            guard.verify_guard(record, code_paths=self.code_paths,
                               config_hash=self.config_hash, inputs=self.inputs)
        self.assertIn("源码哈希变化", str(context.exception))
        self.assertIn("dependency", str(context.exception))

    def test_config_change_is_rejected(self) -> None:
        record = self._record()
        with self.assertRaises(guard.GuardMismatch) as context:
            guard.verify_guard(record, code_paths=self.code_paths,
                               config_hash="f" * 64, inputs=self.inputs)
        self.assertIn("配置哈希变化", str(context.exception))

    def test_input_change_is_rejected(self) -> None:
        record = self._record()
        changed = dict(self.inputs)
        changed["init_checkpoint"] = "0" * 64
        with self.assertRaises(guard.GuardMismatch) as context:
            guard.verify_guard(record, code_paths=self.code_paths,
                               config_hash=self.config_hash, inputs=changed)
        self.assertIn("输入哈希变化", str(context.exception))
        self.assertIn("init_checkpoint", str(context.exception))

    def test_missing_or_extra_files_are_rejected(self) -> None:
        record = self._record()
        extra = dict(self.code_paths)
        extra["third"] = self.driver
        with self.assertRaises(guard.GuardMismatch) as context:
            guard.verify_guard(record, code_paths=extra, config_hash=self.config_hash,
                               inputs=self.inputs)
        self.assertIn("源码清单不一致", str(context.exception))
        with self.assertRaises(guard.GuardMismatch):
            guard.verify_guard(record, code_paths={"driver": self.driver},
                               config_hash=self.config_hash, inputs=self.inputs)

    def test_missing_file_raises_instead_of_skipping(self) -> None:
        with self.assertRaises(FileNotFoundError):
            guard.hash_files({"missing": self.tmp / "nope.py"})

    def test_bad_schema_is_rejected(self) -> None:
        record = self._record()
        record["schema"] = "something-else"
        with self.assertRaises(guard.GuardMismatch):
            guard.verify_guard(record, code_paths=self.code_paths,
                               config_hash=self.config_hash, inputs=self.inputs)

    def test_empty_inputs_rejected_at_build(self) -> None:
        with self.assertRaises(ValueError):
            guard.build_guard(code_paths=self.code_paths, config_hash=self.config_hash, inputs={})

    def test_real_project_files_hash_deterministically(self) -> None:
        paths = {"qat": ROOT / "software" / "role_a" / "qat.py",
                 "quant_math": ROOT / "software" / "role_a" / "quant_math.py"}
        first = guard.hash_files(paths)
        second = guard.hash_files(paths)
        self.assertEqual(first, second)
        self.assertTrue(all(len(value) == 64 for value in first.values()))


if __name__ == "__main__":
    unittest.main()
