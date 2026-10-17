"""``train_overall_v2`` 的关键行为测试（合成小样本；真实数据缺失时明确跳过）。

覆盖 A09 阶段一要求的验收点：

1. 小型合成样本上，90 度整数旋转对 image/mask/valid 逐像素同步，前景不落在 invalid 区；
2. 余弦学习率的边界与单调性，以及非法配置被拒绝；
3. 训练后、评估前中断再 ``--resume``：只补评估，不重复该轮梯度更新；
4. 中断 + ``--resume`` 与不中断训练在相同环境下逐位一致（含每轮 lr 序列）；
5. 配置/输入/代码哈希不符、缺失断点、已完成目录一律拒绝；
6. 旧模型形状与算子白名单不被 v2 改动（部署网络不变）。
"""

from __future__ import annotations

import json
import sys
import unittest
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # 依赖缺失时整组跳过（而不是静默通过）
    import numpy as np
    import torch

    from software.role_a import train_overall_v1 as v1
    from software.role_a import train_overall_v2 as v2
    from software.role_a.data import DatasetItem
    from software.role_a.model import build_model, describe_graph
    from software.role_a.tests import test_train_overall_v1 as harness

    HAVE_DEPENDENCIES = True
except ImportError as error:  # pragma: no cover
    IMPORT_ERROR = str(error)
    HAVE_DEPENDENCIES = False
else:
    IMPORT_ERROR = ""

SKIP_REASON = ("缺少 torch/numpy/Pillow 或本地模块，跳过（不算通过）: " + IMPORT_ERROR
               if not HAVE_DEPENDENCIES else "")

SIZE = harness.SIZE
EPS = 1e-12
REAL_PROJECT = Path(r"E:\嵌赛\pld-medical")
V4_BEST = REAL_PROJECT / "runs" / "role_a_overall_v4_dice2_base8" / "best.pt"
V4_BEST_SHA256 = "65bb913162812ae602aa6ce0f6cda183eb9f8e706af3a1b15a572e7159151b63"


def _v2_config(project, output_dir: Path, **overrides):
    values = {
        "output_dir": Path(output_dir),
        "init": "scratch",
        "base_channels": 8,
        "num_pools": 3,
        "size": SIZE,
        "epochs": 1,
        "batch_size": 3,
        "lr": 0.01,
        "lr_final": 0.001,
        "seed": 7,
        "device": "cpu",
        "num_workers": 0,
        "data_spec": project.spec(),
        "augment": True,
        "resume": False,
        "verify_images": False,
        "rotation": True,
        "lr_schedule": "cosine",
    }
    values.update(overrides)
    return v2.RunConfig(**values)  # type: ignore[arg-type]


def _toy_item(stem: str = "ISIC_0000001") -> DatasetItem:
    """手工构造的方形 letterbox 样本：image 由 mask 直接编码，可精确验对齐。"""
    size = 8
    mask = np.zeros((size, size), dtype=np.float32)
    mask[2:6, 1:3] = 1.0
    valid = np.zeros((size, size), dtype=np.float32)
    valid[1:7, 0:5] = 1.0
    image = np.repeat(mask[:, :, None], 3, axis=2) * 255.0
    return DatasetItem(
        image=torch.from_numpy(image.transpose(2, 0, 1)).float(),
        mask=torch.from_numpy(mask[None]).float(),
        valid=torch.from_numpy(valid[None]).float(),
        filename=stem + ".jpg",
        original_size=(size, size),
        letterbox=None,
        mask_was_binary=True,
    )


# ------------------------------------------------------- 旋转增强对齐
@unittest.skipUnless(HAVE_DEPENDENCIES, SKIP_REASON)
class RotationAlignmentTests(unittest.TestCase):
    def test_rotation_keeps_image_mask_valid_aligned(self) -> None:
        item = _toy_item()
        for quarters in range(v2.ROTATION_QUARTERS):
            rotated = v2.rotate_quarters(item, quarters)
            self.assertEqual(rotated.image.shape, item.image.shape)
            self.assertEqual(rotated.mask.shape, item.mask.shape)
            self.assertEqual(rotated.valid.shape, item.valid.shape)
            # mask/valid 与 image 用同一个 k 同步旋转
            self.assertTrue(torch.equal(rotated.mask,
                                        torch.rot90(item.mask, quarters, dims=(1, 2))))
            self.assertTrue(torch.equal(rotated.valid,
                                        torch.rot90(item.valid, quarters, dims=(1, 2))))
            # 前景绝不落在 invalid 区
            outside = ((rotated.mask > 0.5) & (rotated.valid < 0.5)).sum().item()
            self.assertEqual(outside, 0)
            # image 仍与 mask 逐像素一致（image 由 mask 编码）
            foreground = rotated.image[0] > 127.0
            positive = rotated.mask[0] > 0.5
            self.assertTrue(torch.equal(foreground, positive))
            # 前景像素数不因旋转改变
            self.assertEqual(int(positive.sum().item()),
                             int((item.mask[0] > 0.5).sum().item()))

    def test_rotation_is_mod_four_and_identity_for_zero(self) -> None:
        item = _toy_item()
        same = v2.rotate_quarters(item, 0)
        self.assertTrue(torch.equal(same.image, item.image))
        self.assertTrue(torch.equal(same.mask, item.mask))
        self.assertTrue(torch.equal(same.valid, item.valid))
        for quarters in range(v2.ROTATION_QUARTERS):
            left = v2.rotate_quarters(item, quarters)
            right = v2.rotate_quarters(item, quarters + v2.ROTATION_QUARTERS * 3)
            self.assertTrue(torch.equal(left.image, right.image))
            self.assertTrue(torch.equal(left.mask, right.mask))
            self.assertTrue(torch.equal(left.valid, right.valid))

    def test_dataset_rotation_is_deterministic_per_seed_epoch_index(self) -> None:
        with harness.overall_temp() as root:
            project = harness.SyntheticProject(root / "proj")
            items = list(v1.enumerate_isic_pairs(project.d2016 / "images",
                                                 project.d2016 / "masks", require_mask=True))
            plain = [v1.TrainDataset(items, size=SIZE, augment=False)[index]
                     for index in range(len(items))]
            augmented = [v1.TrainDataset(items, size=SIZE, seed=13, augment=True, epoch=2)[index]
                         for index in range(len(items))]
            dataset = v2.RotatingTrainDataset(items, size=SIZE, seed=13, augment=True,
                                              epoch=2, rotation=True)
            first = [dataset[index] for index in range(len(items))]
            second = [dataset[index] for index in range(len(items))]
            for left, right in zip(first, second):
                self.assertTrue(torch.equal(left.image, right.image))
                self.assertTrue(torch.equal(left.mask, right.mask))
                self.assertTrue(torch.equal(left.valid, right.valid))
            for index, base in enumerate(plain):
                self.assertEqual(first[index].image.shape, base.image.shape)
                self.assertEqual(first[index].valid.shape, base.valid.shape)
                # 旋转发生在 v1 增强之后，因此必须与同 seed/epoch 的 v1 增强结果比较
                expected = v2.rotate_quarters(augmented[index], dataset.quarters(index))
                self.assertTrue(torch.equal(first[index].image, expected.image))
                self.assertTrue(torch.equal(first[index].mask, expected.mask))
                self.assertTrue(torch.equal(first[index].valid, expected.valid))
                self.assertEqual(
                    int(((first[index].mask > 0.5) & (first[index].valid < 0.5)).sum().item()), 0)
            # 每轮每图的旋转参数不随 worker 调度改变：换 epoch 会得到不同的旋转组合
            seen = {tuple(dataset.quarters(index) for index in range(len(items)))}
            for epoch in range(3, 12):
                dataset.set_epoch(epoch)
                seen.add(tuple(dataset.quarters(index) for index in range(len(items))))
            self.assertGreater(len(seen), 1)
            # 关闭旋转时恒为 0，且与 v1 完全一致
            off = v2.RotatingTrainDataset(items, size=SIZE, seed=13, augment=True,
                                          epoch=2, rotation=False)
            for index, base in enumerate(augmented):
                self.assertEqual(off.quarters(index), 0)
                self.assertTrue(torch.equal(off[index].image, base.image))
                self.assertTrue(torch.equal(off[index].mask, base.mask))
                self.assertTrue(torch.equal(off[index].valid, base.valid))

    def test_rotation_never_lands_foreground_on_invalid_for_all_quarters(self) -> None:
        # 非方形有效区 + 非对称前景，覆盖 4 个 k
        size = 8
        mask = np.zeros((size, size), dtype=np.float32)
        mask[0:2, size - 2:size] = 1.0
        valid = np.zeros((size, size), dtype=np.float32)
        valid[0:5, 2:size] = 1.0
        image = np.repeat(mask[:, :, None], 3, axis=2) * 255.0
        item = DatasetItem(
            image=torch.from_numpy(image.transpose(2, 0, 1)).float(),
            mask=torch.from_numpy(mask[None]).float(),
            valid=torch.from_numpy(valid[None]).float(),
            filename="ISIC_0000002.jpg", original_size=(size, size),
            letterbox=None, mask_was_binary=True)
        base_outside = int(((item.mask > 0.5) & (item.valid < 0.5)).sum().item())
        self.assertEqual(base_outside, 0)
        for quarters in range(v2.ROTATION_QUARTERS):
            rotated = v2.rotate_quarters(item, quarters)
            self.assertEqual(
                int(((rotated.mask > 0.5) & (rotated.valid < 0.5)).sum().item()), 0)


# ------------------------------------------------------- 学习率边界
@unittest.skipUnless(HAVE_DEPENDENCIES, SKIP_REASON)
class LearningRateScheduleTests(unittest.TestCase):
    def test_endpoints_and_monotonicity(self) -> None:
        initial, final, epochs = 1e-4, 5e-6, 12
        values = [v2.cosine_learning_rate(epoch, epochs, initial, final)
                  for epoch in range(1, epochs + 1)]
        self.assertAlmostEqual(values[0], initial, places=15)
        self.assertAlmostEqual(values[-1], final, places=15)
        self.assertTrue(all(below < above for above, below in zip(values, values[1:])))
        for epoch, value in zip(range(1, epochs + 1), values):
            self.assertGreaterEqual(value, final - EPS)
            self.assertLessEqual(value, initial + EPS)
            mirror = v2.cosine_learning_rate(epochs - epoch + 1, epochs, initial, final)
            self.assertAlmostEqual(value + mirror, initial + final, places=12)

    def test_single_epoch_returns_initial(self) -> None:
        self.assertAlmostEqual(v2.cosine_learning_rate(1, 1, 1e-4, 5e-6), 1e-4, places=15)

    def test_out_of_range_and_invalid_rates_are_rejected(self) -> None:
        with self.assertRaises(v1.ConfigError):
            v2.cosine_learning_rate(0, 12, 1e-4, 5e-6)
        with self.assertRaises(v1.ConfigError):
            v2.cosine_learning_rate(13, 12, 1e-4, 5e-6)
        with self.assertRaises(v1.ConfigError):
            v2.cosine_learning_rate(1, 0, 1e-4, 5e-6)
        with self.assertRaises(v1.ConfigError):
            v2.cosine_learning_rate(1, 12, 1e-4, 0.0)
        with self.assertRaises(v1.ConfigError):
            v2.cosine_learning_rate(1, 12, 1e-4, 1e-3)

    def test_validate_config_rejects_increasing_cosine(self) -> None:
        with harness.overall_temp() as root:
            project = harness.SyntheticProject(root / "proj")
            good = _v2_config(project, root / "good", epochs=2)
            v2._validate_config(good)
            with self.assertRaises(v1.ConfigError):
                v2._validate_config(replace(good, lr_final=0.02))
            with self.assertRaises(v1.ConfigError):
                v2._validate_config(replace(good, lr_final=0.0))
            with self.assertRaises(v1.ConfigError):
                v2._validate_config(replace(good, lr_schedule="linear"))

    def test_lr_sequence_is_a_pure_function_of_epoch(self) -> None:
        epochs, initial, final = 12, 1e-4, 5e-6
        first = [v2.cosine_learning_rate(e, epochs, initial, final) for e in range(1, epochs + 1)]
        second = [v2.cosine_learning_rate(e, epochs, initial, final) for e in range(1, epochs + 1)]
        self.assertEqual(first, second)


# ------------------------------------------------------- 端到端训练/恢复
@unittest.skipUnless(HAVE_DEPENDENCIES, SKIP_REASON)
class TrainingResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._threads = torch.get_num_threads()

    def tearDown(self) -> None:
        torch.set_num_threads(self._threads)

    def test_one_epoch_writes_protocol_history_and_evidence(self) -> None:
        with harness.overall_temp() as root:
            project = harness.SyntheticProject(root / "proj")
            run = root / "run"
            harness._silent(v2.train, _v2_config(project, run, epochs=1))
            self.assertTrue((run / v2.COMPLETION_NAME).is_file())
            history = json.loads((run / v2.HISTORY_NAME).read_text(encoding="utf-8"))
            self.assertEqual([row["epoch"] for row in history], [1])
            self.assertIn("lr", history[0])
            self.assertAlmostEqual(history[0]["lr"], 0.01, places=15)
            protocol = json.loads((run / v2.PROTOCOL_NAME).read_text(encoding="utf-8"))
            self.assertEqual(protocol["schema"], v2.PROTOCOL_SCHEMA)
            self.assertEqual(protocol["optimizer"]["rotation"], True)
            self.assertEqual(protocol["thresholds"], list(v1.THRESHOLDS))
            checkpoint = v1.load_checkpoint(run / v2.CHECKPOINT_NAME)
            self.assertEqual(checkpoint["schema"], v2.CHECKPOINT_SCHEMA)
            self.assertEqual((checkpoint["epoch"], checkpoint["phase"]), (1, "evaluated"))
            self.assertEqual(sorted(checkpoint["code_hashes"]), sorted(v2.CODE_HASH_FILES))
            self.assertEqual(checkpoint["schedule"]["name"], "cosine")

    def test_lr_sequence_matches_plan_across_epochs(self) -> None:
        with harness.overall_temp() as root:
            project = harness.SyntheticProject(root / "proj")
            run = root / "run"
            harness._silent(v2.train, _v2_config(project, run, epochs=3, lr=0.01, lr_final=0.001))
            history = json.loads((run / v2.HISTORY_NAME).read_text(encoding="utf-8"))
            expected = [v2.cosine_learning_rate(e, 3, 0.01, 0.001) for e in (1, 2, 3)]
            self.assertEqual([row["epoch"] for row in history], [1, 2, 3])
            for row, want in zip(history, expected):
                self.assertAlmostEqual(row["lr"], want, places=15)

    def test_resume_after_crash_before_evaluation_does_not_retrain(self) -> None:
        with harness.overall_temp() as root:
            project = harness.SyntheticProject(root / "proj")
            run = root / "run"
            config = _v2_config(project, run, epochs=2, lr=0.01, lr_final=0.001)
            original_eval = v1.evaluate_development
            original_epoch = v1.run_train_epoch
            train_calls = {"count": 0}
            eval_calls = {"count": 0}

            def counting_epoch(*args, **kwargs):
                train_calls["count"] += 1
                return original_epoch(*args, **kwargs)

            def failing_eval(*args, **kwargs):
                eval_calls["count"] += 1
                if eval_calls["count"] == 2:
                    raise RuntimeError("simulated crash before the epoch-2 evaluation")
                return original_eval(*args, **kwargs)

            v1.run_train_epoch = counting_epoch
            v1.evaluate_development = failing_eval
            try:
                with self.assertRaises(RuntimeError):
                    harness._silent(v2.train, config)
                checkpoint = v1.load_checkpoint(run / v2.CHECKPOINT_NAME)
                self.assertEqual((checkpoint["epoch"], checkpoint["phase"]), (2, "trained"))
                self.assertEqual(train_calls["count"], 2)
                self.assertAlmostEqual(float(checkpoint["lr"]),
                                       v2.cosine_learning_rate(2, 2, 0.01, 0.001), places=15)
                v1.evaluate_development = original_eval
                harness._silent(v2.train, replace(config, resume=True))
            finally:
                v1.run_train_epoch = original_epoch
                v1.evaluate_development = original_eval

            # 恢复只补做第 2 轮评估，绝不重跑梯度
            self.assertEqual(train_calls["count"], 2)
            history = json.loads((run / v2.HISTORY_NAME).read_text(encoding="utf-8"))
            self.assertEqual([row["epoch"] for row in history], [1, 2])
            self.assertTrue((run / v2.COMPLETION_NAME).is_file())
            resume_log = json.loads((run / v2.RESUME_LOG_NAME).read_text(encoding="utf-8"))
            self.assertEqual(resume_log[-1]["pending_eval_epoch"], 2)
            self.assertEqual(resume_log[-1]["next_epoch"], 3)

    def test_resume_after_mid_epoch_crash_matches_uninterrupted_run(self) -> None:
        with harness.overall_temp() as root:
            project = harness.SyntheticProject(root / "proj")
            run_a = root / "run_a"
            run_b = root / "run_b"
            harness._silent(v2.train, _v2_config(project, run_a, epochs=2, lr=0.01, lr_final=0.001))
            self.assertTrue((run_a / v2.COMPLETION_NAME).is_file())

            config_b = _v2_config(project, run_b, epochs=2, lr=0.01, lr_final=0.001)
            original = v1.run_train_epoch
            calls = {"count": 0}

            def interrupting(*args, **kwargs):
                calls["count"] += 1
                outcome = original(*args, **kwargs)
                if calls["count"] == 2:
                    raise RuntimeError("simulated crash after the epoch-2 training pass")
                return outcome

            v1.run_train_epoch = interrupting
            try:
                with self.assertRaises(RuntimeError):
                    harness._silent(v2.train, config_b)
            finally:
                v1.run_train_epoch = original

            self.assertFalse((run_b / v2.COMPLETION_NAME).is_file())
            crashed_best = torch.load(run_b / v2.BEST_NAME, map_location="cpu", weights_only=True)
            self.assertEqual(crashed_best["status"], "candidate")
            self.assertFalse(crashed_best["completed"])
            checkpoint = v1.load_checkpoint(run_b / v2.CHECKPOINT_NAME)
            self.assertEqual((checkpoint["epoch"], checkpoint["phase"]), (1, "evaluated"))

            harness._silent(v2.train, replace(config_b, resume=True))
            self.assertTrue((run_b / v2.COMPLETION_NAME).is_file())
            left = v1.load_checkpoint(run_a / v2.CHECKPOINT_NAME)["model_state"]
            right = v1.load_checkpoint(run_b / v2.CHECKPOINT_NAME)["model_state"]
            self.assertEqual(sorted(left), sorted(right))
            for name in left:
                self.assertTrue(torch.equal(left[name], right[name]), name)
            history_a = json.loads((run_a / v2.HISTORY_NAME).read_text(encoding="utf-8"))
            history_b = json.loads((run_b / v2.HISTORY_NAME).read_text(encoding="utf-8"))
            self.assertEqual(history_a, history_b)
            self.assertEqual([row["lr"] for row in history_a],
                             [v2.cosine_learning_rate(e, 2, 0.01, 0.001) for e in (1, 2)])

    def test_resume_rejects_changed_config_input_or_code(self) -> None:
        with harness.overall_temp() as root:
            project = harness.SyntheticProject(root / "proj")
            run = root / "run"
            config = _v2_config(project, run, epochs=2, lr=0.01, lr_final=0.001)
            original = v1.run_train_epoch
            calls = {"count": 0}

            def interrupting(*args, **kwargs):
                calls["count"] += 1
                outcome = original(*args, **kwargs)
                if calls["count"] == 2:
                    raise RuntimeError("simulated crash after epoch 2 training")
                return outcome

            v1.run_train_epoch = interrupting
            try:
                with self.assertRaises(RuntimeError):
                    harness._silent(v2.train, config)
            finally:
                v1.run_train_epoch = original

            with self.assertRaises(v1.ResumeError):
                v2.train(replace(config, lr=0.02, resume=True))
            with self.assertRaises(v1.ResumeError):
                v2.train(replace(config, rotation=False, resume=True))
            with self.assertRaises(v1.ResumeError):
                v2.train(replace(config, lr_final=0.002, resume=True))

            checkpoint = v1.load_checkpoint(run / v2.CHECKPOINT_NAME)
            input_hashes = v1.assemble(config.data_spec).hashes
            code_hashes = v2.code_hashes_snapshot()
            config_hash = v1.config_digest(config.payload())
            self.assertEqual(
                v1.validate_resume(checkpoint, config_hash=config_hash,
                                   input_hashes=input_hashes, code_hashes=code_hashes,
                                   epochs=2), (1, "evaluated"))
            with self.assertRaises(v1.ResumeError):
                v1.validate_resume(checkpoint, config_hash="0" * 64,
                                   input_hashes=input_hashes, code_hashes=code_hashes,
                                   epochs=2)
            with self.assertRaises(v1.ResumeError):
                v1.validate_resume(
                    checkpoint, config_hash=config_hash,
                    input_hashes={**dict(input_hashes), "exposed_split": "0" * 64},
                    code_hashes=code_hashes, epochs=2)
            with self.assertRaises(v1.ResumeError):
                v1.validate_resume(
                    checkpoint, config_hash=config_hash, input_hashes=input_hashes,
                    code_hashes={**dict(code_hashes), "train_overall_v2.py": "0" * 64}, epochs=2)
            with self.assertRaises(v1.ResumeError):
                v1.validate_resume(checkpoint, config_hash=config_hash,
                                   input_hashes=input_hashes, code_hashes=code_hashes, epochs=0)

    def test_config_hash_reflects_rotation_and_schedule(self) -> None:
        with harness.overall_temp() as root:
            project = harness.SyntheticProject(root / "proj")
            base = _v2_config(project, root / "run", epochs=2)
            base_hash = v1.config_digest(base.payload())
            for changed in (replace(base, rotation=False),
                            replace(base, lr_final=0.002),
                            replace(base, lr_schedule="cosine2")):
                self.assertNotEqual(base_hash, v1.config_digest(changed.payload()))
            self.assertEqual(base_hash, v1.config_digest(_v2_config(
                project, root / "run", epochs=2).payload()))
            # v2 的配置指纹必须与 v1 区分开（schema/trial/增强/lr 计划）
            v1_config = harness._config(project, root / "run", epochs=2)
            self.assertNotEqual(base_hash, v1.config_digest(v1_config.payload()))
            self.assertEqual(base.payload()["schema"], v2.CONFIG_SCHEMA)
            self.assertEqual(base.payload()["trial"], v2.TRIAL_NAME)

    def test_completed_and_partial_directories_are_refused(self) -> None:
        with harness.overall_temp() as root:
            project = harness.SyntheticProject(root / "proj")
            run = root / "run"
            config = _v2_config(project, run, epochs=1)
            harness._silent(v2.train, config)
            before = (run / v2.BEST_NAME).read_bytes()
            with self.assertRaises(v1.TrialStateError):
                v2.train(replace(config, resume=False))
            result = harness._silent(v2.train, replace(config, resume=True))
            self.assertEqual(result["status"], "already-complete")
            self.assertEqual(before, (run / v2.BEST_NAME).read_bytes())

            partial = root / "partial"
            original_eval = v1.evaluate_development

            def failing_eval(*args, **kwargs):
                raise RuntimeError("simulated crash before evaluation")

            v1.evaluate_development = failing_eval
            try:
                with self.assertRaises(RuntimeError):
                    harness._silent(v2.train, _v2_config(project, partial, epochs=1))
            finally:
                v1.evaluate_development = original_eval
            self.assertEqual(v1.plan_mode(partial, False), "partial")
            self.assertEqual(v1.plan_mode(partial, True), "resume")
            with self.assertRaises(v1.TrialStateError):
                v2.train(_v2_config(project, partial, epochs=1))

            missing = root / "missing"
            with self.assertRaises(v1.ResumeError):
                v2.train(_v2_config(project, missing, epochs=1, resume=True))

    def test_validation_split_is_never_used_for_gradients(self) -> None:
        with harness.overall_temp() as root:
            project = harness.SyntheticProject(root / "proj")
            run = root / "run"
            seen: list[str] = []
            original_epoch = v1.run_train_epoch

            def recording_epoch(model, loader, *args, **kwargs):
                for batch in loader:
                    seen.extend(list(batch["filename"]))
                return original_epoch(model, loader, *args, **kwargs)

            v1.run_train_epoch = recording_epoch
            try:
                harness._silent(v2.train, _v2_config(project, run, epochs=1))
            finally:
                v1.run_train_epoch = original_epoch
            split = json.loads((run / v2.SPLIT_NAME).read_text(encoding="utf-8"))
            development = {stem for ids in split["development_ids"].values() for stem in ids}
            self.assertTrue(seen)
            # batch["filename"] 是文件名；按 stem 比较
            stems = {Path(name).stem for name in seen}
            self.assertEqual(stems & development, set())
            self.assertTrue(stems <= set(split["train_ids"]))


# ------------------------------------------------- 恢复边界回归
@unittest.skipUnless(HAVE_DEPENDENCIES, SKIP_REASON)
class ResumeBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._threads = torch.get_num_threads()

    def tearDown(self) -> None:
        torch.set_num_threads(self._threads)

    def _crash_before_epoch_training_end(self, config, epoch: int) -> None:
        """跑完第 ``epoch`` 轮训练后立刻抛错，留下第 ``epoch-1`` 轮的完整断点。"""
        original = v1.run_train_epoch
        calls = {"count": 0}

        def interrupting(*args, **kwargs):
            calls["count"] += 1
            outcome = original(*args, **kwargs)
            if calls["count"] == epoch:
                raise RuntimeError("simulated crash after training pass")
            return outcome

        v1.run_train_epoch = interrupting
        try:
            with self.assertRaises(RuntimeError):
                harness._silent(v2.train, config)
        finally:
            v1.run_train_epoch = original

    def test_pending_evaluation_resume_keeps_checkpoint_lr_consistent(self) -> None:
        with harness.overall_temp() as root:
            project = harness.SyntheticProject(root / "proj")
            run = root / "run"
            config = _v2_config(project, run, epochs=3, lr=0.01, lr_final=0.001)

            original_eval = v1.evaluate_development
            eval_calls = {"count": 0}

            def failing_eval(*args, **kwargs):
                eval_calls["count"] += 1
                if eval_calls["count"] == 2:
                    raise RuntimeError("simulated crash before the epoch-2 evaluation")
                return original_eval(*args, **kwargs)

            v1.evaluate_development = failing_eval
            try:
                with self.assertRaises(RuntimeError):
                    harness._silent(v2.train, config)
            finally:
                v1.evaluate_development = original_eval
            crashed = v1.load_checkpoint(run / v2.CHECKPOINT_NAME)
            self.assertEqual((crashed["epoch"], crashed["phase"]), (2, "trained"))

            # 恢复：补做第 2 轮评估，然后在第 3 轮梯度更新之前再次中断
            original_epoch = v1.run_train_epoch
            started = {"count": 0}

            def interrupting(*args, **kwargs):
                started["count"] += 1
                raise RuntimeError("simulated crash at the start of epoch 3")

            v1.run_train_epoch = interrupting
            try:
                with self.assertRaises(RuntimeError):
                    harness._silent(v2.train, replace(config, resume=True))
            finally:
                v1.run_train_epoch = original_epoch
            self.assertEqual(started["count"], 1)

            # 新落盘的断点必须自洽：lr / schedule.applied_lr / optimizer param_groups[*].lr
            after = v1.load_checkpoint(run / v2.CHECKPOINT_NAME)
            self.assertEqual((after["epoch"], after["phase"]), (2, "evaluated"))
            want = v2.cosine_learning_rate(2, 3, 0.01, 0.001)
            self.assertAlmostEqual(float(after["lr"]), want, places=15)
            self.assertAlmostEqual(float(after["schedule"]["applied_lr"]), want, places=15)
            groups = after["optimizer_state"]["param_groups"]
            self.assertTrue(groups)
            for group in groups:
                self.assertAlmostEqual(float(group["lr"]), want, places=15)
            self.assertFalse((run / v2.COMPLETION_NAME).is_file())
            history = json.loads((run / v2.HISTORY_NAME).read_text(encoding="utf-8"))
            self.assertEqual([row["epoch"] for row in history], [1, 2])
            self.assertAlmostEqual(history[1]["lr"], want, places=15)

            # 再恢复一次跑完第 3 轮：lr 序列仍是纯函数，未因恢复而漂移
            harness._silent(v2.train, replace(config, resume=True))
            history = json.loads((run / v2.HISTORY_NAME).read_text(encoding="utf-8"))
            self.assertEqual([row["epoch"] for row in history], [1, 2, 3])
            self.assertEqual([row["lr"] for row in history],
                             [v2.cosine_learning_rate(e, 3, 0.01, 0.001) for e in (1, 2, 3)])
            final = v1.load_checkpoint(run / v2.CHECKPOINT_NAME)
            self.assertEqual((final["epoch"], final["phase"]), (3, "evaluated"))
            self.assertAlmostEqual(float(final["lr"]),
                                   v2.cosine_learning_rate(3, 3, 0.01, 0.001), places=15)

    def test_resume_preserves_source_checkpoint_metadata(self) -> None:
        with harness.overall_temp() as root:
            project = harness.SyntheticProject(root / "proj")
            seed_run = root / "seed"
            harness._silent(v2.train, _v2_config(project, seed_run, epochs=1))
            seed_best = seed_run / v2.BEST_NAME
            self.assertTrue(seed_best.is_file())

            run = root / "run"
            config = _v2_config(project, run, epochs=2, init="checkpoint",
                                init_checkpoint=seed_best, lr=0.01, lr_final=0.001)
            self._crash_before_epoch_training_end(config, 2)

            before = v1.load_checkpoint(run / v2.CHECKPOINT_NAME)["source_checkpoint"]
            self.assertIsInstance(before, dict)
            self.assertEqual(before["source_checkpoint_sha256"], v1.sha256_file(seed_best))
            self.assertEqual(int(before["source_base_channels"]), 8)
            self.assertEqual(int(before["source_num_pools"]), 3)

            # 强制恢复期间每轮都写 best.pt，确保覆盖 resume 分支写 best 的路径
            original_better = v1._better_than
            v1._better_than = lambda candidate, best: True
            try:
                harness._silent(v2.train, replace(config, resume=True))
            finally:
                v1._better_than = original_better

            after = v1.load_checkpoint(run / v2.CHECKPOINT_NAME)["source_checkpoint"]
            self.assertEqual(after, before)
            best = torch.load(run / v2.BEST_NAME, map_location="cpu", weights_only=True)
            self.assertEqual(best["source_checkpoint"], before)
            self.assertEqual(int(best["epoch"]), 2)
            protocol = json.loads((run / v2.PROTOCOL_NAME).read_text(encoding="utf-8"))
            self.assertEqual(protocol["source_checkpoint"], before)

    def test_resume_rejects_internally_inconsistent_checkpoint_lr(self) -> None:
        with harness.overall_temp() as root:
            project = harness.SyntheticProject(root / "proj")
            run = root / "run"
            config = _v2_config(project, run, epochs=2, lr=0.01, lr_final=0.001)
            self._crash_before_epoch_training_end(config, 2)

            path = run / v2.CHECKPOINT_NAME
            payload = torch.load(path, map_location="cpu", weights_only=True)
            for group in payload["optimizer_state"]["param_groups"]:
                group["lr"] = 0.123456
            torch.save(payload, path)
            with self.assertRaises(v1.ResumeError):
                harness._silent(v2.train, replace(config, resume=True))

            payload = torch.load(path, map_location="cpu", weights_only=True)
            for group in payload["optimizer_state"]["param_groups"]:
                group["lr"] = float(payload["lr"])
            del payload["lr"]
            torch.save(payload, path)
            with self.assertRaises(v1.ResumeError):
                harness._silent(v2.train, replace(config, resume=True))

            payload = torch.load(path, map_location="cpu", weights_only=True)
            payload["lr"] = 0.5
            torch.save(payload, path)
            with self.assertRaises(v1.ResumeError):
                harness._silent(v2.train, replace(config, resume=True))


# ------------------------------------------------- 旧模型形状/算子白名单
@unittest.skipUnless(HAVE_DEPENDENCIES, SKIP_REASON)
class ModelCompatibilityTests(unittest.TestCase):
    def test_v2_does_not_change_deploy_graph_or_operators(self) -> None:
        model = build_model(base_channels=8, num_pools=3, input_shape=(3, SIZE, SIZE))
        model.assert_shapes(batch_size=1, device=torch.device("cpu"))
        graph = describe_graph(model)
        self.assertTrue(bool(graph["allowed_only"]))
        self.assertEqual(list(graph["forbidden"]), [])
        # v2 只改数据增强与学习率，不新增任何层
        self.assertEqual(graph["layers"], describe_graph(
            build_model(base_channels=8, num_pools=3, input_shape=(3, SIZE, SIZE)))["layers"])
        self.assertEqual(v2.ROTATION_QUARTERS, 4)

    @unittest.skipUnless(V4_BEST.is_file(), "缺少 v4 best.pt，跳过真实权重形状检查")
    def test_real_v4_weights_load_strictly_into_unchanged_model(self) -> None:
        self.assertEqual(v1.sha256_file(V4_BEST), V4_BEST_SHA256)
        checkpoint = v1.load_checkpoint(V4_BEST)
        config = checkpoint["config"]
        model = build_model(base_channels=int(config["base_channels"]),
                            num_pools=int(config["num_pools"]),
                            input_shape=(3, int(config["size"]), int(config["size"])))
        model.load_state_dict(checkpoint["model_state"], strict=True)
        graph = describe_graph(model)
        self.assertTrue(bool(graph["allowed_only"]))
        self.assertEqual(list(graph["forbidden"]), [])
        expected = v1.estimate_model(int(config["base_channels"]), int(config["num_pools"]),
                                     int(config["size"]))
        self.assertEqual(int(model.num_parameters()), int(expected["parameters"]))


# --------------------------------------------------------- 真实数据 dry-run
def _real_project_available() -> bool:
    return (REAL_PROJECT / "data" / "raw" / "isic2016" / "provenance.json").is_file()


@unittest.skipUnless(HAVE_DEPENDENCIES, SKIP_REASON)
@unittest.skipUnless(_real_project_available(), "缺少真实项目数据，跳过真实装配检查")
class RealDataDryRunTests(unittest.TestCase):
    def test_real_assembly_counts_and_start_point(self) -> None:
        spec = v1.build_default_spec(REAL_PROJECT)
        assembly = v1.assemble(spec, verify_frozen=True, verify_images=False)
        counts = dict(assembly.counts)
        self.assertEqual(int(counts["train_total"]), v1.TRAIN_TOTAL)
        self.assertEqual(int(counts["development_total"]), v1.DEVELOPMENT_TOTAL)
        self.assertEqual(len(assembly.train_items), v1.TRAIN_TOTAL)
        self.assertEqual(sum(len(d.items) for d in assembly.dev_sets.values()),
                         v1.DEVELOPMENT_TOTAL)
        self.assertTrue(V4_BEST.is_file())
        self.assertEqual(v1.sha256_file(V4_BEST), V4_BEST_SHA256)
        config = v2.RunConfig(
            output_dir=REAL_PROJECT / "runs" / "role_a_overall_v5_rotcos8",
            init="checkpoint", init_checkpoint=V4_BEST, base_channels=8, num_pools=3,
            size=256, epochs=12, batch_size=4, lr=0.0001, lr_final=5.0e-6, seed=v1.DEFAULT_SEED,
            device="cuda", num_workers=2, data_spec=spec, augment=True, resume=False,
            verify_images=False, rotation=True, lr_schedule="cosine",
            bce_weight=1.0, dice_weight=2.0, grad_clip=5.0)
        v2._validate_config(config)
        payload = config.payload()
        self.assertEqual(payload["rotation"], True)
        self.assertEqual(payload["lr_final"], 5.0e-6)
        self.assertEqual(payload["epochs"], 12)
        self.assertEqual(payload["bce_weight"], 1.0)
        self.assertEqual(payload["dice_weight"], 2.0)
        self.assertEqual(payload["grad_clip"], 5.0)
        self.assertEqual(sorted(v2.code_hashes_snapshot()), sorted(v2.CODE_HASH_FILES))


if __name__ == "__main__":
    unittest.main()