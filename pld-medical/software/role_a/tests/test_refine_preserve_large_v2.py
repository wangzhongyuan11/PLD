"""Checkpoint atomicity and resume-gate tests for the v2 repair trial.

These tests never touch ISIC data and never train the real model: they exercise the
atomic save/load helpers and the ``--resume`` admission rules with tiny tensors, so they
stay fast and safe to run at any time.
"""
from __future__ import annotations

import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import WeightedRandomSampler  # noqa: E402

from software.role_a import refine_preserve_large_v2 as v2  # noqa: E402


class TrialTargetTests(unittest.TestCase):
    def test_trial_starts_from_first_round_candidate_in_a_new_directory(self) -> None:
        self.assertEqual(v2.OUT.name, "role_a_preserve_large_v2")
        self.assertEqual(v2.START.parent.name, "role_a_full2017")
        self.assertEqual(v2.START.name, "best.pt")
        self.assertEqual(v2.EPOCHS, 2)
        self.assertEqual(v2.BLENDS, (0.25, 0.5, 1.0))
        self.assertEqual(v2.THRESHOLDS, (0.5, 0.53, 0.56, 0.6))

    def test_disclosure_states_the_real_resume_limitation(self) -> None:
        text = v2.RESUME_SCOPE
        self.assertIn("只能从已完成轮次继续", text)
        self.assertIn("不保证逐位一致", text)
        self.assertIn("工作进程", text)

    def test_config_digest_is_stable_and_sensitive(self) -> None:
        self.assertEqual(v2.config_digest({"a": 1, "b": [1, 2]}),
                         v2.config_digest({"b": [1, 2], "a": 1}))
        self.assertNotEqual(v2.config_digest({"a": 1}), v2.config_digest({"a": 2}))


class AtomicWriteTests(unittest.TestCase):
    def test_atomic_torch_save_replaces_previous_and_leaves_no_temp(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / v2.CHECKPOINT_NAME
            v2.atomic_torch_save({"value": torch.tensor([1.0])}, path)
            v2.atomic_torch_save({"value": torch.tensor([2.0])}, path)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(float(loaded["value"][0]), 2.0)
            self.assertEqual([p.name for p in Path(temp).iterdir() if p.suffix == ".tmp"], [])

    def test_failed_save_keeps_the_last_valid_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / v2.CHECKPOINT_NAME
            v2.atomic_torch_save({"value": torch.tensor([1.0])}, path)

            def failing_serializer(_payload: object, _target: Path) -> None:
                raise OSError("simulated disk failure")

            with self.assertRaises(OSError):
                v2.atomic_torch_save({"value": torch.tensor([2.0])}, path,
                                     serializer=failing_serializer)
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(float(loaded["value"][0]), 1.0)
            self.assertEqual([p.name for p in Path(temp).iterdir() if p.suffix == ".tmp"], [])

    def test_atomic_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "history.json"
            v2.atomic_write_json(path, [{"epoch": 1, "accepted": False}])
            v2.atomic_write_json(path, [{"epoch": 2, "accepted": True}])
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),
                             [{"epoch": 2, "accepted": True}])
            self.assertEqual([p.name for p in Path(temp).iterdir() if p.suffix == ".tmp"], [])


class RandomStateTests(unittest.TestCase):
    def test_rng_snapshot_reproduces_python_numpy_and_torch_draws(self) -> None:
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        snapshot = v2.capture_rng_state()
        first = [random.random(), float(np.random.rand()), float(torch.rand(1))]
        v2.restore_rng_state(snapshot)
        second = [random.random(), float(np.random.rand()), float(torch.rand(1))]
        self.assertEqual(first, second)

    def test_sampler_state_reproduces_the_index_order(self) -> None:
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, 8, replacement=True,
                                        generator=torch.Generator().manual_seed(99))
        list(iter(sampler))
        payload = v2.sampler_state_payload(sampler)
        expected = list(iter(sampler))
        replay = WeightedRandomSampler(weights, 8, replacement=True,
                                       generator=torch.Generator().manual_seed(0))
        v2.restore_sampler_state(replay, payload)
        self.assertEqual(list(iter(replay)), expected)

    def test_sampler_state_rejects_a_different_spec(self) -> None:
        weights = torch.tensor([1.0, 2.0], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, 4, replacement=True,
                                        generator=torch.Generator().manual_seed(3))
        payload = v2.sampler_state_payload(sampler)
        other = WeightedRandomSampler(weights, 5, replacement=True,
                                      generator=torch.Generator().manual_seed(3))
        with self.assertRaises(v2.ResumeError):
            v2.restore_sampler_state(other, payload)


class PlanModeTests(unittest.TestCase):
    def test_empty_directory_is_a_fresh_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(v2.plan_mode(Path(temp), False), "fresh")
            self.assertEqual(v2.plan_mode(Path(temp), True), "missing-checkpoint")

    def test_partial_checkpoint_needs_explicit_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / v2.CHECKPOINT_NAME).write_bytes(b"stub")
            self.assertEqual(v2.plan_mode(root, False), "partial-without-resume")
            self.assertEqual(v2.plan_mode(root, True), "resume")

    def test_completion_marker_blocks_a_silent_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / v2.CHECKPOINT_NAME).write_bytes(b"stub")
            (root / v2.COMPLETION_NAME).write_text("{}", encoding="utf-8")
            self.assertEqual(v2.plan_mode(root, False), "complete")
            self.assertEqual(v2.plan_mode(root, True), "complete")


class ResumeValidationTests(unittest.TestCase):
    CONFIG_HASH = "config-hash"
    INPUTS = {"start_checkpoint": "aaa", "exclusions": "bbb"}

    def _payload(self, **overrides: object) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": v2.CHECKPOINT_SCHEMA,
            "epoch": 1,
            "phase": "evaluated",
            "config_hash": self.CONFIG_HASH,
            "input_hashes": dict(self.INPUTS),
            "model_state": {},
            "optimizer_state": {},
            "sampler_state": {},
            "rng": {},
            "history": [],
        }
        payload.update(overrides)
        return payload

    def test_accepts_a_matching_checkpoint(self) -> None:
        self.assertEqual(v2.validate_resume(self._payload(), config_hash=self.CONFIG_HASH,
                                            input_hashes=self.INPUTS), (1, "evaluated"))

    def test_rejects_a_changed_config(self) -> None:
        with self.assertRaises(v2.ResumeError):
            v2.validate_resume(self._payload(), config_hash="other", input_hashes=self.INPUTS)

    def test_rejects_changed_inputs(self) -> None:
        with self.assertRaises(v2.ResumeError):
            v2.validate_resume(self._payload(), config_hash=self.CONFIG_HASH,
                               input_hashes={"start_checkpoint": "changed"})

    def test_rejects_a_missing_required_field(self) -> None:
        payload = self._payload()
        del payload["optimizer_state"]
        with self.assertRaises(v2.ResumeError):
            v2.validate_resume(payload, config_hash=self.CONFIG_HASH, input_hashes=self.INPUTS)

    def test_rejects_an_unknown_schema(self) -> None:
        with self.assertRaises(v2.ResumeError):
            v2.validate_resume(self._payload(schema="other"), config_hash=self.CONFIG_HASH,
                               input_hashes=self.INPUTS)

    def test_rejects_an_out_of_range_epoch(self) -> None:
        for epoch in (0, v2.EPOCHS + 1):
            with self.assertRaises(v2.ResumeError):
                v2.validate_resume(self._payload(epoch=epoch), config_hash=self.CONFIG_HASH,
                                   input_hashes=self.INPUTS)

    def test_rejects_an_unknown_phase(self) -> None:
        with self.assertRaises(v2.ResumeError):
            v2.validate_resume(self._payload(phase="half"), config_hash=self.CONFIG_HASH,
                               input_hashes=self.INPUTS)

    def test_rejects_weights_that_do_not_fit_the_model(self) -> None:
        model = torch.nn.Linear(3, 2)
        with self.assertRaises(v2.ResumeError):
            v2.restore_model_state(model, {"weight": torch.zeros(5, 5)})


class CheckpointRoundTripTests(unittest.TestCase):
    INPUTS = {"start_checkpoint": "start", "exclusions": "excl"}

    @staticmethod
    def _train_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = (model(torch.ones(4, 3)) ** 2).mean()
        loss.backward()
        optimizer.step()

    def test_round_trip_restores_unmixed_weights_optimizer_and_sampler(self) -> None:
        torch.manual_seed(5)
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        self._train_step(model, optimizer)
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, 6, replacement=True,
                                        generator=torch.Generator().manual_seed(11))
        list(iter(sampler))
        inputs = dict(self.INPUTS)
        config_hash = v2.config_digest({"demo": 1})
        payload = {
            "schema": v2.CHECKPOINT_SCHEMA,
            "epoch": 1,
            "phase": "trained",
            "config": {"demo": 1},
            "config_hash": config_hash,
            "input_hashes": inputs,
            "model_state": {name: value.detach().cpu().clone()
                            for name, value in model.state_dict().items()},
            "optimizer_state": optimizer.state_dict(),
            "sampler_state": v2.sampler_state_payload(sampler),
            "rng": v2.capture_rng_state(),
            "history": [{"epoch": 1, "alpha": 1.0, "accepted": False}],
            "best": None,
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / v2.CHECKPOINT_NAME
            v2.atomic_torch_save(payload, path)
            loaded = v2.load_checkpoint(path)
            self.assertEqual(v2.validate_resume(loaded, config_hash=config_hash,
                                                input_hashes=inputs), (1, "trained"))
            self.assertEqual(loaded["history"], payload["history"])
            self.assertIsNone(loaded["best"])

            restored_model = torch.nn.Linear(3, 2)
            v2.restore_model_state(restored_model, loaded["model_state"])
            restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.5)
            restored_optimizer.load_state_dict(loaded["optimizer_state"])
            replay = WeightedRandomSampler(weights, 6, replacement=True,
                                           generator=torch.Generator().manual_seed(0))
            v2.restore_sampler_state(replay, loaded["sampler_state"])

        for original, restored in zip(model.parameters(), restored_model.parameters()):
            self.assertTrue(torch.equal(original.detach(), restored.detach()))
        self.assertAlmostEqual(restored_optimizer.param_groups[0]["lr"], 0.01)
        self.assertTrue(torch.equal(optimizer.state_dict()["state"][0]["exp_avg"],
                                    restored_optimizer.state_dict()["state"][0]["exp_avg"]))
        self.assertEqual(list(iter(replay)), list(iter(sampler)))


if __name__ == "__main__":
    unittest.main()
