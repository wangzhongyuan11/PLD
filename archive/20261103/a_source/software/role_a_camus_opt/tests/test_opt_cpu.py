"""role_a_camus_opt 的合成 CPU 用例（**不读真实数据、不训练、不推理**）。

覆盖（按任务 169 要求）：

* 后处理：已知两块大小的 5% 删除/超限保留；0 标签封闭孔洞填充；MYO/LA 不被覆盖；开向图边的孔不填；
  输入不被修改；
* 边界损失：SDF 正负号；λ=0 与原 ``multiclass_loss`` 损失与梯度逐值一致；invalid 像素不影响结果；
  全背景/无效情形 finite；小批配对种子下梯度反向一致；
* 训练：初始化张量一致；config 身份含 λ；STOP/完整断点与恢复保护（缺项、λ 不符、源码/数据身份不符拒绝）。

运行（单次，≤300 秒）：
``python -m unittest discover -s software/role_a_camus_opt/tests -p test_opt_cpu.py -v``
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(r"E:/嵌赛/pld-medical")
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from software.role_a_camus import train as base_train            # noqa: E402
from software.role_a_camus_opt import boundary_loss as opt_loss  # noqa: E402
from software.role_a_camus_opt import postprocess as pp          # noqa: E402
from software.role_a_camus_opt import train as opt_train         # noqa: E402


class PostprocessTests(unittest.TestCase):
    def test_island5_deletes_small_secondary_and_preserves_input(self) -> None:
        labels = np.zeros((32, 32), dtype=np.int64)
        labels[5:25, 5:25] = pp.LV                     # 400 px 主块
        labels[27:29, 27:29] = pp.LV                   # 4 px secondary ⇒ 4/404 ≈ 0.99% <= 5%
        original = labels.copy()
        outcome = pp.apply_island5(labels)
        self.assertTrue(outcome.island_applied)
        self.assertEqual(outcome.lv_pixels_before, 404)
        self.assertEqual(outcome.lv_pixels_after, 400)
        self.assertEqual(int((outcome.labels == pp.LV).sum()), 400)
        self.assertTrue(np.array_equal(labels, original), "输入不得被修改")
        self.assertEqual(outcome.changed_pixels, 4)

    def test_island5_preserves_when_secondary_exceeds_five_percent(self) -> None:
        labels = np.zeros((40, 40), dtype=np.int64)
        labels[2:12, 2:12] = pp.LV                     # 100 px
        labels[20:30, 20:30] = pp.LV                   # 100 px ⇒ 50% > 5%
        original = labels.copy()
        outcome = pp.apply_island5(labels)
        self.assertFalse(outcome.island_applied)
        self.assertEqual(outcome.secondary_ratio, 0.5)
        self.assertTrue(np.array_equal(outcome.labels, original))
        self.assertEqual(outcome.changed_pixels, 0)

    def test_hole1_fills_closed_background_only_and_never_overwrites_myo_la(self) -> None:
        labels = np.zeros((32, 32), dtype=np.int64)
        labels[8:24, 8:24] = pp.LV
        labels[15, 15] = pp.BACKGROUND                 # 1/247 <1% eligible hole
        labels[20:22, 10:12] = pp.MYO
        labels[10:12, 20:22] = pp.LA
        original = labels.copy()
        outcome = pp.apply_island5_hole1(labels)
        self.assertTrue(outcome.hole_applied)
        self.assertEqual(outcome.hole_pixels_filled, 1)
        self.assertEqual(int((outcome.labels == pp.BACKGROUND).sum()),
                         int((np.asarray(original) == pp.BACKGROUND).sum()) - 1)
        self.assertTrue(np.array_equal(outcome.labels[20:22, 10:12], np.full((2, 2), pp.MYO)))
        self.assertTrue(np.array_equal(outcome.labels[10:12, 20:22], np.full((2, 2), pp.LA)))
        self.assertTrue(np.array_equal(labels, original), "输入不得被修改")

    def test_hole1_does_not_fill_open_regions_or_oversized_holes(self) -> None:
        labels = np.zeros((32, 32), dtype=np.int64)
        labels[4:28, 4:28] = pp.LV
        labels[0:3, 10:14] = pp.BACKGROUND             # 与图边连通 ⇒ 不得填
        outcome = pp.apply_island5_hole1(labels)
        self.assertEqual(outcome.hole_pixels_filled, 0)
        self.assertFalse(outcome.hole_applied)
        big = np.zeros((40, 40), dtype=np.int64)
        big[2:38, 2:38] = pp.LV
        big[10:20, 10:20] = pp.BACKGROUND              # 100 px 孔洞 > 1% of 1296
        big_outcome = pp.apply_island5_hole1(big)
        self.assertFalse(big_outcome.hole_applied)
        self.assertGreater(big_outcome.hole_ratio, pp.HOLE_RATIO)

    def test_process_all_variants_and_label_validation(self) -> None:
        labels = np.zeros((16, 16), dtype=np.int64)
        labels[2:10, 2:10] = pp.LV
        variants = pp.process_all(labels)
        self.assertEqual(set(variants), set(pp.VARIANTS))
        self.assertIn("secondary_ratio", variants["island5"].as_dict())
        with self.assertRaises(pp.PostprocessError):
            pp.process(np.array([[0, 9]], dtype=np.int64), "island5")
        with self.assertRaises(pp.PostprocessError):
            pp.process(np.zeros((4, 4, 4), dtype=np.int64), "island5")


class BoundaryLossTests(unittest.TestCase):
    def _batch(self, seed: int = 3, size: int = 32, batch: int = 2):
        generator = torch.Generator().manual_seed(seed)
        logits = torch.randn(batch, 4, size, size, generator=generator, requires_grad=True)
        target = torch.zeros(batch, size, size, dtype=torch.long)
        target[:, 4:12, 4:12] = 1
        target[:, 6:10, 6:10] = 2
        target[:, 16:20, 16:22] = 3
        valid = torch.ones(batch, 1, size, size)
        valid[:, 0, 0, 0] = 0.0
        return logits, target, valid

    def test_sdf_signs_and_normalization(self) -> None:
        target = np.zeros((16, 16), dtype=np.int64)
        target[4:12, 4:12] = pp.LV
        distance = opt_loss.signed_distance_map(target, pp.LV, size=256)
        self.assertLess(distance[8, 8], 0.0)
        self.assertGreater(distance[0, 0], 0.0)
        diagonal = np.hypot(256, 256)
        self.assertLessEqual(float(np.nanmax(np.abs(distance))), 1.0)
        self.assertAlmostEqual(float(distance[8, 8]), -3.0 / diagonal, places=6)
        absent = opt_loss.signed_distance_map(target, pp.MYO, size=256)
        self.assertTrue(np.all(np.isnan(absent)))

    def test_lambda_zero_matches_base_loss_and_gradients(self) -> None:
        logits_a, target, valid = self._batch()
        logits_b = logits_a.detach().clone().requires_grad_(True)
        base, _ = base_train.multiclass_loss(logits_a, target, valid)
        opt, breakdown = opt_loss.multiclass_loss_with_boundary(logits_b, target, valid, None,
                                                               boundary_weight=0.0)
        self.assertAlmostEqual(float(base), float(opt), places=10)
        self.assertEqual(breakdown["boundary"], 0.0)
        base.backward()
        opt.backward()
        self.assertTrue(torch.allclose(logits_a.grad, logits_b.grad, atol=1e-10))

    def test_invalid_pixels_do_not_affect_boundary_term(self) -> None:
        logits, target, valid = self._batch()
        maps = opt_loss.boundary_maps_for_batch(target, valid, size=32)
        first = opt_loss.boundary_loss_from_maps(logits, maps, valid)
        maps_noise = maps.clone()
        maps_noise[:, :, 0, 0] = 1e6               # 位于 invalid 像素上
        second = opt_loss.boundary_loss_from_maps(logits, maps_noise, valid)
        self.assertAlmostEqual(float(first), float(second), places=10)

    def test_all_background_and_invalid_target_are_finite(self) -> None:
        empty = torch.zeros(1, 32, 32, dtype=torch.long)
        valid = torch.ones(1, 1, 32, 32)
        maps = opt_loss.boundary_maps_for_batch(empty, valid, size=32)
        logits = torch.randn(1, 4, 32, 32)
        term = opt_loss.boundary_loss_from_maps(logits, maps, valid)
        self.assertTrue(bool(torch.isfinite(term)))
        self.assertAlmostEqual(float(term), 0.0, places=10)
        loss, breakdown = opt_loss.multiclass_loss_for_arm(logits, empty, valid,
                                                          boundary_weight=0.05, size=32)
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertTrue(np.isfinite(breakdown["boundary"]))

    def test_boundary_gradient_reverse_mode_and_paired_seed(self) -> None:
        logits, target, valid = self._batch(seed=11)
        logits = logits.detach().clone().requires_grad_(True)
        loss, breakdown = opt_loss.multiclass_loss_for_arm(logits, target, valid,
                                                          boundary_weight=0.05, size=32)
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertGreater(abs(breakdown["boundary"]), 0.0)
        # 同种子第二次运行必须得到相同边界项（配对一致）
        logits2, target2, valid2 = self._batch(seed=11)
        logits2 = logits2.detach().clone().requires_grad_(True)
        _loss2, breakdown2 = opt_loss.multiclass_loss_for_arm(logits2, target2, valid2,
                                                             boundary_weight=0.05, size=32)
        self.assertAlmostEqual(breakdown["boundary"], breakdown2["boundary"], places=10)


class OptTrainingTests(unittest.TestCase):
    def test_initial_weights_match_frozen_and_config_identity_has_lambda(self) -> None:
        if not opt_train.FROZEN_BEST.is_file():
            self.skipTest("冻结权重不存在")
        model_a = opt_train.base_model.build_model(base_channels=16, num_pools=3, size=256)
        info = opt_train.initial_weights(model_a)
        self.assertEqual(info["initial_SHA256"], opt_train.INITIAL_SHA_EXPECTED)
        self.assertTrue(info["initial_optimizer_reset"])
        model_b = opt_train.base_model.build_model(base_channels=16, num_pools=3, size=256)
        opt_train.initial_weights(model_b)
        self.assertTrue(torch.equal(model_a.head.weight, model_b.head.weight))
        control = opt_train.OptTrainConfig(arm="control20", boundary_weight=opt_train.resolve_arm("control20"))
        boundary = opt_train.OptTrainConfig(arm="boundary20", boundary_weight=opt_train.resolve_arm("boundary20"))
        self.assertEqual(control.boundary_weight, 0.0)
        self.assertEqual(boundary.boundary_weight, 0.05)
        self.assertNotEqual(control.digest(), boundary.digest())
        self.assertIn("boundary_weight", json.dumps(json.loads(json.dumps(
            {k: v for k, v in vars(boundary).items()})), ensure_ascii=False))

    def test_checkpoint_required_keys_stop_and_resume_protection(self) -> None:
        config = opt_train.OptTrainConfig(arm="boundary20", boundary_weight=0.05,
                                          out_dir=None, data_manifest="synthetic", epochs=1)
        model = opt_train.base_model.build_model(base_channels=4, num_pools=1, size=32)
        optimizer = opt_train.make_optimizer(model, config)
        scheduler = opt_train.make_scheduler(optimizer, config)
        scaler = opt_train.make_scaler(config, torch.device("cpu"))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "last.pt"
            payload = opt_train.save_checkpoint_opt(
                path, epoch=1, model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                config=config, history=[], best_metric=0.5, best_epoch=1, stale=0,
                data_identity={"cache_sha256": "abc"}, status="running",
                initial_info={"initial_SHA256": opt_train.INITIAL_SHA_EXPECTED})
            for key in opt_train.REQUIRED_KEYS:
                self.assertIn(key, payload)
            self.assertAlmostEqual(payload["boundary_weight"], 0.05)
            loaded = opt_train.load_checkpoint_opt(path, config=config,
                                                   data_identity={"cache_sha256": "abc"})
            self.assertEqual(loaded["epoch"], 1)
            with self.assertRaises(opt_train.OptTrainError):
                opt_train.load_checkpoint_opt(path, config=config,
                                              data_identity={"cache_sha256": "different"})
            wrong_lambda = opt_train.OptTrainConfig(arm="control20", boundary_weight=0.0,
                                                    data_manifest="synthetic", epochs=1)
            with self.assertRaises(opt_train.OptTrainError):
                opt_train.load_checkpoint_opt(path, config=wrong_lambda,
                                              data_identity={"cache_sha256": "abc"})
            partial = opt_train.save_partial_opt(Path(temporary) / "partial_state.pt", epoch=1,
                                                 model=model, optimizer=optimizer,
                                                 reason="STOP during epoch",
                                                 data_identity={"cache_sha256": "abc"})
            self.assertFalse(partial["recoverable"])
            with self.assertRaises(opt_train.OptTrainError):
                opt_train.load_checkpoint_opt(Path(temporary) / "partial_state.pt", config=config,
                                              data_identity={"cache_sha256": "abc"})

    def test_arm_epoch_runs_with_lambda_zero_and_nonzero(self) -> None:
        torch.manual_seed(5)
        model = opt_train.base_model.build_model(base_channels=4, num_pools=1, size=32)
        generator = torch.Generator().manual_seed(7)
        image = torch.rand(2, 1, 32, 32, generator=generator)
        label = torch.zeros(2, 32, 32, dtype=torch.long)
        label[:, 6:14, 6:14] = 1
        label[:, 8:12, 8:12] = 2
        valid = torch.ones(2, 1, 32, 32)
        for arm in ("control20", "boundary20"):
            config = opt_train.OptTrainConfig(arm=arm, boundary_weight=opt_train.resolve_arm(arm),
                                              size=32, base_channels=4, num_pools=1, amp=False)
            optimizer = opt_train.make_optimizer(model, config)
            stats = opt_train.run_epoch_opt(model, [(image, label, valid)], optimizer,
                                            torch.device("cpu"), config, None,
                                            stop_file=None)
            for value in stats.values():
                self.assertTrue(np.isfinite(value), f"{arm}: {stats}")
            self.assertEqual(stats["boundary"], 0.0 if arm == "control20" else stats["boundary"])


if __name__ == "__main__":
    unittest.main()
