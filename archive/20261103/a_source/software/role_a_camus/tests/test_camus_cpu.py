"""CAMUS 合成 CPU 用例（**只用合成数据**；不读真实标签、不调用官方体积数学内部）。

覆盖：4×4 手算与 valid/ignore、run_epoch 接口统一（``loss``/``ce``/``foreground_dice``）、
半像素中心重采样与原网格重建、官方 12 文件命名（含 half_sequence）/质量字段/封条强绑定、
原生 GT 缓存、模型算子白名单与真实预算、输入不被修改。

运行（一次，≤300 秒）：
``python -m unittest discover -s software/role_a_camus/tests -p test_camus_cpu.py -v``
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from software.role_a_camus import data as camus_data          # noqa: E402
from software.role_a_camus import evaluate as camus_eval      # noqa: E402
from software.role_a_camus import geometry as geo             # noqa: E402
from software.role_a_camus import metrics as camus_metrics    # noqa: E402
from software.role_a_camus import model as camus_model        # noqa: E402
from software.role_a_camus import official_volume as official # noqa: E402
from software.role_a_camus import train as camus_train        # noqa: E402


# ---------------------------------------------------------------- 手算依据（写入用例注释）
# GT   = {(1,1), (1,2), (2,2)} 为 LV 腔；预测 = {(1,0), (1,1), (1,2)}
#   TP = 2（(1,1),(1,2)）、pred = 3、GT = 3 ⇒ Dice = 2*2/(3+3) = 2/3，recall = 2/3
# valid 屏蔽 (2,2)：它是 FN，屏蔽后 TP = 2、pred = 3、GT = 2 ⇒ Dice = 4/5
# ignore(=255) 屏蔽 (0,0)：LA 仍有 3 个 GT 与 3 个预测（补边既不记 GT 也不记 pred）
HAND_TRUTH = np.array([[0, 0, 0, 0],
                       [0, 1, 1, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 0]], dtype=np.int64)
HAND_PREDICTION = np.array([[0, 0, 0, 0],
                            [1, 1, 1, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]], dtype=np.int64)


class FourClassHandCalculationTests(unittest.TestCase):
    def test_lv_cavity_dice_and_recall(self) -> None:
        result = camus_metrics.class_scores(HAND_PREDICTION, HAND_TRUTH, geo.LV_CAVITY)
        self.assertEqual((result.intersection, result.pred_pixels, result.gt_pixels), (2, 3, 3))
        self.assertAlmostEqual(result.dice, 2.0 / 3.0, places=12)
        self.assertAlmostEqual(result.recall, 2.0 / 3.0, places=12)

    def test_valid_mask_hides_fn_pixel(self) -> None:
        valid = np.ones_like(HAND_TRUTH, dtype=np.float32)
        valid[2, 2] = 0.0                                   # (2,2) 是 FN，被 valid 屏蔽
        result = camus_metrics.class_scores(HAND_PREDICTION, HAND_TRUTH, geo.LV_CAVITY, valid=valid)
        self.assertEqual((result.intersection, result.pred_pixels, result.gt_pixels), (2, 3, 2))
        self.assertAlmostEqual(result.dice, 4.0 / 5.0, places=12)

    def test_ignore_index_excludes_padding_from_both_sides(self) -> None:
        truth = np.array([[255, 3, 3], [255, 3, 0], [0, 0, 0]], dtype=np.int64)
        prediction = np.array([[3, 3, 3], [3, 3, 0], [0, 0, 0]], dtype=np.int64)
        result = camus_metrics.class_scores(prediction, truth, geo.LEFT_ATRIUM)
        self.assertEqual((result.intersection, result.pred_pixels, result.gt_pixels), (3, 3, 3))
        self.assertAlmostEqual(result.dice, 1.0, places=12)

    def test_empty_gt_with_false_positive_is_zero_not_null(self) -> None:
        empty_truth = np.zeros((3, 3), dtype=np.int64)
        prediction = np.zeros((3, 3), dtype=np.int64)
        prediction[0, 0] = geo.LV_CAVITY
        result = camus_metrics.class_scores(prediction, empty_truth, geo.LV_CAVITY)
        self.assertEqual(result.dice, 0.0)                   # 假阳性不得被忽略
        self.assertIsNone(result.recall)
        both_empty = camus_metrics.class_scores(empty_truth, empty_truth, geo.LV_CAVITY)
        self.assertIsNone(both_empty.dice)                   # 双方都空 → null 并计数
        summary = camus_metrics.evaluate_frames([{"foreground_macro_dice": None, "per_class": {}}])
        self.assertEqual(summary["patient_balanced"]["null_frames"], 1)

    def test_three_foreground_macro(self) -> None:
        truth = np.zeros((4, 4), dtype=np.int64)
        truth[0, 0] = geo.LV_CAVITY
        truth[1, 1] = geo.MYOCARDIUM
        truth[2, 2] = geo.LEFT_ATRIUM
        prediction = truth.copy()
        scores = camus_metrics.frame_scores(prediction, truth)
        self.assertAlmostEqual(scores["foreground_macro_dice"], 1.0, places=12)


class RunEpochInterfaceTests(unittest.TestCase):
    def test_breakdown_keys_match_run_epoch(self) -> None:
        logits = torch.randn(2, 4, 16, 16)
        target = torch.zeros(2, 16, 16, dtype=torch.long)
        target[:, 4:8, 4:8] = geo.LV_CAVITY
        valid = torch.ones(2, 1, 16, 16)
        loss, breakdown = camus_train.multiclass_loss(logits, target, valid)
        self.assertAlmostEqual(breakdown["loss"], float(loss.detach()), places=6)
        self.assertEqual(set(breakdown), {"loss", "ce", "foreground_dice"})
        model = camus_model.build_model(base_channels=4, num_pools=1, size=16)
        stats = camus_train.run_epoch(model, [(torch.randn(2, 1, 16, 16), target, valid)],
                                      None, torch.device("cpu"),
                                      camus_train.TrainConfig(size=16, amp=False))
        self.assertEqual(set(stats), {"loss", "ce", "foreground_dice"})
        self.assertTrue(all(np.isfinite(v) for v in stats.values()))

    def test_padding_pixels_do_not_enter_loss(self) -> None:
        logits = torch.zeros(1, 4, 8, 8)
        target = torch.zeros(1, 8, 8, dtype=torch.long)
        target[:, 0, 0] = geo.LV_CAVITY
        valid = torch.ones(1, 1, 8, 8)
        valid[:, 0, 0] = 0.0                                 # 标记为补边
        loss_with, _ = camus_train.multiclass_loss(logits, target, valid)
        target_masked = target.clone()
        target_masked[:, 0, 0] = camus_train.IGNORE_INDEX
        loss_expected, _ = camus_train.multiclass_loss(logits, target_masked, valid)
        self.assertAlmostEqual(float(loss_with), float(loss_expected), places=6)

    def test_synthetic_backward_updates_weights_and_keeps_input(self) -> None:
        torch.manual_seed(7)
        model = camus_model.build_model(base_channels=4, num_pools=1, size=32)
        image, label, valid = camus_model.synthetic_batch(batch=2, size=32, seed=11)
        snapshot = image.clone()
        before = model.head.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        stats = camus_train.run_epoch(model, [(image, label, valid)], optimizer,
                                      torch.device("cpu"),
                                      camus_train.TrainConfig(size=32, amp=False))
        self.assertTrue(np.isfinite(stats["loss"]))
        self.assertFalse(torch.equal(before, model.head.weight.detach()))
        self.assertTrue(torch.equal(snapshot, image))        # 输入不被修改


class ResamplingTests(unittest.TestCase):
    def test_half_pixel_center_linear_map(self) -> None:
        # 目标 2 → 源 4：src = (j+0.5)*2-0.5 ⇒ 0.5 与 2.5
        source = np.arange(4, dtype=np.float32).reshape(1, 4)
        resized = geo.resize_bilinear(source, 1, 2)
        self.assertTrue(np.allclose(resized, [[0.5, 2.5]], atol=1e-6))

    def test_upsample_roundtrip_is_close_for_smooth_signal(self) -> None:
        rows = np.linspace(0.0, 1.0, 24, dtype=np.float32)[:, None]
        cols = np.linspace(0.0, 1.0, 18, dtype=np.float32)[None, :]
        image = rows * cols                                     # 双线性可精确表示
        canvas, valid, meta = geo.letterbox(image, size=64)
        restored = geo.restore_to_native(canvas, meta, mode="bilinear")
        self.assertEqual(restored.shape, image.shape)
        # For 24->64 and 18->48, half-pixel edge clamping moves the
        # first reconstructed coordinate by 5/96 original pixels on each axis.
        # This is lossy at the edges even for a bilinear source.
        delta_y = (5.0 / 96.0) / 23.0
        delta_x = (5.0 / 96.0) / 17.0
        self.assertAlmostEqual(float(restored[0, 0]), delta_y * delta_x, places=6)
        self.assertAlmostEqual(float(restored[-1, -1]), (1-delta_y)*(1-delta_x), places=6)
        self.assertTrue(np.allclose(restored[1:-1, 1:-1], image[1:-1, 1:-1], atol=2e-6))
        self.assertEqual(int(valid.sum()), meta["new_height"] * meta["new_width"])

    def test_nearest_keeps_discrete_classes(self) -> None:
        labels = np.zeros((6, 10), dtype=np.int64)
        labels[1:4, 2:8] = geo.MYOCARDIUM
        canvas, _valid, meta = geo.letterbox(np.zeros((6, 10), dtype=np.float32), size=32)
        label_canvas = geo.letterbox_labels(labels, meta)
        self.assertTrue(set(np.unique(label_canvas)).issubset({0, geo.MYOCARDIUM, 255}))
        restored = geo.restore_to_native(label_canvas, meta, mode="nearest")
        self.assertEqual(restored.shape, labels.shape)
        self.assertTrue(set(np.unique(restored)).issubset({0, geo.MYOCARDIUM}))

    def test_argmax_after_native_reconstruction_order(self) -> None:
        # 两个通道在 letterbox 中数值接近，先放大再 argmax 与先 argmax 再放大结果不同
        canvas = np.zeros((2, 4, 4), dtype=np.float32)
        canvas[0, :, :2] = 1.0
        canvas[1, :, 2:] = 0.9
        meta = {"original_height": 3, "original_width": 5, "scale": 1.0, "new_height": 4,
                "new_width": 4, "top": 0, "left": 0, "size": 4}
        native = geo.argmax_on_native(canvas, meta)
        self.assertEqual(native.shape, (3, 5))
        self.assertTrue((native[:, :2] == 0).all() and (native[:, -2:] == 1).all())

    def test_label_validation_and_input_immutability(self) -> None:
        with self.assertRaises(geo.LabelError):
            geo.validate_label(np.array([[0, 7]], dtype=np.int64))
        with self.assertRaises(geo.LabelError):
            geo.validate_label(np.array([[-1, 0]], dtype=np.int64))
        with self.assertRaises(geo.LabelError):
            geo.validate_label(np.array([[0.5, 1.0]], dtype=np.float32))
        with self.assertRaises(geo.LabelError):
            geo.validate_label(np.array([[np.nan, 1.0]], dtype=np.float32))
        with self.assertRaises(geo.LabelError):
            geo.resize_bilinear(np.zeros((3, 3, 3), dtype=np.float32), 2, 2)
        image = np.arange(16, dtype=np.float32).reshape(4, 4)
        snapshot = image.copy()
        geo.normalize_gray(image)
        geo.letterbox(image, size=8)
        self.assertTrue(np.array_equal(image, snapshot))

    def test_hd95_units_and_null(self) -> None:
        block = np.zeros((8, 8), dtype=bool)
        block[2:6, 2:6] = True
        half = np.zeros((8, 8), dtype=bool)
        half[2:6, 2:4] = True
        self.assertIsNone(geo.hd95_mm(np.zeros((8, 8), dtype=bool), block, 1.0, 1.0))
        self.assertAlmostEqual(geo.hd95_mm(half, block, 1.0, 1.0), 2.0, places=6)
        self.assertAlmostEqual(geo.hd95_mm(half, block, 2.0, 2.0), 4.0, places=6)


class FilenameAndSplitTests(unittest.TestCase):
    def test_filename_regex_parses_all_official_kinds(self) -> None:
        cases = {"patient0001_2CH_ED.nii.gz": ("patient0001", "2CH", "ED", ""),
                 "patient0001_4CH_ES_gt.nii.gz": ("patient0001", "4CH", "ES", "_gt"),
                 "patient0001_2CH_half_sequence.nii.gz": ("patient0001", "2CH", None,
                                                             "_half_sequence"),
                 "patient0001_4CH_half_sequence_gt.nii.gz": ("patient0001", "4CH", None,
                                                                "_half_sequence_gt")}
        for name, expected in cases.items():
            identity = camus_data.parse_filename(Path(name))
            self.assertEqual((identity.patient, identity.view, identity.phase, identity.kind),
                             expected)
        with self.assertRaises(camus_data.SplitError):
            camus_data.parse_filename(Path("patient0001_2CH_XX.nii.gz"))
        with self.assertRaises(camus_data.SplitError):
            camus_data.parse_filename(Path("patient1_2CH_ED.nii.gz"))

    def test_saved_official_listing_matches_twelve_real_nifti_names(self) -> None:
        # Metadata retrieved from the official public API; no voxel/data read.
        items=json.loads((PROJECT/"runs/role_a_camus/20261012/preparation/source_review/patient0001_items.json").read_text(encoding="utf-8"))
        names=[x["name"] for x in items if x["name"].endswith(".nii.gz")]
        identities=[camus_data.parse_filename(Path(name)) for name in names]
        self.assertEqual(len(identities),12)
        self.assertEqual(sum(x.is_half_sequence for x in identities),4)
        self.assertEqual(sum(x.is_endpoint_image for x in identities),4)
        self.assertEqual(sum(x.is_endpoint_gt for x in identities),4)
        with self.assertRaises(camus_data.SplitError):
            camus_data.parse_filename(Path("patient0001_2CH_ED_half_sequence.nii.gz"))

    def test_patient_dir_allows_official_twelve_and_rejects_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "database_nifti" / "patient0001"
            root.mkdir(parents=True)
            for view in geo.VIEWS:
                for phase in geo.PHASES:
                    (root / f"patient0001_{view}_{phase}.nii.gz").write_bytes(b"x")
                    (root / f"patient0001_{view}_{phase}_gt.nii.gz").write_bytes(b"x")
                (root / f"patient0001_{view}_half_sequence.nii.gz").write_bytes(b"x")
                (root / f"patient0001_{view}_half_sequence_gt.nii.gz").write_bytes(b"x")
            for count, view in enumerate(geo.VIEWS):
                (root / camus_data.QUALITY_CONFIGS[view]).write_text(
                    f"ImageQuality: {'Good' if count == 0 else 'Poor'}\nSpacing: 0.308 0.308\n",
                    encoding="utf-8")
            listing = camus_data.validate_patient_dir(root, "patient0001")
            self.assertEqual(len(listing["endpoints"]), 4)
            self.assertEqual(len(listing["half_sequence"]), 4)
            quality = camus_data.patient_quality(root)
            self.assertEqual(quality["2CH"]["ImageQuality"], "Good")
            self.assertEqual(quality["4CH"]["ImageQuality"], "Poor")
            (root / "patient0001_2CH_ED_unknown.nii.gz").write_bytes(b"x")
            with self.assertRaises(camus_data.SplitError):
                camus_data.validate_patient_dir(root, "patient0001")
            with self.assertRaises(camus_data.SplitError):
                camus_data.quality_tag(np.zeros((4, 4), dtype=np.float32))

    def test_quality_config_requires_official_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "Info_2CH.cfg"
            path.write_text("Spacing: 0.3 0.3\n", encoding="utf-8")
            with self.assertRaises(camus_data.SplitError):
                camus_data.parse_quality_config(path)
            path.write_text("ImageQuality: Excellent\n", encoding="utf-8")
            with self.assertRaises(camus_data.SplitError):
                camus_data.parse_quality_config(path)

    def test_split_binding_and_cross_group_rejection(self) -> None:
        patients = tuple(f"patient{index:04d}" for index in range(1, 6))
        split = camus_data.PatientSplit(training=patients[:2], validation=patients[2:3],
                                        testing=patients[3:],
                                        source_sha256={"subgroup_training.txt": "a"})
        split.validate(expect={"training": 2, "validation": 1, "testing": 2})
        self.assertEqual(split.owner_of("patient0003"), "validation")
        self.assertIsNone(camus_data.assert_no_patient_cross_split(
            split, {"patient0001":"training","patient0003":"validation","patient0004":"testing"}))
        with self.assertRaises(camus_data.SplitError):
            camus_data.assert_no_patient_cross_split(split, {"patient9999":"training"})
        with self.assertRaises(camus_data.SplitError):
            camus_data.verify_split_binding(Path("nowhere"), split)   # 无源 SHA/文件即拒绝
        with self.assertRaises(camus_data.SplitError):
            camus_data.load_frame(Path("x"), None, split="training", patient=patients[2],
                                  split_object=split, camus_root=Path("nowhere"))
        with self.assertRaises(camus_data.SplitError):
            camus_data.parse_patient_list("# empty\n")
        with self.assertRaises(camus_data.SplitError):
            camus_data.assert_no_patient_cross_split(split, {patients[2]: "training"})
        with self.assertRaises(camus_data.SplitError):
            camus_data.PatientSplit(("patient0001",), ("patient0001",), ("patient0002",))\
                .validate(expect={"training": 1, "validation": 1, "testing": 1})

    def test_test_gt_sealed_until_owner_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertFalse(camus_data.test_release_granted(root))
            with self.assertRaises(camus_data.TestSealedError):
                camus_data.assert_test_gt_allowed(root, "patient0001")
            (root / camus_data.TEST_RELEASE_FILE).write_text(
                json.dumps({"allow_test_gt": True}), encoding="utf-8")
            self.assertTrue(camus_data.test_release_granted(root))
            camus_data.assert_test_gt_allowed(root, "patient0001")
        with self.assertRaises(camus_eval.EvaluationRefused):
            camus_eval.evaluate_dataset([], "testing")
        self.assertIsInstance(camus_eval.evaluate_dataset([], "validation"), dict)


class NativeCacheTests(unittest.TestCase):
    def _frame(self) -> camus_data.CamusFrame:
        native = np.linspace(0.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
        native_label = np.zeros((3, 4), dtype=np.int64)
        native_label[1, 1:3] = geo.LV_CAVITY
        canvas, valid, meta = geo.letterbox(native, size=8)
        label_canvas = geo.letterbox_labels(native_label, meta)
        return camus_data.CamusFrame(
            patient="patient0001", split="training", view="2CH", phase="ED", image=canvas,
            valid=valid, label=label_canvas, meta=meta, spacing_mm=(0.308, 0.308),
            native_image=native, native_label=native_label, native_shape=(3, 4),
            affine=((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0),
                    (0.0, 0.0, 0.0, 1.0)),
            source_sha256={"image": "deadbeef"}, quality="Good",
            quality_source="Info_2CH.cfg", histogram={"lv_cavity": 2})

    def test_cache_preserves_native_gt_and_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            frame = self._frame()
            manifest = camus_data.cache_frames([frame], Path(temporary))
            record = manifest["records"][0]
            self.assertEqual(record["native_shape"], [3, 4])
            self.assertEqual(record["letterbox_shape"], [8, 8])
            self.assertEqual(record["quality"], "Good")
            self.assertEqual(record["quality_source"], "Info_2CH.cfg")
            payload = camus_data.load_cached_frame(Path(record["cache_file"]))
            self.assertTrue(np.array_equal(payload["native_label"], frame.native_label))
            self.assertEqual(payload["native_image"].shape, (3, 4))
            self.assertEqual(payload["spacing_mm"], (0.308, 0.308))
            self.assertEqual(len(payload["affine"]), 4)
            self.assertTrue(manifest["native_gt_retained"])

    def test_native_metric_uses_native_grid_not_resized_label(self) -> None:
        frame = self._frame()
        record = camus_metrics.native_frame_scores(
            frame.native_label, frame.native_label, frame.spacing_mm, quality=frame.quality,
            patient=frame.patient, view=frame.view, phase=frame.phase, frame_id="p1_2CH_ED")
        self.assertAlmostEqual(record["foreground_macro_dice"], 1.0, places=12)
        self.assertAlmostEqual(record["gt_area_mm2"]["lv_cavity"], 2 * 0.308 * 0.308, places=12)
        self.assertEqual(record["quality"], "Good")


class ModelOperatorAndBudgetTests(unittest.TestCase):
    def test_operator_whitelist_and_no_transpose_conv(self) -> None:
        model = camus_model.build_model(base_channels=8, num_pools=2, size=64)
        model.assert_deployable_operators()
        self.assertFalse(any(isinstance(m, torch.nn.ConvTranspose2d) for m in model.modules()))
        with self.assertRaises(AssertionError):
            model.decoders[0].add_module("bad", torch.nn.ConvTranspose2d(4, 4, 2, stride=2))
            model.assert_deployable_operators()

    def test_forward_shape_and_budget_from_actual_model(self) -> None:
        report = camus_model.budget_report(base_channels=16, num_pools=3, size=256)
        model = camus_model.build_model(base_channels=16, num_pools=3, size=256)
        self.assertEqual(report["parameters"], sum(p.numel() for p in model.parameters()))
        self.assertEqual(report["int8_weight_bytes"], report["parameters"])
        self.assertEqual(report["output_bytes"], 4 * 256 * 256 * 4)
        self.assertGreater(report["macs"], 0)
        logits = model(torch.zeros(1, 1, 256, 256))
        self.assertEqual(tuple(logits.shape), (1, 4, 256, 256))


class OfficialVolumeContractTests(unittest.TestCase):
    """官方体积：只按 Codex 给出的契约断言（20-disk 数值/单位/输入不变；失败 null；canonical 对拍）。"""

    def test_twenty_disk_numeric_and_units_mL(self) -> None:
        diameters = np.full(20, 10.0)                        # 10 mm
        other = np.full(20, 20.0)                            # 20 mm
        step_mm = 2.0
        expected = round(float(np.sum((diameters / 1000.0) * (other / 1000.0))
                               * (step_mm / 1000.0) * np.pi / 4.0 * 1e6))
        self.assertEqual(expected, 6)                        # π/4×10×20×20×2/1000 ≈ 6.283 → 6
        first = official.volumes_by_instant(diameters, other, step_mm)
        second = official.volumes_by_instant(diameters, other, step_mm)
        self.assertEqual(first, 6)
        self.assertEqual(second, 6)
        with self.assertRaises(ValueError):
            official.volumes_by_instant(np.full(19, 10.0), other, step_mm)
        with self.assertRaises(ValueError):
            official.volumes_by_instant(diameters, other, 0.0)

    def test_biplane_report_null_on_empty_broken_and_nan_spacing(self) -> None:
        keys = ("2CH_ED", "2CH_ES", "4CH_ED", "4CH_ES")
        good_mask = np.zeros((8, 8), dtype=np.uint8)
        good_mask[2:6, 3:5] = 1
        spacing = {key: (0.3, 0.3) for key in keys}

        empty = official.biplane_report({key: np.zeros((8, 8), dtype=np.uint8) for key in keys},
                                        spacing)
        self.assertIsNone(empty["edv_ml"])
        self.assertIsNotNone(empty["reason"])
        self.assertIs(empty["fallback_used"], False)

        broken = np.zeros((8, 8), dtype=np.uint8)
        broken[1, 1] = 1
        broken[5, 5] = 1                                     # 两个连通域
        two_components = official.biplane_report({**{key: good_mask for key in keys},
                                                  "4CH_ES": broken}, spacing)
        self.assertIsNone(two_components["edv_ml"])
        self.assertIn("components", two_components["reason"])

        nan_spacing = {**spacing, "2CH_ED": (float("nan"), 0.3)}
        nan_report = official.biplane_report({key: good_mask for key in keys}, nan_spacing)
        self.assertIsNone(nan_report["edv_ml"])
        self.assertIsNotNone(nan_report["reason"])

        zero_spacing = {**spacing, "2CH_ED": (0.0, 0.3)}
        self.assertIsNone(official.biplane_report({key: good_mask for key in keys},
                                                  zero_spacing)["edv_ml"])

    def test_inputs_not_modified_by_volume_path(self) -> None:
        keys = ("2CH_ED", "2CH_ES", "4CH_ED", "4CH_ES")
        masks = {}
        for index, key in enumerate(keys):
            mask = np.zeros((10, 10), dtype=np.uint8)
            mask[2:8, 3 + index % 2:5 + index % 2] = 1
            masks[key] = mask
        spacing = {key: (0.31, 0.31) for key in keys}
        masks_copy = {key: value.copy() for key, value in masks.items()}
        spacing_copy = {key: tuple(value) for key, value in spacing.items()}
        diameters = np.full(20, 10.0)
        official.volumes_by_instant(diameters, np.full(20, 20.0), 2.0)
        official.biplane_report(masks, spacing)
        for key in keys:
            self.assertTrue(np.array_equal(masks[key], masks_copy[key]))
        self.assertEqual({key: tuple(value) for key, value in spacing.items()}, spacing_copy)
        self.assertTrue(np.allclose(diameters, 10.0))        # 直径数组未被原地 /1000

    def test_canonical_diameters_match_reference_functions_numerically(self) -> None:
        """canonical 提取算法：与 reference 原件**只抽指定函数**的同一原算法逐数值比。"""
        reference_path = (PROJECT / "runs/role_a_camus/20261012/preparation/source_review"
                          / "official_ef_code_reference.py.txt")
        self.assertTrue(reference_path.is_file(), "缺少官方 reference 原件")
        source = reference_path.read_text(encoding="utf-8")
        import ast
        wanted={"resize_image","resize_image_to_isotropic","_compute_diameters",
                "_find_distance_to_edge","_distance_line_to_points","_get_angle_of_lines_to_point"}
        selected=[node for node in ast.parse(source).body
                  if isinstance(node,ast.FunctionDef) and node.name in wanted]
        self.assertEqual({node.name for node in selected},wanted)
        # Compile only explicitly selected FunctionDef nodes. No original import,
        # assignment, demo call or data-loading function reaches the compiler.
        namespace: dict[str, object] = {"__name__": "camus_reference_functions"}
        exec("from typing import Tuple\nimport numpy as np\nimport PIL\n"
             "from PIL.Image import Resampling\nfrom skimage.measure import find_contours",namespace)
        extracted=ast.Module(body=selected,type_ignores=[])
        exec(compile(extracted, str(reference_path), "exec"), namespace)
        compute_reference = namespace["_compute_diameters"]
        resize_reference = namespace["resize_image_to_isotropic"]

        mask = np.zeros((60, 44), dtype=np.uint8)            # 非对称、非方
        rows = np.arange(60)[:, None]
        cols = np.arange(44)[None, :]
        mask[((rows - 34.0) / 18.0) ** 2 + ((cols - 20.0) / 9.0) ** 2 <= 1.0] = 1
        mask[12:30, 18:22] = 1
        spacing = (0.62, 0.31)                               # 各向异性，触发各向同性重采样

        reference_diameters, reference_step = compute_reference(mask.copy(), spacing)
        canonical_diameters, canonical_step = official._compute_diameters(mask.copy(), spacing)
        self.assertEqual(canonical_diameters.shape, (20,))
        self.assertTrue(np.allclose(canonical_diameters, reference_diameters, rtol=0, atol=1e-9))
        self.assertAlmostEqual(canonical_step, reference_step, places=12)

        reference_iso, reference_iso_spacing = resize_reference(mask.copy(), spacing)
        canonical_iso, canonical_iso_spacing = official.resize_image_to_isotropic(mask.copy(), spacing)
        self.assertTrue(np.array_equal(canonical_iso, reference_iso))
        self.assertAlmostEqual(canonical_iso_spacing, reference_iso_spacing, places=12)

    def test_report_requires_all_four_masks(self) -> None:
        masks = {"2CH_ED": np.ones((4, 4), dtype=np.uint8)}
        report = camus_eval.official_biplane_report(masks, {"2CH_ED": (0.3, 0.3)})
        self.assertIsNone(report["edv_ml"])
        self.assertIsNotNone(report["reason"])
        self.assertIs(report["fallback_used"], False)




class CompleteEpochCheckpointTests(unittest.TestCase):
    def test_mid_epoch_stop_preserves_initial_checkpoint_and_can_restore(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            stop=root/"STOP_REQUEST.txt"
            config=camus_train.TrainConfig(epochs=2,batch_size=1,size=16,
                base_channels=2,num_pools=1,amp=False,out_dir=str(root),stop_file=str(stop))
            camus_train.seed_everything(19)
            model=camus_model.build_model(base_channels=2,num_pools=1,size=16)
            initial={k:v.clone() for k,v in model.state_dict().items()}
            image=torch.rand(1,1,16,16);label=torch.randint(0,4,(1,16,16));valid=torch.ones(1,1,16,16)
            class StopAfterOneBatch:
                def __len__(self):return 2
                def __iter__(self):
                    yield image,label,valid
                    stop.write_text("stop after one completed optimizer step",encoding="utf-8")
                    yield image,label,valid
            identity={"case":"synthetic"}
            result=camus_train.fit(StopAfterOneBatch(),[],config,model,"cpu",
                                  data_identity=identity,source_identity=identity,max_seconds=30)
            self.assertEqual(result["status"],"stopped")
            last=camus_train.load_checkpoint(root/"last.pt",config=config,
                               source_identity=identity,data_identity=identity)
            self.assertEqual(last["epoch"],0)
            self.assertTrue(all(torch.equal(initial[k],last["model_state"][k]) for k in initial))
            partial=torch.load(root/"partial_state.pt",map_location="cpu",weights_only=False)
            self.assertFalse(partial["recoverable"])
            self.assertTrue(any(not torch.equal(initial[k],partial["model_state"][k]) for k in initial))
            restored=camus_model.build_model(base_channels=2,num_pools=1,size=16)
            opt=camus_train.build_optimizer(restored,config);sched=camus_train.build_scheduler(opt,config)
            scaler=torch.amp.GradScaler("cuda",enabled=False)
            state=camus_train.restore_training_state(last,model=restored,optimizer=opt,
                                                     scheduler=sched,scaler=scaler)
            self.assertEqual(state["epoch"],0)
            self.assertTrue(all(torch.equal(initial[k],restored.state_dict()[k]) for k in initial))
            with self.assertRaises(camus_train.ResumeRejected):
                camus_train.load_checkpoint(root/"last.pt",config=config,source_identity={"case":"wrong"})



class MixedPrecisionLargeReductionTests(unittest.TestCase):
    def test_half_logits_full_batch_reduction_matches_hand_counts(self):
        import math
        logits=torch.zeros(8,4,256,256,dtype=torch.float16,requires_grad=True)
        target=(torch.arange(256)[None,:]%4).expand(256,256)[None].expand(8,-1,-1)
        valid=torch.ones(8,1,256,256)
        loss,parts=camus_train.multiclass_loss(logits,target,valid)
        # 524288 valid pixels: each class has 131072, uniform p=1/4.
        expected_dice=1-(2*32768+1)/(262144+1)
        self.assertAlmostEqual(parts["ce"],math.log(4),places=5)
        self.assertAlmostEqual(parts["foreground_dice"],expected_dice,places=6)
        self.assertAlmostEqual(parts["loss"],math.log(4)+expected_dice,places=5)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad!=0).any())

if __name__ == "__main__":
    unittest.main()
