"""数据层测试：配对错误、划分可复现、letterbox 对齐。

运行（项目根目录）::

    py -m unittest discover -s software/role_a/tests -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _support import MISSING_DEPENDENCY, TemporaryRoot, ensure_role_a_on_path  # noqa: E402

ensure_role_a_on_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from data import (  # noqa: E402
    DataLayoutError,
    ISICDataset,
    enumerate_isic_pairs,
    export_split,
    isic_collate,
    letterbox,
    load_split,
    render_split,
)
from _support import make_dataset, write_pair  # noqa: E402


@unittest.skipIf(MISSING_DEPENDENCY is not None, MISSING_DEPENDENCY or "")
class PairingTests(unittest.TestCase):
    def test_pairs_are_found_and_sorted(self) -> None:
        with TemporaryRoot() as root:
            make_dataset(root, count=4, stems=["ISIC_0000010", "ISIC_0000002",
                                               "ISIC_0000007", "ISIC_0000001"])
            pairs = enumerate_isic_pairs(root / "images", root / "masks")
            self.assertEqual([p.stem for p in pairs],
                             ["ISIC_0000001", "ISIC_0000002", "ISIC_0000007", "ISIC_0000010"])
            self.assertTrue(all(p.has_mask for p in pairs))

    def test_missing_mask_raises(self) -> None:
        with TemporaryRoot() as root:
            make_dataset(root, count=3)
            (root / "masks" / "ISIC_0000002_Segmentation.png").unlink()
            with self.assertRaises(DataLayoutError) as ctx:
                enumerate_isic_pairs(root / "images", root / "masks")
            self.assertIn("缺少掩膜", str(ctx.exception))
            self.assertIn("ISIC_0000002", str(ctx.exception))

    def test_optional_masks_are_kept_when_present(self) -> None:
        with TemporaryRoot() as root:
            make_dataset(root, count=2)
            (root / "masks" / "ISIC_0000002_Segmentation.png").unlink()
            pairs = enumerate_isic_pairs(root / "images", root / "masks", require_mask=False)
            self.assertTrue(pairs[0].has_mask)
            self.assertFalse(pairs[1].has_mask)

    def test_orphan_mask_raises(self) -> None:
        with TemporaryRoot() as root:
            make_dataset(root, count=2)
            write_pair(root, "ISIC_0000099")
            (root / "images" / "ISIC_0000099.jpg").unlink()
            with self.assertRaises(DataLayoutError) as ctx:
                enumerate_isic_pairs(root / "images", root / "masks")
            self.assertIn("没有对应图像", str(ctx.exception))

    def test_duplicate_identifier_raises(self) -> None:
        with TemporaryRoot() as root:
            make_dataset(root, count=2)
            duplicate = root / "images" / "nested" / "ISIC_0000001.jpeg"
            duplicate.parent.mkdir(parents=True, exist_ok=True)
            duplicate.write_bytes((root / "images" / "ISIC_0000001.jpg").read_bytes())
            with self.assertRaises(DataLayoutError) as ctx:
                enumerate_isic_pairs(root / "images", root / "masks")
            self.assertIn("重复标识", str(ctx.exception))

    def test_error_message_names_the_offending_paths(self) -> None:
        with TemporaryRoot() as root:
            missing_dir = root / "does_not_exist"
            with self.assertRaises(DataLayoutError) as ctx:
                enumerate_isic_pairs(missing_dir, root / "masks")
            self.assertIn("does_not_exist", str(ctx.exception))


@unittest.skipIf(MISSING_DEPENDENCY is not None, MISSING_DEPENDENCY or "")
class LetterboxTests(unittest.TestCase):
    def test_valid_region_matches_letterbox_geometry(self) -> None:
        with TemporaryRoot() as root:
            pair = write_pair(root, "ISIC_0000001", width=100, height=50)
            for size in (32, 64, 256):
                image, mask, valid, info, was_binary = letterbox(pair.image, pair.mask, size=size)
                left, top = info.offset
                width, height = info.resized
                expected = np.zeros((size, size), dtype=np.uint8)
                expected[top:top + height, left:left + width] = 1
                np.testing.assert_array_equal(valid, expected)
                self.assertEqual(image.shape, (size, size, 3))
                self.assertEqual(mask.shape, (size, size))
                self.assertEqual(info.original_size, (100, 50))
                self.assertEqual(info.canvas, (size, size))
                self.assertTrue(was_binary)
                # 掩膜前景必须完全落在有效区域内：补边没有病灶。
                self.assertEqual(int((mask * (1 - valid)).sum()), 0)
                # 图像补边必须恰好为 0（任务要求补零）。
                self.assertEqual(int(image[valid == 0].sum()), 0)

    def test_aspect_ratio_is_preserved(self) -> None:
        with TemporaryRoot() as root:
            pair = write_pair(root, "ISIC_0000002", width=100, height=50)
            _, _, valid, info, _ = letterbox(pair.image, pair.mask, size=64)
            self.assertEqual(info.resized, (64, 32))
            self.assertEqual(info.offset, (0, 16))
            # 有效区域既不碰左右边（整宽），也不该上下满高。
            self.assertEqual(int(valid[:, 0].sum()), 32)
            self.assertEqual(int(valid[0, :].sum()), 0)

    def test_padding_value_applies_to_image_only(self) -> None:
        with TemporaryRoot() as root:
            pair = write_pair(root, "ISIC_0000003", width=64, height=32)
            image, _, valid, _, _ = letterbox(pair.image, pair.mask, size=64, pad_value=128)
            self.assertTrue(bool((image[valid == 0] == 128).all()))

    def test_mask_and_image_letterbox_stay_aligned(self) -> None:
        """把原图先水平翻转再 letterbox，掩膜前景应跟着翻到另一侧（最近邻、同一几何）。"""
        with TemporaryRoot() as root:
            pair = write_pair(root, "ISIC_0000004", width=96, height=48,
                              box=(0, 0, 24, 48))  # 前景贴左边
            _, mask_left, _, _, _ = letterbox(pair.image, pair.mask, size=64)
            flipped_image = root / "images" / "flipped.jpg"
            flipped_mask = root / "masks" / "flipped_Segmentation.png"
            with Image.open(pair.image) as handle:
                handle.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(flipped_image)
            with Image.open(pair.mask) as handle:
                handle.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(flipped_mask)
            _, mask_right, _, _, _ = letterbox(flipped_image, flipped_mask, size=64)
            midpoint = 64 // 2
            self.assertGreater(int(mask_left[:, :midpoint].sum()), 0)
            self.assertEqual(int(mask_left[:, midpoint:].sum()), 0)
            self.assertEqual(int(mask_right[:, :midpoint].sum()), 0)
            self.assertGreater(int(mask_right[:, midpoint:].sum()), 0)

    def test_mask_is_binarized(self) -> None:
        with TemporaryRoot() as root:
            # 前景 200（>=127 → 1），背景 30（<127 → 0），并如实报告"非严格二值"。
            pair = write_pair(root, "ISIC_0000005", width=32, height=32,
                              mask_values=(30, 200))
            _, mask, _, _, was_binary = letterbox(pair.image, pair.mask, size=32)
            self.assertFalse(was_binary)
            self.assertEqual(sorted(set(np.unique(mask).tolist())), [0, 1])
            self.assertGreater(int(mask.sum()), 0)

    def test_size_mismatch_raises(self) -> None:
        with TemporaryRoot() as root:
            pair = write_pair(root, "ISIC_0000006", width=64, height=64)
            other = write_pair(root, "ISIC_0000007", width=32, height=32)
            with self.assertRaises(DataLayoutError) as ctx:
                letterbox(pair.image, other.mask, size=64)
            self.assertIn("尺寸不一致", str(ctx.exception))

    def test_invalid_size_raises(self) -> None:
        with TemporaryRoot() as root:
            pair = write_pair(root, "ISIC_0000008")
            with self.assertRaises(DataLayoutError):
                letterbox(pair.image, pair.mask, size=0)


@unittest.skipIf(MISSING_DEPENDENCY is not None, MISSING_DEPENDENCY or "")
class SplitTests(unittest.TestCase):
    def _pairs(self, root: Path, count: int = 10):
        make_dataset(root, count=count)
        return enumerate_isic_pairs(root / "images", root / "masks")

    def test_same_seed_gives_identical_split(self) -> None:
        with TemporaryRoot() as root:
            pairs = self._pairs(root)
            reference = render_split(pairs, seed=7, val_fraction=0.2, stratify_by_area=False)
            for _ in range(3):
                again = render_split(pairs, seed=7, val_fraction=0.2, stratify_by_area=False)
                self.assertEqual(reference, again)

    def test_different_seed_changes_split(self) -> None:
        with TemporaryRoot() as root:
            pairs = self._pairs(root, count=20)
            first = render_split(pairs, seed=1, val_fraction=0.25, stratify_by_area=False)
            second = render_split(pairs, seed=2, val_fraction=0.25, stratify_by_area=False)
            self.assertNotEqual(first["val"], second["val"])

    def test_no_overlap_and_full_coverage(self) -> None:
        for stratify in (False, True):
            with self.subTest(stratify=stratify):
                with TemporaryRoot() as root:
                    pairs = self._pairs(root, count=9)
                    split = render_split(pairs, seed=3, val_fraction=0.22,
                                         stratify_by_area=stratify)
                    train, val = set(split["train"]), set(split["val"])
                    self.assertFalse(train & val)
                    self.assertEqual(train | val, {p.stem for p in pairs})
                    self.assertEqual(len(split["val"]), 2)

    def test_val_fraction_out_of_range_raises(self) -> None:
        with TemporaryRoot() as root:
            pairs = self._pairs(root, count=4)
            for bad in (0.0, 1.0, -0.1, 1.5):
                with self.subTest(val_fraction=bad):
                    with self.assertRaises(DataLayoutError):
                        render_split(pairs, seed=0, val_fraction=bad)

    def test_too_few_samples_raises(self) -> None:
        with TemporaryRoot() as root:
            pairs = self._pairs(root, count=1)
            with self.assertRaises(DataLayoutError):
                render_split(pairs, seed=0, val_fraction=0.5)

    def test_export_and_load_roundtrip(self) -> None:
        with TemporaryRoot() as root:
            pairs = self._pairs(root, count=8)
            split = render_split(pairs, seed=11, val_fraction=0.25, stratify_by_area=False)
            target = root / "out" / "split.json"
            export_split(split, target, seed=11, pairs=pairs)
            loaded = load_split(target)
            self.assertEqual(loaded["train"], sorted(split["train"]))
            self.assertEqual(loaded["val"], sorted(split["val"]))
            self.assertNotIn("test", loaded)

    def test_export_rejects_overlap(self) -> None:
        with TemporaryRoot() as root:
            pairs = self._pairs(root, count=4)
            stems = [p.stem for p in pairs]
            with self.assertRaises(DataLayoutError):
                export_split({"train": stems, "val": stems[:2]}, root / "s.json", seed=0)
            with self.assertRaises(DataLayoutError):
                export_split({"train": stems[:2], "val": stems[2:],
                              "test": stems[:1]}, root / "s.json", seed=0)

    def test_load_split_rejects_bad_payload(self) -> None:
        with TemporaryRoot() as root:
            bad = root / "bad.json"
            bad.write_text('{"splits": {"train": ["A"]}}', encoding="utf-8")
            with self.assertRaises(DataLayoutError):
                load_split(bad)
            bad.write_text('{"splits": {"train": ["A"], "val": ["B"], "test": ["A"]}}',
                           encoding="utf-8")
            with self.assertRaises(DataLayoutError):
                load_split(bad)
            with self.assertRaises(DataLayoutError):
                load_split(root / "missing.json")


@unittest.skipIf(MISSING_DEPENDENCY is not None, MISSING_DEPENDENCY or "")
class DatasetTests(unittest.TestCase):
    def test_item_shapes_and_dtypes(self) -> None:
        with TemporaryRoot() as root:
            make_dataset(root, count=2)
            pairs = enumerate_isic_pairs(root / "images", root / "masks")
            dataset = ISICDataset(pairs, root=root, size=64)
            item = dataset[0]
            self.assertEqual(tuple(item.image.shape), (3, 64, 64))
            self.assertEqual(item.image.dtype, torch.float32)
            self.assertLessEqual(float(item.image.max()), 1.0)
            self.assertGreaterEqual(float(item.image.min()), 0.0)
            self.assertEqual(tuple(item.mask.shape), (1, 64, 64))
            self.assertEqual(sorted(set(item.mask.unique().tolist())), [0.0, 1.0])
            self.assertEqual(tuple(item.valid.shape), (1, 64, 64))
            self.assertEqual(item.filename, "ISIC_0000001.jpg")
            self.assertEqual(item.original_size, (96, 64))
            self.assertTrue(item.mask_was_binary)

    def test_mask_only_where_valid(self) -> None:
        with TemporaryRoot() as root:
            make_dataset(root, count=2)
            pairs = enumerate_isic_pairs(root / "images", root / "masks")
            dataset = ISICDataset(pairs, root=root, size=48)
            for index in range(len(dataset)):
                item = dataset[index]
                self.assertEqual(float((item.mask * (1 - item.valid)).sum()), 0.0)

    def test_augment_flip_keeps_mask_alignment(self) -> None:
        with TemporaryRoot() as root:
            write_pair(root, "ISIC_0000001", width=64, height=64,
                       box=(0, 0, 16, 64))  # 前景贴左边，翻转后应对称
            pairs = enumerate_isic_pairs(root / "images", root / "masks")
            dataset = ISICDataset(pairs, root=root, size=64, augment=True)
            torch.manual_seed(0)
            any_flipped = False
            for _ in range(20):
                item = dataset[0]
                left = float(item.mask[..., :32].sum())
                right = float(item.mask[..., 32:].sum())
                if right > left:
                    any_flipped = True
                # 无论是否翻转，掩膜都只能落在有效区域内（图像与掩膜同一几何）。
                self.assertEqual(float((item.mask * (1 - item.valid)).sum()), 0.0)
                self.assertNotEqual(left + right, 0.0)
            self.assertTrue(any_flipped, "20 次采样都没有触发水平翻转，增强逻辑可疑")

    def test_collate_batches(self) -> None:
        with TemporaryRoot() as root:
            make_dataset(root, count=3)
            pairs = enumerate_isic_pairs(root / "images", root / "masks")
            dataset = ISICDataset(pairs, root=root, size=32)
            batch = isic_collate([dataset[0], dataset[1], dataset[2]])
            self.assertEqual(tuple(batch["image"].shape), (3, 3, 32, 32))
            self.assertEqual(tuple(batch["mask"].shape), (3, 1, 32, 32))
            self.assertEqual(tuple(batch["valid"].shape), (3, 1, 32, 32))
            self.assertEqual(len(batch["filename"]), 3)
            self.assertEqual(batch["filename"][0], "ISIC_0000001.jpg")

    def test_empty_dataset_raises(self) -> None:
        with self.assertRaises(DataLayoutError):
            ISICDataset([], size=32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
