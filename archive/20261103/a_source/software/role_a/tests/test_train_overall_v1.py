"""``train_overall_v1`` 的关键行为测试（合成小样本，不读真实数据、不训练真实模型）。

覆盖的验收点：

1. 主指标是三套开发集逐图 Dice 的平均（按图等权，而不是按来源等权），
   且只在五个全局统一阈值里选最优；
2. 面积分组按每套开发集自身的 q10/q50 切分，组内 n/Dice/IoU 为逐图等权平均；
3. 训练/开发 ID 重叠、排除清单含未知 ID、冻结 split/基线的哈希链被改写时立即报错；
4. 8/12 通道的静态估算与 torch 模型参数量一致，且限制在 (8, 12)；
5. 起点权重结构不兼容时抛 :class:`IncompatibleWeightsError`，绝不部分加载；
6. 合成小样本上跑通一轮训练，并验证"中断 + --resume"与不中断训练逐位一致；
7. 已完成/未完成/状态不明的输出目录被拒绝覆盖。

真实数据检查在数据缺失时**明确跳过并说明原因**，不假装通过。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import contextmanager, redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # 依赖缺失时整组跳过（而不是静默通过）
    import numpy as np
    import torch
    from PIL import Image

    from software.role_a import train_overall_v1 as overall
    from software.role_a.data import DataLayoutError, ISICDataset
    from software.role_a.model import build_model

    HAVE_DEPENDENCIES = True
except ImportError as error:  # pragma: no cover
    IMPORT_ERROR = str(error)
    HAVE_DEPENDENCIES = False
else:
    IMPORT_ERROR = ""

SKIP_REASON = ("缺少 torch/numpy/Pillow 或本地模块，跳过（不算通过）: " + IMPORT_ERROR
               if not HAVE_DEPENDENCIES else "")

SIZE = 16
PIXELS = SIZE * SIZE

DEV_2016_IDS = ["ISIC_0001001", "ISIC_0001002", "ISIC_0001003", "ISIC_0001004"]
DEV_2017VAL_IDS = ["ISIC_0002001", "ISIC_0002002", "ISIC_0002003", "ISIC_0002004"]
TRAIN_2016_IDS = [f"ISIC_000000{index}" for index in range(1, 7)]
TRAIN_2017NEW_IDS = [f"ISIC_001200{index}" for index in range(1, 6)]
EXPOSED_IDS = [f"ISIC_0013{index:03d}" for index in range(1, 11)]
EXPOSED_TRAIN_IDS = EXPOSED_IDS[2:7]
EXPOSED_VAL_IDS = EXPOSED_IDS[7:10]
EXPOSED_CLEAN_EXCLUDED = EXPOSED_IDS[0:2]

DEV_2016_COUNTS = dict(zip(DEV_2016_IDS, [26, 77, 154, 205]))
DEV_2017VAL_COUNTS = dict(zip(DEV_2017VAL_IDS, [13, 64, 141, 217]))
EXPOSED_COUNTS = dict(zip(EXPOSED_IDS, [8, 20, 33, 46, 59, 72, 85, 120, 160, 210]))
TRAIN_2016_COUNTS = dict(zip(TRAIN_2016_IDS, [40, 60, 80, 100, 120, 140]))
TRAIN_2017NEW_COUNTS = dict(zip(TRAIN_2017NEW_IDS, [30, 70, 110, 150, 190]))

TRAIN_TOTAL = 14
DEV_TOTAL = 9
EXPOSED_CLEAN_TOTAL = 8


# --------------------------------------------------------------- 合成小样本
def _write_pair(folder: Path, stem: str, count: int, seed: int) -> float:
    """写出一对合成 ISIC 文件，掩膜前景像素数严格等于 ``count``。"""
    images = folder / "images"
    masks = folder / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rgb = rng.integers(0, 256, size=(SIZE, SIZE, 3), dtype=np.uint8)
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    flat = mask.reshape(-1)
    flat[: max(1, min(PIXELS, int(count)))] = 255
    Image.fromarray(rgb, mode="RGB").save(images / f"{stem}.jpg", format="JPEG", quality=95)
    Image.fromarray(mask, mode="L").save(masks / f"{stem}_Segmentation.png", format="PNG")
    return float(count / PIXELS)


def _silent(function, *args, **kwargs):
    """吞掉训练脚本的正常打印，便于断言；异常仍会抛出。"""
    buffer = StringIO()
    with redirect_stdout(buffer):
        return function(*args, **kwargs)


@contextmanager
def overall_temp():
    """合成项目用的临时目录（退出即清理，不写项目内路径）。"""
    with tempfile.TemporaryDirectory(prefix="role_a_overall_v1_") as name:
        yield Path(name)


class SyntheticProject:
    """与真实项目结构一致、但只有 23 张图的小项目，用于跑通全部校验与一轮训练。"""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.s2016train = self.root / "s2016train"
        self.s2017new = self.root / "s2017new"
        self.exposed = self.root / "exposed2017"
        self.d2016 = self.root / "d2016"
        self.d2017val = self.root / "d2017val"
        self.overlap_exclusions = self.root / "overlap_exclusions.json"
        self.exposed_exclusions = self.root / "exposed_exclusions.json"
        self.exposed_split = self.root / "split.json"
        self.development_baseline = self.root / "development_baseline.json"

        self._write_pairs(self.s2016train, TRAIN_2016_COUNTS)
        self._write_pairs(self.s2017new, TRAIN_2017NEW_COUNTS)
        self._write_pairs(self.exposed, EXPOSED_COUNTS)
        self._write_pairs(self.d2016, DEV_2016_COUNTS)
        self._write_pairs(self.d2017val, DEV_2017VAL_COUNTS)
        self._write_provenance(self.s2016train, TRAIN_2016_COUNTS, with_items=False)
        self._write_provenance(self.d2016, DEV_2016_COUNTS, with_items=False)
        self._write_provenance(self.s2017new, TRAIN_2017NEW_COUNTS, with_items=True)
        self._write_provenance(self.d2017val, DEV_2017VAL_COUNTS, with_items=True)
        self._write_provenance(self.exposed, EXPOSED_COUNTS, with_items=True)
        self.overlap_exclusions.write_text(json.dumps({
            "schema": "synthetic-overlap-v1",
            "external_validation": {DEV_2017VAL_IDS[0]: "synthetic"},
            "supplemental_training": {TRAIN_2017NEW_IDS[0]: "synthetic"},
            "isic2016_training": {TRAIN_2016_IDS[0]: "synthetic"},
            "official_test_clean_subset_exclusions": {DEV_2016_IDS[0]: "synthetic"},
        }, indent=2), encoding="utf-8")
        self.exposed_exclusions.write_text(json.dumps({
            "schema": "synthetic-exposed-exclusions-v1",
            "clean_exclusions": list(EXPOSED_CLEAN_EXCLUDED),
        }, indent=2), encoding="utf-8")
        self.write_split()
        self.write_development_baseline()

    # ------------------------------------------------------------- 构造磁盘
    def _write_pairs(self, folder: Path, counts: dict[str, int]) -> None:
        for index, stem in enumerate(sorted(counts)):
            _write_pair(folder, stem, counts[stem], seed=index + 1)

    def _write_provenance(self, folder: Path, counts: dict[str, int],
                          *, with_items: bool) -> None:
        payload: dict[str, object] = {"source": "synthetic-role-a-overall-v1",
                                      "complete": True, "image_count": len(counts)}
        if with_items:
            items = {}
            for stem in sorted(counts):
                items[stem] = {
                    "area_ratio": float(counts[stem] / PIXELS),
                    "image_sha256": overall.sha256_file(folder / "images" / f"{stem}.jpg"),
                    "mask_sha256": overall.sha256_file(
                        folder / "masks" / f"{stem}_Segmentation.png"),
                }
            payload["items"] = items
        (folder / "provenance.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_split(self, train_ids: list[str] | None = None,
                    validation_ids: list[str] | None = None,
                    *, tamper_provenance_hash: bool = False) -> dict[str, object]:
        payload = {
            "schema": "synthetic-exposed-split-v1",
            "source_provenance_sha256": ("0" * 64 if tamper_provenance_hash
                                         else overall.sha256_file(
                                             self.exposed / "provenance.json")),
            "source_overlap_exclusions_sha256": overall.sha256_file(self.exposed_exclusions),
            "train_ids": list(EXPOSED_TRAIN_IDS if train_ids is None else train_ids),
            "validation_ids": list(EXPOSED_VAL_IDS if validation_ids is None else validation_ids),
        }
        self.exposed_split.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def write_development_baseline(self, *, count: int | None = None,
                                   area_delta: float = 0.0,
                                   baseline_sha256: str | None = None) -> None:
        split = json.loads(self.exposed_split.read_text(encoding="utf-8"))
        validation_ids = list(split["validation_ids"])
        areas = {stem: float(EXPOSED_COUNTS[stem] / PIXELS) + float(area_delta)
                 for stem in validation_ids}
        cuts = overall.area_cuts(areas)
        records = [{"stem": stem, "area_ratio": areas[stem],
                    "band": overall.band_of(areas[stem], cuts)}
                   for stem in sorted(areas)]
        self.development_baseline.write_text(json.dumps({
            "split_sha256": overall.sha256_file(self.exposed_split),
            "baseline_sha256": baseline_sha256,
            "count": len(validation_ids) if count is None else int(count),
            "records": records,
        }, indent=2), encoding="utf-8")

    # --------------------------------------------------------------- 变体
    def copy_pair(self, source: Path, stem: str, target: Path) -> None:
        for folder, name in (("images", f"{stem}.jpg"),
                             ("masks", f"{stem}_Segmentation.png")):
            (target / folder).mkdir(parents=True, exist_ok=True)
            (target / folder / name).write_bytes((source / folder / name).read_bytes())

    def add_bogus_exclusion(self, key: str = "supplemental_training",
                            stem: str = "ISIC_9999999") -> None:
        payload = json.loads(self.overlap_exclusions.read_text(encoding="utf-8"))
        payload[key][stem] = "synthetic bogus id"
        self.overlap_exclusions.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def tamper_image_hash(self) -> None:
        path = self.exposed / "provenance.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        first = sorted(payload["items"])[0]
        payload["items"][first]["image_sha256"] = "0" * 64
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # --------------------------------------------------------------- 规格
    def spec(self, *, baseline_checkpoint: Path | None = None,
             development_baseline: Path | None = None) -> "overall.DataSpec":
        return overall.DataSpec(
            train_sources=(
                overall.SourceSpec(name="isic2016_train", role="train",
                                   image_dir=self.s2016train / "images",
                                   mask_dir=self.s2016train / "masks",
                                   provenance=self.s2016train / "provenance.json",
                                   provenance_count=None, expected_raw=6,
                                   exclusions_key="isic2016_training", expected_kept=5),
                overall.SourceSpec(name="isic2017_new_train", role="train",
                                   image_dir=self.s2017new / "images",
                                   mask_dir=self.s2017new / "masks",
                                   provenance=self.s2017new / "provenance.json",
                                   provenance_count=5, expected_raw=5,
                                   exclusions_key="supplemental_training", expected_kept=4),
                overall.SourceSpec(name="isic2017_exposed_train", role="train",
                                   image_dir=self.exposed / "images",
                                   mask_dir=self.exposed / "masks",
                                   provenance=self.exposed / "provenance.json",
                                   provenance_count=10, expected_raw=10,
                                   exclusions_key=None, expected_kept=5,
                                   id_list_key="train_ids"),
            ),
            dev_sources=(
                overall.SourceSpec(name="isic2016_test", role="dev",
                                   image_dir=self.d2016 / "images",
                                   mask_dir=self.d2016 / "masks",
                                   provenance=self.d2016 / "provenance.json",
                                   provenance_count=None, expected_raw=4,
                                   exclusions_key="official_test_clean_subset_exclusions",
                                   expected_kept=3),
                overall.SourceSpec(name="isic2017_validation", role="dev",
                                   image_dir=self.d2017val / "images",
                                   mask_dir=self.d2017val / "masks",
                                   provenance=self.d2017val / "provenance.json",
                                   provenance_count=4, expected_raw=4,
                                   exclusions_key="external_validation", expected_kept=3),
                overall.SourceSpec(name="isic2017_exposed_validation", role="dev",
                                   image_dir=self.exposed / "images",
                                   mask_dir=self.exposed / "masks",
                                   provenance=self.exposed / "provenance.json",
                                   provenance_count=10, expected_raw=10,
                                   exclusions_key=None, expected_kept=3,
                                   id_list_key="validation_ids"),
            ),
            overlap_exclusions=self.overlap_exclusions,
            exposed_split=self.exposed_split,
            exposed_exclusions=self.exposed_exclusions,
            development_baseline=(self.development_baseline if development_baseline is None
                                  else Path(development_baseline)),
            baseline_checkpoint=None if baseline_checkpoint is None else Path(baseline_checkpoint),
            expected_train_total=TRAIN_TOTAL,
            expected_dev_total=DEV_TOTAL,
            expected_exposed_clean=EXPOSED_CLEAN_TOTAL,
        )


def _config(project: SyntheticProject, output_dir: Path, **overrides) -> "overall.RunConfig":
    values: dict[str, object] = {
        "output_dir": Path(output_dir),
        "init": "scratch",
        "base_channels": 8,
        "num_pools": 3,
        "size": SIZE,
        "epochs": 1,
        "batch_size": 3,
        "lr": 0.01,
        "seed": 7,
        "device": "cpu",
        "num_workers": 0,
        "data_spec": project.spec(),
        "augment": True,
        "resume": False,
        "verify_images": False,
    }
    values.update(overrides)
    return overall.RunConfig(**values)  # type: ignore[arg-type]


def _fake_records(source: str, count: int, per_threshold: list[float],
                  band: str = "large") -> list[dict[str, object]]:
    return [{
        "source": source,
        "stem": f"{source}_{index:05d}",
        "area_ratio": 0.5,
        "band": band,
        "dice": [float(value) for value in per_threshold],
        "iou": [float(value) - 0.1 for value in per_threshold],
        "empty_prediction": [False] * len(per_threshold),
    } for index in range(count)]


# ------------------------------------------------------------------- 指标
@unittest.skipUnless(HAVE_DEPENDENCIES, SKIP_REASON)
class MetricTests(unittest.TestCase):
    def test_primary_metric_is_638_image_mean_and_drives_threshold(self) -> None:
        records = (_fake_records("isic2016_test", 376, [0.70, 0.60, 0.75, 0.50, 0.50])
                   + _fake_records("isic2017_validation", 145, [0.70, 0.90, 0.70, 0.50, 0.50])
                   + _fake_records("isic2017_exposed_validation", 117,
                                   [0.70, 0.90, 0.70, 0.50, 0.50]))
        self.assertEqual(len(records), 638)
        table = overall.threshold_table(records)
        self.assertEqual(table[0.45]["n"], 638)
        self.assertAlmostEqual(
            table[0.53]["dice"], (376 * 0.75 + 145 * 0.70 + 117 * 0.70) / 638, places=12)
        self.assertAlmostEqual(
            table[0.50]["dice"], (376 * 0.60 + 145 * 0.90 + 117 * 0.90) / 638, places=12)
        # 主指标按图等权 => 0.53 最优
        self.assertEqual(overall.select_threshold(table), 0.53)
        # 若按"来源平均"排序则会选 0.50，说明主指标确实不是按来源等权
        balanced_best = max(overall.THRESHOLDS,
                            key=lambda t: table[t]["source_dice_balanced"])
        self.assertEqual(balanced_best, 0.50)

    def test_tie_break_prefers_source_balanced_then_smaller_threshold(self) -> None:
        heavy = 376
        middle = 145
        exposed = 117
        # 让 0.50 的逐图平均与 0.70 相等，但三来源均值更高
        heavy_value = (0.70 * 638 - (middle + exposed) * 0.95) / heavy
        records = (_fake_records("isic2016_test", heavy, [0.70, heavy_value, 0.70, 0.70, 0.70])
                   + _fake_records("isic2017_validation", middle, [0.70, 0.95, 0.70, 0.70, 0.70])
                   + _fake_records("isic2017_exposed_validation", exposed,
                                   [0.70, 0.95, 0.70, 0.70, 0.70]))
        table = overall.threshold_table(records)
        for threshold in overall.THRESHOLDS:
            self.assertAlmostEqual(table[threshold]["dice"], 0.70, places=6)
        self.assertGreater(table[0.50]["source_dice_balanced"], 0.70)
        self.assertEqual(overall.select_threshold(table), 0.50)

        flat = overall.threshold_table(
            _fake_records("isic2016_test", 376, [0.70] * 5)
            + _fake_records("isic2017_validation", 145, [0.70] * 5)
            + _fake_records("isic2017_exposed_validation", 117, [0.70] * 5))
        self.assertEqual(overall.select_threshold(flat), 0.45)

    def test_area_bands_follow_each_source_own_quantiles(self) -> None:
        areas = {f"S{index}": value for index, value in enumerate(
            [0.02, 0.05, 0.09, 0.12, 0.20, 0.31, 0.45, 0.55, 0.62, 0.80])}
        cuts = overall.area_cuts(areas)
        bands = [overall.band_of(value, cuts) for value in areas.values()]
        self.assertEqual(sum(band == "tiny" for band in bands), 1)
        self.assertEqual(sum(band == "medium" for band in bands), 4)
        self.assertEqual(sum(band == "large" for band in bands), 5)

        records = [{
            "source": "s", "stem": stem, "area_ratio": areas[stem],
            "band": overall.band_of(areas[stem], cuts),
            "dice": [areas[stem]] * 5, "iou": [areas[stem]] * 5,
            "empty_prediction": [False] * 5,
        } for stem in sorted(areas)]
        summary = overall.summarize_records(records, 0)
        self.assertEqual(summary["tiny"]["n"], 1)
        self.assertEqual(summary["medium"]["n"], 4)
        # 逐图等权平均 => (0.05+0.09+0.12+0.20)/4；按面积加权会得到 0.1413…
        self.assertAlmostEqual(summary["medium"]["dice"], (0.05 + 0.09 + 0.12 + 0.20) / 4,
                               places=12)

    def test_report_keeps_per_source_cuts_and_counts(self) -> None:
        dev_a = overall.DevSet(name="a", items=(), areas={"a1": 0.05, "a2": 0.60},
                               cuts=(0.10, 0.50))
        dev_b = overall.DevSet(name="b", items=(), areas={"b1": 0.90}, cuts=(0.80, 0.95))
        records = [
            {"source": "a", "stem": "a1", "area_ratio": 0.05, "band": "tiny",
             "dice": [0.2] * 5, "iou": [0.1] * 5, "empty_prediction": [False] * 5},
            {"source": "a", "stem": "a2", "area_ratio": 0.60, "band": "large",
             "dice": [0.4] * 5, "iou": [0.3] * 5, "empty_prediction": [True] * 5},
            {"source": "b", "stem": "b1", "area_ratio": 0.90, "band": "medium",
             "dice": [0.9] * 5, "iou": [0.8] * 5, "empty_prediction": [False] * 5},
        ]
        report = overall.build_development_report(records, {"a": dev_a, "b": dev_b})
        self.assertEqual(report["primary"]["n"], 3)
        self.assertEqual(report["sources"]["a"]["area_cuts"], {"q10": 0.10, "q50": 0.50})
        self.assertEqual(report["sources"]["b"]["area_cuts"], {"q10": 0.80, "q50": 0.95})
        self.assertEqual(report["sources"]["a"]["all"]["n"], 2)
        self.assertEqual(report["sources"]["a"]["tiny"]["n"], 1)
        self.assertEqual(report["sources"]["b"]["medium"]["n"], 1)
        self.assertEqual(report["area_groups_pooled"]["tiny"]["n"], 1)
        self.assertAlmostEqual(report["sources"]["a"]["all"]["dice"], 0.3, places=12)


# ------------------------------------------------------------- 数据装配
@unittest.skipUnless(HAVE_DEPENDENCIES, SKIP_REASON)
class AssemblyTests(unittest.TestCase):
    def test_synthetic_spec_assembles_with_expected_isolation(self) -> None:
        with overall_temp() as root:
            project = SyntheticProject(root / "proj")
            assembly = overall.assemble(project.spec())
            self.assertEqual(len(assembly.train_items), TRAIN_TOTAL)
            self.assertEqual(sum(len(dev.items) for dev in assembly.dev_sets.values()), DEV_TOTAL)
            self.assertEqual(assembly.counts["train_total"], TRAIN_TOTAL)
            self.assertEqual(assembly.counts["development_total"], DEV_TOTAL)
            self.assertEqual(assembly.counts["exposed_clean_total"], EXPOSED_CLEAN_TOTAL)
            self.assertFalse(set(assembly.train_ids) & set(assembly.dev_ids["isic2016_test"]))
            self.assertEqual(len(assembly.dev_ids["isic2016_test"]), 3)
            self.assertEqual(len(assembly.dev_ids["isic2017_validation"]), 3)
            self.assertEqual(len(assembly.dev_ids["isic2017_exposed_validation"]), 3)
            # 面积分组来自每套开发集自身，且三组都非空
            for dev in assembly.dev_sets.values():
                bands = {overall.band_of(area, dev.cuts) for area in dev.areas.values()}
                self.assertEqual(bands, {"tiny", "medium", "large"})

    def test_training_development_overlap_is_rejected(self) -> None:
        with overall_temp() as root:
            project = SyntheticProject(root / "proj")
            project.copy_pair(project.d2016, DEV_2016_IDS[1], project.s2016train)
            spec = project.spec()
            spec = replace(
                spec,
                train_sources=(replace(spec.train_sources[0], expected_raw=7, expected_kept=6),
                               *spec.train_sources[1:]),
                expected_train_total=TRAIN_TOTAL + 1,
            )
            with self.assertRaises(overall.AssemblyError) as context:
                overall.assemble(spec)
            self.assertIn("重叠", str(context.exception))

    def test_frozen_split_train_and_validation_must_stay_disjoint(self) -> None:
        with overall_temp() as root:
            project = SyntheticProject(root / "proj")
            # 让 3 个训练 ID 同时出现在 validation_ids 里（数量仍为 3，先触发重叠检查）
            project.write_split(validation_ids=EXPOSED_TRAIN_IDS[:3])
            project.write_development_baseline()
            with self.assertRaises(overall.AssemblyError) as context:
                overall.assemble(project.spec())
            message = str(context.exception)
            self.assertTrue("重叠" in message or "干净暴露集合" in message, message)

    def test_unknown_exclusion_id_is_rejected(self) -> None:
        with overall_temp() as root:
            project = SyntheticProject(root / "proj")
            project.add_bogus_exclusion()
            with self.assertRaises(overall.AssemblyError) as context:
                overall.assemble(project.spec())
            self.assertIn("不存在", str(context.exception))

    def test_source_count_mismatch_is_rejected(self) -> None:
        with overall_temp() as root:
            project = SyntheticProject(root / "proj")
            spec = project.spec()
            spec = replace(spec, dev_sources=(replace(spec.dev_sources[0], expected_raw=99),
                                              *spec.dev_sources[1:]))
            with self.assertRaises(overall.AssemblyError):
                overall.assemble(spec)

    def test_missing_mask_file_is_reported(self) -> None:
        with overall_temp() as root:
            project = SyntheticProject(root / "proj")
            (project.d2016 / "masks" / f"{DEV_2016_IDS[2]}_Segmentation.png").unlink()
            with self.assertRaises(DataLayoutError):
                overall.assemble(project.spec())

    def test_frozen_split_hash_chain_is_enforced(self) -> None:
        with overall_temp() as root:
            project = SyntheticProject(root / "proj")
            project.write_split(tamper_provenance_hash=True)
            project.write_development_baseline()
            with self.assertRaises(overall.AssemblyError) as context:
                overall.assemble(project.spec())
            self.assertIn("source_provenance_sha256", str(context.exception))

    def test_frozen_development_baseline_is_enforced(self) -> None:
        with overall_temp() as root:
            project = SyntheticProject(root / "proj")
            project.write_development_baseline(count=4)
            with self.assertRaises(overall.AssemblyError) as context:
                overall.assemble(project.spec())
            self.assertIn("count", str(context.exception))
            project.write_development_baseline(area_delta=0.05)
            with self.assertRaises(overall.AssemblyError) as context:
                overall.assemble(project.spec())
            self.assertIn("面积", str(context.exception))

    def test_image_level_sha256_check_is_live(self) -> None:
        with overall_temp() as root:
            project = SyntheticProject(root / "proj")
            overall.assemble(project.spec(), verify_images=True)
            project.tamper_image_hash()
            project.write_split()
            project.write_development_baseline()
            overall.assemble(project.spec(), verify_images=False)
            with self.assertRaises(overall.AssemblyError) as context:
                overall.assemble(project.spec(), verify_images=True)
            self.assertIn("SHA256", str(context.exception))

    def test_default_spec_matches_frozen_protocol(self) -> None:
        spec = overall.build_default_spec()
        self.assertEqual(spec.expected_train_total, 2309)
        self.assertEqual(spec.expected_dev_total, 638)
        self.assertEqual(spec.expected_exposed_clean, 583)
        self.assertEqual(overall.THRESHOLDS, (0.45, 0.50, 0.53, 0.56, 0.60))
        expected = {
            "isic2016_train": (900, 897),
            "isic2017_new_train": (949, 946),
            "isic2017_exposed_train": (600, 466),
            "isic2016_test": (379, 376),
            "isic2017_validation": (147, 145),
            "isic2017_exposed_validation": (600, 117),
        }
        found = {source.name: (source.expected_raw, source.expected_kept)
                 for source in (*spec.train_sources, *spec.dev_sources)}
        self.assertEqual(found, expected)
        self.assertEqual(sum(source.expected_kept for source in spec.train_sources), 2309)
        self.assertEqual(sum(source.expected_kept for source in spec.dev_sources), 638)
        self.assertIsNotNone(spec.development_baseline)
        self.assertIsNotNone(spec.baseline_checkpoint)
        # 2018 留出集绝不能出现在任何路径里
        self.assertNotIn("2018", json.dumps(overall.spec_payload(spec)))


# ------------------------------------------------------------------- 模型
@unittest.skipUnless(HAVE_DEPENDENCIES, SKIP_REASON)
class ModelBudgetTests(unittest.TestCase):
    def test_estimates_match_protocol_budget_and_torch(self) -> None:
        expected = {
            8: (122113, 675282944, 1572864),
            12: (274321, 1508376576, 2359296),
        }
        for channels, (parameters, macs, peak_bytes) in expected.items():
            estimates = overall.estimate_model(channels, 3, 256)
            self.assertEqual(estimates["parameters"], parameters)
            self.assertEqual(estimates["conv_macs"], macs)
            self.assertEqual(estimates["peak_feature_bytes_int8"], peak_bytes)
            model = build_model(base_channels=channels, num_pools=3,
                                input_shape=(3, 256, 256))
            self.assertEqual(model.num_parameters(), parameters)
            self.assertTrue(overall.describe_graph(model)["allowed_only"])

    def test_base_channels_are_restricted_to_eight_or_twelve(self) -> None:
        self.assertEqual(overall.ALLOWED_BASE_CHANNELS, (8, 12))
        with overall_temp() as root:
            project = SyntheticProject(root / "proj")
            config = _config(project, root / "run", base_channels=10)
            with self.assertRaises(overall.ConfigError):
                overall.train(config)

    def test_incompatible_baseline_weights_are_reported(self) -> None:
        with overall_temp() as root:
            project = SyntheticProject(root / "proj")
            run = root / "seed_run"
            _silent(overall.train, _config(project, run, epochs=1))
            checkpoint_path = run / "best.pt"
            self.assertTrue(checkpoint_path.is_file())

            # 8 通道可以严格加载
            model = build_model(base_channels=8, num_pools=3, input_shape=(3, SIZE, SIZE))
            info = overall.load_baseline_weights(checkpoint_path, model, base_channels=8,
                                                 num_pools=3, size=SIZE)
            self.assertEqual(info["source_base_channels"], 8)

            # 12 通道必须报错，且不能部分加载
            wide = build_model(base_channels=12, num_pools=3, input_shape=(3, SIZE, SIZE))
            with self.assertRaises(overall.IncompatibleWeightsError):
                overall.load_baseline_weights(checkpoint_path, wide, base_channels=12,
                                             num_pools=3, size=SIZE)

            # config 声明 8 通道但权重是 12 通道：结构核对通过，strict 载入必须失败
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            payload["model_state"] = wide.state_dict()
            broken = root / "broken_baseline.pt"
            torch.save(payload, broken)
            target = build_model(base_channels=8, num_pools=3, input_shape=(3, SIZE, SIZE))
            with self.assertRaises(overall.IncompatibleWeightsError):
                overall.load_baseline_weights(broken, target, base_channels=8, num_pools=3,
                                             size=SIZE)


# ------------------------------------------------------------------- 增强
@unittest.skipUnless(HAVE_DEPENDENCIES, SKIP_REASON)
class AugmentationTests(unittest.TestCase):
    def test_geometry_keeps_image_mask_valid_aligned(self) -> None:
        import random as python_random

        mask = np.zeros((5, 7), dtype=np.uint8)
        mask[1:3, 2:5] = 1
        valid = np.zeros((5, 7), dtype=np.uint8)
        valid[1:4, 1:6] = 1
        image = np.repeat((mask * 255).astype(np.uint8)[:, :, None], 3, axis=2)
        flips = 0
        for seed in range(60):
            out_image, out_mask, out_valid = overall.augment_geometry(
                image, mask.copy(), valid.copy(), python_random.Random(seed))
            self.assertEqual(out_image.shape, image.shape)
            self.assertTrue(np.array_equal(out_image[:, :, 0] > 127, out_mask > 0))
            self.assertEqual(int(((out_mask > 0) & (out_valid == 0)).sum()), 0)
            if not np.array_equal(out_mask, mask):
                flips += 1
        self.assertGreater(flips, 0)

    def test_photometry_only_changes_image(self) -> None:
        import random as python_random

        image = np.full((4, 4, 3), 100, dtype=np.uint8)
        mask = np.ones((4, 4), dtype=np.uint8)
        valid = np.ones((4, 4), dtype=np.uint8)
        seen_different = False
        for seed in range(40):
            rng = python_random.Random(seed)
            out_image, out_mask, out_valid = overall.augment_arrays(
                image.copy(), mask.copy(), valid.copy(), rng)
            self.assertEqual(out_image.dtype, np.uint8)
            self.assertTrue(np.array_equal(out_mask, mask))
            self.assertTrue(np.array_equal(out_valid, valid))
            if not np.array_equal(out_image, image):
                seen_different = True
        self.assertTrue(seen_different)

    def test_train_dataset_without_augment_matches_original_preprocessing(self) -> None:
        with overall_temp() as root:
            project = SyntheticProject(root / "proj")
            items = list(overall.enumerate_isic_pairs(project.d2016 / "images",
                                                      project.d2016 / "masks", require_mask=True))
            mine = overall.TrainDataset(items, size=SIZE, seed=3, augment=False)
            reference = ISICDataset(items, size=SIZE, augment=False)
            for index in range(len(items)):
                left, right = mine[index], reference[index]
                self.assertTrue(torch.equal(left.image, right.image))
                self.assertTrue(torch.equal(left.mask, right.mask))
                self.assertTrue(torch.equal(left.valid, right.valid))

    def test_augmentation_is_reproducible_per_seed_epoch_and_index(self) -> None:
        with overall_temp() as root:
            project = SyntheticProject(root / "proj")
            items = list(overall.enumerate_isic_pairs(project.d2016 / "images",
                                                      project.d2016 / "masks", require_mask=True))
            plain = [overall.TrainDataset(items, size=SIZE, augment=False)[index].image
                     for index in range(len(items))]
            dataset = overall.TrainDataset(items, size=SIZE, seed=11, augment=True, epoch=3)
            first = [dataset[index].image.clone() for index in range(len(items))]
            second = [dataset[index].image.clone() for index in range(len(items))]
            for left, right in zip(first, second):
                self.assertTrue(torch.equal(left, right))
            changed = False
            for epoch in range(1, 9):
                dataset.set_epoch(epoch)
                if any(not torch.equal(dataset[index].image, plain[index])
                       for index in range(len(items))):
                    changed = True
            self.assertTrue(changed)


# ----------------------------------------------------------------- 端到端
@unittest.skipUnless(HAVE_DEPENDENCIES, SKIP_REASON)
class TrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._threads = torch.get_num_threads()
        torch.set_num_threads(1)

    def tearDown(self) -> None:
        torch.set_num_threads(self._threads)

    def test_finetune_uses_explicit_checkpoint_and_records_its_hash(self) -> None:
        with overall_temp() as root:
            project = SyntheticProject(root / "proj")
            first = root / "first"
            second = root / "second"
            _silent(overall.train, _config(project, first, epochs=1))
            source = first / overall.BEST_NAME
            config = _config(project, second, init="checkpoint", init_checkpoint=source,
                             epochs=1, lr=0.001)
            _silent(overall.train, config)
            protocol = json.loads((second / overall.PROTOCOL_NAME).read_text(encoding="utf-8"))
            expected = overall.sha256_file(source)
            self.assertEqual(protocol["source_checkpoint"]["source_checkpoint_sha256"], expected)
            self.assertEqual(protocol["input_hashes"]["init_checkpoint"], expected)
            self.assertEqual(protocol["config"]["init_checkpoint"], source.as_posix())
            with self.assertRaises(overall.ConfigError):
                _silent(overall.train, _config(project, root / "invalid", init="checkpoint"))

    def test_one_epoch_writes_protocol_checkpoints_and_evidence(self) -> None:
        with overall_temp() as root:
            project = SyntheticProject(root / "proj")
            run = root / "run"
            result = _silent(overall.train, _config(project, run, epochs=1))
            self.assertEqual(result["status"], "completed")
            for name in (overall.PROTOCOL_NAME, overall.SPLIT_NAME, overall.CHECKPOINT_NAME,
                         overall.BEST_NAME, overall.HISTORY_NAME, overall.DEV_EVIDENCE_NAME,
                         overall.COMPLETION_NAME):
                self.assertTrue((run / name).is_file(), f"缺少产物 {name}")

            protocol = json.loads((run / overall.PROTOCOL_NAME).read_text(encoding="utf-8"))
            self.assertEqual(protocol["counts"]["train_total"], TRAIN_TOTAL)
            self.assertEqual(protocol["counts"]["development_total"], DEV_TOTAL)
            self.assertEqual(protocol["thresholds"], list(overall.THRESHOLDS))
            self.assertEqual(protocol["optimizer"]["lr"], 0.01)
            self.assertEqual(protocol["optimizer"]["name"], "torch.optim.Adam")
            self.assertEqual(protocol["model"]["parameters"],
                             build_model(base_channels=8, num_pools=3,
                                         input_shape=(3, SIZE, SIZE)).num_parameters())
            self.assertEqual(protocol["model"]["conv_macs"],
                             overall.estimate_model(8, 3, SIZE)["conv_macs"])
            self.assertIn("train_ids", protocol["split_digest"])
            self.assertEqual(protocol["input_hashes"]["exposed_split"],
                             overall.sha256_file(project.exposed_split))
            json.dumps(protocol, allow_nan=False)

            split_used = json.loads((run / overall.SPLIT_NAME).read_text(encoding="utf-8"))
            self.assertEqual(len(split_used["train_ids"]), TRAIN_TOTAL)
            self.assertEqual(sum(len(ids) for ids in split_used["development_ids"].values()),
                             DEV_TOTAL)
            self.assertFalse(set(split_used["train_ids"]) &
                             {stem for ids in split_used["development_ids"].values()
                              for stem in ids})

            checkpoint = torch.load(run / overall.CHECKPOINT_NAME, map_location="cpu",
                                    weights_only=True)
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["phase"], "evaluated")
            self.assertEqual(checkpoint["schema"], "role-a-overall-v1-checkpoint-v1")
            self.assertIn("optimizer_state", checkpoint)
            self.assertIn("rng", checkpoint)
            self.assertEqual(checkpoint["config_hash"], protocol["config_hash"])

            best = torch.load(run / overall.BEST_NAME, map_location="cpu", weights_only=True)
            self.assertEqual(best["status"], "candidate")
            self.assertFalse(best["completed"])
            self.assertIn(best["selected_threshold"], overall.THRESHOLDS)
            self.assertEqual(best["development_images"], DEV_TOTAL)
            self.assertEqual(sorted(best["sources"]), sorted(split_used["development_ids"]))
            for name, info in best["sources"].items():
                for band in overall.SUMMARY_BANDS:
                    self.assertIn(band, info, f"{name}/{band}")
                self.assertEqual(info["all"]["n"], len(split_used["development_ids"][name]))
                self.assertTrue(all(info[band]["n"] > 0 for band in overall.BAND_NAMES))

            history = json.loads((run / overall.HISTORY_NAME).read_text(encoding="utf-8"))
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["epoch"], 1)

            evidence = json.loads((run / overall.DEV_EVIDENCE_NAME).read_text(encoding="utf-8"))
            self.assertEqual(len(evidence["records"]), DEV_TOTAL)
            self.assertIn("band", evidence["records"][0])

            completion = json.loads((run / overall.COMPLETION_NAME).read_text(encoding="utf-8"))
            self.assertEqual(completion["completed_epochs"], 1)
            self.assertIsNotNone(completion["best"])

    def test_resume_after_mid_epoch_crash_matches_uninterrupted_run(self) -> None:
        with overall_temp() as root:
            project = SyntheticProject(root / "proj")
            run_a = root / "run_a"
            _silent(overall.train, _config(project, run_a, epochs=2))
            self.assertTrue((run_a / overall.COMPLETION_NAME).is_file())

            run_b = root / "run_b"
            config_b = _config(project, run_b, epochs=2)
            original = overall.run_train_epoch
            calls = {"count": 0}

            def interrupting(*args, **kwargs):
                calls["count"] += 1
                outcome = original(*args, **kwargs)
                if calls["count"] == 2:
                    raise RuntimeError("simulated crash after the epoch-2 training pass")
                return outcome

            overall.run_train_epoch = interrupting
            try:
                with self.assertRaises(RuntimeError):
                    _silent(overall.train, config_b)
            finally:
                overall.run_train_epoch = original

            # 异常中断绝不能留下完成标记，best.pt 必须仍是候选
            self.assertFalse((run_b / overall.COMPLETION_NAME).is_file())
            crashed_best = torch.load(run_b / overall.BEST_NAME, map_location="cpu",
                                      weights_only=True)
            self.assertEqual(crashed_best["status"], "candidate")
            self.assertFalse(crashed_best["completed"])
            checkpoint = torch.load(run_b / overall.CHECKPOINT_NAME, map_location="cpu",
                                    weights_only=True)
            self.assertEqual((checkpoint["epoch"], checkpoint["phase"]), (1, "evaluated"))

            _silent(overall.train, replace(config_b, resume=True))
            self.assertTrue((run_b / overall.COMPLETION_NAME).is_file())
            left = torch.load(run_a / overall.CHECKPOINT_NAME, map_location="cpu",
                              weights_only=True)["model_state"]
            right = torch.load(run_b / overall.CHECKPOINT_NAME, map_location="cpu",
                               weights_only=True)["model_state"]
            self.assertEqual(sorted(left), sorted(right))
            for name in left:
                self.assertTrue(torch.equal(left[name], right[name]), name)
            self.assertEqual(
                json.loads((run_a / overall.HISTORY_NAME).read_text(encoding="utf-8")),
                json.loads((run_b / overall.HISTORY_NAME).read_text(encoding="utf-8")))

    def test_resume_after_crash_before_evaluation_does_not_retrain(self) -> None:
        with overall_temp() as root:
            project = SyntheticProject(root / "proj")
            run = root / "run"
            config = _config(project, run, epochs=2)
            original_eval = overall.evaluate_development
            original_epoch = overall.run_train_epoch
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

            overall.run_train_epoch = counting_epoch
            overall.evaluate_development = failing_eval
            try:
                with self.assertRaises(RuntimeError):
                    _silent(overall.train, config)
                checkpoint = torch.load(run / overall.CHECKPOINT_NAME, map_location="cpu",
                                        weights_only=True)
                self.assertEqual((checkpoint["epoch"], checkpoint["phase"]), (2, "trained"))
                self.assertEqual(train_calls["count"], 2)
                # 恢复时让评估恢复正常；计数包装保持开启以证明没有重跑第 2 轮
                overall.evaluate_development = original_eval
                _silent(overall.train, replace(config, resume=True))
            finally:
                overall.run_train_epoch = original_epoch
                overall.evaluate_development = original_eval

            # 恢复只补做评估，不重跑第 2 轮
            self.assertEqual(train_calls["count"], 2)
            history = json.loads((run / overall.HISTORY_NAME).read_text(encoding="utf-8"))
            self.assertEqual([row["epoch"] for row in history], [1, 2])
            self.assertTrue((run / overall.COMPLETION_NAME).is_file())

    def test_resume_rejects_changed_config_input_or_code(self) -> None:
        with overall_temp() as root:
            project = SyntheticProject(root / "proj")
            run = root / "run"
            config = _config(project, run, epochs=2)
            original = overall.run_train_epoch
            calls = {"count": 0}

            def interrupting(*args, **kwargs):
                calls["count"] += 1
                outcome = original(*args, **kwargs)
                if calls["count"] == 2:
                    raise RuntimeError("simulated crash after epoch 2 training")
                return outcome

            overall.run_train_epoch = interrupting
            try:
                with self.assertRaises(RuntimeError):
                    _silent(overall.train, config)
            finally:
                overall.run_train_epoch = original

            # 第 1 轮已完整落盘，可以直接用 --resume 检查配置/输入/代码守卫
            with self.assertRaises(overall.ResumeError):
                overall.train(replace(config, lr=0.02, resume=True))

            checkpoint = overall.load_checkpoint(run / overall.CHECKPOINT_NAME)
            input_hashes = overall.assemble(config.data_spec).hashes
            code_hashes = overall.code_hashes_snapshot()
            config_hash = overall.config_digest(config.payload())
            self.assertEqual(
                overall.validate_resume(checkpoint, config_hash=config_hash,
                                        input_hashes=input_hashes, code_hashes=code_hashes,
                                        epochs=2), (1, "evaluated"))
            with self.assertRaises(overall.ResumeError):
                overall.validate_resume(checkpoint, config_hash="0" * 64,
                                        input_hashes=input_hashes, code_hashes=code_hashes,
                                        epochs=2)
            with self.assertRaises(overall.ResumeError):
                overall.validate_resume(
                    checkpoint, config_hash=config_hash,
                    input_hashes={**dict(input_hashes), "exposed_split": "0" * 64},
                    code_hashes=code_hashes, epochs=2)
            with self.assertRaises(overall.ResumeError):
                overall.validate_resume(
                    checkpoint, config_hash=config_hash, input_hashes=input_hashes,
                    code_hashes={**dict(code_hashes), "model.py": "0" * 64}, epochs=2)
            with self.assertRaises(overall.ResumeError):
                overall.validate_resume(checkpoint, config_hash=config_hash,
                                        input_hashes=input_hashes, code_hashes=code_hashes,
                                        epochs=0)

    def test_completed_partial_and_unknown_directories_are_refused(self) -> None:
        with overall_temp() as root:
            project = SyntheticProject(root / "proj")
            run = root / "run"
            config = _config(project, run, epochs=1)
            _silent(overall.train, config)
            before = (run / overall.BEST_NAME).read_bytes()

            with self.assertRaises(overall.TrialStateError):
                overall.train(replace(config, resume=False))
            result = _silent(overall.train, replace(config, resume=True))
            self.assertEqual(result["status"], "already-complete")
            self.assertEqual(before, (run / overall.BEST_NAME).read_bytes())

            partial = root / "partial"
            original_eval = overall.evaluate_development

            def failing_eval(*args, **kwargs):
                raise RuntimeError("simulated crash before evaluation")

            overall.evaluate_development = failing_eval
            try:
                with self.assertRaises(RuntimeError):
                    _silent(overall.train, _config(project, partial, epochs=1))
            finally:
                overall.evaluate_development = original_eval
            # 完整一轮已落盘但没有评估/完成标记 => partial，必须显式 --resume
            self.assertEqual(overall.plan_mode(partial, False), "partial")
            self.assertEqual(overall.plan_mode(partial, True), "resume")
            with self.assertRaises(overall.TrialStateError):
                overall.train(_config(project, partial, epochs=1))

            unknown = root / "unknown"
            unknown.mkdir(parents=True)
            (unknown / "stray.txt").write_text("not a trial", encoding="utf-8")
            with self.assertRaises(overall.TrialStateError):
                overall.train(_config(project, unknown, epochs=1))

            missing = root / "missing"
            with self.assertRaises(overall.ResumeError):
                overall.train(_config(project, missing, epochs=1, resume=True))


# --------------------------------------------------------------------- CLI
@unittest.skipUnless(HAVE_DEPENDENCIES, SKIP_REASON)
class CliTests(unittest.TestCase):
    def test_cli_parses_the_required_options(self) -> None:
        parser = overall.build_parser()
        args = parser.parse_args([
            "--output-dir", "runs/example", "--init", "scratch", "--base-channels", "12",
            "--epochs", "3", "--resume", "--lr", "0.002", "--batch-size", "2",
            "--device", "cpu", "--num-workers", "0", "--no-augment",
        ])
        config = overall.config_from_args(args)
        self.assertEqual(config.init, "scratch")
        self.assertEqual(config.base_channels, 12)
        self.assertEqual(config.epochs, 3)
        self.assertTrue(config.resume)
        self.assertAlmostEqual(config.lr, 0.002)
        self.assertEqual(config.batch_size, 2)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.num_workers, 0)
        self.assertFalse(config.augment)
        self.assertEqual(config.thresholds, overall.THRESHOLDS)
        self.assertEqual(config.data_spec.expected_train_total, 2309)
        self.assertEqual(config.data_spec.expected_dev_total, 638)

    def test_cli_rejects_unknown_init_and_unlisted_channels(self) -> None:
        parser = overall.build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["--output-dir", "x", "--init", "pretrained"])
        args = parser.parse_args(["--output-dir", "x", "--base-channels", "16"])
        with self.assertRaises(overall.ConfigError):
            overall.train(overall.config_from_args(args))


# --------------------------------------------------------------- 真实数据
def _real_project_available() -> bool:
    if not HAVE_DEPENDENCIES:  # pragma: no cover
        return False
    spec = overall.build_default_spec()
    paths = [spec.overlap_exclusions, spec.exposed_split, spec.exposed_exclusions,
             spec.development_baseline, spec.baseline_checkpoint]
    paths += [source.provenance for source in (*spec.train_sources, *spec.dev_sources)
              if source.provenance is not None]
    paths += [source.image_dir for source in (*spec.train_sources, *spec.dev_sources)]
    return all(Path(path).exists() for path in paths if path is not None)


REAL_DATA_AVAILABLE = _real_project_available()


@unittest.skipUnless(REAL_DATA_AVAILABLE,
                     "本地真实数据或冻结产物不完整，跳过真实装配检查（不算通过）")
class RealDataAssemblyTests(unittest.TestCase):
    """真实数据上的装配检查：验证冻结计数、哈希链与 ID 隔离确实成立。

    这是集成检查，需要本地 ISIC 2016/2017 数据与冻结产物；缺失时明确跳过。
    """

    def test_real_project_satisfies_frozen_constraints(self) -> None:
        assembly = overall.assemble(overall.build_default_spec(), verify_frozen=True)
        self.assertEqual(len(assembly.train_items), 2309)
        self.assertEqual(assembly.counts["train_total"], 2309)
        self.assertEqual(assembly.counts["development_total"], 638)
        self.assertEqual(assembly.counts["exposed_clean_total"], 583)
        self.assertEqual(len(assembly.dev_ids["isic2016_test"]), 376)
        self.assertEqual(len(assembly.dev_ids["isic2017_validation"]), 145)
        self.assertEqual(len(assembly.dev_ids["isic2017_exposed_validation"]), 117)
        every_dev = set()
        for ids in assembly.dev_ids.values():
            every_dev |= set(ids)
        self.assertEqual(len(every_dev), 638)
        self.assertFalse(set(assembly.train_ids) & every_dev)


if __name__ == "__main__":
    unittest.main(verbosity=2)
