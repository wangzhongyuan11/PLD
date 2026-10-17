"""Offline tests for the full ISIC 2017 training survey and its disk budget.

Everything here runs without network access: HEAD lookups are replaced by fakes,
ZIPs are synthesised in a temporary directory, and the "project" being measured
is a temporary tree, never the real repository.  The tests therefore prove the
survey/budget *logic* only; they say nothing about the real official object
store or about the real project's byte count.
"""
from __future__ import annotations

import contextlib
from io import BytesIO
import io as _io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

if __package__ in (None, ""):  # allow `python software/role_a/tests/test_...py`
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _support import ensure_role_a_on_path  # noqa: E402

ensure_role_a_on_path()

from PIL import Image  # noqa: E402

from software.role_a import external2017 as ext  # noqa: E402
from software.role_a import survey_full2017 as sf  # noqa: E402

GIB = 1024 * 1024 * 1024


def mask_png(width: int = 4, height: int = 3, foreground: int = 255,
             box: tuple[int, int, int, int] | None = None) -> bytes:
    """A tiny PNG mask; ``box`` is an inclusive-exclusive foreground rectangle."""
    image = Image.new("L", (width, height), 0)
    if box is None:
        box = (0, 0, max(1, width // 2), max(1, height // 2))
    x0, y0, x1, y1 = box
    for y in range(y0, min(y1, height)):
        for x in range(x0, min(x1, width)):
            image.putpixel((x, y), foreground)
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def write_mask_zip(path: Path, stems: list[str], **kwargs: object) -> Path:
    """Write ``ISIC_<stem>_segmentation.png`` members in the official layout."""
    with ZipFile(path, "w") as zip_file:
        for stem in stems:
            zip_file.writestr(f"{stem}_segmentation.png", mask_png(**kwargs))
    return path


def write_jpeg(path: Path, length: int) -> None:
    """Write a JPEG whose exact byte length is ``length`` (valid, just padded)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (4, 4), (10, 20, 30))
    stream = BytesIO()
    image.save(stream, format="JPEG")
    data = stream.getvalue()
    if len(data) > length:
        raise ValueError("requested length is too small for a JPEG header")
    path.write_bytes(data + b"\x00" * (length - len(data)))


def make_project(root: Path) -> dict[str, int]:
    """Create a synthetic project tree with a known byte total."""
    (root / "software" / "role_a").mkdir(parents=True)
    (root / "software" / "role_a" / "code.py").write_bytes(b"x" * 100)
    (root / "data" / "raw" / "isic2016" / "train" / "images").mkdir(parents=True)
    (root / "data" / "raw" / "isic2016" / "train" / "masks").mkdir(parents=True)
    (root / "data" / "raw" / "isic2016" / "test" / "images").mkdir(parents=True)
    (root / "data" / "raw" / "isic2016" / "test" / "masks").mkdir(parents=True)
    (root / "data" / "raw" / "isic2016" / "downloads").mkdir(parents=True)
    (root / "data" / "raw" / "isic2017_validation" / "images").mkdir(parents=True)
    (root / "data" / "raw" / "isic2017_validation" / "masks").mkdir(parents=True)
    (root / "data" / "raw" / "isic2017_training" / "images").mkdir(parents=True)
    (root / "data" / "raw" / "isic2017_training" / "masks").mkdir(parents=True)
    (root / "data" / "raw" / "isic2017_training" / "downloads").mkdir(parents=True)
    (root / "data" / "raw" / "isic2016" / "downloads" / "train_images.zip").write_bytes(b"z" * 7)
    return {"outside_code": 100, "raw_zip": 7}


def fake_lookup(row: dict[str, object]):
    """A deterministic stand-in for ``ext.official_object_metadata``."""

    def lookup(stem: str) -> tuple[str, dict[str, object]]:
        return stem, {
            "url": f"https://{ext.IMAGE_HOST}/images/{stem}.jpg",
            "expected_bytes": int(row[stem]),
            "license": "CC-0",
        }

    return lookup


class MaskIndexTests(unittest.TestCase):
    def test_unsafe_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "masks.zip"
            with ZipFile(archive, "w") as zip_file:
                zip_file.writestr("../ISIC_0000001_segmentation.png", mask_png())
            with self.assertRaisesRegex(ValueError, "unsafe ZIP member"):
                sf.mask_index(archive)

    def test_duplicate_and_unexpected_members_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "masks.zip"
            with ZipFile(archive, "w") as zip_file:
                zip_file.writestr("a/ISIC_0000001_segmentation.png", mask_png())
                zip_file.writestr("b/ISIC_0000001_segmentation.png", mask_png())
            with self.assertRaisesRegex(ValueError, "duplicate mask ID"):
                sf.mask_index(archive)
            with ZipFile(archive, "w") as zip_file:
                zip_file.writestr("ISIC_0000001.jpg", mask_png())
            with self.assertRaisesRegex(ValueError, "unexpected ZIP member"):
                sf.mask_index(archive)

    def test_uncompressed_size_limit_is_checked_before_reading(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            archive = write_mask_zip(Path(temp) / "masks.zip", ["ISIC_0000001"])
            with patch.object(sf, "MAX_TOTAL_MASK_BYTES", 1):
                with self.assertRaisesRegex(ValueError, "size limit"):
                    sf.mask_index(archive)

    def test_index_is_sorted_and_carries_real_area(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            archive = write_mask_zip(Path(temp) / "masks.zip",
                                     ["ISIC_0000002", "ISIC_0000001"],
                                     width=4, height=4, box=(0, 0, 2, 2))
            index = sf.mask_index(archive)
            self.assertEqual(list(index), ["ISIC_0000001", "ISIC_0000002"])
            self.assertEqual(index["ISIC_0000001"].area_ratio, 0.25)
            self.assertEqual(index["ISIC_0000001"].width, 4)
            self.assertEqual(index["ISIC_0000001"].height, 4)
            self.assertEqual(index["ISIC_0000001"].bytes, len(mask_png(width=4, height=4,
                                                                      box=(0, 0, 2, 2))))
            self.assertEqual(index["ISIC_0000001"].sha256,
                             sf._sha256(mask_png(width=4, height=4, box=(0, 0, 2, 2))))


class SelectionTests(unittest.TestCase):
    def test_all_new_ids_are_kept_and_only_excluded_ones_dropped(self) -> None:
        masks = {stem: object() for stem in
                 ("ISIC_0000001", "ISIC_0000002", "ISIC_0000003", "ISIC_0000004")}
        chosen = sf.select_new_ids(masks, old_2016=["ISIC_0000002"],
                                   validation=["ISIC_0000003"])
        self.assertEqual(chosen, ["ISIC_0000001", "ISIC_0000004"])

    def test_bogus_ids_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid ISIC image ID"):
            sf.select_new_ids({"ISIC_0000001": object()}, old_2016=["nope"], validation=[])
        with self.assertRaisesRegex(ValueError, "invalid ISIC image ID"):
            sf.select_new_ids({"nope": object()}, old_2016=[], validation=[])

    def test_missing_mask_in_index_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = {"items": {"ISIC_0000001": {"expected_bytes": 5}}}
            with self.assertRaisesRegex(ValueError, "missing from ZIP index"):
                sf.estimate_disk(manifest, {}, root=Path(temp))


class HeadMetadataTests(unittest.TestCase):
    def test_lookup_is_deterministic_sorted_and_bounded(self) -> None:
        sizes = {f"ISIC_{index:07d}": index for index in range(3, 0, -1)}
        rows = sf.head_metadata(sizes, workers=4, lookup=fake_lookup(sizes))
        self.assertEqual(list(rows), sorted(sizes))
        self.assertEqual(rows["ISIC_0000001"]["expected_bytes"], 1)
        self.assertEqual(rows["ISIC_0000001"]["license"], "CC-0")

    def test_worker_ceiling_and_zero_are_refused(self) -> None:
        sizes = {"ISIC_0000001": 10}
        for workers in (0, -1, sf.MAX_HEAD_WORKERS + 1):
            with self.assertRaises(ValueError):
                sf.head_metadata(sizes, workers=workers, lookup=fake_lookup(sizes))

    def test_id_mismatch_and_wrong_host_are_refused(self) -> None:
        sizes = {"ISIC_0000001": 10}

        def bad_stem(stem: str):
            return "ISIC_0000002", fake_lookup(sizes)(stem)[1]

        with self.assertRaisesRegex(ValueError, "ID mismatch"):
            sf.head_metadata(sizes, lookup=bad_stem)

        def bad_url(stem: str):
            return stem, {"url": "https://example.org/x.jpg", "expected_bytes": 10}

        with self.assertRaisesRegex(ValueError, "non-official HEAD URL"):
            sf.head_metadata(sizes, lookup=bad_url)

    def test_progress_callback_sees_every_item_once(self) -> None:
        sizes = {f"ISIC_{index:07d}": index for index in range(1, 4)}
        seen: list[tuple[int, int]] = []
        sf.head_metadata(sizes, lookup=fake_lookup(sizes),
                         progress=lambda done, total: seen.append((done, total)))
        self.assertEqual(sorted(done for done, _ in seen), [1, 2, 3])
        self.assertTrue(all(total == 3 for _, total in seen))


class DiskEstimateTests(unittest.TestCase):
    def test_existing_images_and_masks_are_not_counted_twice(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / "project"
            (project / "data" / "raw" / "isic2017_survey_full" / "images").mkdir(parents=True)
            (project / "data" / "raw" / "isic2017_survey_full" / "masks").mkdir(parents=True)
            (project / "data" / "raw" / "isic2016" / "train" / "images").mkdir(parents=True)
            (project / "code.py").write_bytes(b"x" * 10)
            root = project / "data" / "raw" / "isic2017_survey_full"
            (root / "images" / "ISIC_0000001.jpg").write_bytes(b"x" * 40)
            (root / "images" / "ISIC_0000002.jpg").write_bytes(b"x" * 50)
            (root / "masks" / "ISIC_0000001_Segmentation.png").write_bytes(b"m" * 5)
            (root / "masks" / "ISIC_0000002_Segmentation.png").write_bytes(b"m" * 7)
            masks = {
                "ISIC_0000001": sf.ZipMask("ISIC_0000001_segmentation.png", "a", 5, 4, 4, 0.5),
                "ISIC_0000002": sf.ZipMask("ISIC_0000002_segmentation.png", "b", 7, 4, 4, 0.5),
                "ISIC_0000003": sf.ZipMask("ISIC_0000003_segmentation.png", "c", 11, 4, 4, 0.5),
                "ISIC_0000004": sf.ZipMask("ISIC_0000004_segmentation.png", "d", 13, 4, 4, 0.5),
            }
            manifest = {
                "items": {
                    "ISIC_0000001": {"expected_bytes": 40},   # already on disk, exact
                    "ISIC_0000002": {"expected_bytes": 60},   # already on disk, wrong length
                    "ISIC_0000003": {"expected_bytes": 100},  # new image, mask present
                    "ISIC_0000004": {"expected_bytes": 200},  # new image, mask missing
                }
            }
            estimate = sf.estimate_disk(manifest, masks, root=project, data_root=root)
            self.assertEqual(estimate["manifest_bytes"], 400)
            self.assertEqual(estimate["existing_image_count"], 1)
            self.assertEqual(estimate["existing_image_bytes"], 40)
            self.assertEqual(estimate["required_image_bytes"], 360)
            self.assertEqual(estimate["corrupt_existing_images"], ["ISIC_0000002"])
            self.assertEqual(estimate["existing_mask_count"], 2)
            self.assertEqual(estimate["existing_mask_bytes"], 12)
            self.assertEqual(estimate["missing_mask_count"], 2)
            self.assertEqual(estimate["missing_mask_bytes"], 24)
            self.assertEqual(estimate["added_bytes"], 384)
            self.assertEqual(estimate["project_bytes_before"], sf.tree_bytes(project))
            self.assertEqual(estimate["project_bytes_after"],
                             sf.tree_bytes(project) + 384)
            sf.assert_budget(estimate)  # small but internally consistent

    def test_measure_splits_raw_data_from_the_rest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp) / "project"
            expected = make_project(project)
            (project / ".venv" / "lib").mkdir(parents=True)
            (project / ".venv" / "lib" / "pkg.bin").write_bytes(b"v" * 50)
            measured = sf.measure_project(project)
            self.assertEqual(measured["data_raw_bytes"], expected["raw_zip"])
            self.assertEqual(measured["venv_bytes"], 50)
            self.assertEqual(measured["project_bytes"],
                             expected["outside_code"] + expected["raw_zip"] + 50)
            self.assertEqual(measured["non_data_bytes"],
                             expected["outside_code"] + 50)

    def test_odd_filenames_are_errors_not_silently_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            images = Path(temp) / "images"
            images.mkdir()
            (images / "ISIC_0000001.jpg").write_bytes(b"a")
            (images / "ISIC_0000001_copy.jpg").write_bytes(b"b")
            with self.assertRaisesRegex(ValueError, "unexpected file"):
                sf.image_files(images)
            masks = Path(temp) / "masks"
            masks.mkdir()
            (masks / "ISIC_0000001_Segmentation.png").write_bytes(b"a")
            (masks / "readme.txt").write_bytes(b"b")
            self.assertEqual(len(sf.mask_files(masks)), 1)  # .txt is outside the glob
            (masks / "ISIC_0000002_segmentation.png").write_bytes(b"c")
            with self.assertRaisesRegex(ValueError, "unexpected file"):
                sf.mask_files(masks)

    def test_empty_manifest_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "items are missing or empty"):
            sf.estimate_disk({"items": {}}, {}, data_root=Path("."))


class BudgetTests(unittest.TestCase):
    def test_manifest_limit_is_two_gib_and_blocks_before_download(self) -> None:
        self.assertEqual(sf.MANIFEST_MAX_BYTES, 2 * GIB)
        size = 2 * GIB + 1
        estimate = {
            "manifest_bytes": size,
            "required_image_bytes": size,
            "existing_image_bytes": 0,
            "missing_mask_bytes": 0,
            "added_bytes": size,
            "project_bytes_before": 1,
            "project_bytes_after": 1 + size,
        }
        with self.assertRaisesRegex(sf.BudgetError, "exceed"):
            sf.assert_budget(estimate)
        self.assertFalse(sf.budget_summary(estimate)["confirmed"])

    def test_project_limit_is_exactly_seventeen_billion_bytes(self) -> None:
        self.assertEqual(sf.PROJECT_MAX_BYTES, 17_000_000_000)
        size = 100
        estimate = {
            "manifest_bytes": size,
            "required_image_bytes": size,
            "existing_image_bytes": 0,
            "missing_mask_bytes": 0,
            "added_bytes": size,
            "project_bytes_before": sf.PROJECT_MAX_BYTES - size,
            "project_bytes_after": sf.PROJECT_MAX_BYTES,
        }
        sf.assert_budget(estimate)  # exactly at the limit is allowed
        over = dict(estimate, project_bytes_before=sf.PROJECT_MAX_BYTES - size + 1,
                    project_bytes_after=sf.PROJECT_MAX_BYTES + 1)
        with self.assertRaisesRegex(sf.BudgetError, "project would reach"):
            sf.assert_budget(over)
        summary = sf.budget_summary(over)
        self.assertFalse(summary["project_within_limit"])
        self.assertFalse(summary["confirmed"])

    def test_internally_inconsistent_estimate_is_refused(self) -> None:
        estimate = {
            "manifest_bytes": 10,
            "required_image_bytes": 10,
            "existing_image_bytes": 0,
            "missing_mask_bytes": 0,
            "added_bytes": 10,
            "project_bytes_before": 100,
            "project_bytes_after": 111,  # should be 110
        }
        with self.assertRaisesRegex(ValueError, "accounting mismatch"):
            sf.assert_budget(estimate)

    def test_non_positive_limits_are_refused(self) -> None:
        estimate = {
            "manifest_bytes": 1, "required_image_bytes": 1, "existing_image_bytes": 0,
            "missing_mask_bytes": 0, "added_bytes": 1,
            "project_bytes_before": 0, "project_bytes_after": 1,
        }
        for kwargs in ({"project_max_bytes": 0}, {"manifest_max_bytes": -1}):
            with self.assertRaisesRegex(ValueError, "must be positive"):
                sf.assert_budget(estimate, **kwargs)

    def test_missing_reported_field_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            sf.assert_budget({"manifest_bytes": 1})


class ValidationIdTests(unittest.TestCase):
    def test_validation_ids_merge_zip_and_disk(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            validation = root / "isic2017_validation"
            archive = write_mask_zip(root / "validation_masks.zip",
                                     ["ISIC_0000001", "ISIC_0000002"])
            (validation / "masks").mkdir(parents=True)
            (validation / "images").mkdir(parents=True)
            (validation / "masks" / "ISIC_0000003_Segmentation.png").write_bytes(b"m")
            (validation / "images" / "ISIC_0000004.jpg").write_bytes(b"j")
            with patch.object(ext, "MASK_ZIP", archive), \
                    patch.object(sf, "VALIDATION_ROOT", validation):
                self.assertEqual(sf.validation_ids(),
                                 {"ISIC_0000001", "ISIC_0000002",
                                  "ISIC_0000003", "ISIC_0000004"})


class SurveyTests(unittest.TestCase):
    def _fixtures(self, root: Path) -> tuple[Path, Path]:
        project = root / "project"
        (project / "data" / "raw").mkdir(parents=True)
        archive = write_mask_zip(project / "training_masks.zip",
                                 ["ISIC_0000001", "ISIC_0000002", "ISIC_0000003"])
        train_masks_dir = project / "data" / "raw" / "isic2017_training" / "masks"
        train_masks_dir.mkdir(parents=True)
        # Training masks live in the ZIP, not on disk: the survey must not need them.
        return project, archive

    def test_survey_excludes_2016_and_validation_and_writes_nothing_else(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project, archive = self._fixtures(root)
            out = root / "out"
            sizes = {"ISIC_0000001": 100, "ISIC_0000003": 300}
            with patch.object(ext, "PROJECT", project), \
                    patch.object(ext, "existing_2016_ids",
                                 return_value={"ISIC_0000001"}), \
                    patch.object(sf, "validation_ids",
                                 return_value={"ISIC_0000002"}), \
                    patch.object(sf, "VALIDATION_ROOT", root / "absent"), \
                    patch.object(sf, "ISIC2016_ROOT", root / "absent"), \
                    patch.object(sf, "TRAINING_ROOT", root / "absent"):
                result = sf.survey(root=out, archive=archive, project_root=project,
                                   lookup=fake_lookup(sizes))
            self.assertEqual(result["selected_count"], 1)
            self.assertEqual(list(result["items"]), ["ISIC_0000003"])
            self.assertEqual(result["excluded_2016_ids"], ["ISIC_0000001"])
            self.assertEqual(result["excluded_validation_ids"], ["ISIC_0000002"])
            self.assertEqual(result["estimated_image_bytes"], 300)
            self.assertFalse(result["downloaded"])
            self.assertTrue(result["budget"]["confirmed"])
            self.assertEqual(result["head_workers"], sf.MAX_HEAD_WORKERS)
            written = sorted(path.name for path in out.iterdir())
            self.assertEqual(written, [sf.MANIFEST_NAME, sf.SURVEY_NAME])
            manifest = json.loads((out / sf.MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(manifest["selected_count"], 1)
            self.assertEqual(manifest["items"]["ISIC_0000003"]["license"], "CC-0")
            self.assertEqual(manifest["items"]["ISIC_0000003"]["mask_bytes"],
                             result["items"]["ISIC_0000003"]["mask_bytes"])

    def test_survey_output_is_byte_identical_on_a_second_run(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project, archive = self._fixtures(root)
            out = root / "out"
            sizes = {"ISIC_0000001": 100, "ISIC_0000002": 200, "ISIC_0000003": 300}
            common = dict(root=out, archive=archive, project_root=project,
                          lookup=fake_lookup(sizes))
            with patch.object(sf, "validation_ids", return_value=set()), \
                    patch.object(ext, "existing_2016_ids", return_value=set()), \
                    patch.object(sf, "VALIDATION_ROOT", root / "absent"), \
                    patch.object(sf, "ISIC2016_ROOT", root / "absent"), \
                    patch.object(sf, "TRAINING_ROOT", root / "absent"):
                first = sf.survey(**common)
                survey_bytes = (out / sf.SURVEY_NAME).read_bytes()
                manifest_bytes = (out / sf.MANIFEST_NAME).read_bytes()
                second = sf.survey(**common)
            self.assertEqual(first, second)
            self.assertEqual(survey_bytes, (out / sf.SURVEY_NAME).read_bytes())
            self.assertEqual(manifest_bytes, (out / sf.MANIFEST_NAME).read_bytes())

    def test_rejected_plan_writes_no_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project, archive = self._fixtures(root)
            out = root / "out"
            sizes = {stem: 1 for stem in ("ISIC_0000001", "ISIC_0000002", "ISIC_0000003")}
            with patch.object(sf, "validation_ids", return_value=set()), \
                    patch.object(ext, "existing_2016_ids", return_value=set()), \
                    patch.object(sf, "VALIDATION_ROOT", root / "absent"), \
                    patch.object(sf, "ISIC2016_ROOT", root / "absent"), \
                    patch.object(sf, "TRAINING_ROOT", root / "absent"):
                with self.assertRaises(sf.BudgetError):
                    sf.survey(root=out, archive=archive, project_root=project,
                              lookup=fake_lookup(sizes), manifest_max_bytes=2)
                self.assertFalse(out.exists())
                with self.assertRaises(sf.BudgetError):
                    sf.survey(root=out, archive=archive, project_root=project,
                              lookup=fake_lookup(sizes), project_max_bytes=1)
                self.assertFalse(out.exists())

    def test_dry_run_returns_the_plan_without_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project, archive = self._fixtures(root)
            out = root / "out"
            sizes = {stem: 10 for stem in ("ISIC_0000001", "ISIC_0000002", "ISIC_0000003")}
            with patch.object(sf, "validation_ids", return_value=set()), \
                    patch.object(ext, "existing_2016_ids", return_value=set()), \
                    patch.object(sf, "VALIDATION_ROOT", root / "absent"), \
                    patch.object(sf, "ISIC2016_ROOT", root / "absent"), \
                    patch.object(sf, "TRAINING_ROOT", root / "absent"):
                result = sf.survey(root=out, archive=archive, project_root=project,
                                   lookup=fake_lookup(sizes), write=False)
            self.assertEqual(result["selected_count"], 3)
            self.assertFalse(out.exists())

    def test_no_download_function_is_called_by_survey(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            project, archive = self._fixtures(root)
            sizes = {stem: 10 for stem in ("ISIC_0000001", "ISIC_0000002", "ISIC_0000003")}
            with patch.object(sf, "validation_ids", return_value=set()), \
                    patch.object(ext, "existing_2016_ids", return_value=set()), \
                    patch.object(sf, "VALIDATION_ROOT", root / "absent"), \
                    patch.object(sf, "ISIC2016_ROOT", root / "absent"), \
                    patch.object(sf, "TRAINING_ROOT", root / "absent"), \
                    patch.object(ext, "download", side_effect=AssertionError("download")):
                sf.survey(root=root / "out", archive=archive, project_root=project,
                          lookup=fake_lookup(sizes))

class CliTests(unittest.TestCase):
    def test_only_survey_is_valid_subcommand(self) -> None:
        with contextlib.redirect_stderr(_io.StringIO()):
            for forbidden in ("train", "evaluate", "download"):
                with self.assertRaises(SystemExit):
                    sf.build_parser().parse_args([forbidden])

    def test_survey_help_documents_the_guards(self) -> None:
        with contextlib.redirect_stdout(_io.StringIO()) as captured:
            with self.assertRaises(SystemExit):
                sf.main(["survey", "--help"])
        text = captured.getvalue()
        self.assertIn("--workers", text)
        self.assertIn("--dry-run", text)
        self.assertIn("--project-max-bytes", text)
        self.assertIn("--manifest-max-bytes", text)

    def test_survey_rejects_more_than_four_head_workers(self) -> None:
        with self.assertRaises(SystemExit):
            sf.main(["survey", "--workers", str(sf.MAX_HEAD_WORKERS + 1)])

    def test_cli_survey_passes_choices_to_the_plan(self) -> None:
        seen: dict[str, object] = {}

        def fake_survey(**kwargs):
            seen.update(kwargs)
            return {"items": {}, "selected_count": 0}

        with contextlib.redirect_stdout(_io.StringIO()):
            with patch.object(sf, "survey", side_effect=fake_survey):
                sf.main(["survey", "--workers", "2", "--dry-run"])
        self.assertEqual(seen["workers"], 2)
        self.assertTrue(seen["write"] is False)
        self.assertEqual(seen["manifest_max_bytes"], sf.MANIFEST_MAX_BYTES)

    def test_survey_plan_api_is_importable_without_side_effects(self) -> None:
        self.assertTrue(callable(sf.survey))
        self.assertTrue(callable(sf.estimate_disk))
        self.assertTrue(callable(sf.assert_budget))
        self.assertEqual(sf.ROOT,
                         ext.PROJECT / "data" / "raw" / "isic2017_training")
        self.assertEqual(sf.MAX_HEAD_WORKERS, 4)
        self.assertEqual(sf.MANIFEST_MAX_BYTES, 2 * GIB)
        self.assertEqual(sf.PROJECT_MAX_BYTES, 17_000_000_000)


if __name__ == "__main__":
    unittest.main()
