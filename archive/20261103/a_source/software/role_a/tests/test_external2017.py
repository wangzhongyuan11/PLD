"""Offline checks for the bounded external-validation data workflow."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image

from software.role_a import external2017 as ext


def png_bytes() -> bytes:
    stream = BytesIO()
    Image.new("L", (4, 3), 255).save(stream, format="PNG")
    return stream.getvalue()


class External2017Tests(unittest.TestCase):
    def test_zip_checks_path_duplicate_and_size_before_read(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "masks.zip"
            with ZipFile(archive, "w") as zip_file:
                zip_file.writestr("../ISIC_0000001_segmentation.png", png_bytes())
            with self.assertRaisesRegex(ValueError, "unsafe ZIP member"):
                ext.mask_entries(archive)
            with ZipFile(archive, "w") as zip_file:
                zip_file.writestr("ISIC_0000001_segmentation.png", png_bytes())
            with patch.object(ext, "MAX_TOTAL_MASK_BYTES", 1):
                with self.assertRaisesRegex(ValueError, "size limit"):
                    ext.mask_entries(archive)

    def test_mask_manifest_is_sorted_and_uses_real_area(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "masks.zip"
            with ZipFile(archive, "w") as zip_file:
                zip_file.writestr("ISIC_0000002_segmentation.png", png_bytes())
                zip_file.writestr("ISIC_0000001_segmentation.png", png_bytes())
            rows = ext.mask_entries(archive)
            self.assertEqual(list(rows), ["ISIC_0000001", "ISIC_0000002"])
            self.assertEqual(rows["ISIC_0000001"]["area_ratio"], 1.0)

    def test_budget_rejects_before_download_and_wrong_host(self) -> None:
        stem = "ISIC_0000001"
        row = {"url": f"https://{ext.IMAGE_HOST}/images/{stem}.jpg",
               "expected_bytes": 100, "license": "CC-0"}
        manifest = {"items": {stem: row}, "estimated_image_bytes": 100}
        with self.assertRaisesRegex(ValueError, "exceeds limit"):
            ext.assert_budget(manifest, 99)
        row["url"] = "https://example.org/images/ISIC_0000001.jpg"
        with self.assertRaisesRegex(ValueError, "non-official"):
            ext.assert_budget(manifest, 200)

    def test_survey_excludes_overlap_without_contacting_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            zip_path = root / "masks.zip"
            zip_path.write_bytes(b"mock zip bytes")
            masks = {"ISIC_0000001": {"mask_sha256": "a"},
                     "ISIC_0000002": {"mask_sha256": "b"}}
            def fake_get(stem: str):
                self.assertEqual(stem, "ISIC_0000002")
                return stem, {"url": f"https://{ext.IMAGE_HOST}/images/{stem}.jpg",
                              "expected_bytes": 123, "license": "CC-0"}
            with (patch.object(ext, "ROOT", root), patch.object(ext, "MASK_ZIP", zip_path),
                  patch.object(ext, "mask_entries", return_value=masks),
                  patch.object(ext, "existing_2016_ids", return_value={"ISIC_0000001"}),
                  patch.object(ext, "official_object_metadata", side_effect=fake_get)):
                result = ext.survey()
            self.assertEqual(result["selected_count"], 1)
            self.assertEqual(result["excluded_overlap_2016"], ["ISIC_0000001"])
            self.assertEqual(result["estimated_image_bytes"], 123)


if __name__ == "__main__":
    unittest.main()
