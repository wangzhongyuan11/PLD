"""Plan a new untouched ISIC 2017 Task 1 test holdout under the project cap."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from urllib.request import Request, urlopen
from zipfile import ZipFile

from software.role_a import external2017 as ext
from software.role_a import survey_full2017 as full


ROOT = ext.PROJECT / "data" / "raw" / "isic2017_test"
ARCHIVE = ROOT / "downloads" / "test_masks.zip"
URL = "https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Test_v2_Part1_GroundTruth.zip"
MAX_ARCHIVE_BYTES = 30_000_000
MAX_IMAGE_BYTES = 2 * 1024 * 1024 * 1024
MAX_PROJECT_BYTES = 17_000_000_000


def fetch_masks() -> None:
    if ARCHIVE.exists():
        if ARCHIVE.stat().st_size > MAX_ARCHIVE_BYTES:
            raise ValueError("existing test mask ZIP exceeds reviewed size cap")
        ext.mask_entries(ARCHIVE)
        return
    with urlopen(Request(URL, headers={"User-Agent": "role-a-research/1.0"}), timeout=90) as response:
        if response.status != 200 or response.geturl() != URL:
            raise ValueError("official 2017 test-mask URL redirected or failed")
        data = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError("official test-mask ZIP exceeds reviewed size cap")
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    ext._atomic_bytes(ARCHIVE, data)
    ext.mask_entries(ARCHIVE)


def main() -> None:
    before = full.tree_bytes(ext.PROJECT)
    if before > MAX_PROJECT_BYTES - MAX_ARCHIVE_BYTES - MAX_IMAGE_BYTES:
        raise ValueError("project lacks a safe disk reserve for the test holdout")
    fetch_masks()
    masks = ext.mask_entries(ARCHIVE)
    known = {path.stem for root in (
        ext.PROJECT / "data" / "raw" / "isic2016" / "train" / "images",
        ext.PROJECT / "data" / "raw" / "isic2016" / "test" / "images",
        ext.PROJECT / "data" / "raw" / "isic2017_training" / "images",
        ext.PROJECT / "data" / "raw" / "isic2017_validation" / "images")
        for path in root.glob("ISIC_*.jpg")}
    ids = sorted(set(masks) - known)
    if not ids:
        raise ValueError("no ID-disjoint official test masks")
    rows = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(ext.official_object_metadata, stem): stem for stem in ids}
        for index, future in enumerate(as_completed(futures), 1):
            stem, remote = future.result()
            rows[stem] = {**masks[stem], **remote}
            if index % 50 == 0 or index == len(ids):
                print(f"official test image HEAD {index}/{len(ids)}", flush=True)
    rows = {stem: rows[stem] for stem in sorted(rows)}
    total_images = sum(int(row["expected_bytes"]) for row in rows.values())
    with ZipFile(ARCHIVE) as archive:
        total_masks = sum(archive.getinfo(str(row["mask_zip_member"])).file_size
                          for row in rows.values())
    planned = full.tree_bytes(ext.PROJECT) + total_images + total_masks
    if total_images > MAX_IMAGE_BYTES or planned > MAX_PROJECT_BYTES:
        raise ValueError("ISIC 2017 test holdout exceeds reviewed image or project budget")
    manifest = {
        "schema": "role-a-isic2017-test-v1",
        "source_masks_url": URL,
        "source_masks_sha256": ext.sha256(ARCHIVE.read_bytes()),
        "source_images_pattern": "https://isic-archive.s3.amazonaws.com/images/{isic_id}.jpg",
        "source_license": "https://challenge.isic-archive.com/data/ (2017 Task 1 CC-0)",
        "official_mask_count": len(masks),
        "excluded_known_ids": sorted(set(masks) & known),
        "selected_count": len(rows),
        "estimated_image_bytes": total_images,
        "estimated_mask_bytes": total_masks,
        "estimated_project_bytes_after": planned,
        "items": rows,
    }
    ext.assert_budget(manifest, MAX_IMAGE_BYTES)
    ext._atomic_bytes(ROOT / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8"))
    print(json.dumps({key: manifest[key] for key in ("official_mask_count", "selected_count",
                                                   "estimated_image_bytes", "estimated_mask_bytes",
                                                   "estimated_project_bytes_after")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
