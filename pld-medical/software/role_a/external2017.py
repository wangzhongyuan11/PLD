"""Bounded, auditable access to disjoint ISIC 2017 Task 1 validation images.

Only the official ground-truth ZIP and official ISIC Archive API/S3 are used.
This module does not train, evaluate, or change the ISIC 2016 experiment.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path, PurePosixPath
import re
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zipfile import ZipFile

from PIL import Image


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT / "data" / "raw" / "isic2017_validation"
MASK_ZIP = ROOT / "downloads" / "validation_masks.zip"
API = "https://api.isic-archive.com/api/v2/images/{stem}/"
IMAGE_HOST = "isic-archive.s3.amazonaws.com"
ID_RE = re.compile(r"ISIC_\d{7}")
MASK_RE = re.compile(r"(ISIC_\d{7})_segmentation\.png")
DEFAULT_LIMIT = 200 * 1024 * 1024
MAX_MASK_BYTES = 20 * 1024 * 1024
MAX_TOTAL_MASK_BYTES = 64 * 1024 * 1024


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_bytes(path: Path, data: bytes) -> None:
    part = path.with_name(path.name + ".part")
    with part.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(part, path)


def mask_entries(archive: Path = MASK_ZIP) -> dict[str, dict[str, object]]:
    """Inspect masks in ZIP without extracting arbitrary paths."""
    found: dict[str, dict[str, object]] = {}
    with ZipFile(archive) as zip_file:
        infos = zip_file.infolist()
        if sum(info.file_size for info in infos) > MAX_TOTAL_MASK_BYTES:
            raise ValueError("mask ZIP exceeds uncompressed size limit")
        seen_names: set[str] = set()
        for info in infos:
            name = info.filename
            if name in seen_names:
                raise ValueError(f"duplicate ZIP member: {name}")
            seen_names.add(name)
            path = PurePosixPath(name)
            raw_parts = name.rstrip("/").split("/")
            if ("\\" in name or path.is_absolute() or
                    any(part in ("", ".", "..") for part in raw_parts)):
                raise ValueError(f"unsafe ZIP member: {name}")
            if info.is_dir():
                continue
            if info.file_size > MAX_MASK_BYTES:
                raise ValueError(f"oversized mask ZIP member: {name}")
            match = MASK_RE.fullmatch(path.name)
            if match is None:
                raise ValueError(f"unexpected ZIP member: {name}")
            stem = match.group(1)
            if stem in found:
                raise ValueError(f"duplicate mask ID: {stem}")
            data = zip_file.read(info)
            with Image.open(BytesIO(data)) as image:
                image.load()
                width, height = image.size
                if image.format != "PNG" or width <= 0 or height <= 0:
                    raise ValueError(f"invalid mask: {stem}")
                # The ratio is used only for area-group audit and selection.
                import numpy as np
                ratio = float((np.asarray(image.convert("L")) >= 127).mean())
            found[stem] = {"mask_zip_member": name, "mask_sha256": sha256(data),
                           "width": width, "height": height, "area_ratio": ratio}
    if not found:
        raise ValueError("mask ZIP contains no masks")
    return dict(sorted(found.items()))


def existing_2016_ids() -> set[str]:
    root = PROJECT / "data" / "raw" / "isic2016"
    return {p.stem for role in ("train", "test")
            for p in (root / role / "images").glob("ISIC_*.jpg")}


def validate_image_metadata(stem: str, data: object) -> dict[str, object]:
    if not isinstance(data, dict) or data.get("isic_id") != stem:
        raise ValueError(f"API returned wrong ID for {stem}")
    license_name = data.get("copyright_license")
    if license_name not in ("CC-0", "CC0"):
        raise ValueError(f"unexpected license for {stem}: {license_name}")
    files = data.get("files")
    if not isinstance(files, dict) or not isinstance(files.get("full"), dict):
        raise ValueError(f"missing full file metadata: {stem}")
    full = files["full"]
    url = full.get("url")
    size = full.get("size")
    if not isinstance(url, str) or not isinstance(size, int) or isinstance(size, bool):
        raise ValueError(f"invalid URL/size metadata: {stem}")
    parsed = urlparse(url)
    if (parsed.scheme != "https" or parsed.hostname != IMAGE_HOST or
            parsed.port is not None or parsed.query or parsed.fragment or
            parsed.path != f"/images/{stem}.jpg"):
        raise ValueError(f"non-official image URL: {stem}")
    if not 0 < size <= 20 * 1024 * 1024:
        raise ValueError(f"implausible image size: {stem}: {size}")
    return {"url": url, "expected_bytes": size, "license": license_name}


def _get_json(stem: str) -> tuple[str, dict[str, object]]:
    url = API.format(stem=stem)
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={"User-Agent": "role-a-research/1.0"}), timeout=25) as response:
                if response.status != 200:
                    raise ValueError(f"API HTTP {response.status}: {stem}")
                raw = response.read(256 * 1024 + 1)
                if len(raw) > 256 * 1024:
                    raise ValueError(f"oversized API metadata: {stem}")
            metadata = validate_image_metadata(stem, json.loads(raw))
            head_bytes = _head_length(str(metadata["url"]), stem)
            metadata["api_reported_bytes"] = metadata["expected_bytes"]
            metadata["expected_bytes"] = head_bytes
            return stem, metadata
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f"API lookup failed for {stem}: {last_error}")


def _head_length(url: str, stem: str) -> int:
    """Use the current official object length; Archive metadata can be stale."""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(Request(url, method="HEAD", headers={"User-Agent": "role-a-research/1.0"}),
                         timeout=25) as response:
                if response.status != 200 or response.geturl() != url:
                    raise ValueError(f"unexpected HEAD redirect/status: {stem}")
                length = int(response.headers["Content-Length"])
            if not 0 < length <= 20 * 1024 * 1024:
                raise ValueError(f"implausible HEAD size: {stem}: {length}")
            return length
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f"HEAD lookup failed for {stem}: {last_error}")


def official_object_metadata(stem: str) -> tuple[str, dict[str, object]]:
    """2017 Task 1 CC-0 dataset object, verified by exact S3 HEAD path."""
    if ID_RE.fullmatch(stem) is None:
        raise ValueError(f"invalid image ID: {stem}")
    url = f"https://{IMAGE_HOST}/images/{stem}.jpg"
    length = _head_length(url, stem)
    return stem, {"url": url, "expected_bytes": length, "license": "CC-0"}


def survey() -> dict[str, object]:
    masks = mask_entries()
    old_ids = existing_2016_ids()
    new_ids = sorted(set(masks) - old_ids)
    entries: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(official_object_metadata, stem): stem for stem in new_ids}
        for index, future in enumerate(as_completed(futures), 1):
            stem, remote = future.result()
            entries[stem] = {**masks[stem], **remote}
            if index % 10 == 0 or index == len(new_ids):
                print(f"official object HEAD {index}/{len(new_ids)}", flush=True)
    ordered = {stem: entries[stem] for stem in sorted(entries)}
    result: dict[str, object] = {
        "source_masks_url": "https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Validation_Part1_GroundTruth.zip",
        "source_masks_sha256": sha256(MASK_ZIP.read_bytes()),
        "source_images_pattern": f"https://{IMAGE_HOST}/images/{{isic_id}}.jpg",
        "source_license": "https://challenge.isic-archive.com/data/ (2017 Task 1 CC-0)",
        "official_mask_count": len(masks),
        "excluded_overlap_2016": sorted(set(masks) & old_ids),
        "selected_count": len(ordered),
        "estimated_image_bytes": sum(int(row["expected_bytes"]) for row in ordered.values()),
        "items": ordered,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_bytes(ROOT / "survey.json", json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"))
    return result


def assert_budget(manifest: dict[str, object], maximum: int) -> None:
    if maximum <= 0:
        raise ValueError("max-bytes must be positive")
    items = manifest.get("items")
    if not isinstance(items, dict):
        raise ValueError("manifest items are missing")
    total = 0
    for stem, row in items.items():
        if not isinstance(stem, str) or ID_RE.fullmatch(stem) is None:
            raise ValueError("invalid manifest ID")
        if not isinstance(row, dict):
            raise ValueError(f"invalid manifest item: {stem}")
        validate_image_metadata(stem, {"isic_id": stem, "copyright_license": row.get("license"),
                                       "files": {"full": {"url": row.get("url"), "size": row.get("expected_bytes")}}})
        total += int(row["expected_bytes"])
    if total != manifest.get("estimated_image_bytes"):
        raise ValueError("manifest size sum mismatch")
    if total > maximum:
        raise ValueError(f"estimated download {total} exceeds limit {maximum}; no files downloaded")


def _check_jpeg(data: bytes, expected: tuple[int, int], stem: str) -> None:
    with Image.open(BytesIO(data)) as image:
        image.load()
        if image.format != "JPEG" or image.size != expected:
            raise ValueError(f"JPEG and mask dimension/format mismatch: {stem}")


def download(manifest: dict[str, object], max_bytes: int = DEFAULT_LIMIT, *,
             root: Path = ROOT, archive: Path = MASK_ZIP,
             manifest_path: Path | None = None) -> dict[str, object]:
    assert_budget(manifest, max_bytes)  # Verify full plan before touching disk/network.
    masks = mask_entries(archive)
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict[str, object]] = {}
    actual = 0
    planned = sorted(manifest["items"].items())  # type: ignore[union-attr]
    for index, (stem, row) in enumerate(planned, 1):
        assert isinstance(row, dict)
        mask_row = masks.get(stem)
        if (mask_row is None or row.get("mask_sha256") != mask_row["mask_sha256"] or
                row.get("mask_zip_member") != mask_row["mask_zip_member"] or
                row.get("area_ratio") != mask_row["area_ratio"] or
                row.get("width") != mask_row["width"] or row.get("height") != mask_row["height"]):
            raise ValueError(f"mask changed since survey: {stem}")
        expected = int(row["expected_bytes"])
        image_path = image_dir / f"{stem}.jpg"
        if image_path.exists():
            data = image_path.read_bytes()
            if len(data) != expected:
                raise ValueError(f"existing JPEG length mismatch: {stem}")
            _check_jpeg(data, (int(row["width"]), int(row["height"])), stem)
        else:
            url = str(row["url"])
            with urlopen(Request(url, headers={"User-Agent": "role-a-research/1.0"}), timeout=90) as response:
                if response.status != 200 or response.geturl() != url:
                    raise ValueError(f"unexpected download redirect/status: {stem}")
                data = response.read(expected + 1)
            if len(data) != expected:
                raise ValueError(f"download length mismatch: {stem}: {len(data)} != {expected}")
            _check_jpeg(data, (int(row["width"]), int(row["height"])), stem)
            _atomic_bytes(image_path, data)
        with ZipFile(archive) as z:
            mask_data = z.read(str(row["mask_zip_member"]))
        if sha256(mask_data) != row["mask_sha256"]:
            raise ValueError(f"mask hash mismatch: {stem}")
        mask_path = mask_dir / f"{stem}_Segmentation.png"
        if mask_path.exists() and sha256(mask_path.read_bytes()) != row["mask_sha256"]:
            raise ValueError(f"existing mask changed: {stem}")
        if not mask_path.exists():
            _atomic_bytes(mask_path, mask_data)
        actual += len(data)
        rows[stem] = {"image_sha256": sha256(data), "image_bytes": len(data),
                      "mask_sha256": str(row["mask_sha256"]), "area_ratio": float(row["area_ratio"])}
        if index % 50 == 0 or index == len(planned):
            print(f"verified images {index}/{len(planned)}", flush=True)
    if actual > max_bytes:
        raise ValueError("actual image bytes exceed maximum")
    source_path = manifest_path if manifest_path is not None else root / "survey.json"
    source_key = "source_manifest_sha256" if manifest_path is not None else "source_survey_sha256"
    result: dict[str, object] = {source_key: sha256(source_path.read_bytes()),
                                 "image_count": len(rows), "image_bytes": actual,
                                 "mask_zip_bytes": archive.stat().st_size,
                                 "maximum_image_bytes": max_bytes, "complete": True, "items": rows}
    _atomic_bytes(root / "provenance.json", json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("survey")
    getter = commands.add_parser("download")
    getter.add_argument("--max-bytes", type=int, default=DEFAULT_LIMIT)
    args = parser.parse_args()
    if args.command == "survey":
        result = survey()
        print(json.dumps({k: v for k, v in result.items() if k != "items"}, ensure_ascii=False, indent=2))
    else:
        payload = json.loads((ROOT / "survey.json").read_text(encoding="utf-8"))
        result = download(payload, max_bytes=args.max_bytes)
        print(json.dumps({k: v for k, v in result.items() if k != "items"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
