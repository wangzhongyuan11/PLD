"""Survey the *whole* remaining ISIC 2017 Task 1 training set, without downloading it.

Scope of this module (deliberately narrow):

* Read the ISIC 2017 official **training** mask ZIP that already exists in this
  project (never re-download it).
* Build the new-ID set: every official 2017 training ID that is **not** a 2016
  train/test ID and **not** a 2017 validation ID.  "Select all new IDs" means the
  new-ID set is the selection; there is no sub-sampling here.
* Resolve the official object URL and length for each selected ID with a bounded
  concurrency of at most :data:`MAX_HEAD_WORKERS` (4) HEAD requests, reusing
  :func:`software.role_a.external2017.official_object_metadata`.
* Write a deterministic manifest (sorted keys, no timestamps) so the same inputs
  produce byte-identical output.
* Estimate the extra disk the download *would* need, from the measured project
  size, the images already on disk and the mask bytes still missing, and refuse
  the plan unless

  1. the whole project would stay at or below :data:`PROJECT_MAX_BYTES`
     (17,000,000,000 bytes), and
  2. the manifest's raw-image total stays at or below :data:`MANIFEST_MAX_BYTES`
     (2 GiB).

This module does **not** download images and does **not** train.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Callable, Iterable, Mapping
from zipfile import ZipFile

import numpy as np
from PIL import Image

from software.role_a import external2017 as ext


PROJECT = ext.PROJECT
ROOT = PROJECT / "data" / "raw" / "isic2017_training"
MASK_ZIP = PROJECT / "data" / "raw" / "isic2017_training" / "downloads" / "training_masks.zip"

#: Other official 2017 data sets that must never be touched by this survey.
VALIDATION_ROOT = PROJECT / "data" / "raw" / "isic2017_validation"
TRAINING_ROOT = ROOT
ISIC2016_ROOT = PROJECT / "data" / "raw" / "isic2016"

SURVEY_NAME = "survey.json"
MANIFEST_NAME = "manifest.json"

MASK_SOURCE_URL = (
    "https://isic-archive.s3.amazonaws.com/challenges/2017/"
    "ISIC-2017_Training_Part1_GroundTruth.zip"
)
PROJECT_LICENSE = "https://challenge.isic-archive.com/data/ (2017 Task 1 CC-0)"

#: Whole-project ceiling, in bytes, including this project's existing data.
PROJECT_MAX_BYTES = 17_000_000_000
#: Ceiling for the manifest's raw-image total: 2 GiB.
MANIFEST_MAX_BYTES = 2 * 1024 * 1024 * 1024
#: Hard ceiling on concurrent official-object HEAD requests.
MAX_HEAD_WORKERS = 4

ID_RE = re.compile(r"ISIC_\d{7}")
MASK_MEMBER_RE = re.compile(r"(ISIC_\d{7})_segmentation\.png")
IMAGE_NAME_RE = re.compile(r"(ISIC_\d{7})\.jpg")
MASK_NAME_RE = re.compile(r"(ISIC_\d{7})_Segmentation\.png")

MAX_MASK_MEMBER_BYTES = 20 * 1024 * 1024
MAX_TOTAL_MASK_BYTES = 128 * 1024 * 1024


class BudgetError(ValueError):
    """Raised when a plan would exceed a hard budget limit (nothing is written)."""


@dataclass(frozen=True)
class ZipMask:
    """Immutable mask facts taken from the official ZIP member."""

    member: str
    sha256: str
    bytes: int
    width: int
    height: int
    area_ratio: float

    def as_row(self) -> dict[str, object]:
        return {
            "mask_zip_member": self.member,
            "mask_sha256": self.sha256,
            "mask_bytes": self.bytes,
            "width": self.width,
            "height": self.height,
            "area_ratio": self.area_ratio,
        }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_id(stem: object) -> str:
    if not isinstance(stem, str) or ID_RE.fullmatch(stem) is None:
        raise ValueError(f"invalid ISIC image ID: {stem!r}")
    return stem


def _require_count(value: object, what: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"invalid {what}: {value!r}")
    return value


# --------------------------------------------------------------------------- #
# Official ZIP index (safe, read-only, no extraction)
# --------------------------------------------------------------------------- #
def mask_index(archive: Path = MASK_ZIP) -> dict[str, ZipMask]:
    """Return ``{stem: ZipMask}`` for the official 2017 training mask ZIP.

    Members are validated with the same rules as
    :func:`software.role_a.external2017.mask_entries` (unsafe paths, duplicates,
    oversized members), but the returned facts are frozen so a later comparison
    cannot be fooled by a mutated dict.
    """
    found: dict[str, ZipMask] = {}
    with ZipFile(archive) as zip_file:
        infos = zip_file.infolist()
        total = 0
        for info in infos:
            total += info.file_size
            if total > MAX_TOTAL_MASK_BYTES:
                raise ValueError("mask ZIP exceeds uncompressed size limit")
        seen: set[str] = set()
        for info in infos:
            name = info.filename
            if name in seen:
                raise ValueError(f"duplicate ZIP member: {name}")
            seen.add(name)
            path = PurePosixPath(name)
            parts = name.rstrip("/").split("/")
            if ("\\" in name or path.is_absolute() or
                    any(part in ("", ".", "..") for part in parts)):
                raise ValueError(f"unsafe ZIP member: {name}")
            if info.is_dir():
                continue
            if info.file_size > MAX_MASK_MEMBER_BYTES:
                raise ValueError(f"oversized mask ZIP member: {name}")
            match = MASK_MEMBER_RE.fullmatch(path.name)
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
                ratio = float((np.asarray(image.convert("L")) >= 127).mean())
            found[stem] = ZipMask(member=name, sha256=_sha256(data),
                                  bytes=len(data), width=width, height=height,
                                  area_ratio=ratio)
    if not found:
        raise ValueError("mask ZIP contains no masks")
    return dict(sorted(found.items()))


def mask_files(masks_dir: Path) -> dict[str, int]:
    """Return ``{stem: file_bytes}`` for masks already extracted on disk.

    Any ``ISIC_*.png`` that is not an exact ``ISIC_<7 digits>_Segmentation.png``
    name is a layout error rather than something to ignore: silently skipping it
    could hide a corrupted data set.
    """
    found: dict[str, int] = {}
    masks_dir = Path(masks_dir)
    if not masks_dir.is_dir():
        return found
    for path in sorted(masks_dir.glob("*.png")):
        match = MASK_NAME_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected file in masks directory: {path.name}")
        stem = match.group(1)
        if stem in found:
            raise ValueError(f"duplicate mask file: {stem}")
        found[stem] = path.stat().st_size
    return found


def image_files(images_dir: Path) -> dict[str, int]:
    """Return ``{stem: file_bytes}`` for images already on disk (same strictness)."""
    found: dict[str, int] = {}
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        return found
    for path in sorted(images_dir.glob("*.jpg")):
        match = IMAGE_NAME_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"unexpected file in images directory: {path.name}")
        stem = match.group(1)
        if stem in found:
            raise ValueError(f"duplicate image file: {stem}")
        found[stem] = path.stat().st_size
    return found


# --------------------------------------------------------------------------- #
# ID bookkeeping
# --------------------------------------------------------------------------- #
def validation_ids() -> set[str]:
    """Every 2017 validation ID: official mask ZIP plus anything already on disk."""
    ids = set(mask_index(ext.MASK_ZIP))
    ids |= set(mask_files(VALIDATION_ROOT / "masks"))
    ids |= set(image_files(VALIDATION_ROOT / "images"))
    return ids


def project_image_files() -> dict[str, int]:
    """``{stem: file_bytes}`` for images already present anywhere under ``data/raw``."""
    found: dict[str, int] = {}
    for root in (ISIC2016_ROOT, VALIDATION_ROOT, TRAINING_ROOT):
        for role in ("train", "test"):
            found.update(image_files(root / role / "images"))
        found.update(image_files(root / "images"))
    return found


def project_mask_files() -> dict[str, int]:
    """``{stem: file_bytes}`` for masks already present anywhere under ``data/raw``."""
    found: dict[str, int] = {}
    for root in (ISIC2016_ROOT, VALIDATION_ROOT, TRAINING_ROOT):
        for role in ("train", "test"):
            found.update(mask_files(root / role / "masks"))
        found.update(mask_files(root / "masks"))
    return found


def project_mask_ids() -> set[str]:
    """IDs with a mask already present anywhere under ``data/raw``."""
    return set(project_mask_files())


def select_new_ids(
    masks: Mapping[str, object],
    *,
    old_2016: Iterable[str],
    validation: Iterable[str],
) -> list[str]:
    """All 2017 training IDs that are neither 2016 nor 2017-validation.

    ``masks`` keys are validated so a typo in a hand-built mapping cannot leak a
    bogus ID into the manifest.  The result is sorted, hence deterministic.
    """
    official = {_require_id(stem) for stem in masks}
    forbidden: set[str] = set()
    for stem in old_2016:
        forbidden.add(_require_id(stem))
    for stem in validation:
        forbidden.add(_require_id(stem))
    return sorted(official - forbidden)


# --------------------------------------------------------------------------- #
# Official object HEAD (bounded concurrency, deterministic order)
# --------------------------------------------------------------------------- #
def head_metadata(
    stems: Iterable[str],
    *,
    workers: int = MAX_HEAD_WORKERS,
    lookup: Callable[[str], tuple[str, dict[str, object]]] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, dict[str, object]]:
    """Resolve official object metadata for ``stems`` with at most 4 workers.

    Failures propagate: a partial manifest must never be mistaken for a complete
    one.  The returned mapping is sorted by ID.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers > MAX_HEAD_WORKERS:
        raise ValueError(f"workers must not exceed {MAX_HEAD_WORKERS}")
    lookup = lookup or ext.official_object_metadata
    ordered = sorted(_require_id(stem) for stem in stems)
    if len(set(ordered)) != len(ordered):
        raise ValueError("duplicate ID in HEAD list")
    results: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(lookup, stem): stem for stem in ordered}
        for index, future in enumerate(as_completed(futures), 1):
            stem = futures[future]
            got_stem, row = future.result()
            if got_stem != stem:
                raise ValueError(f"HEAD result ID mismatch: {got_stem!r} != {stem!r}")
            if not isinstance(row, dict):
                raise ValueError(f"invalid HEAD metadata for {stem}")
            if row.get("url") != f"https://{ext.IMAGE_HOST}/images/{stem}.jpg":
                raise ValueError(f"non-official HEAD URL for {stem}")
            expected = _require_count(row.get("expected_bytes"), "expected_bytes")
            results[stem] = {"url": row["url"], "expected_bytes": expected,
                             "license": row.get("license", "CC-0")}
            if progress is not None:
                progress(index, len(ordered))
    return dict(sorted(results.items()))


# --------------------------------------------------------------------------- #
# Disk accounting
# --------------------------------------------------------------------------- #
def tree_bytes(root: Path) -> int:
    """Total size of every file under ``root`` (0 when it does not exist)."""
    root = Path(root)
    if not root.exists():
        return 0
    total = 0
    def raise_walk_error(error: OSError) -> None:
        raise error
    for base, _dirs, names in os.walk(root, onerror=raise_walk_error):
        for name in names:
            path = Path(base) / name
            total += path.stat().st_size
    return total


def _sum_files(paths: Iterable[Path]) -> int:
    total = 0
    for path in paths:
        try:
            total += Path(path).stat().st_size
        except OSError:
            continue
    return total


def measure_project(root: Path = PROJECT) -> dict[str, int]:
    """Measure the bytes already occupied by this project.

    ``data/raw`` is split out because it is the part this survey may grow, and
    ``<root>/.venv`` is reported separately because a local virtual environment
    is not data owned by the project.  ``total`` is the honest whole-tree size
    (every file under ``root``) and is what the budget check uses.
    """
    root = Path(root)
    total = tree_bytes(root)
    data_raw = tree_bytes(root / "data" / "raw")
    venv = tree_bytes(root / ".venv")
    return {
        "project_bytes": total,
        "data_raw_bytes": data_raw,
        "venv_bytes": venv,
        "non_data_bytes": total - data_raw,
    }


def estimate_disk(
    manifest: Mapping[str, object],
    masks: Mapping[str, ZipMask],
    *,
    root: Path = PROJECT,
    data_root: Path | None = None,
    images_dir: Path | None = None,
    masks_dir: Path | None = None,
) -> dict[str, object]:
    """Estimate the extra bytes a download of ``manifest`` would create.

    ``root`` is the project measured for the whole-project limit and ``data_root``
    defaults to its existing ``data/raw/isic2017_training`` directory. A pre-existing
    image whose length differs from the manifest is reported in
    ``corrupt_existing_images`` and its full size is counted as required, because
    re-downloading it is the only safe repair.
    """
    data_root = Path(data_root) if data_root is not None else Path(root) / "data" / "raw" / "isic2017_training"
    images_dir = Path(images_dir) if images_dir is not None else data_root / "images"
    masks_dir = Path(masks_dir) if masks_dir is not None else data_root / "masks"
    project = measure_project(root)

    items = manifest.get("items")
    if not isinstance(items, Mapping) or not items:
        raise ValueError("manifest items are missing or empty")
    unknown_items = sorted(set(items) - set(masks))
    if unknown_items:
        raise ValueError(f"manifest mask missing from ZIP index: {unknown_items[:3]}")

    on_disk_images = image_files(images_dir)
    on_disk_masks = mask_files(masks_dir)

    existing_image_bytes = 0
    existing_image_ids = 0
    corrupt: list[str] = []
    required_image_bytes = 0
    manifest_bytes = 0
    for stem, row in sorted(items.items()):
        _require_id(stem)
        if not isinstance(row, Mapping):
            raise ValueError(f"invalid manifest item: {stem}")
        expected = _require_count(row.get("expected_bytes"), f"expected_bytes for {stem}")
        manifest_bytes += expected
        actual = on_disk_images.get(stem)
        if actual is None:
            required_image_bytes += expected
        elif actual == expected:
            existing_image_bytes += actual
            existing_image_ids += 1
        else:
            corrupt.append(stem)
            required_image_bytes += expected

    selected = set(items)
    existing_mask_ids = {stem for stem in selected if stem in on_disk_masks}
    missing_mask_ids = selected - existing_mask_ids
    existing_mask_bytes = sum(on_disk_masks[stem] for stem in existing_mask_ids)
    missing_mask_bytes = sum(
        masks[stem].bytes for stem in missing_mask_ids if stem in masks
    )
    unknown = sorted(missing_mask_ids - set(masks))
    if unknown:
        raise ValueError(f"manifest mask missing from ZIP index: {unknown[:3]}")

    added_bytes = required_image_bytes + missing_mask_bytes
    project_bytes = _require_count(project.get("project_bytes"), "project_bytes")
    return {
        "project": project,
        "project_bytes_before": project_bytes,
        "selected_count": len(selected),
        "existing_image_count": existing_image_ids,
        "existing_image_bytes": existing_image_bytes,
        "required_image_bytes": required_image_bytes,
        "existing_mask_count": len(existing_mask_ids),
        "existing_mask_bytes": existing_mask_bytes,
        "missing_mask_count": len(missing_mask_ids),
        "missing_mask_bytes": missing_mask_bytes,
        "unknown_mask_ids": unknown,
        "corrupt_existing_images": corrupt,
        "manifest_bytes": manifest_bytes,
        "added_bytes": added_bytes,
        "project_bytes_after": project_bytes + added_bytes,
    }


def assert_budget(
    estimate: Mapping[str, object],
    *,
    project_max_bytes: int = PROJECT_MAX_BYTES,
    manifest_max_bytes: int = MANIFEST_MAX_BYTES,
) -> None:
    """Enforce both hard limits; raise :class:`BudgetError` before any write.

    Checks are intentionally cheap and explicit so a caller cannot accidentally
    download first and discover the limit later.  The estimate's own arithmetic
    is re-verified here so a tampered or stale estimate cannot pass the limits by
    under-reporting its totals.
    """
    if project_max_bytes <= 0 or manifest_max_bytes <= 0:
        raise ValueError("budget limits must be positive")
    manifest_bytes = _require_count(estimate.get("manifest_bytes"), "manifest_bytes")
    added_bytes = _require_count(estimate.get("added_bytes"), "added_bytes")
    after = _require_count(estimate.get("project_bytes_after"), "project_bytes_after")
    before = _require_count(estimate.get("project_bytes_before"), "project_bytes_before")
    existing_image_bytes = _require_count(estimate.get("existing_image_bytes"),
                                          "existing_image_bytes")
    required_image_bytes = _require_count(estimate.get("required_image_bytes"),
                                          "required_image_bytes")
    missing_mask_bytes = _require_count(estimate.get("missing_mask_bytes"),
                                        "missing_mask_bytes")
    if existing_image_bytes + required_image_bytes != manifest_bytes:
        raise ValueError("manifest accounting mismatch")
    if added_bytes != required_image_bytes + missing_mask_bytes:
        raise ValueError("added-size accounting mismatch")
    if before + added_bytes != after:
        raise ValueError("project size accounting mismatch")
    if manifest_bytes > manifest_max_bytes:
        raise BudgetError(
            f"manifest raw images {manifest_bytes} exceed {manifest_max_bytes} bytes; "
            "nothing downloaded"
        )
    if after > project_max_bytes:
        raise BudgetError(
            f"project would reach {after} bytes, over the {project_max_bytes} byte limit; "
            "nothing downloaded"
        )


def budget_summary(estimate: Mapping[str, object], *,
                   project_max_bytes: int = PROJECT_MAX_BYTES,
                   manifest_max_bytes: int = MANIFEST_MAX_BYTES) -> dict[str, object]:
    """Human-readable budget view; never raises, so a report can show overruns."""
    manifest_bytes = int(estimate.get("manifest_bytes", 0))
    after = int(estimate.get("project_bytes_after", 0))
    return {
        "project_max_bytes": project_max_bytes,
        "manifest_max_bytes": manifest_max_bytes,
        "manifest_bytes": manifest_bytes,
        "manifest_within_limit": manifest_bytes <= manifest_max_bytes,
        "project_bytes_after": after,
        "project_within_limit": after <= project_max_bytes,
        "confirmed": manifest_bytes <= manifest_max_bytes and after <= project_max_bytes,
    }


# --------------------------------------------------------------------------- #
# Survey
# --------------------------------------------------------------------------- #
def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with part.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(part, path)


def _write_json(path: Path, payload: object) -> None:
    _atomic_bytes(path, json.dumps(payload, ensure_ascii=False, indent=2,
                                   sort_keys=True).encode("utf-8"))


def _relative_label(path: Path, root: Path) -> str:
    """Project-relative label when ``path`` is inside ``root``, else the full path."""
    path = Path(path)
    root = Path(root)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def survey(
    *,
    root: Path = ROOT,
    archive: Path = MASK_ZIP,
    project_root: Path = PROJECT,
    workers: int = MAX_HEAD_WORKERS,
    lookup: Callable[[str], tuple[str, dict[str, object]]] | None = None,
    project_max_bytes: int = PROJECT_MAX_BYTES,
    manifest_max_bytes: int = MANIFEST_MAX_BYTES,
    write: bool = True,
) -> dict[str, object]:
    """Build the deterministic manifest for every new 2017 training ID.

    Nothing is downloaded.  Both budgets are checked **before** the survey files
    are written, so a rejected plan leaves the download directory untouched.
    """
    index = mask_index(archive)
    old_2016 = ext.existing_2016_ids()
    validation = validation_ids()
    selected = select_new_ids(index, old_2016=old_2016, validation=validation)
    if not selected:
        raise ValueError("no new 2017 training ID left after exclusions")
    def progress(done: int, total: int) -> None:
        if done % 50 == 0 or done == total:
            print(f"official object HEAD {done}/{total}", flush=True)
    remote = head_metadata(selected, workers=workers, lookup=lookup, progress=progress)
    items = {
        stem: {**index[stem].as_row(), **remote[stem]} for stem in selected
    }
    manifest: dict[str, object] = {
        "schema": "role_a.isic2017_full_survey/1",
        "source_masks_url": MASK_SOURCE_URL,
        "source_masks_path": _relative_label(archive, project_root),
        "source_masks_sha256": _sha256(Path(archive).read_bytes()),
        "source_masks_zip_bytes": Path(archive).stat().st_size,
        "source_images_pattern": f"https://{ext.IMAGE_HOST}/images/{{isic_id}}.jpg",
        "source_license": PROJECT_LICENSE,
        "selection_rule": (
            "all official 2017 training mask IDs minus every 2016 train/test ID and "
            "every 2017 validation ID (no sub-sampling)"
        ),
        "official_training_mask_count": len(index),
        "excluded_2016_ids": sorted(set(index) & set(old_2016)),
        "excluded_validation_ids": sorted(set(index) & set(validation)),
        "selected_count": len(items),
        "estimated_image_bytes": sum(int(row["expected_bytes"]) for row in items.values()),
        "head_workers": workers,
        "items": items,
    }
    estimate = estimate_disk(manifest, index, root=project_root)
    assert_budget(estimate, project_max_bytes=project_max_bytes,
                  manifest_max_bytes=manifest_max_bytes)
    result: dict[str, object] = {
        **manifest,
        "disk_estimate": estimate,
        "budget": budget_summary(estimate, project_max_bytes=project_max_bytes,
                                 manifest_max_bytes=manifest_max_bytes),
        "downloaded": False,
    }
    if write:
        _write_json(Path(root) / SURVEY_NAME, result)
        _write_json(
            Path(root) / MANIFEST_NAME,
            {
                "schema": manifest["schema"],
                "source_masks_sha256": manifest["source_masks_sha256"],
                "selection_rule": manifest["selection_rule"],
                "selected_count": manifest["selected_count"],
                "estimated_image_bytes": manifest["estimated_image_bytes"],
                "budget": result["budget"],
                "items": items,
            },
        )
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _printable(result: Mapping[str, object]) -> dict[str, object]:
    return {key: result[key] for key in ("selected_count", "estimated_image_bytes",
                                          "disk_estimate", "budget", "downloaded") if key in result}


def register_commands(commands: argparse._SubParsersAction) -> list[str]:
    """Attach the read-only survey command."""
    survey_cmd = commands.add_parser("survey", help="HEAD every new ID; download nothing")
    survey_cmd.add_argument("--root", type=Path, default=ROOT)
    survey_cmd.add_argument("--mask-zip", type=Path, default=MASK_ZIP)
    survey_cmd.add_argument("--workers", type=int, default=MAX_HEAD_WORKERS)
    survey_cmd.add_argument("--project-max-bytes", type=int, default=PROJECT_MAX_BYTES)
    survey_cmd.add_argument("--manifest-max-bytes", type=int, default=MANIFEST_MAX_BYTES)
    survey_cmd.add_argument("--dry-run", action="store_true",
                            help="print the plan without writing survey files")
    return ["survey"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    register_commands(parser.add_subparsers(dest="command", required=True))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.workers > MAX_HEAD_WORKERS:
        raise SystemExit(f"--workers must not exceed {MAX_HEAD_WORKERS}")
    result = survey(root=args.root, archive=args.mask_zip, workers=args.workers,
                    project_max_bytes=args.project_max_bytes,
                    manifest_max_bytes=args.manifest_max_bytes,
                    write=not args.dry_run)
    print(json.dumps(_printable(result), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
