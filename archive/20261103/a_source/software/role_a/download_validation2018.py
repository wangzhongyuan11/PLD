"""A07: download, verify and expand the official ISIC 2018 Task 1 validation pairs."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import urllib.request
import zipfile
import zlib
from pathlib import Path

from PIL import Image

from software.role_a import refine_small as base
from software.role_a.refine_from_baseline_v3 import atomic_write_json


ROOT = base.PROJECT / "data" / "raw" / "isic2018_validation"
AUDIT = base.PROJECT / "runs" / "role_a_isic2018_validation_audit"
CAP = 17_000_000_000
EXPECTED_PAIRS = 100
SOURCES = {
    "images": ("https://isic-archive.s3.amazonaws.com/challenges/2018/"
               "ISIC2018_Task1-2_Validation_Input.zip", 239_231_159,
               re.compile(r"ISIC_\d{7}\.jpg", re.IGNORECASE)),
    "masks": ("https://isic-archive.s3.amazonaws.com/challenges/2018/"
              "ISIC2018_Task1_Validation_GroundTruth.zip", 759_706,
              re.compile(r"ISIC_\d{7}_segmentation\.png", re.IGNORECASE)),
}


def project_bytes() -> int:
    total = 0
    for directory, subdirs, files in os.walk(base.PROJECT, followlinks=False):
        subdirs[:] = [name for name in subdirs
                      if not os.path.isjunction(os.path.join(directory, name))
                      and not os.path.islink(os.path.join(directory, name))]
        for name in files:
            total += os.path.getsize(os.path.join(directory, name))
    return total


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def crc32_file(path: Path) -> int:
    crc = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            crc = zlib.crc32(block, crc)
    return crc & 0xffffffff


def fetch(url: str, expected: int, target: Path) -> dict:
    """下载（支持断点续传），返回 URL/期望长度/实际字节/SHA256。只在长度与官方 HEAD 一致时接受。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if target.exists():
        size = target.stat().st_size
        if size != expected:
            raise ValueError(f"Existing archive has wrong size: {target} {size} != {expected}")
        return {"url": url, "expected_bytes": expected, "actual_bytes": size,
                "sha256": sha256_file(target), "resumed": False, "reused": True}
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected:
        raise ValueError(f"Partial archive exceeds official HEAD size: {partial}")
    if offset == expected:
        os.replace(partial, target)
        return {"url": url, "expected_bytes": expected, "actual_bytes": offset,
                "sha256": sha256_file(target), "resumed": True, "reused": False}
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=180) as response:
        status = response.status
        if offset and status != 206:
            raise ValueError(f"Server did not honour resume range: {status}")
        if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("Resume Content-Range does not match partial file")
        with partial.open("ab" if offset else "wb") as stream:
            last = offset // 50_000_000
            while block := response.read(1 << 20):
                stream.write(block)
                offset += len(block)
                if offset > expected:
                    raise ValueError("Archive exceeded its official HEAD size")
                milestone = offset // 50_000_000
                if milestone > last:
                    print(f"  download {target.name}: {offset}/{expected}", flush=True)
                    last = milestone
    if offset != expected:
        raise ValueError(f"Download incomplete: {offset}/{expected}; keep {partial} for resume")
    os.replace(partial, target)
    return {"url": url, "expected_bytes": expected, "actual_bytes": offset,
            "sha256": sha256_file(target), "resumed": bool(headers), "reused": False}


def audit_archive(archive: Path, pattern: re.Pattern[str]) -> dict:
    """ZIP 结构安全审计：路径穿越、绝对路径、隐藏成员、重复文件名、CRC。"""
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        matched, skipped = [], []
        for info in infos:
            if info.is_dir():
                continue
            entry = info.filename.replace("\\", "/")
            basename = entry.rsplit("/", 1)[-1]
            (matched if pattern.fullmatch(basename) else skipped).append(entry)
        basenames = [name.rsplit("/", 1)[-1] for name in matched]
        duplicates = sorted({name for name in basenames if basenames.count(name) > 1})
        unsafe = []
        for info in infos:
            entry = info.filename.replace("\\", "/")
            parts = [part for part in entry.split("/") if part not in ("", ".")]
            if (entry.startswith("/") or re.match(r"^[A-Za-z]:", entry)
                    or any(part == ".." for part in parts)):
                unsafe.append(info.filename)
        return {
            "archive": archive.name, "archive_bytes": archive.stat().st_size,
            "sha256": sha256_file(archive),
            "members_total": len(infos), "matched_files": len(matched),
            "skipped_members": sorted(skipped),
            "duplicate_basenames": duplicates,
            "unsafe_paths": unsafe,
            "testzip_bad_member": bundle.testzip(),
            "matched_names": sorted(matched),
            "compress_types": sorted({info.compress_type for info in infos}),
            "encrypted_members": sorted(info.filename for info in infos
                                        if info.flag_bits & 0x1),
        }


def extract_pairs(archive: Path, directory: Path, pattern: re.Pattern[str]) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = [info for info in bundle.infolist()
                   if not info.is_dir() and pattern.fullmatch(Path(info.filename).name)]
        names = [Path(info.filename).name for info in members]
        if len(members) != EXPECTED_PAIRS or len(set(names)) != EXPECTED_PAIRS:
            raise ValueError(
                f"Archive has {len(members)} expected entries, not {EXPECTED_PAIRS} unique files")
        remaining = sum(info.file_size for info in members)
        if project_bytes() + remaining > CAP:
            raise ValueError("Extracted archive would exceed the 17 GB project cap")
        for index, info in enumerate(members, 1):
            target = directory / Path(info.filename).name
            if target.exists() and target.stat().st_size == info.file_size and crc32_file(target) == info.CRC:
                remaining -= info.file_size
                continue
            temporary = target.with_suffix(target.suffix + ".part")
            with bundle.open(info) as source, temporary.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=1 << 20)
            if temporary.stat().st_size != info.file_size or crc32_file(temporary) != info.CRC:
                raise ValueError(f"Extracted file failed CRC: {info.filename}")
            os.replace(temporary, target)
            remaining -= info.file_size
            if index % 25 == 0:
                print(f"  extract {directory.name}: {index}/{EXPECTED_PAIRS}", flush=True)
    return EXPECTED_PAIRS


def verify_pairs(source_hashes: dict) -> dict:
    images = {p.stem: p for p in (ROOT / "images").glob("ISIC_*.jpg")}
    masks = {p.name.removesuffix("_segmentation.png"): p
             for p in (ROOT / "masks").glob("ISIC_*_segmentation.png")}
    if len(images) != EXPECTED_PAIRS or set(images) != set(masks):
        raise ValueError(f"Images and masks do not form {EXPECTED_PAIRS} complete pairs")
    items = {}
    for index, stem in enumerate(sorted(images), 1):
        image, mask = images[stem], masks[stem]
        with Image.open(image) as loaded:
            loaded.verify()
            image_size = list(loaded.size)
        with Image.open(mask) as loaded:
            loaded.verify()
            mask_size = list(loaded.size)
            mask_mode = loaded.mode
        items[stem] = {"image_bytes": image.stat().st_size,
                       "image_sha256": sha256_file(image),
                       "image_size": image_size,
                       "mask_bytes": mask.stat().st_size,
                       "mask_sha256": sha256_file(mask),
                       "mask_size": mask_size,
                       "mask_mode": mask_mode,
                       "area_ratio": base.area_ratio(mask)}
        if index % 25 == 0:
            print(f"  verify pairs: {index}/{EXPECTED_PAIRS}", flush=True)
    return {"schema": "isic2018-task1-validation-provenance-v1",
            "source": {key: {**value, "expected_bytes": SOURCES[key][1]}
                       for key, value in source_hashes.items()},
            "license": "CC0 according to https://challenge.isic-archive.com/data/",
            "count": len(items), "complete": True, "items": items}


def main() -> None:
    provenance = ROOT / "provenance.json"
    if provenance.exists():
        prior = json.loads(provenance.read_text(encoding="utf-8"))
        if prior.get("complete") is True and prior.get("count") == EXPECTED_PAIRS:
            print("Already complete; retained existing verified dataset", flush=True)
            return
    ROOT.mkdir(parents=True, exist_ok=True)
    AUDIT.mkdir(parents=True, exist_ok=True)
    initial = project_bytes()
    peak_estimate = initial + sum(SOURCES[key][1] for key in SOURCES) + 260_000_000
    print(f"preflight project_bytes={initial} estimated_peak={peak_estimate} cap={CAP}", flush=True)
    if peak_estimate > CAP:
        raise ValueError(f"Preflight peak exceeds 17 GB cap: {peak_estimate}")
    source_hashes, archives = {}, {}
    for key, (url, expected, _pattern) in SOURCES.items():
        archive = ROOT / "downloads" / f"{key}.zip"
        source_hashes[key] = fetch(url, expected, archive)
        archives[key] = archive
        print(f"verified archive {key}: {source_hashes[key]['actual_bytes']} bytes "
              f"SHA256 {source_hashes[key]['sha256']}", flush=True)
    manifest = {"schema": "isic2018-validation-zip-manifest-v1",
                "source_page": "https://challenge.isic-archive.com/data/",
                "license": "CC0", "archives": {}}
    for key, (_url, _expected, pattern) in SOURCES.items():
        manifest["archives"][key] = audit_archive(archives[key], pattern)
        entry = manifest["archives"][key]
        if entry["unsafe_paths"] or entry["duplicate_basenames"] or entry["testzip_bad_member"] \
                or entry["encrypted_members"] or entry["matched_files"] != EXPECTED_PAIRS:
            raise ValueError(f"Archive failed safety audit: {key} {entry}")
        print(f"  zip audit {key}: members={entry['members_total']} "
              f"matched={entry['matched_files']} skipped={len(entry['skipped_members'])} "
              f"unsafe=0 duplicates=0 crc=ok", flush=True)
    atomic_write_json(AUDIT / "zip_manifest.json", manifest)
    for key, (_url, _expected, pattern) in SOURCES.items():
        extract_pairs(archives[key], ROOT / key, pattern)
    evidence = verify_pairs(source_hashes)
    atomic_write_json(provenance, evidence)
    print(f"complete pairs={evidence['count']} project_bytes={project_bytes()}", flush=True)


if __name__ == "__main__":
    main()