"""Download and verify the official ISIC 2018 Task 1 test pairs within the project cap."""
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


ROOT = base.PROJECT / "data" / "raw" / "isic2018_test"
CAP = 17_000_000_000
SOURCES = {
    "images": ("https://isic-archive.s3.amazonaws.com/challenges/2018/"
               "ISIC2018_Task1-2_Test_Input.zip", 2_370_457_338,
               re.compile(r"ISIC_\d{7}\.jpg", re.IGNORECASE)),
    "masks": ("https://isic-archive.s3.amazonaws.com/challenges/2018/"
              "ISIC2018_Task1_Test_GroundTruth.zip", 9_680_101,
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


def fetch(url: str, expected: int, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if target.exists():
        if target.stat().st_size != expected:
            raise ValueError(f"Existing archive has wrong size: {target}")
        return sha256_file(target)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > expected:
        raise ValueError(f"Partial archive exceeds official HEAD size: {partial}")
    if offset == expected:
        os.replace(partial, target)
        return sha256_file(target)
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = response.status
        if offset and status != 206:
            raise ValueError(f"Server did not honor resume range: {status}")
        if offset and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("Resume Content-Range does not match partial file")
        with partial.open("ab" if offset else "wb") as stream:
            last_report = offset // 100_000_000
            while block := response.read(1 << 20):
                stream.write(block)
                offset += len(block)
                if offset > expected:
                    raise ValueError("Archive exceeded its official HEAD size")
                milestone = offset // 100_000_000
                if milestone > last_report:
                    print(f"download {target.name}: {offset}/{expected}", flush=True)
                    last_report = milestone
    if offset != expected:
        raise ValueError(f"Download incomplete: {offset}/{expected}; keep {partial} for resume")
    os.replace(partial, target)
    return sha256_file(target)


def crc32_file(path: Path) -> int:
    crc = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            crc = zlib.crc32(block, crc)
    return crc & 0xffffffff


def extract_pairs(archive: Path, directory: Path, pattern: re.Pattern[str]) -> int:
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        members = [info for info in bundle.infolist()
                   if not info.is_dir() and pattern.fullmatch(Path(info.filename).name)]
        names = [Path(info.filename).name for info in members]
        if len(members) != 1000 or len(set(names)) != 1000:
            raise ValueError(f"Archive has {len(members)} expected entries, not 1000 unique files")
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
            if index % 100 == 0:
                print(f"extract {directory.name}: {index}/1000", flush=True)
    return 1000


def verify_pairs(source_hashes: dict[str, str]) -> dict:
    images = {p.stem: p for p in (ROOT / "images").glob("ISIC_*.jpg")}
    masks = {p.name.removesuffix("_segmentation.png"): p
             for p in (ROOT / "masks").glob("ISIC_*_segmentation.png")}
    if len(images) != 1000 or set(images) != set(masks):
        raise ValueError("Images and masks do not form 1000 complete pairs")
    items = {}
    for index, stem in enumerate(sorted(images), 1):
        image, mask = images[stem], masks[stem]
        with Image.open(image) as loaded:
            loaded.verify()
        with Image.open(mask) as loaded:
            loaded.verify()
        items[stem] = {"image_bytes": image.stat().st_size,
                       "image_sha256": sha256_file(image),
                       "mask_bytes": mask.stat().st_size,
                       "mask_sha256": sha256_file(mask),
                       "area_ratio": base.area_ratio(mask)}
        if index % 100 == 0:
            print(f"verify pairs: {index}/1000", flush=True)
    return {"schema": "isic2018-task1-test-provenance-v1",
            "source": {key: {"url": SOURCES[key][0], "zip_bytes": SOURCES[key][1],
                             "zip_sha256": digest} for key, digest in source_hashes.items()},
            "license": "CC0 according to https://challenge.isic-archive.com/data/",
            "count": len(items), "complete": True, "items": items}


def main() -> None:
    provenance = ROOT / "provenance.json"
    if provenance.exists():
        prior = json.loads(provenance.read_text(encoding="utf-8"))
        if prior.get("complete") is True and prior.get("count") == 1000:
            print("Already complete; retained existing verified dataset", flush=True)
            return
    ROOT.mkdir(parents=True, exist_ok=True)
    initial = project_bytes()
    if initial + 5_500_000_000 > CAP:
        raise ValueError(f"Preflight exceeds 17 GB cap: current={initial}")
    print(f"preflight project_bytes={initial} cap={CAP}", flush=True)
    archives = {}
    source_hashes = {}
    for key, (url, expected, _pattern) in SOURCES.items():
        archive = ROOT / "downloads" / f"{key}.zip"
        source_hashes[key] = fetch(url, expected, archive)
        archives[key] = archive
        print(f"verified archive {key}: {expected} bytes SHA256 {source_hashes[key]}", flush=True)
    for key, (_url, _expected, pattern) in SOURCES.items():
        extract_pairs(archives[key], ROOT / key, pattern)
    evidence = verify_pairs(source_hashes)
    atomic_write_json(provenance, evidence)
    for archive in archives.values():
        archive.unlink()
    print(f"complete pairs={evidence['count']} project_bytes={project_bytes()}", flush=True)


if __name__ == "__main__":
    main()
