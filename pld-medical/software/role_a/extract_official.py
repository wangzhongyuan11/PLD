"""Safely unpack the four official ISIC 2016 Task 1 ZIP archives.

Only filenames in the known flat archive layout are accepted. This script
never downloads data and writes a provenance manifest after pair checks pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


BASE_URL = "https://isic-archive.s3.amazonaws.com/challenges/2016/"
ARCHIVES = {
    "train_images.zip": ("ISBI2016_ISIC_Part1_Training_Data.zip", "train/images", 900, r"ISIC_\d+\.jpg"),
    "train_masks.zip": ("ISBI2016_ISIC_Part1_Training_GroundTruth.zip", "train/masks", 900,
                        r"ISIC_\d+_Segmentation\.png"),
    "test_images.zip": ("ISBI2016_ISIC_Part1_Test_Data.zip", "test/images", 379, r"ISIC_\d+\.jpg"),
    "test_masks.zip": ("ISBI2016_ISIC_Part1_Test_GroundTruth.zip", "test/masks", 379,
                       r"ISIC_\d+_Segmentation\.png"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_one(archive: Path, destination: Path, expected: int, pattern: str) -> list[str]:
    with zipfile.ZipFile(archive) as bundle:
        files = [item for item in bundle.infolist() if not item.is_dir()]
        names = [Path(item.filename).name for item in files]
        if len(names) != expected or len(set(names)) != expected:
            raise ValueError(f"{archive}: expected {expected} unique files, found {len(names)}")
        if any(len(Path(item.filename).parts) != 2 or not re.fullmatch(pattern, name)
               for item, name in zip(files, names)):
            raise ValueError(f"{archive}: unsafe or unexpected archive filename")
        destination.mkdir(parents=True, exist_ok=True)
        for item, name in zip(files, names):
            target = destination / name
            if target.exists() and target.stat().st_size == item.file_size:
                continue
            temporary = target.with_suffix(target.suffix + ".part")
            try:
                with bundle.open(item) as source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    return sorted(names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/isic2016"))
    args = parser.parse_args()
    root: Path = args.root
    manifest = {"source": "https://challenge.isic-archive.com/data/",
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(), "year": 2016,
                "task": "Task 1 lesion segmentation", "archives": {}}
    stems: dict[str, set[str]] = {}
    for local_name, (remote_name, subdir, count, pattern) in ARCHIVES.items():
        archive = root / "downloads" / local_name
        if not archive.is_file():
            raise FileNotFoundError(archive)
        names = extract_one(archive, root / subdir, count, pattern)
        stems[subdir] = {name.removesuffix("_Segmentation.png").removesuffix(".jpg")
                         for name in names}
        manifest["archives"][local_name] = {"url": BASE_URL + remote_name,
                                             "sha256": sha256(archive), "files": count,
                                             "destination": subdir}
    for role in ("train", "test"):
        if stems[f"{role}/images"] != stems[f"{role}/masks"]:
            raise ValueError(f"{role} images and masks do not pair exactly")
    if stems["train/images"] & stems["test/images"]:
        raise ValueError("train and test contain overlapping ISIC identifiers")
    output = root / "provenance.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": output.as_posix(), "train_pairs": 900,
                      "test_pairs": 379}, ensure_ascii=False))


if __name__ == "__main__":
    main()
