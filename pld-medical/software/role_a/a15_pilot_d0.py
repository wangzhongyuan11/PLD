"""A15-D0：ISIC 2018 Task1 官方训练掩膜 ZIP + 最多 20 对图的确定性试点。

约束：
* 只下载官方掩膜 ZIP（约 27 MB），**绝不下载 10.4 GB 整包图像**；
* 图像经官方 Archive API 按 ISIC ID 单张获取（GET /api/v2/images/<id>/ -> files.full.url）；
* 逐张缩到最长边 512、保持比例，图/掩膜同尺寸；原图临时副本用后即删；
* 只用文件名 + SHA256 + dHash 与现有 train/dev/final-test **图像**做重复检查；
* 不打开 2018 final test 掩膜，不做任何推理或评分。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import urllib.request
import zipfile
import zlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

from software.role_a import train_overall_v1 as overall
from software.role_a.audit_image_overlap import dhash, records

ROOT = Path("data/raw/isic2018_training_pilot")
AUDIT = Path("runs/role_a_a15_pilot_d0")
ZIP_URL = ("https://isic-archive.s3.amazonaws.com/challenges/2018/"
           "ISIC2018_Task1_Training_GroundTruth.zip")
ZIP_NAME = "ISIC2018_Task1_Training_GroundTruth.zip"
ZIP_BYTES = 27_402_895
API = "https://api.isic-archive.com/api/v2/images/{isic_id}/"
MASK_PATTERN = re.compile(r"ISIC_\d{7}_segmentation\.png", re.IGNORECASE)
SELECT_SEED = 20260926
TAKE = 20
LONG_SIDE = 512
EXISTING = {
    "train2016": Path("data/raw/isic2016/train/images"),
    "train2017": Path("data/raw/isic2017_training/images"),
    "validation2017": Path("data/raw/isic2017_validation/images"),
    "development2016": Path("data/raw/isic2016/test/images"),
    "development2017": Path("data/raw/isic2017_test/images"),
    "final_test2018_images": Path("data/raw/isic2018_test/images"),
}


def _target_size(size: tuple[int, int]) -> tuple[int, int]:
    """最长边缩到 LONG_SIDE，保持比例。"""
    scale = LONG_SIDE / max(size)
    return (max(1, round(size[0] * scale)), max(1, round(size[1] * scale)))


def safe_cleanup(tmp: Path, root: Path) -> int:
    """逐个删除 tmp 内文件后删空目录；严禁用 rmtree，且清理异常绝不忽略。"""
    tmp_resolved, root_resolved = Path(tmp).resolve(), Path(root).resolve()
    if tmp_resolved == root_resolved or root_resolved not in tmp_resolved.parents:
        raise SystemExit(f"拒绝递归清理 {tmp_resolved}：不在试点目录 {root_resolved} 内")
    removed = 0
    for path in sorted(tmp_resolved.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file():
            path.unlink()
            removed += 1
        elif path.is_dir():
            path.rmdir()
    tmp_resolved.rmdir()
    return removed


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_zip(target: Path) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".zip.part")
    if target.exists() and target.stat().st_size == ZIP_BYTES:
        return {"reused": True, "bytes": ZIP_BYTES, "sha256": sha256_file(target)}
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(ZIP_URL, headers=headers)
    with urllib.request.urlopen(request, timeout=180) as response:
        with partial.open("ab" if offset else "wb") as stream:
            while block := response.read(1 << 20):
                stream.write(block)
                offset += len(block)
                if offset > ZIP_BYTES:
                    raise ValueError("ZIP 超过官方 HEAD 长度")
    if offset != ZIP_BYTES:
        raise ValueError(f"下载不完整 {offset}/{ZIP_BYTES}；保留 {partial}")
    os.replace(partial, target)
    return {"reused": False, "bytes": target.stat().st_size, "sha256": sha256_file(target)}


def audit_zip(archive: Path) -> dict[str, Any]:
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        matched, skipped = [], []
        for info in infos:
            if info.is_dir():
                continue
            entry = info.filename.replace("\\", "/")
            base = entry.rsplit("/", 1)[-1]
            (matched if MASK_PATTERN.fullmatch(base) else skipped).append(entry)
        names = [n.rsplit("/", 1)[-1] for n in matched]
        unsafe = [i.filename for i in infos
                  if i.filename.replace("\\", "/").startswith("/")
                  or ".." in i.filename.replace("\\", "/").split("/")]
        order = sorted(range(len(matched)), key=lambda i: names[i])
        return {"members": len(infos), "matched": len(matched),
                "duplicates": sorted({n for n in names if names.count(n) > 1}),
                "unsafe_paths": unsafe, "testzip_bad": bundle.testzip(),
                "skipped": sorted(skipped)[:5],
                "names": [names[i] for i in order],
                "entries": [matched[i] for i in order]}


def api_image_url(isic_id: str) -> tuple[str, dict[str, Any]]:
    request = urllib.request.Request(API.format(isic_id=isic_id),
                                     headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
    files = payload.get("files") or {}
    full = (files.get("full") or {}).get("url")
    if not full:
        raise ValueError(f"{isic_id}: API 未返回 files.full.url")
    return full, {"license": payload.get("copyright_license"),
                  "attribution": payload.get("attribution"),
                  "api_size": (files.get("full") or {}).get("size")}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    args = parser.parse_args(list(argv) if argv is not None else None)
    root, audit = Path(args.root), Path(args.audit)
    for sub in ("downloads", "images", "masks"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    audit.mkdir(parents=True, exist_ok=True)
    tmp = root / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    archive = root / "downloads" / ZIP_NAME
    download = fetch_zip(archive)
    print(json.dumps({"zip": str(archive), **download}, ensure_ascii=False), flush=True)
    info = audit_zip(archive)
    if info["unsafe_paths"] or info["duplicates"] or info["testzip_bad"]:
        raise SystemExit(f"ZIP 安全审计失败: {info}")
    if info["matched"] != 2594:
        raise SystemExit(f"掩膜条目 {info['matched']} != 2594")
    (audit / "zip_manifest.json").write_text(
        json.dumps({"schema": "a15-d0-zip-manifest-v1", "url": ZIP_URL,
                    "bytes": download["bytes"], "sha256": download["sha256"],
                    "matched_masks": info["matched"], "members": info["members"],
                    "unsafe_paths": info["unsafe_paths"], "duplicates": info["duplicates"],
                    "testzip_bad": info["testzip_bad"], "skipped_sample": info["skipped"],
                    "mask_names_sha256": hashlib.sha256(
                        "\n".join(info["names"]).encode()).hexdigest()},
                   ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    picked = sorted(random.Random(SELECT_SEED).sample(info["names"], TAKE))
    entry_of = dict(zip(info["names"], info["entries"]))
    items: list[dict[str, Any]] = []
    temp_bytes = 0
    with zipfile.ZipFile(archive) as bundle:
        for stem_name in picked:
            isic_id = stem_name.removesuffix("_segmentation.png")
            url, meta = api_image_url(isic_id)
            raw = tmp / f"{isic_id}.jpg"
            urllib.request.urlretrieve(url, raw)
            temp_bytes += raw.stat().st_size
            with Image.open(raw) as source:
                source = source.convert("RGB")
                image_original_size = source.size
                target = _target_size(image_original_size)
                resized_image = source.resize(target, Image.Resampling.LANCZOS)
            with bundle.open(entry_of[stem_name]) as stream:
                mask = Image.open(stream).convert("L")
            mask_original_size = mask.size
            if mask_original_size != image_original_size:
                raise SystemExit(f"{isic_id}: 原图 {image_original_size} 与原掩膜 "
                                 f"{mask_original_size} 尺寸不一致，拒绝保存")
            mask_values = set(np.unique(np.asarray(mask)).tolist())
            resized_mask = mask.resize(target, Image.Resampling.NEAREST)
            resized_values = set(np.unique(np.asarray(resized_mask)).tolist())
            image_path = root / "images" / f"{isic_id}.jpg"
            mask_path = root / "masks" / f"{isic_id}_segmentation.png"
            resized_image.save(image_path, format="JPEG", quality=92)
            resized_mask.save(mask_path, format="PNG")
            raw.unlink()
            with Image.open(image_path) as check_image, Image.open(mask_path) as check_mask:
                saved_image_size = tuple(check_image.size)
                saved_mask_size = tuple(check_mask.size)
                saved_mask_values = sorted(set(np.unique(
                    np.asarray(check_mask.convert("L"))).tolist()))
            if not (saved_image_size == saved_mask_size == target):
                raise SystemExit(f"{isic_id}: 落盘后图/掩膜尺寸不一致 "
                                 f"{saved_image_size}/{saved_mask_size} != {target}")
            if not set(saved_mask_values) <= {0, 255}:
                raise SystemExit(f"{isic_id}: 落盘掩膜非二值 {saved_mask_values}")
            items.append({
                "isic_id": isic_id, "zip_name": stem_name,
                "image_original_size": list(image_original_size),
                "mask_original_size": list(mask_original_size),
                "image_resized_size": list(saved_image_size),
                "mask_resized_size": list(saved_mask_size),
                "source_url": url, "api_meta": meta,
                "image_bytes": image_path.stat().st_size,
                "image_sha256": sha256_file(image_path),
                "mask_bytes": mask_path.stat().st_size,
                "mask_sha256": sha256_file(mask_path),
                "mask_values_original": sorted(mask_values),
                "mask_values_resized": sorted(saved_mask_values),
                "image_dhash": dhash(image_path),
            })
    removed_temp_files = safe_cleanup(tmp, root)

    mismatched = [i["isic_id"] for i in items
                  if not (set(i["mask_values_original"]) <= {0, 255})
                  or i["image_bytes"] == 0 or i["mask_bytes"] == 0]
    checks = {
        "pairs": len(items),
        "original_sizes_match": all(i["image_original_size"] == i["mask_original_size"]
                                    for i in items),
        "resized_sizes_match": all(i["image_resized_size"] == i["mask_resized_size"]
                                   for i in items),
        "resized_long_side_max": max(max(i["image_resized_size"]) for i in items),
        "masks_binary": not mismatched,
        "nonbinary_ids": mismatched,
    }

    existing = {name: records(path) for name, path in EXISTING.items() if path.is_dir()}
    overlaps = {}
    for name, rows in existing.items():
        exact, close = [], []
        for item in items:
            for row in rows:
                if row["sha256"] == item["image_sha256"]:
                    exact.append({"isic_id": item["isic_id"], "other": row["stem"]})
                distance = (item["image_dhash"] ^ row["dhash"]).bit_count()
                if distance <= 4:
                    close.append({"isic_id": item["isic_id"], "other": row["stem"],
                                  "distance": distance})
        overlaps[name] = {"other_count": len(rows), "exact_sha256_pairs": exact,
                          "dhash_close_pairs": close}
        print(f"overlap {name}: others={len(rows)} exact={len(exact)} dhash<=4={len(close)}",
              flush=True)

    manifest = {
        "schema": "a15-d0-pilot-manifest-v1",
        "zip": {"url": ZIP_URL, "bytes": download["bytes"], "sha256": download["sha256"],
                "matched_masks": info["matched"]},
        "selection": {"seed": SELECT_SEED, "take": TAKE, "method": "random.Random(seed).sample(sorted(names), 20)"},
        "long_side": LONG_SIDE,
        "temp_originals_downloaded_bytes": temp_bytes,
        "temp_originals_removed": True,
        "items": items,
        "checks": checks,
        "temp_originals_removed_files": removed_temp_files,
        "overlaps_vs_existing_images": overlaps,
        "holdout_masks": "2018 final test masks were NOT opened; only its images were read for dedup",
        "licenses": sorted({i["api_meta"]["license"] for i in items}),
    }
    (audit / "pilot_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    sheet_rows = 5
    cell = 200
    for offset in range(0, len(items), sheet_rows):
        chunk = items[offset:offset + sheet_rows]
        canvas = Image.new("RGB", (cell * 2 + 30, len(chunk) * (cell + 26) + 10), "white")
        draw = ImageDraw.Draw(canvas)
        for index, item in enumerate(chunk):
            top = 8 + index * (cell + 26)
            for column, sub in enumerate(("images", "masks")):
                name = (f"{item['isic_id']}.jpg" if column == 0
                        else f"{item['isic_id']}_segmentation.png")
                with Image.open(root / sub / name) as raw:
                    image = raw.convert("RGB")
                    image.thumbnail((cell, cell), Image.Resampling.LANCZOS)
                x = 6 + column * (cell + 18) + (cell - image.width) // 2
                canvas.paste(image, (x, top + 18 + (cell - image.height) // 2))
            draw.text((6, top), f"{item['isic_id']} "
                                f"orig={item['image_original_size']} -> {item['image_resized_size']}",
                      fill="black")
        path = audit / f"a15_d0_sheet_{offset // sheet_rows + 1}.jpg"
        canvas.save(path, quality=90)
    print(json.dumps({"items": len(items), "licenses": manifest["licenses"],
                      "temp_bytes": temp_bytes,
                      "overlap_summary": {k: {"exact": len(v["exact_sha256_pairs"]),
                                              "close": len(v["dhash_close_pairs"])}
                                          for k, v in overlaps.items()}},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())