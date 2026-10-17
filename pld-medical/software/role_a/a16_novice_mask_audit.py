"""A16-N1：Novice Masks.zip 安全审计 + 全量掩膜质量统计（只在内存读取 PNG，不解压落盘）。"""
from __future__ import annotations
import csv, hashlib, io, json, random, statistics, zipfile
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path("data/raw/isic2018_novice_pilot")
META = Path("data/raw/isic2018_novice_metadata")
AUDIT = Path("runs/role_a_a16_novice_mask_audit")
ZIP = ROOT/"Masks.zip"
SEED = 20260926


def sha256(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as s:
        for b in iter(lambda: s.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


with (META/"Masks_to_IsicId_mapping.csv").open(encoding="utf-8", newline="") as f:
    mapping = list(csv.DictReader(f))
map_name_to_id = {r["Filename"]: r["ISIC_ID"] for r in mapping}
meta_rows = list(csv.DictReader((META/"isic-2018-novice-segmentation-masks.csv").open(encoding="utf-8", newline="")))
meta_by_id = {r["isic_id"]: r for r in meta_rows}

with zipfile.ZipFile(ZIP) as z:
    infos = [i for i in z.infolist() if not i.is_dir()]
    raw_names = [i.filename for i in infos]
    base_names = [n.replace("\\", "/").rsplit("/", 1)[-1] for n in raw_names]
    prefixes = sorted({n.replace("\\", "/").rsplit("/", 1)[0] for n in raw_names if "/" in n.replace("\\", "/")})
    unsafe = []
    for i in infos:
        entry = i.filename.replace("\\", "/")
        parts = [p for p in entry.split("/") if p not in ("", ".")]
        mode = i.external_attr >> 16
        if entry.startswith("/") or ".." in parts or (mode and (mode & 0o170000) == 0o120000):
            unsafe.append(i.filename)
    dup = [n for n, c in Counter(base_names).items() if c > 1]
    non_png = [n for n in base_names if not n.lower().endswith(".png")]
    bad_crc = z.testzip()
    in_map = [n for n in base_names if n not in map_name_to_id]
    zip_set, map_set = set(base_names), set(map_name_to_id)
    manifest = {
        "schema": "a16-n1-zip-manifest-v1",
        "url": "https://isic-archive.s3.amazonaws.com/dois/10.34970-387951/supplements/Masks.zip",
        "bytes": ZIP.stat().st_size, "sha256": sha256(ZIP),
        "entries_total": len(z.infolist()), "entries_files": len(infos),
        "filename_format": {"prefixes": prefixes[:5], "prefix_count": len(prefixes),
                            "sample_raw": raw_names[:3], "sample_normalized": base_names[:3]},
        "safety": {"unsafe_paths": unsafe, "duplicate_basenames": dup, "non_png": non_png,
                   "testzip_bad_member": bad_crc, "symlink_entries": [],
                   "zip_entries_not_in_mapping": in_map,
                   "mapping_not_in_zip": sorted(map_set - zip_set),
                   "counts_equal": len(base_names) == len(mapping)},
    }
    if unsafe or dup or non_png or bad_crc or in_map or (map_set - zip_set) or len(base_names) != len(mapping):
        (AUDIT/"zip_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        raise SystemExit("ZIP 安全/映射审计失败，已保存 manifest 并停止")

    qc = []
    for info, base in zip(infos, base_names):
        isic_id = map_name_to_id[base]
        rec = {"isic_id": isic_id, "zip_name": info.filename, "compressed_bytes": info.compress_size,
               "uncompressed_bytes": info.file_size, "crc32": f"{info.CRC:08x}"}
        try:
            with z.open(info) as stream:
                data = stream.read()
            with Image.open(io.BytesIO(data)) as img:
                rec["format"] = img.format
                rec["mode"] = img.mode
                rec["width"], rec["height"] = img.size
                arr = np.asarray(img.convert("L"))
            uniq = np.unique(arr)
            rec["n_unique"] = int(uniq.size)
            rec["values_head"] = [int(v) for v in uniq[:5]]
            rec["min"] = int(arr.min()); rec["max"] = int(arr.max())
            fg = int((arr > 0).sum())
            rec["area_ratio"] = round(fg / arr.size, 6)
            rec["empty"] = fg == 0
            rec["full"] = fg == arr.size
            rec["binary_0_255"] = bool(set(np.unique(arr).tolist()) <= {0, 255})
            rec["decodable"] = True
        except Exception as error:  # noqa: BLE001
            rec.update({"decodable": False, "error": f"{type(error).__name__}: {error}"})
        qc.append(rec)

ids = {r["isic_id"] for r in qc}
split = json.loads(Path("runs/role_a_overall_v4_dice2_base8/split_used.json").read_text(encoding="utf-8"))
train_ids = set(split["train_ids"]); dev_ids = {i for v in split["development_ids"].values() for i in v}
hold_ids = {p.stem for p in (Path("data/raw/isic2018_test/images")).glob("ISIC_*.jpg")}
leak = {"train2309": sorted(ids & train_ids), "dev638": sorted(ids & dev_ids),
        "final_test2018": sorted(ids & hold_ids)}

anomalies = [r["isic_id"] for r in qc
             if (not r["decodable"]) or r["empty"] or r["full"] or not r.get("binary_0_255", False)]
areas = [r["area_ratio"] for r in qc if r["decodable"]]
result = {
    "schema": "a16-n1-mask-qc-v1",
    "zip": {"bytes": ZIP.stat().st_size, "sha256": sha256(ZIP), "entries": len(qc)},
    "qc": {
        "decodable": sum(1 for r in qc if r["decodable"]),
        "undecodable": sum(1 for r in qc if not r["decodable"]),
        "empty_masks": sum(1 for r in qc if r.get("empty")),
        "full_masks": sum(1 for r in qc if r.get("full")),
        "non_binary": sum(1 for r in qc if r["decodable"] and not r.get("binary_0_255", False)),
        "area_ratio": {"min": min(areas), "p05": float(np.percentile(areas, 5)),
                       "p25": float(np.percentile(areas, 25)), "median": float(np.median(areas)),
                       "p75": float(np.percentile(areas, 75)), "p95": float(np.percentile(areas, 95)),
                       "max": max(areas), "mean": float(np.mean(areas))},
        "size_distribution_top": Counter(f"{r['width']}x{r['height']}" for r in qc if r["decodable"]).most_common(5),
        "modes": dict(Counter(r.get("mode") for r in qc)),
    },
    "ISIC_0035068": next((r for r in qc if r["isic_id"] == "ISIC_0035068"), None),
    "anomaly_ids": anomalies,
    "leak_check": leak,
}
(AUDIT/"zip_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
(AUDIT/"mask_qc.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
with (AUDIT/"mask_qc.csv").open("w", encoding="utf-8", newline="") as f:
    cols = ["isic_id","zip_name","compressed_bytes","uncompressed_bytes","crc32","format","mode",
            "width","height","n_unique","min","max","area_ratio","empty","full","binary_0_255","decodable"]
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(qc)
print("ZIP: entries=%d bytes=%d sha256=%s" % (len(qc), ZIP.stat().st_size, manifest["sha256"][:16]))
print("条目格式: prefix_count=%d 示例=%s" % (len(prefixes), prefixes[:2]))
print("可解码=%d 不可解码=%d | 空掩膜=%d 满掩膜=%d 非二值=%d" % (
    result["qc"]["decodable"], result["qc"]["undecodable"], result["qc"]["empty_masks"],
    result["qc"]["full_masks"], result["qc"]["non_binary"]))
print("面积比 min/median/max = %.6f / %.6f / %.6f" % (
    result["qc"]["area_ratio"]["min"], result["qc"]["area_ratio"]["median"], result["qc"]["area_ratio"]["max"]))
print("尺寸分布 top:", result["qc"]["size_distribution_top"], "| modes:", result["qc"]["modes"])
print("ISIC_0035068:", json.dumps(result["ISIC_0035068"], ensure_ascii=False))
print("泄漏检查:", {k: len(v) for k, v in leak.items()})
print("异常 ID 数:", len(anomalies), anomalies[:10])