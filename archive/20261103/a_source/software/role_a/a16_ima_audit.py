"""A16-I0：IMA++ 元数据审计（ID 交叉、许可与标注分布）。只读 CSV，不下载图像。"""
from __future__ import annotations
import csv, json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("data/raw")
META = ROOT/"ima_plus_plus_metadata"
AUDIT = Path("runs/role_a_a16_ima_metadata")
FROZEN_SPLIT = Path("runs/role_a_overall_v4_dice2_base8/split_used.json")


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


img_rows = read_csv(META/"IMAplusplus_img_metadata.csv")
seg_rows = read_csv(META/"IMAplusplus_seg_metadata.csv")
img_ids = {r["isic_id"] for r in img_rows}
seg_ids = {r["ISIC_id"] for r in seg_rows}

split = json.loads(FROZEN_SPLIT.read_text(encoding="utf-8"))
train_ids = set(split["train_ids"])
dev_ids = {i for v in split["development_ids"].values() for i in v}

LOCAL = {
    "train2309_frozen": train_ids,
    "dev638_frozen": dev_ids,
    "isic2016_train": {p.stem for p in (ROOT/"isic2016/train/images").glob("ISIC_*.jpg")},
    "isic2016_test": {p.stem for p in (ROOT/"isic2016/test/images").glob("ISIC_*.jpg")},
    "isic2017_training": {p.stem for p in (ROOT/"isic2017_training/images").glob("ISIC_*.jpg")},
    "isic2017_validation": {p.stem for p in (ROOT/"isic2017_validation/images").glob("ISIC_*.jpg")},
    "isic2017_test": {p.stem for p in (ROOT/"isic2017_test/images").glob("ISIC_*.jpg")},
    "isic2018_validation_images": {p.stem for p in (ROOT/"isic2018_validation/images").glob("ISIC_*.jpg")},
    "isic2018_test_images_final_holdout": {p.stem for p in (ROOT/"isic2018_test/images").glob("ISIC_*.jpg")},
}
local_union = set().union(*LOCAL.values())

inter = {}
for name, ids in LOCAL.items():
    hit = img_ids & ids
    inter[name] = {"local_unique": len(ids), "overlap_with_ima_img": len(hit),
                   "overlap_ratio_of_local": round(len(hit)/max(1, len(ids)), 4),
                   "overlap_ratio_of_ima": round(len(hit)/max(1, len(img_ids)), 4)}
new_ids = sorted(img_ids - local_union)
covered = len(img_ids & local_union)

lic = Counter((r.get("copyright_license") or "").strip() or "<empty>" for r in img_rows)
annot_per_image = Counter(Counter(r["ISIC_id"] for r in seg_rows).values())
annotators = Counter((r.get("annotator") or "").strip() or "<empty>" for r in seg_rows)
tools = Counter((r.get("tool") or "").strip() or "<empty>" for r in seg_rows)
skills = Counter((r.get("skill_level") or "").strip() or "<empty>" for r in seg_rows)

missing = {}
for col in img_rows[0]:
    n = sum(1 for r in img_rows if (r.get(col) or "").strip() == "")
    if n:
        missing[col] = n

seg_by_id = Counter(r["ISIC_id"] for r in seg_rows)
candidates = []
for r in img_rows:
    i = r["isic_id"]
    if i in local_union:
        continue
    candidates.append({
        "isic_id": i,
        "pixels_x": r.get("pixels_x", ""), "pixels_y": r.get("pixels_y", ""),
        "patient_id": r.get("patient_id", ""),
        "copyright_license_as_recorded": (r.get("copyright_license") or "").strip(),
        "attribution_as_recorded": (r.get("attribution") or "").strip(),
        "image_type": (r.get("image_type") or "").strip(),
        "annotations_in_seg_metadata": seg_by_id.get(i, 0),
    })

result = {
    "schema": "a16-ima-metadata-audit-v1",
    "sources": {
        "img_metadata": {"file": "IMAplusplus_img_metadata.csv", "rows": len(img_rows),
                         "md5": "c27fdbcaab1e2cae48d6ad7edd3d64c4",
                         "sha256": "5ebc23efdd91de68cc6df6e04365023f2318de574cc23b585ea64b5a800c6942"},
        "seg_metadata": {"file": "IMAplusplus_seg_metadata.csv", "rows": len(seg_rows),
                         "md5": "aae21dec31b48e45786da70c4609e45f",
                         "sha256": "e8cfb2f6d9366ab5b06a61883a97798af2a61ef2a714d228674bf40b0ff5c9d0"},
        "zenodo_blocked": "zenodo.org DNS 黑洞；改用作者仓库 raw.githubusercontent.com，MD5 与 Zenodo 公告一致",
    },
    "counts": {"ima_images": len(img_ids), "ima_images_with_annotation_rows": len(seg_ids),
               "ima_annotation_rows": len(seg_rows),
               "collection_description_masks": 17684,
               "note_masks_vs_rows": "集合页描述称 17,684 个掩膜，seg CSV 实有 22,472 行标注；两者口径不同，未做进一步解释"},
    "id_intersection": {**inter,
        "local_union_unique": len(local_union),
        "ima_covered_by_local": covered, "ima_truly_new": len(new_ids)},
    "license_distribution": dict(lic.most_common()),
    "annotations_per_image_distribution": {str(k): v for k, v in sorted(annot_per_image.items())},
    "annotator_distribution": dict(annotators.most_common()),
    "tool_distribution": dict(tools.most_common()),
    "skill_level_distribution": dict(skills.most_common()),
    "missing_fields_img": missing,
    "candidate_new_ids_count": len(candidates),
    "holdout": "2018 final test masks NOT opened; only its 1000 image filenames were read",
}
(AUDIT/"metadata_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
with (AUDIT/"candidate_new_ids.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(candidates[0]) if candidates else ["isic_id"])
    w.writeheader(); w.writerows(candidates)

print("IMA++ 图像:", len(img_ids), "| 有标注行的图像:", len(seg_ids), "| 标注行:", len(seg_rows))
print("本地合并 unique:", len(local_union), "| IMA 被本地覆盖:", covered, "| 真正新增:", len(new_ids))
print()
for k, v in inter.items():
    print("  %-34s local=%5d 交集=%5d (占本地%5.1f%%, 占IMA%5.1f%%)" % (
        k, v["local_unique"], v["overlap_with_ima_img"],
        100*v["overlap_ratio_of_local"], 100*v["overlap_ratio_of_ima"]))
print()
print("许可分布:", json.dumps(dict(lic.most_common()), ensure_ascii=False))
print("每图标注数分布:", json.dumps({str(k): v for k, v in sorted(annot_per_image.items())}, ensure_ascii=False))
print("标注者:", json.dumps(dict(annotators.most_common()), ensure_ascii=False))
print("工具:", json.dumps(dict(tools.most_common()), ensure_ascii=False))
print("熟练度:", json.dumps(dict(skills.most_common()), ensure_ascii=False))
print("img 缺字段(non-empty 计数):", json.dumps(missing, ensure_ascii=False)[:400])
print("候选新增 ID 数:", len(candidates), "| 示例:", [c["isic_id"] for c in candidates[:10]])