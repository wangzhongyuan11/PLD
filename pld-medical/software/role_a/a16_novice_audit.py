"""A16-N0：ISIC 2018 Novice Segmentation masks 元数据审计（只读文件名/元数据，不下载掩膜 ZIP/图像）。"""
from __future__ import annotations
import csv, json
from collections import Counter
from pathlib import Path

ROOT = Path("data/raw")
NOV = ROOT/"isic2018_novice_metadata"
AUDIT = Path("runs/role_a_a16_novice_metadata")
NOT_FOUND = "unknown"


def read_csv(p):
    with Path(p).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


meta = read_csv(NOV/"isic-2018-novice-segmentation-masks.csv")
mapping = read_csv(NOV/"Masks_to_IsicId_mapping.csv")
nov_ids = {r["isic_id"] for r in meta}
map_ids = {r["ISIC_ID"] for r in mapping}

split = json.loads(Path("runs/role_a_overall_v4_dice2_base8/split_used.json").read_text(encoding="utf-8"))
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

# A16 IMA++ 待审候选（已移除 157 隔离 ID 后的 "10,936" 口径）
ima_all = {r["isic_id"] for r in read_csv(AUDIT.parent/"role_a_a16_ima_metadata/candidate_new_ids.csv")}
isolated = {r["isic_id"] for r in read_csv(AUDIT.parent/"role_a_a16_ima_metadata/isolation_ids.csv")}
ima_pending = ima_all - isolated

inter = {name: {"local_unique": len(ids), "overlap": len(nov_ids & ids)}
         for name, ids in LOCAL.items()}
N1 = sorted(nov_ids - local_union)

# lesion_id 关联筛查
lesion = {r["isic_id"]: (r.get("lesion_id") or "").strip() for r in meta}
know_lesion = sum(1 for v in lesion.values() if v)


def flagged(new_ids, ref_ids):
    ref_vals = {lesion.get(i) for i in ref_ids if lesion.get(i)}
    return sorted(i for i in new_ids if lesion.get(i) and lesion[i] in ref_vals), len(ref_vals)


flag_dev, dev_vals = flagged(set(N1), dev_ids)
flag_hold, hold_vals = flagged(set(N1), LOCAL["isic2018_test_images_final_holdout"])
isolate = sorted(set(flag_dev) | set(flag_hold))
final_ids = sorted(set(N1) - set(isolate))

lic = Counter((r.get("copyright_license") or "").strip() or "<empty>" for r in meta)
attr = Counter((r.get("attribution") or "").strip() or "<empty>" for r in meta)
special = sorted(i for i in nov_ids if i == "ISIC_0035068")

result = {
    "schema": "a16-novice-metadata-audit-v1",
    "source": {
        "page": "https://api.isic-archive.com/doi/isic-2018-novice-segmentation-masks/",
        "metadata_csv_url": "https://isic-archive.s3.amazonaws.com/dois/10.34970-387951/isic-2018-novice-segmentation-masks.csv",
        "mapping_csv_url": "https://isic-archive.s3.amazonaws.com/dois/10.34970-387951/supplements/Masks_to_IsicId_mapping.csv",
        "expected_counts": {"masks": 11720, "metadata_mb": 2.7, "mapping_kb": 412, "zip_mb": 19.1},
    },
    "integrity": {
        "metadata_rows": len(meta), "metadata_unique_ids": len(nov_ids),
        "mapping_rows": len(mapping), "mapping_unique_ids": len(map_ids),
        "metadata_equals_mapping_ids": nov_ids == map_ids,
        "mapping_filename_pattern_ok": all(
            r["Filename"] == f"{r['ISIC_ID']}_UNET.png" for r in mapping),
    },
    "id_intersection": {**inter, "local_union_unique": len(local_union),
                        "novice_covered_by_local": len(nov_ids & local_union),
                        "novice_truly_new_N1": len(N1)},
    "vs_ima_pending_10936": {
        "ima_pending_count": len(ima_pending),
        "novice_in_ima_pending": len(nov_ids & ima_pending),
        "novice_new_after_ima": len(set(N1) - ima_pending),
    },
    "lesion_association": {
        "coverage_total": {"total": len(meta), "known": know_lesion, "unknown": len(meta)-know_lesion,
                           "coverage": round(know_lesion/max(1, len(meta)), 4)},
        "dev638": {"reference_known": sum(1 for i in dev_ids if lesion.get(i)),
                   "reference_distinct": dev_vals, "flagged_N1_ids": len(flag_dev)},
        "final_test2018_1000": {"reference_known": sum(1 for i in LOCAL["isic2018_test_images_final_holdout"] if lesion.get(i)),
                                "reference_distinct": hold_vals, "flagged_N1_ids": len(flag_hold)},
        "union_isolated": len(isolate),
        "patient_id": "该源元数据不含 patient_id 字段 => 无法做同患者筛查（未知）",
    },
    "final_usable": {"N1_truly_new": len(N1), "minus_lesion_isolated": len(final_ids),
                     "note": "仍需像素级 SHA256 + dHash 与人工视觉复核后才算可用"},
    "quality_limitations_official": [
        "掩膜由 U-Net 初步生成，再由人工审查/修正（novice segmentation）",
        "可能出现中心偏置（center bias）",
        "边缘与毛发（hair）伪影可能被误标",
        "ISIC_0035068 为全黑掩膜特例",
        "不得把这些掩膜当作无误真值（not flawless ground truth）",
    ],
    "license": {
        "csv_recorded": dict(lic.most_common()),
        "attribution_top": dict(attr.most_common(3)),
        "note": "官方页面标注 CC-BY-NC；用户已说明竞赛为非商用；本报告保留归属信息",
    },
    "special_case_ids": special,
    "holdout": "2018 final test masks NOT opened; only image filenames were read",
    "isolation_ids": isolate,
}
(AUDIT).mkdir(parents=True, exist_ok=True)
(AUDIT/"novice_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
(AUDIT/"novice_final_usable_ids.csv").write_text(
    "isic_id\n" + "\n".join(final_ids) + ("\n" if final_ids else ""), encoding="utf-8")
(AUDIT/"novice_isolation_ids.csv").write_text(
    "isic_id\n" + "\n".join(isolate) + ("\n" if isolate else ""), encoding="utf-8")

print("metadata 行=%d 唯一ID=%d | mapping 行=%d 唯一ID=%d | 两侧 ID 集合相同=%s | 命名规则一致=%s" % (
    len(meta), len(nov_ids), len(mapping), len(map_ids),
    nov_ids == map_ids, result["integrity"]["mapping_filename_pattern_ok"]))
print("本地合并 unique=%d | novice 被覆盖=%d | N1 真正新增=%d" % (
    len(local_union), len(nov_ids & local_union), len(N1)))
for k, v in inter.items():
    print("   %-34s overlap=%5d" % (k, v["overlap"]))
print("与 IMA++ 待审候选(10,936) 交集=%d | 去掉后仍有=%d" % (
    result["vs_ima_pending_10936"]["novice_in_ima_pending"], result["vs_ima_pending_10936"]["novice_new_after_ima"]))
print("lesion_id 覆盖=%d/%d (%.1f%%) | 关联命中 dev638=%d final1000=%d | 隔离并集=%d" % (
    know_lesion, len(meta), 100*know_lesion/len(meta), len(flag_dev), len(flag_hold), len(isolate)))
print("最终可筛查独立 ID 数 = %d" % len(final_ids))
print("许可:", dict(lic.most_common()), "| 归属 top:", attr.most_common(1))
print("全黑特例存在:", special)