"""A16-I0 补充：patient_id / lesion_id 覆盖率与跨集合同患者/同病灶关联筛查（只读，不下载）。

隐私约束：仅输出聚合计数与 ISIC ID 隔离清单，**不输出任何 patient_id / lesion_id 取值**。
"""
from __future__ import annotations
import csv, json
from collections import defaultdict
from pathlib import Path

META = Path("data/raw/ima_plus_plus_metadata")
AUDIT = Path("runs/role_a_a16_ima_metadata")
ROOT = Path("data/raw")


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def norm(rows, key):
    return {r["isic_id"]: (r.get(key) or "").strip() for r in rows if r.get("isic_id")}


img = read_csv(META/"IMAplusplus_img_metadata.csv")
split = json.loads(Path("runs/role_a_overall_v4_dice2_base8/split_used.json").read_text(encoding="utf-8"))
dev_ids = {i for v in split["development_ids"].values() for i in v}
holdout_ids = {p.stem for p in (ROOT/"isic2018_test/images").glob("ISIC_*.jpg")}
cand = list(csv.DictReader((AUDIT/"candidate_new_ids.csv").open(encoding="utf-8", newline="")))
cand_ids = {r["isic_id"] for r in cand}

patient = norm(img, "patient_id")
lesion = norm(img, "lesion_id")


def coverage(mapping):
    known = sum(1 for v in mapping.values() if v)
    return {"total": len(mapping), "known": known,
            "unknown": len(mapping)-known, "coverage": round(known/max(1, len(mapping)), 4),
            "distinct_values": len({v for v in mapping.values() if v})}


def linked(new_ids, ref_ids, mapping):
    """new_ids 中与 ref_ids 共享同一 patient/lesion 值的 ID。"""
    ref_vals = {mapping.get(i) for i in ref_ids if mapping.get(i)}
    hit = sorted(i for i in new_ids if mapping.get(i) and mapping[i] in ref_vals)
    return hit, len(ref_vals)


report = {
    "schema": "a16-ima-patient-lesion-audit-v1",
    "privacy_note": "仅聚合计数与 ISIC ID 清单；不含 patient_id/lesion_id 取值",
    "coverage": {
        "all_ima_images": {"patient_id": coverage(patient), "lesion_id": coverage(lesion)},
        "candidate_new_ids": {
            "patient_id": {"total": len(cand_ids),
                           "known": sum(1 for i in cand_ids if patient.get(i)),
                           "unknown": sum(1 for i in cand_ids if not patient.get(i))},
            "lesion_id": {"total": len(cand_ids),
                          "known": sum(1 for i in cand_ids if lesion.get(i)),
                          "unknown": sum(1 for i in cand_ids if not lesion.get(i))}},
    },
    "association_screen": {},
}
for ref_name, ref_ids in (("dev638_frozen", dev_ids), ("final_test2018_1000", holdout_ids)):
    block = {}
    for key, mapping in (("patient_id", patient), ("lesion_id", lesion)):
        hit, ref_distinct = linked(cand_ids, ref_ids, mapping)
        block[key] = {
            "reference_ids": len(ref_ids),
            "reference_with_field": sum(1 for i in ref_ids if mapping.get(i)),
            "reference_distinct_values": ref_distinct,
            "candidate_new_ids_flagged": len(hit),
            "candidate_new_ids_screenable": sum(1 for i in cand_ids if mapping.get(i)),
            "flagged_ids": hit,
            "note_unknown": f"候选新增中 {sum(1 for i in cand_ids if not mapping.get(i))} 个该字段为空 => 未知，无法用此字段筛查",
        }
    both = sorted(set(block["patient_id"]["flagged_ids"]) | set(block["lesion_id"]["flagged_ids"]))
    block["union_flagged_ids"] = both
    block["union_flagged_count"] = len(both)
    report["association_screen"][ref_name] = block

all_flagged = sorted(set(report["association_screen"]["dev638_frozen"]["union_flagged_ids"]) |
                     set(report["association_screen"]["final_test2018_1000"]["union_flagged_ids"]))
report["isolation_union"] = {"count": len(all_flagged), "ids": all_flagged}
report["limitations"] = [
    "patient_id/lesion_id 覆盖率低（候选新增多数为空）=> 只能拦截少数关联，其余标为未知",
    "该筛查只覆盖元数据关联；同图/近重复与同一病灶的重拍仍需像素级 SHA256 + dHash 与人工视觉复核",
    "本次未下载任何图像或掩膜",
]
(AUDIT/"patient_lesion_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n",
                                               encoding="utf-8")
with (AUDIT/"isolation_ids.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f); w.writerow(["isic_id"])
    for i in all_flagged:
        w.writerow([i])

c = report["coverage"]
print("== 覆盖率（img 14,967）==")
print("  patient_id:", json.dumps(c["all_ima_images"]["patient_id"], ensure_ascii=False))
print("  lesion_id :", json.dumps(c["all_ima_images"]["lesion_id"], ensure_ascii=False))
print("== 候选新增 11,093 的字段可用性 ==")
print("  patient_id:", json.dumps(c["candidate_new_ids"]["patient_id"], ensure_ascii=False))
print("  lesion_id :", json.dumps(c["candidate_new_ids"]["lesion_id"], ensure_ascii=False))
print()
for name in ("dev638_frozen", "final_test2018_1000"):
    b = report["association_screen"][name]
    print("== 关联筛查 vs %s ==" % name)
    for key in ("patient_id", "lesion_id"):
        v = b[key]
        print("  %-10s 参考集已知=%d 不同值=%d | 候选可筛=%d 命中候选ID=%d | 候选未知=%s" % (
            key, v["reference_with_field"], v["reference_distinct_values"],
            v["candidate_new_ids_screenable"], v["candidate_new_ids_flagged"],
            "是" if "为空" in v["note_unknown"] else "否"))
    print("  并集命中的候选 ID 数:", b["union_flagged_count"])
print()
print("== 需隔离并集 ==", len(all_flagged), "个 ID（写入 isolation_ids.csv）")
print("  示例前 10:", all_flagged[:10])