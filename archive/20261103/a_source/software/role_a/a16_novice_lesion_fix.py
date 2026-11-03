"""A16-N0 更正：用**跨源** IMA++ 元数据为参考集取 lesion_id/patient_id，重算与 novice 的关联筛查。

错因：上一版把 `lesion` 映射只建立在 novice 元数据上（`meta`），而 dev638/final1000 的 ID 不在 novice 集合内
（交集本来就为 0），于是 `ref_vals` 恒为空集，`reference_known=0`、命中数=0 —— 结论无效。
"""
from __future__ import annotations
import csv, json
from collections import Counter
from pathlib import Path

ROOT = Path("data/raw")
NOV = ROOT/"isic2018_novice_metadata"
IMA = ROOT/"ima_plus_plus_metadata"
AUDIT = Path("runs/role_a_a16_novice_metadata")


def read_csv(p):
    with Path(p).open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


nov = read_csv(NOV/"isic-2018-novice-segmentation-masks.csv")
ima = read_csv(IMA/"IMAplusplus_img_metadata.csv")
nov_ids = {r["isic_id"] for r in nov}
nov_lesion = {r["isic_id"]: (r.get("lesion_id") or "").strip() for r in nov}
ima_lesion = {r["isic_id"]: (r.get("lesion_id") or "").strip() for r in ima}
ima_patient = {r["isic_id"]: (r.get("patient_id") or "").strip() for r in ima}

split = json.loads(Path("runs/role_a_overall_v4_dice2_base8/split_used.json").read_text(encoding="utf-8"))
dev_ids = {i for v in split["development_ids"].values() for i in v}
hold_ids = {p.stem for p in (ROOT/"isic2018_test/images").glob("ISIC_*.jpg")}


def screen(ref_ids, ref_map, new_map, new_ids):
    ref_known = {i: ref_map.get(i, "") for i in ref_ids if ref_map.get(i)}
    ref_vals = set(ref_known.values())
    hits = sorted(i for i in new_ids if new_map.get(i) and new_map[i] in ref_vals)
    return {
        "reference_total": len(ref_ids),
        "reference_known": len(ref_known),
        "reference_unknown_missing_in_reference_source": len(ref_ids) - len(ref_known),
        "reference_distinct_values": len(ref_vals),
        "novice_ids_screenable": sum(1 for i in new_ids if new_map.get(i)),
        "novice_ids_unknown": sum(1 for i in new_ids if not new_map.get(i)),
        "hits": len(hits),
        "hit_ids_sha256_note": "命中 ID 清单见 hit_ids 字段（无隐私取值）",
        "hit_ids": hits,
        "reference_source": "data/raw/ima_plus_plus_metadata/IMAplusplus_img_metadata.csv",
    }


res = {
    "schema": "a16-novice-metadata-audit-v1",
    "correction_note": ("上一版 lesion_association 的参考集 lesion_id 取自 novice 元数据本身（参考 ID 不在该表内），"
                        "导致 reference_known=0、命中数恒为 0，结论无效。本版改由 IMA++ img 元数据为参考集取字段后重算。"),
    "lesion_association_corrected": {
        "dev638": screen(dev_ids, ima_lesion, nov_lesion, nov_ids),
        "final_test2018_1000": screen(hold_ids, ima_lesion, nov_lesion, nov_ids),
        "patient_id": ("novice 元数据不含 patient_id；IMA++ 侧 patient_id 覆盖率低（全集 13.9%、候选 6.5%）"
                       "=> 同患者筛查无法完成，标记未知"),
        "isolation_union": 0,
    },
    "reference_coverage_check": {
        "dev638_lesion_known": sum(1 for i in dev_ids if ima_lesion.get(i)),
        "dev638_lesion_distinct": len({ima_lesion[i] for i in dev_ids if ima_lesion.get(i)}),
        "final1000_lesion_known": sum(1 for i in hold_ids if ima_lesion.get(i)),
        "final1000_lesion_distinct": len({ima_lesion[i] for i in hold_ids if ima_lesion.get(i)}),
    },
}
(AUDIT/"lesion_association_corrected.json").write_text(json.dumps(res, ensure_ascii=False, indent=2)+"\n",
                                                       encoding="utf-8")
print("== 参考集覆盖（取自 IMA++ img 元数据）==")
print(json.dumps(res["reference_coverage_check"], ensure_ascii=False))
for name in ("dev638", "final_test2018_1000"):
    b = res["lesion_association_corrected"][name]
    print("== %s ==" % name)
    for k in ("reference_total", "reference_known", "reference_unknown_missing_in_reference_source",
              "reference_distinct_values", "novice_ids_screenable", "novice_ids_unknown", "hits"):
        print("   %-46s %s" % (k, b[k]))
    print("   hit_ids:", b["hit_ids"][:5])