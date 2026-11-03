"""A16-N1 抽样：排除异常掩膜后，按面积段 + 解剖部位 + 诊断做确定性抽样（>=48，微小段 >=16）。"""
from __future__ import annotations
import csv, json, random
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

AUDIT = Path("runs/role_a_a16_novice_mask_audit")
META = Path("data/raw/isic2018_novice_metadata")
SEED = 20260926
TOTAL, TINY_MIN = 48, 16

qc = {r["isic_id"]: r for r in csv.DictReader((AUDIT/"mask_qc.csv").open(encoding="utf-8", newline=""))}
meta = {r["isic_id"]: r for r in csv.DictReader(
    (META/"isic-2018-novice-segmentation-masks.csv").open(encoding="utf-8", newline=""))}
anomalies = set(json.loads((AUDIT/"mask_qc.json").read_text(encoding="utf-8"))["anomaly_ids"])
valid = [i for i, r in qc.items() if i not in anomalies and r["decodable"] == "True"]
areas = np.array([float(qc[i]["area_ratio"]) for i in valid])
q33, q67 = float(np.percentile(areas, 33.333)), float(np.percentile(areas, 66.667))


def band(a):
    return "low" if a <= q33 else ("mid" if a <= q67 else "high")


groups = defaultdict(list)
for i in valid:
    groups[band(float(qc[i]["area_ratio"]))].append(i)
for k in groups:
    groups[k].sort(key=lambda i: (meta[i].get("anatom_site_1", ""), meta[i].get("diagnosis_1", ""), i))

plan = {"low": TINY_MIN, "mid": (TOTAL - TINY_MIN) // 2, "high": TOTAL - TINY_MIN - (TOTAL - TINY_MIN) // 2}
rng = random.Random(SEED)
selected, rows = [], []
for name in ("low", "mid", "high"):
    pool = groups[name]
    take = plan[name]
    idx = np.linspace(0, len(pool) - 1, num=take, dtype=int).tolist()
    picked = [pool[j] for j in idx]
    for rank, i in enumerate(picked):
        a = float(qc[i]["area_ratio"])
        sel = ("最小/常见面积段(tiny)" if name == "low" else
               ("中等面积段" if name == "mid" else "大面积段"))
        rows.append({"isic_id": i, "area_band": name, "area_ratio": a,
                     "width": qc[i]["width"], "height": qc[i]["height"],
                     "anatom_site_1": meta[i].get("anatom_site_1", ""),
                     "diagnosis_1": meta[i].get("diagnosis_1", ""),
                     "copyright_license": meta[i].get("copyright_license", ""),
                     "sampling_reason": f"{sel}；按(部位,诊断,ID)排序后在该段内等间隔取第 {rank+1}/{take} 个"})
        selected.append(i)

summary = {
    "schema": "a16-n1-pilot-selection-v1",
    "seed": SEED, "total_selected": len(rows),
    "band_boundaries": {"q33_3": q33, "q66_7": q67},
    "counts_by_band": dict(Counter(r["area_band"] for r in rows)),
    "counts_by_site": dict(Counter(r["anatom_site_1"] for r in rows)),
    "counts_by_diagnosis": dict(Counter(r["diagnosis_1"] for r in rows)),
    "tiny_requirement_met": sum(1 for r in rows if r["area_band"] == "low") >= TINY_MIN,
    "excluded_anomalies": len(anomalies),
    "excluded_anomaly_ids": sorted(anomalies),
    "not_downloaded": "本阶段不下载这些 ID 的原图；下一阶段需另批",
    "note": "抽样不依赖开发集 Dice 或最终测试集表现",
}
(AUDIT/"pilot_selection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
with (AUDIT/"pilot_selection.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(json.dumps({k: summary[k] for k in ("total_selected","band_boundaries","counts_by_band",
                                          "counts_by_site","tiny_requirement_met","excluded_anomalies")},
                 ensure_ascii=False, indent=2))
print("样例:", [r["isic_id"] for r in rows[:6]])