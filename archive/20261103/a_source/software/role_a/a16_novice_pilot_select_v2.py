"""A16-N1 抽样更正：按项目 dev 口径 tiny=q10 重新生成 48 张确定性清单（不下载任何文件、不使用随机数）。"""
from __future__ import annotations
import csv, hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np

AUDIT = Path("runs/role_a_a16_novice_mask_audit")
META = Path("data/raw/isic2018_novice_metadata")
TINY_N, MID_N, LARGE_N = 16, 16, 16

old_csv = AUDIT/"pilot_selection.csv"
old_bytes = old_csv.read_bytes() if old_csv.exists() else b""
old_hash = hashlib.sha256(old_bytes).hexdigest()
old_rows = list(csv.DictReader(old_bytes.decode("utf-8").splitlines()))

qc = {r["isic_id"]: r for r in csv.DictReader((AUDIT/"mask_qc.csv").open(encoding="utf-8", newline=""))}
meta = {r["isic_id"]: r for r in csv.DictReader(
    (META/"isic-2018-novice-segmentation-masks.csv").open(encoding="utf-8", newline=""))}

# 有效掩膜：可解码 + 非空 + 非满 + 二值（与 A16-N1 异常口径一致）
valid = sorted(i for i, r in qc.items()
               if r["decodable"] == "True" and r["empty"] == "False"
               and r["full"] == "False" and r["binary_0_255"] == "True")
areas = np.array([float(qc[i]["area_ratio"]) for i in valid])
q10, q50 = float(np.percentile(areas, 10)), float(np.percentile(areas, 50))


def band(a):
    if a <= q10:
        return "tiny"
    return "medium" if a <= q50 else "large"


groups = {b: [] for b in ("tiny", "medium", "large")}
for i in valid:
    groups[band(float(qc[i]["area_ratio"]))].append(i)
for b in groups:  # 确定性排序，无随机数
    groups[b].sort(key=lambda i: (meta[i].get("anatom_site_1", ""), meta[i].get("diagnosis_1", ""), i))

target = {"tiny": TINY_N, "medium": MID_N, "large": LARGE_N}
rows = []
for b, pool in groups.items():
    take = min(target[b], len(pool))
    idx = np.linspace(0, len(pool) - 1, num=take, dtype=int).tolist()
    for rank, j in enumerate(idx):
        i = pool[j]
        a = float(qc[i]["area_ratio"])
        rows.append({"isic_id": i, "area_band": b, "area_ratio": a,
                     "width": qc[i]["width"], "height": qc[i]["height"],
                     "anatom_site_1": meta[i].get("anatom_site_1", ""),
                     "diagnosis_1": meta[i].get("diagnosis_1", ""),
                     "copyright_license": meta[i].get("copyright_license", ""),
                     "sampling_reason": (f"面积段={b}（tiny: <=q10 {q10:.6f}；medium: q10~q50 {q50:.6f}；large: >q50）；"
                                         f"该段内按(部位,诊断,ID)排序后等间隔取第 {rank+1}/{take} 个（确定性，无随机数）")})

old_tiny = sum(1 for r in old_rows if float(r["area_ratio"]) <= q10)
summary = {
    "schema": "a16-n1-pilot-selection-v2",
    "revision": "v2（更正：旧 v1 用面积三分位并把 low 段误称 tiny；本版按项目 dev 口径 tiny=q10）",
    "algorithm": {"type": "确定性等间隔抽样", "random_numbers_used": False,
                  "seed": None,
                  "rule": "有效掩膜先按 tiny(<=q10)/medium(q10,q50]/large(>q50) 分段；每段内按 (anatom_site_1, diagnosis_1, isic_id) 排序后 np.linspace 等间隔取点"},
    "valid_mask_count": len(valid),
    "band_thresholds": {"q10": q10, "q50": q50},
    "counts_by_band": dict(Counter(r["area_band"] for r in rows)),
    "tiny_requirement_met": sum(1 for r in rows if r["area_band"] == "tiny") >= TINY_N,
    "counts_by_site": dict(Counter(r["anatom_site_1"] for r in rows)),
    "counts_by_diagnosis": dict(Counter(r["diagnosis_1"] for r in rows)),
    "old_list_change_record": {
        "old_file": "pilot_selection.csv (v1)",
        "old_sha256": old_hash,
        "old_rows": len(old_rows),
        "old_rows_within_true_q10": old_tiny,
        "invalidated_reason": "旧版把面积最小三分之一(low) 误等同于项目 dev 的微小病灶 tiny=q10；48 张中仅 3 张真正落在 q10 以内，未重点覆盖微小病灶",
        "archived_as": "pilot_selection_v1_invalidated.csv",
    },
    "not_downloaded": "本阶段不下载任何原图",
}
(AUDIT/"pilot_selection_v1_invalidated.csv").write_bytes(old_bytes)
with (AUDIT/"pilot_selection.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(AUDIT/"pilot_selection.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n",
                                          encoding="utf-8")
print("有效掩膜 =", len(valid))
print("q10 = %.6f | q50 = %.6f" % (q10, q50))
print("各段可用池:", {b: len(v) for b, v in groups.items()})
print("选中分布:", summary["counts_by_band"], "| tiny 达标:", summary["tiny_requirement_met"])
print("旧 v1 清单 SHA256 =", old_hash)
print("旧 48 张中真正 <= q10 的 =", old_tiny)
print("新 48 张面积比区间: tiny %.6f~%.6f / medium %.6f~%.6f / large %.6f~%.6f" % (
    min(r["area_ratio"] for r in rows if r["area_band"] == "tiny"),
    max(r["area_ratio"] for r in rows if r["area_band"] == "tiny"),
    min(r["area_ratio"] for r in rows if r["area_band"] == "medium"),
    max(r["area_ratio"] for r in rows if r["area_band"] == "medium"),
    min(r["area_ratio"] for r in rows if r["area_band"] == "large"),
    max(r["area_ratio"] for r in rows if r["area_band"] == "large")))