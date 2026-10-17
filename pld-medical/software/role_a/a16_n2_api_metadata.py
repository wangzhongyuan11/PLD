"""A16-N2 步骤 0：48 张清单核对 + 官方 API 元数据（只读，不下载图像）。"""
from __future__ import annotations
import csv, hashlib, json, urllib.request
from pathlib import Path

PILOT = Path("runs/role_a_a16_novice_mask_audit/pilot_selection.csv")
OUT = Path("runs/role_a_a16_novice_image_pilot")
OUT.mkdir(parents=True, exist_ok=True)
API = "https://api.isic-archive.com/api/v2/images/{id}/"

rows = list(csv.DictReader(PILOT.open(encoding="utf-8", newline="")))
print("pilot_selection.csv sha256 =", hashlib.sha256(PILOT.read_bytes()).hexdigest())
print("rows =", len(rows), "| bands =", {b: sum(1 for r in rows if r["area_band"] == b)
                                        for b in ("tiny", "medium", "large")})

api_rows = []
for r in rows:
    req = urllib.request.Request(API.format(id=r["isic_id"]), headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    files = payload.get("files") or {}
    full = files.get("full") or {}
    api_rows.append({"isic_id": r["isic_id"], "area_band": r["area_band"],
                     "area_ratio": float(r["area_ratio"]),
                     "full_url": full.get("url"), "full_size": full.get("size"),
                     "license": payload.get("copyright_license"),
                     "attribution": payload.get("attribution"),
                     "image_type": payload.get("image_type"),
                     "metadata": payload.get("metadata") or {}})
    print("  %-14s %-6s size=%s license=%s" % (r["isic_id"], r["area_band"], full.get("size"),
                                               payload.get("copyright_license")))

total = sum(int(x["full_size"] or 0) for x in api_rows)
missing = [x["isic_id"] for x in api_rows if not x["full_url"] or not x["full_size"]]
report = {"schema": "a16-n2-pilot-api-v1",
          "pilot_selection_sha256": hashlib.sha256(PILOT.read_bytes()).hexdigest(),
          "count": len(api_rows), "images": api_rows,
          "sum_full_size_bytes": total,
          "estimated_normalized_bytes": len(api_rows) * 150_000,
          "estimated_peak_bytes": total + len(api_rows) * 150_000 + 20_000_000,
          "budget": {"persistent_target": 250_000_000, "temp_peak_target": 450_000_000},
          "missing_full_url_or_size": missing,
          "licenses": sorted({x["license"] for x in api_rows}),
          "attributions": sorted({x["attribution"] for x in api_rows})}
(OUT/"api_metadata.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
print()
print("48 张 files.full.size 合计 = %d 字节 (%.1f MB)" % (total, total/1e6))
print("估算临时峰值 = %d 字节 (%.1f MB) | 目标 <450MB:" % (
    report["estimated_peak_bytes"], report["estimated_peak_bytes"]/1e6),
    report["estimated_peak_bytes"] < 450_000_000)
print("许可:", report["licenses"], "| 归属:", report["attributions"])
print("缺失 url/size:", missing)