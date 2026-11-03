"""A16-N2 后处理（纯本地）：逐图复验 -> 分组对照页 -> 与 train/dev/final 仅图像重复筛查。"""
from __future__ import annotations
import csv, hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

from software.role_a.audit_image_overlap import dhash, records
from software.role_a import train_overall_v1 as overall

ROOT = Path("data/raw/isic2018_novice_pilot")
AUDIT = Path("runs/role_a_a16_novice_image_pilot")
PILOT = Path("runs/role_a_a16_novice_mask_audit/pilot_selection.csv")
PANEL, ROWS_PER_SHEET = 200, 8
REFERENCE = {
    "train2016": Path("data/raw/isic2016/train/images"),
    "development2016": Path("data/raw/isic2016/test/images"),
    "train2017": Path("data/raw/isic2017_training/images"),
    "validation2017": Path("data/raw/isic2017_validation/images"),
    "development2017": Path("data/raw/isic2017_test/images"),
    "final_test2018_images": Path("data/raw/isic2018_test/images"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with Path(p).open("rb") as s:
        for b in iter(lambda: s.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    audit = json.loads((AUDIT/"pair_audit_full.json").read_text(encoding="utf-8"))
    rows = {r["isic_id"]: r for r in csv.DictReader(PILOT.open(encoding="utf-8", newline=""))}
    problems, pairs_out = [], []
    for rec in audit["pairs"]:
        i = rec["isic_id"]
        raw = ROOT/"raw_images"/f"{i}.jpg"
        norm = ROOT/"images"/f"{i}.png"
        msk = ROOT/"masks"/f"{i}_UNET.png"
        item = {"isic_id": i, "area_band": rec["area_band"]}
        for key, path in (("raw", raw), ("normalized", norm), ("mask", msk)):
            if not path.is_file():
                problems.append(f"{i}: 缺 {key}"); continue
            item[f"{key}_bytes"] = path.stat().st_size
            item[f"{key}_sha256"] = sha256(path)
            item[f"{key}_sha256_matches_audit"] = (
                item[f"{key}_sha256"] == rec.get(f"{key}_sha256", item[f"{key}_sha256"]))
        if raw.is_file():
            with Image.open(raw) as im:
                item["raw_size"] = list(im.size); item["raw_format"] = im.format
        if msk.is_file():
            with Image.open(msk) as im:
                arr = np.asarray(im.convert("L"))
                item["mask_size"] = list(im.size)
                item["mask_values"] = sorted(set(np.unique(arr).tolist()))
                item["mask_area_ratio"] = round(float((arr > 0).mean()), 6)
        if norm.is_file():
            with Image.open(norm) as im:
                item["normalized_size"] = list(im.size)
        item["area_ratio_matches_selection"] = (
            abs(item.get("mask_area_ratio", -1) - float(rows[i]["area_ratio"])) <= 2e-3)
        pairs_out.append(item)

    # 对照页：按面积段分组
    sheets = []
    for band in ("tiny", "medium", "large"):
        sel = [p for p in pairs_out if p["area_band"] == band]
        sel.sort(key=lambda p: p["isic_id"])
        for offset in range(0, len(sel), ROWS_PER_SHEET):
            chunk = sel[offset:offset + ROWS_PER_SHEET]
            canvas = Image.new("RGB", (PANEL*3 + 60, len(chunk)*(PANEL + 44) + 10), "white")
            draw = ImageDraw.Draw(canvas)
            for r, item in enumerate(chunk):
                i = item["isic_id"]
                top = 8 + r*(PANEL + 44)
                with Image.open(ROOT/"images"/f"{i}.png") as img:
                    base = img.convert("RGB").resize((PANEL, PANEL), Image.Resampling.LANCZOS)
                with Image.open(ROOT/"masks"/f"{i}_UNET.png") as mk:
                    m = mk.convert("L").resize((PANEL, PANEL), Image.Resampling.NEAREST)
                overlay = base.copy()
                arr = np.asarray(m) > 127
                ov = np.asarray(overlay).copy()
                ov[arr] = (0.55*ov[arr] + 0.45*np.array([255, 0, 0])).astype(np.uint8)
                pad = np.pad(arr, 1, constant_values=False)
                eroded = (pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:])
                edge = arr & ~eroded          # 前景边界像素（4 邻域至少一个背景）
                ov[edge] = (255, 255, 0)
                overlay = Image.fromarray(ov)
                binary = Image.fromarray((arr.astype(np.uint8)*255)).convert("RGB")
                for c, im in enumerate((base, overlay, binary)):
                    canvas.paste(im, (10 + c*(PANEL + 20), top + 26))
                draw.text((10, top), f"{i}  band={item['area_band']}  area={item['mask_area_ratio']:.4f}  "
                                     f"600x450  mask={item['mask_values']}", fill="black")
            for c, title in enumerate(("derived image (600x450)", "image + mask contour", "binary mask")):
                draw.text((10 + c*(PANEL + 20), 10 + len(chunk)*(PANEL + 44)), title, fill="black")
            path = AUDIT/f"sheet_{band}_{offset//ROWS_PER_SHEET + 1}.jpg"
            canvas.save(path, quality=90); sheets.append(path.name)

    # 重复筛查（仅图像）
    pilot_records = records(ROOT/"raw_images")
    screen = {"schema": "a16-n2-overlap-screen-v1", "pilot_count": len(pilot_records),
              "comparisons": {}, "intra_pilot": {"exact": [], "dhash_le4": []}}
    suspicious = []
    for name, d in REFERENCE.items():
        ref = records(d)
        exact, close = [], []
        for a in pilot_records:
            for b in ref:
                if a["sha256"] == b["sha256"]:
                    exact.append({"pilot": a["stem"], "ref": b["stem"], "set": name})
                dist = (a["dhash"] ^ b["dhash"]).bit_count()
                if dist <= 4:
                    close.append({"pilot": a["stem"], "ref": b["stem"], "set": name, "distance": dist})
        screen["comparisons"][name] = {"reference_count": len(ref), "exact": exact, "dhash_le4": close}
        suspicious.extend(close)
    for x in range(len(pilot_records)):
        for y in range(x + 1, len(pilot_records)):
            a, b = pilot_records[x], pilot_records[y]
            if a["sha256"] == b["sha256"]:
                screen["intra_pilot"]["exact"].append({"a": a["stem"], "b": b["stem"]})
            dist = (a["dhash"] ^ b["dhash"]).bit_count()
            if dist <= 4:
                screen["intra_pilot"]["dhash_le4"].append({"a": a["stem"], "b": b["stem"], "distance": dist})
    screen["suspicious_count"] = len(suspicious)

    # 可疑项并排页
    side = []
    if suspicious:
        uniq = []
        for s in suspicious:
            if s["pilot"] not in uniq:
                uniq.append(s["pilot"])
        for offset in range(0, len(uniq), 4):
            chunk = uniq[offset:offset + 4]
            canvas = Image.new("RGB", (PANEL*2 + 40, len(chunk)*(PANEL + 44) + 10), "white")
            draw = ImageDraw.Draw(canvas)
            for r, pid in enumerate(chunk):
                best = min((s for s in suspicious if s["pilot"] == pid), key=lambda s: s["distance"])
                top = 8 + r*(PANEL + 44)
                with Image.open(ROOT/"raw_images"/f"{pid}.jpg") as a, \
                     Image.open(REFERENCE[best["set"]]/f"{best['ref']}.jpg") as b:
                    canvas.paste(a.convert("RGB").resize((PANEL, PANEL), Image.Resampling.LANCZOS), (10, top + 26))
                    canvas.paste(b.convert("RGB").resize((PANEL, PANEL), Image.Resampling.LANCZOS), (PANEL + 30, top + 26))
                draw.text((10, top), f"pilot {pid}  vs  {best['set']}/{best['ref']}  dHash={best['distance']}", fill="black")
            path = AUDIT/f"suspicious_pairs_{offset//4 + 1}.jpg"
            canvas.save(path, quality=90); side.append(path.name)
    screen["suspicious_sheets"] = side

    summary = {"schema": "a16-n2-pair-audit-summary-v1", "count": len(pairs_out),
               "bands": dict(Counter(p["area_band"] for p in pairs_out)),
               "problems": problems,
               "hashes_matching_audit": all(
                   all(p.get(f"{k}_sha256_matches_audit") for k in ("raw", "normalized", "mask"))
                   for p in pairs_out if f"raw_sha256_matches_audit" in p),
               "registration_ok_all": all(p.get("normalized_size") == p.get("mask_size") for p in pairs_out),
               "areas_match_selection": all(p["area_ratio_matches_selection"] for p in pairs_out),
               "sheets": sheets,
               "licenses": {"image": "CC-BY-NC / MILK study team",
                            "mask": "CC-BY-NC / ViDIR Group, Department of Dermatology, Medical University of Vienna"}}
    (AUDIT/"pair_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (AUDIT/"pair_audit_recheck.json").write_text(json.dumps({"pairs": pairs_out}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    (AUDIT/"overlap_screen.json").write_text(json.dumps(screen, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"pairs": len(pairs_out), "problems": len(problems), "sheets": len(sheets),
                      "suspicious": len(suspicious), "suspicious_sheets": len(side),
                      "overlap": {k: {"exact": len(v["exact"]), "dhash_le4": len(v["dhash_le4"])}
                                  for k, v in screen["comparisons"].items()},
                      "intra": {k: len(v) for k, v in screen["intra_pilot"].items()}},
                     ensure_ascii=False, indent=2))
    return 0 if not problems else 4


if __name__ == "__main__":
    raise SystemExit(main())