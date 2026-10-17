"""A07: ID / SHA256 / dHash / colour overlap audit for the ISIC 2018 validation supplement.

只读既有 2018 最终测试【图像】，绝不读取其掩膜或成绩。
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from software.role_a.audit_close_pairs import pixels
from software.role_a.audit_image_overlap import compare, records
from software.role_a.refine_small import PROJECT


VALIDATION = PROJECT / "data" / "raw" / "isic2018_validation" / "images"
OUT = PROJECT / "runs" / "role_a_isic2018_validation_audit"
AUDIT = OUT / "overlap_audit.json"
SIMILARITY = OUT / "overlap_similarity.json"
SUMMARY = OUT / "summary.json"
SHEET_SIZE = 220
PER_SHEET = 6
DHASH_THRESHOLD = 4
OTHERS = {
    "train2016": PROJECT / "data" / "raw" / "isic2016" / "train" / "images",
    "train2017": PROJECT / "data" / "raw" / "isic2017_training" / "images",
    "validation2017": PROJECT / "data" / "raw" / "isic2017_validation" / "images",
    "development2016": PROJECT / "data" / "raw" / "isic2016" / "test" / "images",
    "development2017": PROJECT / "data" / "raw" / "isic2017_test" / "images",
    "final_test2018_images": PROJECT / "data" / "raw" / "isic2018_test" / "images",
}


@lru_cache(maxsize=2048)
def cached_pixels(path: Path) -> np.ndarray:
    return pixels(path)


def render_sheets(rows: list) -> list:
    written = []
    for offset in range(0, len(rows), PER_SHEET):
        chunk = rows[offset:offset + PER_SHEET]
        canvas = Image.new("RGB", (SHEET_SIZE * 2 + 30, len(chunk) * (SHEET_SIZE + 38)), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (group, row) in enumerate(chunk):
            y = index * (SHEET_SIZE + 38)
            paths = (VALIDATION / (row["left"] + ".jpg"), OTHERS[group] / (row["right"] + ".jpg"))
            for column, path in enumerate(paths):
                with Image.open(path) as raw:
                    image = ImageOps.contain(ImageOps.exif_transpose(raw).convert("RGB"),
                                             (SHEET_SIZE, SHEET_SIZE))
                x = 5 + column * (SHEET_SIZE + 20) + (SHEET_SIZE - image.width) // 2
                canvas.paste(image, (x, y + 18 + (SHEET_SIZE - image.height) // 2))
            draw.text((5, y + 3),
                      f"{row['left']} / {row['right']} {group} "
                      f"mae={row['rgb_mae']:.3f} corr={row['rgb_correlation']:.3f}",
                      fill="black")
        path = OUT / ("validation2018_overlap_sheet_" + str(offset // PER_SHEET + 1) + ".jpg")
        canvas.save(path, quality=90)
        written.append(path.name)
    return written


def main() -> None:
    provenance = json.loads((VALIDATION.parent / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("complete") is not True or provenance.get("count") != 100:
        raise ValueError("ISIC 2018 validation data are not completely verified")
    validation = records(VALIDATION)
    if len(validation) != 100 or {row["stem"] for row in validation} != set(provenance["items"]):
        raise ValueError("ISIC 2018 validation images differ from verified provenance")
    result = {"schema": "isic2018-validation-overlap-audit-v1",
              "purpose": "image-only screening of a candidate training supplement",
              "dhash_threshold": DHASH_THRESHOLD,
              "validation_count": len(validation),
              "note": "dHash 近邻只是筛查候选，不能证明同一病灶或同一患者；本次未做视觉初审",
              "comparisons": {}}
    for name, directory in OTHERS.items():
        other = records(directory)
        row = {"other_count": len(other),
               "id_intersection": sorted({r["stem"] for r in validation} &
                                         {r["stem"] for r in other}),
               **compare(validation, other, threshold=DHASH_THRESHOLD)}
        result["comparisons"][name] = row
        print(name, "other", len(other), "id_overlap", len(row["id_intersection"]),
              "exact", len(row["exact_sha256_pairs"]),
              "dhash<=4", len(row["dhash_close_pairs"]), flush=True)
    AUDIT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    similarity = {"schema": "isic2018-validation-overlap-similarity-v1", "comparisons": {}}
    for name, group in result["comparisons"].items():
        rows = []
        for pair in group["dhash_close_pairs"]:
            left = cached_pixels(VALIDATION / (pair["left"] + ".jpg"))
            right = cached_pixels(OTHERS[name] / (pair["right"] + ".jpg"))
            mae = float(np.abs(left - right).mean())
            a, b = left - left.mean(), right - right.mean()
            correlation = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
            rows.append({**pair, "rgb_mae": mae, "rgb_correlation": correlation})
        rows.sort(key=lambda row: (row["rgb_mae"], -row["rgb_correlation"]))
        similarity["comparisons"][name] = rows
        strong = [r for r in rows if r["rgb_mae"] <= 0.12 and r["rgb_correlation"] >= 0.9]
        print(name, "screen_pairs", len(rows), "strong_pairs", len(strong), flush=True)
    SIMILARITY.write_text(json.dumps(similarity, ensure_ascii=False, indent=2), encoding="utf-8")

    candidates = [(group, row) for group, rows in similarity["comparisons"].items() for row in rows
                  if row["rgb_mae"] <= 0.18 and row["rgb_correlation"] >= 0.9]
    candidates.sort(key=lambda entry: entry[1]["rgb_mae"])
    sheets = render_sheets(candidates)
    exact = {group: [a + "|" + b for a, b in group_row["exact_sha256_pairs"]]
             for group, group_row in result["comparisons"].items()
             if group_row["exact_sha256_pairs"]}
    ids = {group: group_row["id_intersection"]
           for group, group_row in result["comparisons"].items() if group_row["id_intersection"]}
    strong = [{"group": group, **row} for group, row in candidates]
    uncertain = sorted({row["left"] for row in strong})
    summary = {
        "schema": "isic2018-validation-overlap-summary-v1",
        "validation_count": 100,
        "exact_sha256_overlap": exact,
        "shared_ids": ids,
        "strong_similarity_candidates": strong,
        "strong_candidate_count": len(strong),
        "screened_dhash_pairs": sum(len(rows) for rows in similarity["comparisons"].values()),
        "contact_sheets": sheets,
        "uncertain_ids": uncertain,
        "uncertain_count": len(uncertain),
        "candidate_supplement_after_screen": 100 - len(uncertain),
        "visual_review": "未执行：本执行助手无法查看图像内容，未做视觉初审；对照页仅供 Codex 人工复核",
        "criteria": "exact = 文件 SHA256 相同；strong = dHash<=4 且 RGB MAE<=0.18 且 RGB 相关系数>=0.9；"
                    "strong 候选记为不确定，不自动排除",
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("exact_overlap_groups", list(exact), "shared_id_groups", list(ids),
          "strong_candidates", len(strong), "uncertain_ids", len(uncertain),
          "contact_sheets", len(sheets), flush=True)


if __name__ == "__main__":
    main()