"""Evaluate the preselected unchanged-weight 0.65 control on ISIC 2016 test."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from software.role_a import refine_small as base
from software.role_a.data import enumerate_isic_pairs


OUT = base.PROJECT / "runs" / "role_a_full2017"
THRESHOLDS = (0.5, 0.65)


@torch.inference_mode()
def main() -> None:
    model, _ = base.load_model(base.OLD, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    device = next(model.parameters()).device
    model.eval()
    items = enumerate_isic_pairs(base.TEST / "images", base.TEST / "masks", require_mask=True)
    if len(items) != 379:
        raise ValueError("ISIC 2016 test count changed")
    areas = base.area_map(items)
    full_cuts = base.cuts(areas)
    excluded = set(json.loads((OUT / "overlap_exclusions.json").read_text(encoding="utf-8"))
                   ["official_test_clean_subset_exclusions"])
    clean_areas = {stem: area for stem, area in areas.items() if stem not in excluded}
    clean_cuts = base.cuts(clean_areas)
    records = {str(t): [] for t in THRESHOLDS}
    for batch in base.loader(items):
        logits = model(batch["image"].to(device)).cpu()
        for i, filename in enumerate(batch["filename"]):
            stem = Path(filename).stem
            for t in THRESHOLDS:
                records[str(t)].append({"stem": stem, "area_ratio": areas[stem],
                                        "band": base.band(areas[stem], full_cuts),
                                        **base.paired_score(logits[i], batch["mask"][i],
                                                            batch["valid"][i], t)})
    summary = {t: base.summarize(rows, full_cuts) for t, rows in records.items()}
    clean_records = {t: [{**row, "band": base.band(row["area_ratio"], clean_cuts)}
                         for row in rows if row["stem"] not in excluded]
                     for t, rows in records.items()}
    clean_summary = {t: base.summarize(rows, clean_cuts) for t, rows in clean_records.items()}
    base._save_json(OUT / "calibrated_test_control.json", {
        "selection_source": "calibrated_control.json, saved before candidate test evaluation",
        "original_checkpoint": str(base.OLD), "thresholds": THRESHOLDS,
        "full_summaries": summary, "full_records": records,
        "clean_excluded_ids": sorted(excluded),
        "clean_summaries": clean_summary, "clean_records": clean_records,
    })
    for title, group in (("full", summary), ("clean", clean_summary)):
        print(title, {name: (round(group["0.5"][name]["dice"], 6),
                             round(group["0.65"][name]["dice"], 6))
                      for name in ("tiny", "medium", "large", "all")})


if __name__ == "__main__":
    main()
