"""Reclassify the now-exposed 2016 test as development feedback, never final holdout."""
from __future__ import annotations

import json

import torch

from software.role_a import refine_small as base
from software.role_a.data import enumerate_isic_pairs
from software.role_a.refine_conservative import evaluate_grid
from software.role_a.refine_full2017 import OUT, THRESHOLDS


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = base.load_model(OUT / "best.pt", device)
    items = enumerate_isic_pairs(base.TEST / "images", base.TEST / "masks", require_mask=True)
    areas = base.area_map(items)
    original, _ = base.load_model(base.OLD, device)
    original_grid = evaluate_grid(original, items, areas, base.cuts(areas), device, (0.5,))
    grid = evaluate_grid(model, items, areas, base.cuts(areas), device, THRESHOLDS)
    excluded = set(json.loads((OUT / "overlap_exclusions.json").read_text(encoding="utf-8"))
                   ["official_test_clean_subset_exclusions"])
    clean_areas = {stem: area for stem, area in areas.items() if stem not in excluded}
    clean_cuts = base.cuts(clean_areas)

    def clean_summary(rows):
        selected = [{**row, "band": base.band(row["area_ratio"], clean_cuts)}
                    for row in rows if row["stem"] not in excluded]
        return base.summarize(selected, clean_cuts)

    baseline = clean_summary(original_grid[0.5][1])
    candidates = {str(threshold): {"full": metrics, "clean": clean_summary(rows)}
                  for threshold, (metrics, rows) in grid.items()}
    base._save_json(OUT / "development_threshold_grid2016.json", {
        "warning": "ISIC 2016 test has been exposed; use only as development feedback",
        "baseline_clean": baseline, "candidates": candidates,
        "excluded_confirmed_duplicate_test_ids": sorted(excluded),
    })
    for threshold, result in candidates.items():
        metrics = result["clean"]
        print(threshold, {name: round(metrics[name]["dice"] - baseline[name]["dice"], 6)
                          for name in ("tiny", "medium", "large", "all")})


if __name__ == "__main__":
    main()
