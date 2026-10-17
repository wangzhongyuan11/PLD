"""Evaluate a fixed repaired checkpoint on development sets only."""
from __future__ import annotations

import json

import torch

from software.role_a import refine_small as base
from software.role_a.data import enumerate_isic_pairs
from software.role_a.refine_conservative import evaluate_grid
from software.role_a.refine_full2017 import EXCLUSIONS, OUT as FIRST_OUT
from software.role_a.refine_preserve_large_v2 import OUT


THRESHOLDS = (0.35, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.53, 0.56, 0.6)
CANDIDATE = OUT / "candidates" / "epoch01_alpha0.25.pt"
OUTPUT = OUT / "threshold_exploration.json"


def main() -> None:
    exclusions = json.loads(EXCLUSIONS.read_text(encoding="utf-8"))
    protocol = json.loads((FIRST_OUT / "protocol.json").read_text(encoding="utf-8"))
    comparison = json.loads((FIRST_OUT / "official_test_comparison.json")
                            .read_text(encoding="utf-8"))
    baseline = {"external": protocol["baseline"],
                "development2016": comparison["clean_subset_summaries"]["original"]}
    external_items, external_areas = base.external_items()
    external_items = [item for item in external_items
                      if item.stem not in exclusions["external_validation"]]
    external_areas = {stem: area for stem, area in external_areas.items()
                      if stem not in exclusions["external_validation"]}
    development_items = enumerate_isic_pairs(base.TEST / "images", base.TEST / "masks",
                                              require_mask=True)
    development_items = [item for item in development_items
                         if item.stem not in exclusions["official_test_clean_subset_exclusions"]]
    development_areas = base.area_map(development_items)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = base.load_model(CANDIDATE, device)
    grids = {
        "external": evaluate_grid(model, external_items, external_areas,
                                  base.cuts(external_areas), device, THRESHOLDS),
        "development2016": evaluate_grid(model, development_items, development_areas,
                                         base.cuts(development_areas), device, THRESHOLDS),
    }
    rows = []
    for threshold in THRESHOLDS:
        metrics = {name: grids[name][threshold][0] for name in grids}
        rows.append({"threshold": threshold, "metrics": metrics,
                     "accepted": all(base.acceptance(metrics[name], baseline[name])
                                     for name in metrics),
                     "dice_delta": {name: {band: metrics[name][band]["dice"] -
                                            baseline[name][band]["dice"]
                                          for band in ("all", "tiny", "medium", "large")}
                                    for name in metrics}})
    OUTPUT.write_text(json.dumps({"candidate": str(CANDIDATE),
                                  "purpose": "development exploration; official 2017 test untouched",
                                  "thresholds": THRESHOLDS, "rows": rows},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    for row in rows:
        print(row["threshold"], row["accepted"], row["dice_delta"], flush=True)


if __name__ == "__main__":
    main()
