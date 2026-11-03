"""Search a fixed compact area-threshold grid on all three v4 development sets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from software.role_a import refine_small as base
from software.role_a.data import enumerate_isic_pairs
from software.role_a.refine_full2017 import EXCLUSIONS, OUT as FIRST_OUT
from software.role_a.refine_augmented_v4 import OUT, EXPOSED_2017, SPLIT, NEW_BASELINE
from software.role_a.search_adaptive_thresholds import (AREA_CUTS, SMALL_THRESHOLDS,
                                                      LARGE_THRESHOLDS, collect, summary)


def main(candidate: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Adaptive search report already exists: {output}")
    exclusions = json.loads(EXCLUSIONS.read_text(encoding="utf-8"))
    first = json.loads((FIRST_OUT / "protocol.json").read_text(encoding="utf-8"))
    old_development = json.loads((FIRST_OUT / "official_test_comparison.json")
                                 .read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    new_baseline = json.loads(NEW_BASELINE.read_text(encoding="utf-8"))
    baselines = {"external": first["baseline"],
                 "development2016": old_development["clean_subset_summaries"]["original"],
                 "exposed2017_validation": new_baseline["summary"]}
    ext_items, ext_areas = base.external_items()
    ext_items = [item for item in ext_items if item.stem not in exclusions["external_validation"]]
    ext_areas = {stem: area for stem, area in ext_areas.items()
                 if stem not in exclusions["external_validation"]}
    dev_items = enumerate_isic_pairs(base.TEST / "images", base.TEST / "masks", require_mask=True)
    dev_items = [item for item in dev_items
                 if item.stem not in exclusions["official_test_clean_subset_exclusions"]]
    dev_areas = base.area_map(dev_items)
    new_ids = set(split["validation_ids"])
    new_items = [item for item in enumerate_isic_pairs(EXPOSED_2017 / "images",
                                                       EXPOSED_2017 / "masks", require_mask=True)
                 if item.stem in new_ids]
    if len(new_items) != 117:
        raise ValueError("v4 new development set changed")
    new_areas = base.area_map(new_items)
    datasets = {"external": (ext_items, ext_areas),
                "development2016": (dev_items, dev_areas),
                "exposed2017_validation": (new_items, new_areas)}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = base.load_model(candidate, device)
    material = {}
    for name, (items, areas) in datasets.items():
        cuts = base.cuts(areas)
        material[name] = (collect(model, items, areas, cuts, device), cuts)
        print(f"{name}: scored {len(material[name][0])} images", flush=True)
    results = []
    for area_cut in AREA_CUTS:
        for small_threshold in SMALL_THRESHOLDS:
            for large_threshold in LARGE_THRESHOLDS:
                metrics = {name: summary(rows, cuts, area_cut, small_threshold, large_threshold)
                           for name, (rows, cuts) in material.items()}
                accepted = all(base.acceptance(metrics[name], baselines[name]) for name in metrics)
                deltas = {name: {band: metrics[name][band]["dice"] - baselines[name][band]["dice"]
                                 for band in ("all", "tiny", "medium", "large")}
                          for name in metrics}
                results.append({"area_cut": area_cut, "small_threshold": small_threshold,
                                "large_threshold": large_threshold, "accepted": accepted,
                                "dice_delta": deltas,
                                "min_tiny_gain": min(deltas[name]["tiny"] for name in deltas),
                                "min_other_margin": min(deltas[name][band] for name in deltas
                                                        for band in ("all", "medium", "large"))})
    accepted_rows = sorted((row for row in results if row["accepted"]),
                           key=lambda row: (row["min_other_margin"], row["min_tiny_gain"]),
                           reverse=True)
    output.write_text(json.dumps({"candidate": str(candidate),
                                  "development_only": True,
                                  "official_test2018_scores_viewed": False,
                                  "grid": {"area_cut": AREA_CUTS,
                                           "small_threshold": SMALL_THRESHOLDS,
                                           "large_threshold": LARGE_THRESHOLDS},
                                  "accepted_count": len(accepted_rows),
                                  "best_robust_accepted": accepted_rows[0] if accepted_rows else None,
                                  "results": results}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print("accepted", len(accepted_rows), flush=True)
    print("best_robust", accepted_rows[0] if accepted_rows else None, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.candidate, args.output)
