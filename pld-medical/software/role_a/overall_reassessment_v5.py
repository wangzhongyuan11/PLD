"""Compare the frozen v5 candidates with the old model on one threshold grid.

This reads development masks only. The isolated 2018 test set is never opened.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from software.role_a import refine_small as base
from software.role_a.data import enumerate_isic_pairs
from software.role_a.refine_conservative import evaluate_grid
from software.role_a.refine_retention_v5 import (
    EXCLUSIONS, EXPOSED_2017, OUT, SPLIT, TEST2017_EXCLUSIONS,
    THRESHOLDS, sha256_file,
)


def clean(items, excluded):
    return [item for item in items if item.stem not in excluded]


def main() -> None:
    exclusions = json.loads(EXCLUSIONS.read_text(encoding="utf-8"))
    ext_items, ext_areas = base.external_items()
    ext_items = clean(ext_items, set(exclusions["external_validation"]))
    ext_areas = {item.stem: ext_areas[item.stem] for item in ext_items}
    test_items = enumerate_isic_pairs(base.TEST / "images", base.TEST / "masks", require_mask=True)
    test_items = clean(test_items, set(exclusions["official_test_clean_subset_exclusions"]))
    exposed_items = enumerate_isic_pairs(EXPOSED_2017 / "images", EXPOSED_2017 / "masks", require_mask=True)
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    frozen_excluded = set(json.loads(TEST2017_EXCLUSIONS.read_text(encoding="utf-8"))["clean_exclusions"])
    exposed_ids = set(split["validation_ids"])
    assert not (exposed_ids & frozen_excluded)
    exposed_items = [item for item in exposed_items if item.stem in exposed_ids]
    datasets = {
        "external": (ext_items, ext_areas),
        "development2016": (test_items, base.area_map(test_items)),
        "exposed2017_validation": (exposed_items, base.area_map(exposed_items)),
    }
    assert [len(datasets[name][0]) for name in datasets] == [145, 376, 117]
    stems = [set(item.stem for item in items) for items, _ in datasets.values()]
    assert not (stems[0] & stems[1] or stems[0] & stems[2] or stems[1] & stems[2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = base.load_model(base.OLD, device)
    grids = {}
    for name, (items, areas) in datasets.items():
        grid = evaluate_grid(model, items, areas, base.cuts(areas), device, THRESHOLDS)
        grids[name] = {str(t): grid[t][0] for t in THRESHOLDS}
        print(f"baseline {name}: {len(items)} evaluated", flush=True)
    history = json.loads((OUT / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 2 * 4 * len(THRESHOLDS)
    names = tuple(datasets)
    total = sum(len(datasets[name][0]) for name in names)
    def pooled(summaries):
        return sum(summaries[name]["all"]["dice"] * len(datasets[name][0]) for name in names) / total
    baseline_rows = []
    for threshold in THRESHOLDS:
        summaries = {name: grids[name][str(threshold)] for name in names}
        baseline_rows.append({"threshold": threshold, "pooled_dice": pooled(summaries), "sources": summaries})
    candidate_rows = []
    for row in history:
        summaries = {name: row[name] for name in names}
        candidate_rows.append({"epoch": row["epoch"], "alpha": row["alpha"],
                               "threshold": row["threshold"], "candidate": row["candidate"],
                               "pooled_dice": pooled(summaries), "sources": summaries})
    # Equal-score ties retain the earliest predeclared threshold and history row.
    old_best = max(baseline_rows, key=lambda row: row["pooled_dice"])
    new_best = max(candidate_rows, key=lambda row: row["pooled_dice"])
    payload = {
        "schema": "overall-reassessment-v5-v1", "metric": "mean per-image Dice over 638 frozen development images",
        "thresholds": THRESHOLDS, "source_counts": {name: len(datasets[name][0]) for name in names},
        "input_hashes": {"baseline_checkpoint": sha256_file(base.OLD), "split": sha256_file(SPLIT),
                         "history": sha256_file(OUT / "history.json"),
                         "candidate_checkpoint": sha256_file(Path(new_best["candidate"]))},
        "baseline_grid": baseline_rows, "baseline_best": old_best,
        "candidate_best": new_best, "candidate_gain": new_best["pooled_dice"] - old_best["pooled_dice"],
        "candidate_count": len(candidate_rows),
        "holdout": "official ISIC 2018 Task 1 test has not been evaluated",
    }
    target = OUT / "overall_reassessment.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"old={old_best['pooled_dice']:.6f}@{old_best['threshold']} "
          f"candidate={new_best['pooled_dice']:.6f}@{new_best['threshold']} "
          f"gain={payload['candidate_gain']:.6f}", flush=True)


if __name__ == "__main__":
    main()
