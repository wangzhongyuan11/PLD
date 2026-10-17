"""Freeze a stratified split of the now-exposed ISIC 2017 test set for v4 development."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import torch

from software.role_a import refine_small as base
from software.role_a.data import enumerate_isic_pairs
from software.role_a.refine_from_baseline_v3 import atomic_write_json, sha256_file


DATA = base.PROJECT / "data" / "raw" / "isic2017_test"
EXCLUSIONS = base.PROJECT / "runs" / "role_a_full2017" / "test2017_overlap_exclusions.json"
OUT = base.PROJECT / "runs" / "role_a_augmented_v4"
SEED = "role-a-v4-20260924"


def main() -> None:
    provenance = json.loads((DATA / "provenance.json").read_text(encoding="utf-8"))
    excluded = set(json.loads(EXCLUSIONS.read_text(encoding="utf-8"))["clean_exclusions"])
    if provenance.get("complete") is not True or len(provenance["items"]) != 600:
        raise ValueError("2017 test provenance is incomplete")
    items = enumerate_isic_pairs(DATA / "images", DATA / "masks", require_mask=True)
    if len(items) != 600 or {item.stem for item in items} != set(provenance["items"]):
        raise ValueError("2017 test files differ from verified provenance")
    areas = {stem: float(row["area_ratio"]) for stem, row in provenance["items"].items()
             if stem not in excluded}
    q10, q50 = np.quantile(list(areas.values()), [0.1, 0.5])
    bands = {
        "tiny": [stem for stem, area in areas.items() if area <= q10],
        "medium": [stem for stem, area in areas.items() if q10 < area <= q50],
        "large": [stem for stem, area in areas.items() if area > q50],
    }
    validation = []
    for ids in bands.values():
        ids.sort(key=lambda stem: hashlib.sha256(f"{SEED}:{stem}".encode()).hexdigest())
        validation.extend(ids[:round(0.2 * len(ids))])
    validation = sorted(validation)
    training = sorted(set(areas) - set(validation))
    if len(training) != 466 or len(validation) != 117 or set(training) & set(validation):
        raise ValueError("Unexpected v4 split sizes or overlap")
    choice = {"schema": "role-a-augmented-v4-split-v1",
              "disclosure": "ISIC 2017 official test was scored once and is now exposed development data",
              "source_provenance_sha256": sha256_file(DATA / "provenance.json"),
              "source_overlap_exclusions_sha256": sha256_file(EXCLUSIONS),
              "seed": SEED, "method": "per-ground-truth-area band, SHA256(seed:id) sort, 20% validation",
              "area_band_cuts": {"q10": float(q10), "q50": float(q50)},
              "band_sizes": {name: len(ids) for name, ids in bands.items()},
              "train_ids": training, "validation_ids": validation,
              "final_holdout": "official ISIC 2018 Task 1 test; no scores viewed"}
    OUT.mkdir(parents=True, exist_ok=True)
    split_path = OUT / "split.json"
    if split_path.exists() and json.loads(split_path.read_text(encoding="utf-8")) != choice:
        raise ValueError("Existing v4 split differs; refusing overwrite")
    atomic_write_json(split_path, choice)
    validation_items = [item for item in items if item.stem in set(validation)]
    validation_areas = {stem: areas[stem] for stem in validation}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = base.load_model(base.OLD, device)
    summary, records = base.evaluate(model, validation_items, validation_areas,
                                     base.cuts(validation_areas), device)
    atomic_write_json(OUT / "validation_baseline.json", {
        "baseline_sha256": sha256_file(base.OLD),
        "split_sha256": sha256_file(split_path),
        "count": len(validation_items), "summary": summary, "records": records})
    print(f"v4 split train={len(training)} validation={len(validation)} "
          f"band_sizes={choice['band_sizes']}", flush=True)
    print("baseline Dice", {key: summary[key]["dice"] for key in
                            ("all", "tiny", "medium", "large")}, flush=True)


if __name__ == "__main__":
    main()
