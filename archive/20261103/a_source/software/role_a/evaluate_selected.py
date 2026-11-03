"""One-time paired evaluation of the selected model on ISIC 2016 test masks.

No threshold or checkpoint is selected with these test results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

from software.role_a import refine_small as base
from software.role_a.data import enumerate_isic_pairs


OUTPUT = base.PROJECT / "runs" / "role_a_full2017"
SELECTED = OUTPUT / "best.pt"


@torch.inference_mode()
def run() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    items = enumerate_isic_pairs(base.TEST / "images", base.TEST / "masks", require_mask=True)
    stems = [item.stem for item in items]
    if len(stems) != 379 or len(set(stems)) != 379:
        raise ValueError("expected 379 unique ISIC 2016 test image-mask pairs")
    selected, checkpoint = base.load_model(SELECTED, device)
    threshold = float(checkpoint["threshold"])
    if not 0.0 < threshold < 1.0 or checkpoint.get("selection") != "strict area-group gate":
        raise ValueError("selected checkpoint lacks a valid, gated threshold")
    original, _ = base.load_model(base.OLD, device)
    original.eval()
    selected.eval()
    areas = base.area_map(items)
    group_cuts = base.cuts(areas)
    records = {"original": [], "selected": []}
    for batch in base.loader(items):
        images = batch["image"].to(device)
        logits = {"original": original(images).cpu(), "selected": selected(images).cpu()}
        for i, filename in enumerate(batch["filename"]):
            stem = Path(filename).stem
            shared = {"stem": stem, "area_ratio": areas[stem], "band": base.band(areas[stem], group_cuts)}
            for name, cutoff in (("original", 0.5), ("selected", threshold)):
                records[name].append({**shared,
                                      **base.paired_score(logits[name][i], batch["mask"][i],
                                                          batch["valid"][i], cutoff)})
    summaries = {name: base.summarize(rows, group_cuts) for name, rows in records.items()}
    exclusions = json.loads((OUTPUT / "overlap_exclusions.json").read_text(encoding="utf-8"))
    excluded_test = set(exclusions["official_test_clean_subset_exclusions"])
    if not excluded_test <= set(stems):
        raise ValueError("test overlap exclusion ID absent from official data")
    clean_areas = {stem: area for stem, area in areas.items() if stem not in excluded_test}
    clean_cuts = base.cuts(clean_areas)
    clean_records = {name: [{**row, "band": base.band(row["area_ratio"], clean_cuts)}
                            for row in rows if row["stem"] not in excluded_test]
                     for name, rows in records.items()}
    clean_summaries = {name: base.summarize(rows, clean_cuts)
                       for name, rows in clean_records.items()}
    base._save_json(OUTPUT / "official_test_comparison.json", {
        "dataset": "ISIC 2016 Task 1 official test",
        "test_image_count": len(items),
        "test_area_cuts": {"q10": group_cuts[0], "q50": group_cuts[1]},
        "original_checkpoint_sha256": hashlib.sha256(base.OLD.read_bytes()).hexdigest(),
        "selected_checkpoint_sha256": hashlib.sha256(SELECTED.read_bytes()).hexdigest(),
        "thresholds": {"original": 0.5, "selected": threshold},
        "model_selection_uses_test": False,
        "summaries": summaries,
        "records": records,
        "clean_subset_excluded_test_ids": sorted(excluded_test),
        "clean_subset_area_cuts": {"q10": clean_cuts[0], "q50": clean_cuts[1]},
        "clean_subset_summaries": clean_summaries,
        "clean_subset_records": clean_records,
    })
    for group in ("all", "tiny", "medium", "large"):
        before = summaries["original"][group]["dice"]
        after = summaries["selected"][group]["dice"]
        print(f"{group}: original={before:.6f} selected={after:.6f} delta={after-before:+.6f}")
    print("clean subset (confirmed exact overlaps removed):")
    for group in ("all", "tiny", "medium", "large"):
        before = clean_summaries["original"][group]["dice"]
        after = clean_summaries["selected"][group]["dice"]
        print(f"{group}: original={before:.6f} selected={after:.6f} delta={after-before:+.6f}")


if __name__ == "__main__":
    run()
