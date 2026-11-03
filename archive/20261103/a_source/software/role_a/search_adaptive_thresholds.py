"""Search a compact single-network area-conditioned threshold rule on development sets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from software.role_a import refine_small as base
from software.role_a.data import enumerate_isic_pairs
from software.role_a.refine_full2017 import EXCLUSIONS, OUT as FIRST_OUT
from software.role_a.refine_preserve_large_v2 import OUT


CANDIDATE = OUT / "candidates" / "epoch01_alpha0.25.pt"
OUTPUT = OUT / "adaptive_threshold_search.json"
AREA_CUTS = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)
SMALL_THRESHOLDS = (0.5, 0.53, 0.56, 0.6)
LARGE_THRESHOLDS = (0.3, 0.35, 0.4, 0.45, 0.5)
THRESHOLDS = tuple(sorted(set(SMALL_THRESHOLDS + LARGE_THRESHOLDS)))


@torch.inference_mode()
def collect(model, items, areas, cuts, device):
    model.eval()
    rows = []
    for batch in base.loader(items):
        logits = model(batch["image"].to(device)).cpu()
        for index, filename in enumerate(batch["filename"]):
            stem = Path(filename).stem
            valid = batch["valid"][index]
            prediction = (torch.sigmoid(logits[index]) >= 0.5) & (valid > 0)
            pred_area = float(prediction.sum() / (valid > 0).sum())
            rows.append({"stem": stem, "area_ratio": areas[stem],
                         "band": base.band(areas[stem], cuts),
                         "predicted_area_at_0_5": pred_area,
                         "scores": {threshold: base.paired_score(logits[index],
                                                                  batch["mask"][index], valid,
                                                                  threshold)
                                    for threshold in THRESHOLDS}})
    return rows


def summary(rows, cuts, area_cut, small_threshold, large_threshold):
    selected = []
    for row in rows:
        threshold = (large_threshold if row["predicted_area_at_0_5"] >= area_cut
                     else small_threshold)
        selected.append({"stem": row["stem"], "area_ratio": row["area_ratio"],
                         "band": row["band"], **row["scores"][threshold]})
    return base.summarize(selected, cuts)


def main(candidate: Path = CANDIDATE, output: Path = OUTPUT) -> None:
    exclusions = json.loads(EXCLUSIONS.read_text(encoding="utf-8"))
    first = json.loads((FIRST_OUT / "protocol.json").read_text(encoding="utf-8"))
    comparison = json.loads((FIRST_OUT / "official_test_comparison.json")
                            .read_text(encoding="utf-8"))
    baselines = {"external": first["baseline"],
                 "development2016": comparison["clean_subset_summaries"]["original"]}
    ext_items, ext_areas = base.external_items()
    ext_items = [item for item in ext_items if item.stem not in exclusions["external_validation"]]
    ext_areas = {stem: area for stem, area in ext_areas.items()
                 if stem not in exclusions["external_validation"]}
    dev_items = enumerate_isic_pairs(base.TEST / "images", base.TEST / "masks", require_mask=True)
    dev_items = [item for item in dev_items
                 if item.stem not in exclusions["official_test_clean_subset_exclusions"]]
    dev_areas = base.area_map(dev_items)
    datasets = {"external": (ext_items, ext_areas), "development2016": (dev_items, dev_areas)}
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
                                "min_tiny_gain": min(deltas[name]["tiny"] for name in deltas)})
    accepted_rows = sorted((row for row in results if row["accepted"]),
                           key=lambda row: row["min_tiny_gain"], reverse=True)
    output.write_text(json.dumps({"candidate": str(candidate),
                                  "only_one_network": True,
                                  "area_estimator": "fraction of valid pixels predicted positive at threshold 0.5",
                                  "rule": "if predicted_area >= area_cut use large_threshold else small_threshold",
                                  "development_only": True,
                                  "official_test2017_scores_viewed": False,
                                  "grid": {"area_cut": AREA_CUTS,
                                           "small_threshold": SMALL_THRESHOLDS,
                                           "large_threshold": LARGE_THRESHOLDS},
                                  "accepted_count": len(accepted_rows),
                                  "best_accepted": accepted_rows[0] if accepted_rows else None,
                                  "results": results}, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print("accepted", len(accepted_rows), flush=True)
    print("best", accepted_rows[0] if accepted_rows else None, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=CANDIDATE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    main(args.candidate, args.output)
