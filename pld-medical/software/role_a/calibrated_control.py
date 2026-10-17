"""Check whether a threshold change alone explains the validated gain."""
from __future__ import annotations

import json

import torch

from software.role_a import refine_small as base
from software.role_a.refine_conservative import evaluate_grid


OUT = base.PROJECT / "runs" / "role_a_full2017"


def main() -> None:
    excluded = set(json.loads((OUT / "overlap_exclusions.json").read_text(encoding="utf-8"))
                   ["external_validation"])
    items, areas = base.external_items()
    items = [item for item in items if item.stem not in excluded]
    areas = {stem: area for stem, area in areas.items() if stem not in excluded}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = base.load_model(base.OLD, device)
    grid = evaluate_grid(model, items, areas, base.cuts(areas), device, (0.5, 0.6, 0.65))
    base._save_json(OUT / "calibrated_control.json", {
        "description": "unchanged original weights; validation threshold only",
        "excluded_external_ids": sorted(excluded),
        "thresholds": {str(threshold): {"metrics": metrics, "records": records}
                       for threshold, (metrics, records) in grid.items()},
    })
    for threshold, (metrics, _) in grid.items():
        print(threshold, {name: round(metrics[name]["dice"], 6)
                          for name in ("tiny", "medium", "large", "all")})


if __name__ == "__main__":
    main()
