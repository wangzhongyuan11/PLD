"""Choose deterministic, area-balanced training-only images for INT8 calibration."""
from __future__ import annotations

import json

import numpy as np

from software.role_a import refine_small as base


OUT = base.PROJECT / "runs" / "role_a_full2017"


def spaced(stems: list[str], count: int) -> list[str]:
    ordered = sorted(stems)
    if len(ordered) < count:
        raise ValueError("not enough training images in calibration area band")
    return [ordered[int(index)] for index in np.linspace(0, len(ordered) - 1, count, dtype=int)]


def main() -> None:
    survey = json.loads((base.SUPPLEMENT / "survey.json").read_text(encoding="utf-8"))
    exclusions = json.loads((OUT / "overlap_exclusions.json").read_text(encoding="utf-8"))
    excluded = set(exclusions["supplemental_training"])
    groups = {"tiny": [], "medium": [], "large": []}
    for stem, row in survey["items"].items():
        if stem in excluded:
            continue
        area = float(row["area_ratio"])
        group = "tiny" if area <= 0.02 else "medium" if area <= 0.10 else "large"
        groups[group].append(stem)
    chosen = spaced(groups["tiny"], 24) + spaced(groups["medium"], 20) + spaced(groups["large"], 20)
    if len(chosen) != 64 or len(set(chosen)) != 64:
        raise ValueError("calibration IDs are not 64 distinct training-only samples")
    base._save_json(OUT / "calibration_split.json", {
        "dataset": "official ISIC 2017 training, reviewed unique IDs only",
        "selection": "24 tiny <=2%, 20 medium <=10%, 20 large >10%, spaced by sorted ID",
        "splits": {"train": sorted(chosen), "val": []},
    })
    print({name: len(rows) for name, rows in groups.items()})
    print("selected 64 distinct training images for INT8 calibration")


if __name__ == "__main__":
    main()
