"""Freeze an image-only clean holdout before any 2018 segmentation evaluation."""
from __future__ import annotations

import json

from software.role_a.audit_test2018_similarity import OUTPUT as SIMILARITY
from software.role_a.audit_test2018_overlap import OUTPUT as AUDIT
from software.role_a.refine_from_baseline_v3 import atomic_write_json, sha256_file


OUTPUT = AUDIT.with_name("test2018_overlap_exclusions.json")


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Frozen exclusion list already exists: {OUTPUT}")
    data = json.loads(SIMILARITY.read_text(encoding="utf-8"))
    selected = sorted({row["left"] for rows in data["comparisons"].values() for row in rows
                       if row["distance"] <= 4 and row["rgb_mae"] <= 0.18 and
                       row["rgb_correlation"] >= 0.9})
    if len(selected) != 41:
        raise ValueError(f"Expected 41 reviewed image-only screen hits, saw {len(selected)}")
    atomic_write_json(OUTPUT, {
        "schema": "isic2018-test-clean-overlap-exclusions-v1",
        "frozen_before_segmentation_evaluation": True,
        "method": "dHash distance <=4 then color MAE <=0.18 and RGB correlation >=0.9; 41 contact-sheet pairs visually reviewed, uncertain matches excluded conservatively",
        "source_audit": str(AUDIT), "source_audit_sha256": sha256_file(AUDIT),
        "source_similarity": str(SIMILARITY),
        "source_similarity_sha256": sha256_file(SIMILARITY),
        "contact_sheets": [f"test2018_overlap_sheet_{index}.jpg" for index in range(1, 8)],
        "excluded_test_ids": selected,
        "full_count": 1000, "clean_count_expected": 1000 - len(selected),
        "limitation": "This screen does not prove all near-duplicates have been found. It excludes high-similarity candidates before any model score is viewed."
    })
    print(f"frozen {len(selected)} exclusions; clean holdout {1000-len(selected)}", flush=True)
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
