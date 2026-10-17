"""Image-only duplicate screen before any ISIC 2018 test segmentation scoring."""
from __future__ import annotations

import json
from pathlib import Path

from software.role_a.audit_image_overlap import compare, records
from software.role_a.refine_small import PROJECT


TEST = PROJECT / "data" / "raw" / "isic2018_test" / "images"
OUTPUT = PROJECT / "runs" / "role_a_from_baseline_v3" / "test2018_overlap_audit.json"
OTHERS = {
    "train2016": PROJECT / "data" / "raw" / "isic2016" / "train" / "images",
    "train2017": PROJECT / "data" / "raw" / "isic2017_training" / "images",
    "validation2017": PROJECT / "data" / "raw" / "isic2017_validation" / "images",
    "development2016": PROJECT / "data" / "raw" / "isic2016" / "test" / "images",
    "development2017": PROJECT / "data" / "raw" / "isic2017_test" / "images",
}


def main() -> None:
    provenance = json.loads((TEST.parent / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("complete") is not True or provenance.get("count") != 1000:
        raise ValueError("ISIC 2018 test data are not completely verified")
    test = records(TEST)
    if len(test) != 1000 or {row["stem"] for row in test} != set(provenance["items"]):
        raise ValueError("ISIC 2018 test images differ from verified provenance")
    result = {"schema": "isic2018-test-overlap-audit-v1",
              "purpose": "image-only screening before final holdout scoring",
              "dhash_threshold": 4,
              "test_count": len(test), "comparisons": {}}
    for name, directory in OTHERS.items():
        other = records(directory)
        result["comparisons"][name] = {"other_count": len(other), **compare(test, other)}
        row = result["comparisons"][name]
        print(name, len(other), "exact", len(row["exact_sha256_pairs"]),
              "dhash<=4", len(row["dhash_close_pairs"]), flush=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
