"""Color-aware second screen for ISIC 2018 test image-only overlap candidates."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from software.role_a.audit_close_pairs import pixels
from software.role_a.audit_test2018_overlap import OTHERS, OUTPUT as AUDIT, TEST


OUTPUT = AUDIT.with_name("test2018_overlap_similarity.json")


@lru_cache(maxsize=1200)
def cached_pixels(path: Path) -> np.ndarray:
    return pixels(path)


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    result = {"schema": "isic2018-test-overlap-similarity-v1", "comparisons": {}}
    for name, group in audit["comparisons"].items():
        rows = []
        for pair in group["dhash_close_pairs"]:
            left = cached_pixels(TEST / f"{pair['left']}.jpg")
            right = cached_pixels(OTHERS[name] / f"{pair['right']}.jpg")
            mae = float(np.abs(left - right).mean())
            a, b = left - left.mean(), right - right.mean()
            correlation = float(np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b)))
            rows.append({**pair, "rgb_mae": mae, "rgb_correlation": correlation})
        result["comparisons"][name] = sorted(rows, key=lambda row: (row["rgb_mae"],
                                                              -row["rgb_correlation"]))
        strong = [row for row in rows if row["rgb_mae"] <= 0.12 and
                  row["rgb_correlation"] >= 0.9]
        print(name, "screen_pairs", len(rows), "strong_pairs", len(strong), flush=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
