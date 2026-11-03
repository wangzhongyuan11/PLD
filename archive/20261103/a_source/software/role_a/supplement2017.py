"""Select a bounded ISIC 2017 *training* subset rich in real tiny lesions."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np

from software.role_a import external2017 as ext


ROOT = ext.PROJECT / "data" / "raw" / "isic2017_training"
MASK_ZIP = ROOT / "downloads" / "training_masks.zip"
MICRO_MAX = 0.02054309  # q10 of disjoint 2017 validation masks, fixed pre-training.
BRIDGE_MAX = 0.05563571  # q10 of official 2016 training masks.
MEDIUM_MAX = 0.08513896  # q50 of disjoint 2017 validation masks.
MICRO_N = 80
BRIDGE_N = 40
MEDIUM_N = 30
LARGE_N = 40
MAX_BYTES = 130 * 1024 * 1024


def spaced(rows: list[tuple[str, dict[str, object]]], number: int) -> list[tuple[str, dict[str, object]]]:
    if len(rows) < number:
        raise ValueError(f"only {len(rows)} eligible masks for requested {number}")
    indices = np.linspace(0, len(rows) - 1, number, dtype=int)
    if len(set(int(i) for i in indices)) != number:
        raise ValueError("selection indices are not unique")
    return [rows[int(index)] for index in indices]


def selection() -> dict[str, dict[str, object]]:
    masks = ext.mask_entries(MASK_ZIP)
    forbidden = ext.existing_2016_ids() | set(ext.mask_entries())
    eligible = sorted(((stem, row) for stem, row in masks.items() if stem not in forbidden),
                      key=lambda item: (float(item[1]["area_ratio"]), item[0]))
    micro = [(stem, row) for stem, row in eligible if float(row["area_ratio"]) <= MICRO_MAX]
    bridge = [(stem, row) for stem, row in eligible if MICRO_MAX < float(row["area_ratio"]) <= BRIDGE_MAX]
    medium = [(stem, row) for stem, row in eligible if BRIDGE_MAX < float(row["area_ratio"]) <= MEDIUM_MAX]
    large = [(stem, row) for stem, row in eligible if float(row["area_ratio"]) > MEDIUM_MAX]
    chosen = (spaced(micro, MICRO_N) + spaced(bridge, BRIDGE_N) +
              spaced(medium, MEDIUM_N) + spaced(large, LARGE_N))
    return {stem: row for stem, row in sorted(chosen)}


def survey() -> dict[str, object]:
    chosen = selection()
    remote: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(ext.official_object_metadata, stem): stem for stem in chosen}
        for index, future in enumerate(as_completed(futures), 1):
            stem, row = future.result()
            remote[stem] = row
            if index % 10 == 0 or index == len(chosen):
                print(f"metadata {index}/{len(chosen)}", flush=True)
    items = {stem: {**chosen[stem], **remote[stem]} for stem in sorted(chosen)}
    manifest: dict[str, object] = {
        "source_masks_url": "https://isic-archive.s3.amazonaws.com/challenges/2017/ISIC-2017_Training_Part1_GroundTruth.zip",
        "source_masks_sha256": ext.sha256(MASK_ZIP.read_bytes()),
        "selection_rule": "80 micro, 40 bridge, 30 medium, 40 large, evenly spaced by raw mask area; exclude all 2016 and 2017 validation IDs",
        "micro_area_upper": MICRO_MAX,
        "bridge_area_upper": BRIDGE_MAX,
        "medium_area_upper": MEDIUM_MAX,
        "micro_count": MICRO_N,
        "bridge_count": BRIDGE_N,
        "medium_count": MEDIUM_N,
        "large_count": LARGE_N,
        "selected_count": len(items),
        "estimated_image_bytes": sum(int(row["expected_bytes"]) for row in items.values()),
        "items": items,
    }
    ext.assert_budget(manifest, MAX_BYTES)
    ext._atomic_bytes(ROOT / "survey.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("survey", "download"))
    args = parser.parse_args()
    if args.command == "survey":
        result = survey()
    else:
        manifest = json.loads((ROOT / "survey.json").read_text(encoding="utf-8"))
        result = ext.download(manifest, max_bytes=MAX_BYTES, root=ROOT, archive=MASK_ZIP)
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
