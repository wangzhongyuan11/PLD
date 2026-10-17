"""Rank dHash screening hits with a second, color-aware image comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from software.role_a import refine_small as ref


AUDIT = ref.PROJECT / "runs" / "role_a_small_balanced_grid" / "overlap_audit.json"
OUTPUT = AUDIT.with_name("overlap_similarity.json")
GROUPS = {
    "train2016_vs_test": (ref.TRAIN / "images", (ref.TEST / "images",)),
    "external_vs_train": (ref.EXTERNAL / "images", (ref.TRAIN / "images", ref.SUPPLEMENT / "images")),
    "supplement_vs_test": (ref.SUPPLEMENT / "images", (ref.TEST / "images",)),
    "external_vs_test": (ref.EXTERNAL / "images", (ref.TEST / "images",)),
    "test2017_vs_train": (ref.PROJECT / "data" / "raw" / "isic2017_test" / "images",
                          (ref.TRAIN / "images", ref.SUPPLEMENT / "images")),
    "test2017_vs_external": (ref.PROJECT / "data" / "raw" / "isic2017_test" / "images",
                             (ref.EXTERNAL / "images",)),
    "test2017_vs_test2016": (ref.PROJECT / "data" / "raw" / "isic2017_test" / "images",
                             (ref.TEST / "images",)),
}


def pixels(path):
    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw).convert("RGB").resize((128, 128), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def main(audit_path: Path = AUDIT, output_path: Path = OUTPUT) -> None:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    result = {}
    for name, (left_dir, right_dirs) in GROUPS.items():
        rows = []
        for pair in audit[name]["dhash_close_pairs"]:
            left = left_dir / (pair["left"] + ".jpg")
            right = next(path for directory in right_dirs
                         if (path := directory / (pair["right"] + ".jpg")).exists())
            a, b = pixels(left), pixels(right)
            distance = float(np.abs(a - b).mean())
            centered_a, centered_b = a - a.mean(), b - b.mean()
            correlation = float(np.sum(centered_a * centered_b) /
                                np.sqrt(np.sum(centered_a ** 2) * np.sum(centered_b ** 2)))
            rows.append({**pair, "rgb_mae": distance, "rgb_correlation": correlation})
        result[name] = sorted(rows, key=lambda row: row["rgb_mae"])
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, rows in result.items():
        print(name, [(row["left"], row["right"], round(row["rgb_mae"], 3),
                      round(row["rgb_correlation"], 3)) for row in rows])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=AUDIT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    main(args.audit, args.output)
