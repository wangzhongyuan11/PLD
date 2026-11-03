"""Audit exact and visually close image overlap across experiment partitions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from software.role_a import refine_small as ref


def dhash(path: Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(gray.getdata())
    bits = 0
    for y in range(8):
        for x in range(8):
            bits = (bits << 1) | int(pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return bits


def records(directory: Path) -> list[dict[str, object]]:
    output = []
    for path in sorted(directory.glob("ISIC_*.jpg")):
        output.append({"stem": path.stem, "path": str(path),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                       "dhash": dhash(path)})
    return output


def compare(left, right, threshold: int = 4):
    exact = []
    close = []
    for a in left:
        for b in right:
            if a["sha256"] == b["sha256"]:
                exact.append((a["stem"], b["stem"]))
            distance = (a["dhash"] ^ b["dhash"]).bit_count()
            if distance <= threshold:
                close.append({"left": a["stem"], "right": b["stem"], "distance": distance})
    return {"exact_sha256_pairs": exact, "dhash_close_pairs": close}


def main(output: Path | None = None) -> None:
    data = {
        "train2016": records(ref.TRAIN / "images"),
        "train2017": records(ref.SUPPLEMENT / "images"),
        "test2016": records(ref.TEST / "images"),
        "external2017": records(ref.EXTERNAL / "images"),
        "test2017": records(ref.PROJECT / "data" / "raw" / "isic2017_test" / "images"),
    }
    result = {"counts": {name: len(rows) for name, rows in data.items()},
              "train2016_vs_test": compare(data["train2016"], data["test2016"]),
              "external_vs_train": compare(data["external2017"], data["train2016"] + data["train2017"]),
              "supplement_vs_test": compare(data["train2017"], data["test2016"]),
              "external_vs_test": compare(data["external2017"], data["test2016"]),
              "test2017_vs_train": compare(data["test2017"], data["train2016"] + data["train2017"]),
              "test2017_vs_external": compare(data["test2017"], data["external2017"]),
              "test2017_vs_test2016": compare(data["test2017"], data["test2016"]),
              "dhash_threshold": 4,
              "note": "dHash close pairs are screening candidates, not proof of shared patient or duplicate image"}
    if output is None:
        output = ref.PROJECT / "runs" / "role_a_small_balanced_grid" / "overlap_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"counts": result["counts"],
                      **{name: {key: len(value) for key, value in result[name].items()}
                         for name in ("train2016_vs_test", "external_vs_train", "supplement_vs_test", "external_vs_test",
                                      "test2017_vs_train", "test2017_vs_external", "test2017_vs_test2016")}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="审计输出路径；默认沿用旧报告路径")
    main(parser.parse_args().output)
