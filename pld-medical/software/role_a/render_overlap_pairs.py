"""Render small contact sheets for human review of test-set overlap candidates."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from software.role_a.audit_close_pairs import GROUPS
from software.role_a.refine_small import PROJECT


AUDIT = PROJECT / "runs" / "role_a_full2017" / "test2017_overlap_similarity.json"
OUT = PROJECT / "runs" / "role_a_full2017"
PAIRS_PER_SHEET = 6
SIZE = 240


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    pairs = [(name, row) for name in ("test2017_vs_train", "test2017_vs_external",
                                      "test2017_vs_test2016") for row in audit[name]]
    for offset in range(0, len(pairs), PAIRS_PER_SHEET):
        chunk = pairs[offset:offset + PAIRS_PER_SHEET]
        canvas = Image.new("RGB", (SIZE * 2 + 40, len(chunk) * (SIZE + 45)), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (group, row) in enumerate(chunk):
            left_dir, right_dirs = GROUPS[group]
            left = left_dir / f"{row['left']}.jpg"
            right = next(directory / f"{row['right']}.jpg" for directory in right_dirs
                         if (directory / f"{row['right']}.jpg").exists())
            y = index * (SIZE + 45)
            for x, path in ((10, left), (SIZE + 30, right)):
                with Image.open(path) as raw:
                    image = ImageOps.contain(ImageOps.exif_transpose(raw).convert("RGB"),
                                             (SIZE, SIZE))
                canvas.paste(image, (x + (SIZE - image.width) // 2,
                                     y + 22 + (SIZE - image.height) // 2))
            draw.text((10, y + 4), f"{row['left']} / {row['right']}  {group}  "
                                  f"mae={row['rgb_mae']:.3f} corr={row['rgb_correlation']:.3f}",
                      fill="black")
        path = OUT / f"test2017_overlap_sheet_{offset // PAIRS_PER_SHEET + 1}.jpg"
        canvas.save(path, quality=90)
        print(path)


if __name__ == "__main__":
    main()
