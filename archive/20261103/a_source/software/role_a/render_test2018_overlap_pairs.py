"""Contact sheets for human review of high-confidence 2018 test overlap candidates."""
from __future__ import annotations

import json

from PIL import Image, ImageDraw, ImageOps

from software.role_a.audit_test2018_overlap import OTHERS, OUTPUT as AUDIT, TEST


OUT = AUDIT.parent
SIMILARITY = AUDIT.with_name("test2018_overlap_similarity.json")
SIZE = 220
PER_SHEET = 6


def main() -> None:
    data = json.loads(SIMILARITY.read_text(encoding="utf-8"))
    pairs = [(group, row) for group, rows in data["comparisons"].items() for row in rows
             if row["rgb_mae"] <= 0.18 and row["rgb_correlation"] >= 0.9]
    pairs.sort(key=lambda entry: entry[1]["rgb_mae"])
    for offset in range(0, len(pairs), PER_SHEET):
        chunk = pairs[offset:offset + PER_SHEET]
        canvas = Image.new("RGB", (SIZE * 2 + 30, len(chunk) * (SIZE + 38)), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (group, row) in enumerate(chunk):
            y = index * (SIZE + 38)
            paths = (TEST / f"{row['left']}.jpg", OTHERS[group] / f"{row['right']}.jpg")
            for column, path in enumerate(paths):
                with Image.open(path) as raw:
                    image = ImageOps.contain(ImageOps.exif_transpose(raw).convert("RGB"),
                                             (SIZE, SIZE))
                x = 5 + column * (SIZE + 20) + (SIZE - image.width) // 2
                canvas.paste(image, (x, y + 18 + (SIZE - image.height) // 2))
            draw.text((5, y + 3),
                      f"{row['left']} / {row['right']} {group} "
                      f"mae={row['rgb_mae']:.3f} corr={row['rgb_correlation']:.3f}",
                      fill="black")
        path = OUT / f"test2018_overlap_sheet_{offset // PER_SHEET + 1}.jpg"
        canvas.save(path, quality=90)
        print(path, flush=True)


if __name__ == "__main__":
    main()
