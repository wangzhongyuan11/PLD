"""Lightweight structural and thumbnail QA for the B-group textbook PDF."""

from pathlib import Path

from PIL import Image, ImageDraw
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
pdf = ROOT / "output/pdf/B组FPGA加速器从零到独立实现教材.pdf"
scratch = ROOT / "tmp/pdfs/b_textbook"
reader = PdfReader(pdf)
texts = [page.extract_text() or "" for page in reader.pages]
assert len(texts) >= 25, len(texts)
assert all(len(s) > 500 for s in texts[1:]), [
    (i + 1, len(s)) for i, s in enumerate(texts) if i > 0 and len(s) <= 500
]
assert len(reader.outline) >= 18
print("pages", len(texts), "outline_entries", len(reader.outline))
print("text_chars", sum(map(len, texts)))
print("contains_B_title", "B" in texts[0])

paths = sorted(scratch.glob("review-*.png"))
assert len(paths) == len(texts), (len(paths), len(texts))
thumb_w, thumb_h, cols = 240, 340, 4
rows = (len(paths) + cols - 1) // cols
sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + 22)), "#e6ecf0")
draw = ImageDraw.Draw(sheet)
for index, path in enumerate(paths):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((thumb_w - 12, thumb_h - 10))
        x = (index % cols) * thumb_w + (thumb_w - image.width) // 2
        y = (index // cols) * (thumb_h + 22) + 3
        sheet.paste(image, (x, y))
        draw.text((index % cols * thumb_w + 8, y + thumb_h - 3), str(index + 1), fill="black")
out = scratch / "contact_sheet.png"
sheet.save(out)
print(out, out.stat().st_size)
