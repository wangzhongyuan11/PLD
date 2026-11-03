"""Optional Pillow converter. It does not download any images."""
import argparse
import json
from pathlib import Path
from baseline import write_pnm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.size <= 0:
        parser.error("size must be positive")
    try:
        from PIL import Image
    except ImportError:
        parser.error("Install optional dependency first: py -m pip install -r software/requirements-optional.txt")
    with Image.open(args.image) as source:
        source = source.convert("RGB")
        original = source.size
        factor = args.size / max(original)
        resized = tuple(max(1, round(v * factor)) for v in original)
        offset = tuple((args.size - v) // 2 for v in resized)
        canvas = Image.new("RGB", (args.size, args.size), (255, 255, 255))
        canvas.paste(source.resize(resized, Image.Resampling.BILINEAR), offset)
    valid = Image.new("L", canvas.size, 0)
    valid.paste(255, (offset[0], offset[1], offset[0] + resized[0], offset[1] + resized[1]))
    mask_canvas = None
    if args.mask:
        with Image.open(args.mask) as mask:
            if mask.size != original:
                parser.error("Original image and mask dimensions differ")
            mask = mask.convert("L")
            if any(v not in (0, 255) for v in mask.getdata()):
                parser.error("Ground truth must contain only 0 and 255")
            mask_canvas = Image.new("L", canvas.size, 0)
            mask_canvas.paste(mask.resize(resized, Image.Resampling.NEAREST), offset)
    args.output.mkdir(parents=True, exist_ok=True)
    write_pnm(args.output / "input.ppm", args.size, args.size, list(canvas.getdata()), rgb=True)
    write_pnm(args.output / "valid.pgm", args.size, args.size, list(valid.getdata()))
    if mask_canvas is not None:
        write_pnm(args.output / "truth.pgm", args.size, args.size, list(mask_canvas.getdata()))
    metadata = {"source": str(args.image.resolve()), "original_size": original,
                "resized_size": resized, "offset": offset, "canvas_size": args.size,
                "mask_source": str(args.mask.resolve()) if args.mask else None,
                "exif_orientation": "not applied; preserve raw image and mask coordinates"}
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
