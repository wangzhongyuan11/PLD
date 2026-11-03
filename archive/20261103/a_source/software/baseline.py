"""Readable, standard-library-only image processing reference; not a speed benchmark."""
import argparse
import json
from pathlib import Path
from time import perf_counter


def read_pnm(path, magic):
    tokens = []
    for line in Path(path).read_text(encoding="ascii").splitlines():
        tokens.extend(line.split("#", 1)[0].split())
    if len(tokens) < 4 or tokens[0] != magic:
        raise ValueError(f"Expected ASCII {magic} image: {path}")
    width, height, maximum = map(int, tokens[1:4])
    if width <= 0 or height <= 0 or maximum != 255:
        raise ValueError("Expected positive dimensions and maximum value 255")
    values = list(map(int, tokens[4:]))
    channels = 3 if magic == "P3" else 1
    if len(values) != width * height * channels or any(v < 0 or v > 255 for v in values):
        raise ValueError("Invalid pixel count or range")
    if channels == 3:
        values = [tuple(values[i:i + 3]) for i in range(0, len(values), 3)]
    return width, height, values


def write_pnm(path, width, height, values, rgb=False):
    channels = 3 if rgb else 1
    flat = [c for pixel in values for c in pixel] if rgb else values
    if len(flat) != width * height * channels:
        raise ValueError("Pixel count does not match dimensions")
    header = f"{'P3' if rgb else 'P2'}\n{width} {height}\n255\n"
    Path(path).write_text(header + " ".join(map(str, flat)) + "\n", encoding="ascii")


def grayscale(rgb):
    return [(77 * r + 150 * g + 29 * b + 128) >> 8 for r, g, b in rgb]


def gaussian(pixels, width, height):
    result = []
    weights = (1, 2, 1)
    for y in range(height):
        for x in range(width):
            total = 0
            for dy in range(-1, 2):
                yy = min(height - 1, max(0, y + dy))
                for dx in range(-1, 2):
                    xx = min(width - 1, max(0, x + dx))
                    total += pixels[yy * width + xx] * weights[dy + 1] * weights[dx + 1]
            result.append((total + 8) >> 4)
    return result


def morphology(mask, width, height, dilate=False):
    result = []
    for y in range(height):
        for x in range(width):
            window = [
                mask[yy * width + xx] if 0 <= xx < width and 0 <= yy < height else 0
                for yy in range(y - 1, y + 2) for xx in range(x - 1, x + 2)
            ]
            result.append(int(any(window) if dilate else all(window)))
    return result


def overlap(prediction, truth, valid):
    if not (len(prediction) == len(truth) == len(valid)):
        raise ValueError("Mask sizes differ")
    pairs = [(bool(p), bool(g)) for p, g, v in zip(prediction, truth, valid) if v]
    if not pairs:
        raise ValueError("Valid region is empty")
    p_count = sum(p for p, _ in pairs)
    g_count = sum(g for _, g in pairs)
    intersection = sum(p and g for p, g in pairs)
    union = p_count + g_count - intersection
    return {"dice": 2 * intersection / (p_count + g_count) if p_count + g_count else 1.0,
            "iou": intersection / union if union else 1.0}


def synthetic(width=64, height=64):
    truth = [int(((x - width / 2) / (width * .23)) ** 2 +
                 ((y - height / 2) / (height * .3)) ** 2 <= 1)
             for y in range(height) for x in range(width)]
    rgb = [(85, 55, 45) if inside else (215, 180, 160) for inside in truth]
    return width, height, rgb, truth


def preview(path, width, height, rgb, filtered, mask, contour, demo):
    # Vector rectangles are used so no image library is needed for this learning demo.
    panel = 256
    scale_x, scale_y = panel / width, panel / height
    panels = [rgb, [(g, g, g) for g in filtered],
              [(255, 50, 50) if c else tuple((v + t) // 2 for v, t in zip(p, (0, 220, 80)))
               if m else p for p, m, c in zip(rgb, mask, contour)],
              [(255 * m,) * 3 for m in mask]]
    title = "SYNTHETIC DEMO - NOT MEDICAL VALIDATION" if demo else "SOFTWARE REFERENCE - NOT FPGA OUTPUT"
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1104" height="360" viewBox="0 0 1104 360">',
             '<rect width="1104" height="360" fill="#142031"/>',
             f'<text x="20" y="28" fill="white" font-size="18">{title}</text>']
    for k, (label, colors) in enumerate(zip(("Input", "Filtered", "Overlay", "Mask"), panels)):
        origin = 20 + 272 * k
        parts.append(f'<text x="{origin}" y="58" fill="white" font-size="16">{label}</text>')
        parts.append(f'<g transform="translate({origin},72) scale({scale_x},{scale_y})" shape-rendering="crispEdges">')
        for i, (r, g, b) in enumerate(colors):
            parts.append(f'<rect x="{i % width}" y="{i // width}" width="1" height="1" fill="rgb({r},{g},{b})"/>')
        parts.append('</g>')
    parts.append('</svg>')
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true")
    source.add_argument("--input", type=Path, help="ASCII P3 RGB image")
    parser.add_argument("--mask", type=Path, help="Optional ASCII P2 ground truth")
    parser.add_argument("--valid-mask", type=Path, help="ASCII P2 region excluding padding")
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--no-opening", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    args = parser.parse_args()
    if not 0 <= args.threshold <= 255:
        parser.error("threshold must be between 0 and 255")
    if args.demo and (args.mask or args.valid_mask):
        parser.error("demo already includes synthetic truth and valid region")
    if args.demo:
        width, height, rgb, truth = synthetic()
    else:
        width, height, rgb = read_pnm(args.input, "P3")
        truth = None
    valid = [1] * (width * height)
    for name in ("mask", "valid_mask"):
        source_path = getattr(args, name)
        if source_path:
            w, h, values = read_pnm(source_path, "P2")
            if (w, h) != (width, height):
                parser.error(f"{name} dimensions differ from input")
            if any(v not in (0, 255) for v in values):
                parser.error(f"{name} must contain only 0 and 255")
            if name == "mask":
                truth = [int(v > 0) for v in values]
            else:
                valid = [int(v > 0) for v in values]
    if not any(valid):
        parser.error("valid region is empty")
    started = perf_counter()
    gray = grayscale(rgb)
    filtered = gaussian(gray, width, height)
    mask = [int(p < args.threshold and v) for p, v in zip(filtered, valid)]
    if not args.no_opening:
        mask = morphology(morphology(mask, width, height), width, height, dilate=True)
    mask = [m * v for m, v in zip(mask, valid)]
    eroded = morphology(mask, width, height)
    contour = [m - e for m, e in zip(mask, eroded)]
    elapsed = perf_counter() - started
    metrics = {"synthetic": args.demo, "backend": "python_reference", "width": width,
               "height": height, "threshold": args.threshold, "opening": not args.no_opening,
               "foreground_pixels": sum(mask), "valid_pixels": sum(valid),
               "area_fraction": sum(mask) / sum(valid), "software_processing_ms": elapsed * 1000,
               "timing_scope": "gray through contour, excluding IO, metrics and rendering",
               "evaluation_space": "processed valid region", "empty_pair_score": 1.0,
               "input": str(args.input) if args.input else "synthetic ellipse",
               "ground_truth": str(args.mask) if args.mask else "synthetic" if args.demo else None,
               "scores": overlap(mask, truth, valid) if truth is not None else None,
               "warning": "Not medical validation or FPGA measurement" if args.demo else "Not FPGA measurement"}
    args.output.mkdir(parents=True, exist_ok=True)
    write_pnm(args.output / "input.ppm", width, height, rgb, rgb=True)
    for name, pixels in (("gray", gray), ("filtered", filtered),
                         ("mask", [255 * m for m in mask]), ("contour", [255 * m for m in contour])):
        write_pnm(args.output / f"{name}.pgm", width, height, pixels)
    if truth is not None:
        write_pnm(args.output / "truth.pgm", width, height, [255 * v for v in truth])
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    preview(args.output / "preview.svg", width, height, rgb, filtered, mask, contour, args.demo)
    print(json.dumps(metrics, indent=2))
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
