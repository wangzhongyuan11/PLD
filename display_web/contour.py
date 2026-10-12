"""Binary lesion contours on CPU; no PYNQ/OpenCV dependency or model inference.

Contours are raster foreground boundaries (including holes), not ordered polygons.
Image exterior is background. Thickness is measured inward from the boundary.
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
from pld_hdmi import binary_mask, require_rgb, read_rgb, fit_rgb
from pld_result import load_result


def extract_contour(mask, connectivity=8, thickness=1):
    """Return a new bool boundary mask; preserve all components and hole edges."""
    m = binary_mask(mask).copy()
    if type(connectivity) is not int or connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if type(thickness) is not int or not 1 <= thickness <= 32:
        raise ValueError("thickness must be an integer in [1,32]")
    inner = m.copy()
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    h, w = m.shape
    for _ in range(thickness):
        p = np.pad(inner, 1, constant_values=False)
        eroded = inner.copy()
        for dy, dx in offsets:
            eroded &= p[1+dy:1+dy+h, 1+dx:1+dx+w]
        inner = eroded
    return m & ~inner


def draw_contour(image, mask, connectivity=8, thickness=2, color=(255, 220, 0)):
    """Draw only the contour on an aligned RGB copy, without filling the lesion."""
    rgb = require_rgb(image)
    m = binary_mask(mask, rgb.shape[:2])
    c = np.asarray(color)
    if c.shape != (3,) or not np.issubdtype(c.dtype, np.integer) or (c < 0).any() or (c > 255).any():
        raise ValueError("color must contain three integer RGB values in [0,255]")
    out = rgb.copy()
    out[extract_contour(m, connectivity, thickness)] = c.astype(np.uint8)
    return out


def process(mask, image=None, connectivity=8, thickness=2):
    """Web/HDMI-ready API returning arrays and JSON-compatible statistics."""
    started = perf_counter()
    m = binary_mask(mask).copy()
    edge = extract_contour(m, connectivity, thickness)
    rgb = np.repeat((m.astype(np.uint8)*255)[:, :, None], 3, axis=2) if image is None else require_rgb(image)
    binary_mask(m, rgb.shape[:2])
    outlined = rgb.copy()
    outlined[edge] = (255, 220, 0)
    cutout = rgb.copy()
    cutout[~m] = 0
    stats = {"foreground_pixels": int(m.sum()), "image_pixels": int(m.size),
             "area_percent": float(100*m.sum()/m.size), "boundary_pixels": int(edge.sum()),
             "connectivity": connectivity, "inward_thickness_pixels": thickness,
             "original_rgb_supplied": image is not None,
             "cpu_postprocess_seconds": perf_counter()-started,
             "scope": "CPU raster boundary; includes holes; no inference, IO or display timing"}
    return {"mask": m, "contour": edge, "outlined": outlined, "cutout": cutout, "statistics": stats}


def export(result, destination):
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    for key in ("mask", "contour", "outlined", "cutout"):
        a = result[key]
        Image.fromarray(a.astype(np.uint8)*255 if a.dtype == bool else a).save(out/(key+".png"))
    panel = Image.new("RGB", (1200, 450), (18, 25, 35))
    d = ImageDraw.Draw(panel)
    for i, (key, label) in enumerate((("mask", "SEGMENTATION MASK"), ("contour", "BOUNDARY"), ("outlined", "CONTOUR VIEW"))):
        a = result[key]
        if a.ndim == 2:
            a = np.repeat((a.astype(np.uint8)*255)[:, :, None], 3, axis=2)
        panel.paste(Image.fromarray(fit_rgb(a, 380, 360, nearest=True)), (10+i*400, 45))
        d.text((15+i*400, 20), label, fill="white")
    s = result["statistics"]
    d.text((15, 420), "Boundary pixels: {} | CPU {:.2f} ms | {}".format(s["boundary_pixels"], s["cpu_postprocess_seconds"]*1000, "Original RGB" if s["original_rgb_supplied"] else "Real mask only; original RGB not supplied"), fill="white")
    panel.save(out/"preview.png")
    (out/"report.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--result", type=Path, help="B frame JSON or BIN; validates hashes and removes letterbox padding")
    source.add_argument("--mask", type=Path, help="Binary grayscale PNG: only 0/1 or 0/255")
    parser.add_argument("--image", type=Path, help="Optional aligned original RGB; hash checked with --result")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--thickness", type=int, default=2)
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    args = parser.parse_args()
    if args.result:
        b = load_result(args.result, args.image)
        mask, rgb = b.original_mask(), b.rgb
    else:
        with Image.open(args.mask) as im:
            mask = np.asarray(im).copy()
        rgb = read_rgb(args.image) if args.image else None
    result = process(mask, rgb, args.connectivity, args.thickness)
    if args.result:
        result["statistics"]["B_valid_region_statistics"] = b.stats
        result["statistics"]["source_head_sha256"] = b.metadata["output_sha256"]
    export(result, args.output)
    print(json.dumps(result["statistics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
