"""C-side adapter for B's frozen c_v3_20260929 head/result contract.

No PYNQ import, inference, bitstream loading, or DMA ownership here.
Postprocessing runs on the PS/CPU; it is not PL postprocessing.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from pld_hdmi import require_rgb, fit_rgb, read_rgb, _font, _resample

SPEC_SHA256 = "fe137ca5b128fc1983fabad9bee39b3208cda4256e0d04c4e27cef5c07666aef"
MANIFEST_SHA256 = "e47eb18b94369451b36790971f4e0db62b2d128f424937e00f2768e866a30936"
HEAD_SCALE = 0.1995087721216397


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError("{} must be an integer >= {}".format(name, minimum))
    return value


def validate_metadata(metadata):
    """Reject failed frames, other models, and inconsistent letterbox geometry."""
    m = metadata
    expected = {"status": "success", "shape": [1, 1, 256, 256],
                "dtype": "int8", "layout": "NCHW", "comparison": ">=",
                "model_name": "soup_v5_a050", "model_spec_sha256": SPEC_SHA256,
                "model_manifest_sha256": MANIFEST_SHA256, "threshold_q": -1,
                "threshold_probability": 0.45}
    for key, value in expected.items():
        if m.get(key) != value:
            raise ValueError("Unsupported B result field {}: {!r}".format(key, m.get(key)))
    scale = m.get("head_output_scale")
    if not isinstance(scale, (int, float)) or not math.isclose(scale, HEAD_SCALE, rel_tol=1e-12):
        raise ValueError("head_output_scale does not match frozen final model")
    _integer(m.get("frame_id"), "frame_id", 1)
    rect = m.get("valid_rect")
    if not isinstance(rect, list) or len(rect) != 4:
        raise ValueError("valid_rect must be [left, top, width, height]")
    left, top, width, height = [_integer(v, "valid_rect", 0 if i < 2 else 1)
                                for i, v in enumerate(rect)]
    if left + width > 256 or top + height > 256:
        raise ValueError("valid_rect lies outside the 256 x 256 head")
    g = m.get("input_metadata")
    if not isinstance(g, dict) or g.get("model_spec_sha256") != SPEC_SHA256:
        raise ValueError("Missing or mismatched input_metadata")
    ow = _integer(g.get("original_width"), "original_width", 1)
    oh = _integer(g.get("original_height"), "original_height", 1)
    ratio = min(256 / ow, 256 / oh)
    rw, rh = max(1, min(256, round(ow * ratio))), max(1, min(256, round(oh * ratio)))
    if rect != [(256-rw)//2, (256-rh)//2, rw, rh]:
        raise ValueError("valid_rect does not match B's letterbox geometry")
    if [g.get(k) for k in ("left", "top", "resized_width", "resized_height")] != rect:
        raise ValueError("input geometry disagrees with valid_rect")
    if g.get("valid_rect") != rect or g.get("valid_pixels") != width * height:
        raise ValueError("input valid rectangle/pixel count mismatch")
    for value in (m.get("output_sha256"), g.get("image_sha256"), g.get("input_sha256")):
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Missing or invalid SHA256 identity")
    for name in ("online_b_seconds", "inference_seconds"):
        value = m.get(name)
        if value is not None and (not isinstance(value, (int, float)) or
                                  not math.isfinite(value) or value <= 0):
            raise ValueError("Invalid timing field " + name)
    return left, top, width, height


class SegmentationResult:
    def __init__(self, head, metadata, rgb=None):
        # Deep-copy metadata and head: caller may reuse its next result buffers.
        self.metadata = json.loads(json.dumps(metadata))
        left, top, width, height = validate_metadata(self.metadata)
        q = np.asarray(head)
        if q.dtype != np.int8 or q.shape != (1, 1, 256, 256):
            raise ValueError("head must be signed np.int8 [1,1,256,256]")
        if (q == -128).any():
            raise ValueError("-128 is outside the frozen [-127,127] quantization range")
        if hashlib.sha256(q.tobytes()).hexdigest() != self.metadata["output_sha256"]:
            raise ValueError("head SHA256 mismatch; wrong or incomplete frame")
        self.head = q.copy()
        self.mask256 = np.zeros((256, 256), dtype=bool)
        self.mask256[top:top+height, left:left+width] = q[0, 0, top:top+height, left:left+width] >= -1
        self.valid_mask = self.mask256[top:top+height, left:left+width].copy()
        count = int(self.valid_mask.sum())
        self.stats = {"mask_pixels": count, "valid_pixels": width * height,
                      "area_percent": 100.0 * count / (width * height),
                      "area_scope": "256 letterbox valid region; padding excluded"}
        self.rgb = None
        if rgb is not None:
            self.rgb = require_rgb(rgb).copy()
            g = self.metadata["input_metadata"]
            if self.rgb.shape[:2] != (g["original_height"], g["original_width"]):
                raise ValueError("RGB dimensions must match EXIF-oriented original image")

    def original_mask(self):
        g = self.metadata["input_metadata"]
        return np.asarray(Image.fromarray(self.valid_mask.astype(np.uint8) * 255).resize(
            (g["original_width"], g["original_height"]), _resample("NEAREST"))) != 0

    def cutout(self):
        if self.rgb is None:
            raise ValueError("Supply the matching original --image for a color cutout")
        out = self.rgb.copy()
        out[~self.original_mask()] = 0
        return out

    def pixels(self, view):
        if view == "mask":
            return np.repeat((self.valid_mask.astype(np.uint8) * 255)[:, :, None], 3, axis=2)
        if self.rgb is None:
            raise ValueError("This view requires the matching original --image")
        if view == "original":
            return self.rgb.copy()
        if view == "cutout":
            return self.cutout()
        if view == "crop":
            mask = self.original_mask()
            ys, xs = np.nonzero(mask)
            if not len(xs):
                return np.zeros((1, 1, 3), dtype=np.uint8)
            return self.cutout()[ys.min():ys.max()+1, xs.min():xs.max()+1].copy()
        raise ValueError("Unknown view: " + view)

    def screen(self, view="cutout", width=640, height=480):
        if view != "dashboard":
            return fit_rgb(self.pixels(view), width, height, nearest=(view == "mask"))
        if width < 640 or height < 480:
            raise ValueError("Dashboard needs at least 640 x 480")
        canvas = Image.new("RGB", (width, height), (12, 22, 36))
        d = ImageDraw.Draw(canvas)
        d.text((22, 20), "PLD / SEGMENTATION RESULT", font=_font(23), fill=(238, 245, 255))
        d.text((22, 57), "Frame {}  /  soup_v5_a050  /  q >= -1".format(self.metadata["frame_id"]),
               font=_font(16), fill=(75, 214, 176))
        pw = (width - 56) // 2
        ph = min(height - 260, pw)
        for i, (name, label) in enumerate((("original", "ORIGINAL"), ("cutout", "CUTOUT"))):
            x = 22 + i * (pw + 12)
            d.text((x, 104), label, font=_font(14), fill=(238, 245, 255))
            canvas.paste(Image.fromarray(fit_rgb(self.pixels(name), pw, ph)), (x, 132))
        y = 154 + ph
        d.text((22, y), "Valid-region area: {:.2f}%  ({} / {})".format(
            self.stats["area_percent"], self.stats["mask_pixels"], self.stats["valid_pixels"]),
            font=_font(18), fill=(75, 214, 176))
        seconds = self.metadata.get("online_b_seconds")
        label = "B time: not supplied" if seconds is None else "B time: {:.1f} ms (excludes display)".format(seconds * 1000)
        d.text((22, y+35), label, font=_font(16), fill=(238, 245, 255))
        d.text((22, height-48), "PS postprocess / black background outside segmentation", font=_font(13), fill=(147, 165, 190))
        d.text((22, height-27), "Pixel ratio only; no physical area calibration", font=_font(13), fill=(147, 165, 190))
        return np.asarray(canvas).copy()


def load_result(result_path, image_path=None):
    """Read frame_XXXXXX.json + .bin; verify source file identity if supplied."""
    path = Path(result_path)
    if path.suffix.lower() not in (".json", ".bin"):
        raise ValueError("--result must name a frame .json or .bin")
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    validate_metadata(metadata)
    raw = path.with_suffix(".bin").read_bytes()
    if len(raw) != 65536:
        raise ValueError("B head file must contain exactly 65536 bytes")
    rgb = None
    if image_path is not None:
        source = Path(image_path)
        if hashlib.sha256(source.read_bytes()).hexdigest() != metadata["input_metadata"]["image_sha256"]:
            raise ValueError("Original image SHA256 mismatch; select the JPEG used for this result")
        rgb = read_rgb(source)
    return SegmentationResult(np.frombuffer(raw, dtype=np.int8).reshape(1, 1, 256, 256), metadata, rgb)


def save_result_images(result, destination, view="cutout"):
    """Create a new output directory so a failed run cannot reuse old images."""
    screen = result.screen(view)
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    Image.fromarray(result.mask256.astype(np.uint8) * 255).save(out / "mask_256.png")
    Image.fromarray(result.original_mask().astype(np.uint8) * 255).save(out / "mask_original.png")
    Image.fromarray(screen).save(out / "screen.png")
    if result.rgb is not None:
        for name in ("cutout", "crop"):
            Image.fromarray(result.pixels(name)).save(out / (name + ".png"))
        rgba = np.dstack((result.rgb, result.original_mask().astype(np.uint8) * 255))
        Image.fromarray(rgba).save(out / "cutout_transparent.png")
        Image.fromarray(result.screen("dashboard")).save(out / "dashboard.png")
    report = {"frame_id": result.metadata["frame_id"], "source_name": result.metadata.get("source_name"),
              "model_spec_sha256": SPEC_SHA256, "output_sha256": result.metadata["output_sha256"],
              "statistics": result.stats, "postprocessing": "PS/CPU signed threshold and nearest mask mapping",
              "online_b_seconds": result.metadata.get("online_b_seconds"),
              "original_image_supplied": result.rgb is not None,
              "hardware_verified": False, "view": view}
    (out / "display_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return screen
