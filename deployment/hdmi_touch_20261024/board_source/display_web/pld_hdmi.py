"""PYNQ-Z2 HDMI output and board-independent display composition.

Public image contract: uint8 RGB, shape (height, width, 3).
Public mask contract: aligned 2-D bool / binary 0,1 / binary 0,255.
Importing this module never loads a bitstream or imports pynq.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


def require_rgb(image):
    a = np.asarray(image)
    if a.ndim != 3 or a.shape[2] != 3 or min(a.shape[:2]) < 1:
        raise ValueError("Expected nonempty H x W x 3 RGB image")
    if a.dtype != np.uint8:
        raise ValueError("RGB image must be uint8, with values 0..255")
    return a


def binary_mask(mask, shape=None):
    """Reject ambiguous scores instead of silently thresholding them."""
    a = np.asarray(mask)
    if a.ndim != 2 or min(a.shape) < 1:
        raise ValueError("Mask must be a nonempty 2-D array")
    if shape is not None and a.shape != tuple(shape):
        raise ValueError("Image and mask must have identical height/width; align them first")
    if a.dtype == np.bool_:
        return a
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError("Use mask_from_probabilities() for float probability maps")
    if not (np.isin(a, [0, 1]).all() or np.isin(a, [0, 255]).all()):
        raise ValueError("Mask must contain only 0/1 or 0/255; decode model output explicitly")
    return a != 0


def mask_from_probabilities(probabilities, threshold=0.5):
    """Explicit probability >= threshold; logits/quantized scores are not accepted."""
    p = np.asarray(probabilities)
    if p.ndim != 2 or min(p.shape) < 1 or not np.issubdtype(p.dtype, np.floating):
        raise ValueError("Probabilities must be a nonempty 2-D floating-point array")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0, 1]")
    return p >= threshold


def mask_statistics(mask):
    """Pixel area only: no physical area calibration and no inference timing."""
    m = binary_mask(mask)
    count = int(np.count_nonzero(m))
    return {"mask_pixels": count, "total_pixels": int(m.size),
            "area_percent": 100.0 * count / m.size}


def mask_boundary(mask):
    """Inside boundary using 4-neighbour erosion; outside the image is background."""
    m = binary_mask(mask)
    p = np.pad(m, 1, constant_values=False)
    interior = (m & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:])
    return m & ~interior


def overlay_mask(image, mask, alpha=0.35):
    rgb = require_rgb(image)
    m = binary_mask(mask, rgb.shape[:2])
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0, 1]")
    out = rgb.copy()
    tint = np.array([40, 220, 145], dtype=np.float32)
    out[m] = np.rint((1 - alpha) * rgb[m].astype(np.float32) + alpha * tint).astype(np.uint8)
    out[mask_boundary(m)] = (255, 220, 70)
    return out


def _resample(name):
    return getattr(getattr(Image, "Resampling", Image), name)


def fit_rgb(image, width, height, nearest=False):
    """Aspect-preserving fit with black padding, RGB throughout."""
    rgb = require_rgb(image)
    if width <= 0 or height <= 0:
        raise ValueError("Output dimensions must be positive")
    scale = min(width / rgb.shape[1], height / rgb.shape[0])
    target = (max(1, min(width, round(rgb.shape[1] * scale))),
              max(1, min(height, round(rgb.shape[0] * scale))))
    resized = Image.fromarray(rgb).resize(target, _resample("NEAREST" if nearest else "BILINEAR"))
    canvas = Image.new("RGB", (width, height))
    canvas.paste(resized, ((width - target[0]) // 2, (height - target[1]) // 2))
    return np.asarray(canvas).copy()


def read_rgb(path):
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB")).copy()


def read_mask(path):
    # No lossy JPEG masks or palette conversion that could silently alter labels.
    with Image.open(path) as im:
        a = np.asarray(im).copy()
    return binary_mask(a)


def test_pattern(width=640, height=480, pattern="bars"):
    if width < 8 or height < 8:
        raise ValueError("Test pattern dimensions must be at least 8")
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    if pattern == "rectangle":
        frame[height // 4:3 * height // 4, width // 4:3 * width // 4] = 255
    elif pattern == "bars":
        colors = np.array([(255, 255, 255), (255, 255, 0), (0, 255, 255),
                           (0, 255, 0), (255, 0, 255), (255, 0, 0),
                           (0, 0, 255), (0, 0, 0)], dtype=np.uint8)
        for i in range(8):
            frame[:, i * width // 8:(i + 1) * width // 8] = colors[i]
    else:
        raise ValueError("pattern must be bars or rectangle")
    return frame


def synthetic_sample():
    """Deterministic geometric test fixture; no real patient/model output."""
    y, x = np.mgrid[:256, :256]
    m = ((x - 132) / 77.0) ** 2 + ((y - 121) / 91.0) ** 2 < 1
    m &= ~(((x - 190) / 32.0) ** 2 + ((y - 73) / 29.0) ** 2 < 1)
    a = np.empty((256, 256, 3), dtype=np.uint8)
    a[:, :, 0] = 65 + x // 3
    a[:, :, 1] = 100 + y // 4
    a[:, :, 2] = 150 + (x + y) // 8
    a[m] = (67, 54, 115)
    return a, m


def _font(size):
    for candidate in ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                      "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            pass
    return ImageFont.load_default()


def compose_dashboard(image, mask, width=640, height=480, latency_ms=None, demo=False):
    """Original, nearest-neighbour mask, overlay, and source-resolution statistics.

    latency_ms means caller-measured inference latency, never HDMI refresh rate.
    No timing is invented when no accelerator has been run.
    """
    rgb = require_rgb(image)
    m = binary_mask(mask, rgb.shape[:2])
    if width < 640 or height < 480:
        raise ValueError("Dashboard requires at least 640 x 480")
    if latency_ms is not None and (not math.isfinite(latency_ms) or latency_ms <= 0):
        raise ValueError("latency_ms must be positive and finite, or None")
    stats = mask_statistics(m)
    canvas = Image.new("RGB", (width, height), (12, 22, 36))
    d = ImageDraw.Draw(canvas)
    title, regular, small = _font(23), _font(16), _font(13)
    d.text((22, 20), "PLD / SEGMENTATION DISPLAY", fill=(238, 245, 255), font=title)
    tag = "SYNTHETIC DEMO - NO INFERENCE" if demo else "IMAGE + SUPPLIED MASK"
    d.text((22, 57), tag, fill=(75, 214, 176), font=regular)
    d.text((22, 85), "PYNQ-Z2  /  RGB 24-bit  /  {} x {}".format(width, height),
           fill=(147, 165, 190), font=small)
    gap, margin = 12, 22
    panel_w = (width - 2 * margin - 2 * gap) // 3
    panel_h = min(height - 265, panel_w)
    mask_rgb = np.repeat((m.astype(np.uint8) * 255)[:, :, None], 3, axis=2)
    panels = [("01  ORIGINAL", rgb, False), ("02  BINARY MASK", mask_rgb, True),
              ("03  OVERLAY", overlay_mask(rgb, m), False)]
    for i, (label, pixels, nearest) in enumerate(panels):
        x = margin + i * (panel_w + gap)
        d.text((x, 119), label, fill=(219, 230, 247), font=small)
        canvas.paste(Image.fromarray(fit_rgb(pixels, panel_w, panel_h, nearest)), (x, 143))
    sy = 143 + panel_h + 20
    d.line((22, sy, width - 22, sy), fill=(44, 62, 83), width=1)
    d.text((22, sy + 12), "MASK AREA", fill=(147, 165, 190), font=small)
    d.text((22, sy + 33), "{:.2f}%".format(stats["area_percent"]), fill=(75, 214, 176), font=title)
    d.text((200, sy + 12), "PIXELS (SOURCE)", fill=(147, 165, 190), font=small)
    d.text((200, sy + 37), "{} / {}".format(stats["mask_pixels"], stats["total_pixels"]),
           fill=(238, 245, 255), font=regular)
    d.text((422, sy + 12), "INFERENCE", fill=(147, 165, 190), font=small)
    label = "Not measured" if latency_ms is None else "{:.2f} ms".format(latency_ms)
    d.text((422, sy + 37), label, fill=(238, 245, 255), font=regular)
    d.text((22, height - 27), "Green: mask  |  Yellow: boundary  |  Area is a pixel ratio",
           fill=(147, 165, 190), font=small)
    return np.asarray(canvas).copy(), stats


class PynqHdmiDisplay:
    """Own one output pipeline; the caller owns/loading the overlay.

    Use synchronous show() in CLI and `await show_async()` in Jupyter.
    close() must finish before the caller downloads a different bitstream.
    """
    SUPPORTED_MODES = {(640, 480), (800, 600), (1280, 720), (1280, 1024), (1920, 1080)}

    def __init__(self, hdmi_out, width=640, height=480):
        if (width, height) not in self.SUPPORTED_MODES:
            raise ValueError("Unsupported resolution; start with 640 x 480")
        self.hdmi_out = hdmi_out
        self.width, self.height = width, height
        self.running = False
        self._configured = False

    def start(self):
        if self.running:
            return self
        from pynq.lib.video import VideoMode, PIXEL_RGB
        # Configure explicitly for NumPy/Pillow RGB rather than default BGR.
        self._configured = True
        try:
            self.hdmi_out.configure(VideoMode(self.width, self.height, 24, fps=60),
                                    pixelformat=PIXEL_RGB)
            self.hdmi_out.start()
            self.running = True
        except BaseException:
            self.close()
            raise
        return self

    def _prepare(self, image):
        if not self.running:
            raise RuntimeError("Call start() before show()")
        rgb = require_rgb(image)
        if rgb.shape != (self.height, self.width, 3):
            raise ValueError("Frame must match HDMI resolution; use fit_rgb() first")
        frame = self.hdmi_out.newframe()
        try:
            np.copyto(frame, rgb)
        except BaseException:
            frame.freebuffer()  # This buffer has not been submitted to VDMA.
            raise
        return frame

    def show(self, image):
        frame = self._prepare(image)
        # Ownership transfers to the driver. Never modify/free a submitted frame.
        try:
            self.hdmi_out.writeframe(frame)
        except BaseException:
            # Ownership may already have transferred: stop hardware, do not
            # manually free the same frame twice or retry that frame.
            self.close()
            raise

    async def show_async(self, image):
        writer = self.hdmi_out.writeframe_async
        frame = self._prepare(image)
        try:
            await writer(frame)
        except BaseException:
            self.close()
            raise

    def blank(self):
        """Hide the picture without stopping/reconfiguring the HDMI/VDMA link.

        Keep close() for final shutdown. Repeated physical stop/start on the
        board can leave a subsequent writeframe waiting for an interrupt.
        """
        if self.running:
            self.show(np.zeros((self.height, self.width, 3), dtype=np.uint8))

    def close(self):
        if self._configured:
            self.hdmi_out.close()  # Official close stops VDMA then HDMI frontend.
            self._configured = False
            self.running = False

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
