"""EXIF -> RGB -> bilinear letterbox -> NCHW signed INT8.

JPEG decoder builds can differ by a byte. Validate the frozen source image
against A's input_full.bin on each runtime; never silently patch pixels to
match the oracle. Windows validation uses the Pillow 12.3.0 wheel.
"""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps


def decode_rgb(image_path):
    """Decode and apply EXIF before the online preprocessing timing boundary."""
    with Image.open(image_path) as source:
        oriented = ImageOps.exif_transpose(source)
        if oriented.mode == "RGB":
            return oriented
        try:
            return oriented.convert("RGB")
        finally:
            oriented.close()


def _letterbox_canvas(image, size, resizer):
    if size <= 0 or image.mode != "RGB":
        raise ValueError("positive size and decoded RGB image required")
    width, height = image.size
    ratio = min(size / width, size / height)
    new_width = max(1, min(size, round(width * ratio)))
    new_height = max(1, min(size, round(height * ratio)))
    left, top = (size-new_width)//2, (size-new_height)//2
    # PYNQ's Pillow 9.0 uses the legacy constant; both select the same filter.
    resampling = getattr(Image, "Resampling", Image)
    resized = (image.resize((new_width, new_height), resampling.BILINEAR) if resizer is None
               else resizer.resize(image, (new_width, new_height)))
    canvas = Image.new("RGB", (size,size), (0,0,0))
    canvas.paste(resized,(left,top))
    valid = np.zeros((size,size),dtype=np.bool_)
    valid[top:top+new_height,left:left+new_width] = True
    metadata = {"original_width":width,"original_height":height,
                "resized_width":new_width,"resized_height":new_height,"left":left,"top":top}
    return canvas, valid, metadata


def letterbox_rgb(image, size=256, resizer=None):
    """Return float32 NCHW RGB, valid mask and geometry, without quantization."""
    canvas, valid, metadata = _letterbox_canvas(image,size,resizer)
    real = np.asarray(canvas,dtype=np.float32) / np.float32(255)
    return np.ascontiguousarray(real.transpose(2,0,1)[None]), valid, metadata


def quantize_input(real, input_scale):
    if not np.isfinite(input_scale) or input_scale <= 0:
        raise ValueError("invalid input scale")
    return np.clip(np.rint(real / np.float32(input_scale)), -127, 127).astype(np.int8)


@lru_cache(maxsize=8)
def _rgb8_quant_lut(input_scale):
    values=np.arange(256,dtype=np.float32) / np.float32(255)
    return np.clip(np.rint(values / np.float32(input_scale)), -127, 127).astype(np.int8)


def letterbox_int8(image, input_scale, size=256, resizer=None):
    """Pixel-exact RGB8 lookup version of letterbox_rgb + quantize_input."""
    if not np.isfinite(input_scale) or input_scale <= 0:
        raise ValueError("invalid input scale")
    canvas, valid, metadata = _letterbox_canvas(image,size,resizer)
    pixels=np.asarray(canvas)
    table=_rgb8_quant_lut(float(input_scale))
    result=np.empty((1,3,size,size),dtype=np.int8)
    for channel in range(3):
        result[0,channel]=table[pixels[:,:,channel]]
    return result, valid, metadata


def preprocess(image_path, input_scale, size=256, resizer=None):
    real, valid, metadata = letterbox_rgb(decode_rgb(image_path), size, resizer=resizer)
    return quantize_input(real, input_scale), valid, metadata
