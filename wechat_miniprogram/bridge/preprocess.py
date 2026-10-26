# -*- coding: utf-8 -*-
"""Skin-model input preparation, mirroring the web frontend's upload-copy rules.

The board contract (display_web/server.py) for mode=infer:
  - original JPEG bytes (FFD8 magic), single file <= 8 MB
  - <= 3,000,000 pixels in board mode
So the bridge always hands the server a JPEG copy <= 3 MP, leaving the
user's original file untouched.
"""
import io

MAX_PIXELS = 3_000_000
MAX_BYTES = 8 * 1024 * 1024


def prepare_skin_image(raw):
    """Return JPEG bytes obeying the board contract. Raises ValueError."""
    try:
        from PIL import Image, ImageOps
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ValueError("桥接电脑缺少 Pillow，无法准备皮肤镜输入：%s" % exc) from exc
    try:
        with Image.open(io.BytesIO(raw)) as im:
            # The web frontend keeps valid JPEGs byte-for-byte, including EXIF.
            if im.format == "JPEG" and im.width * im.height <= MAX_PIXELS and len(raw) <= MAX_BYTES:
                im.verify()
                return raw
            im.load()
            oriented = ImageOps.exif_transpose(im)
            if oriented.mode != "RGB":
                oriented = oriented.convert("RGB")
            w, h = oriented.size
            if w * h > MAX_PIXELS:
                scale = (MAX_PIXELS / float(w * h)) ** 0.5
                oriented = oriented.resize((max(1, int(w * scale)),
                                            max(1, int(h * scale))))
            for quality in (92, 85, 78, 70):
                buf = io.BytesIO()
                oriented.save(buf, format="JPEG", quality=quality)
                data = buf.getvalue()
                if len(data) <= MAX_BYTES:
                    return data
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("图片无法解码：%s" % exc) from exc
    raise ValueError("图片压缩后仍超过 8MB，请换用更小的图片")
