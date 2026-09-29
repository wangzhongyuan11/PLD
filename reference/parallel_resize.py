"""Parallel horizontal rows, then the original Pillow vertical filter.

Pillow's separable 8-bit intermediate is retained. Integer source row boxes
avoid changing the horizontal sampling grid. Verify pixels on each runtime.
Reference: Pillow 9.0.1 src/libImaging/Resample.c ImagingResampleInner.
"""
from concurrent.futures import ThreadPoolExecutor
from PIL import Image


class ParallelBilinear:
    def __init__(self):
        self.pool = ThreadPoolExecutor(max_workers=2)

    def resize(self, image, size):
        filters = getattr(Image, "Resampling", Image)
        width, height = size
        if image.mode != "RGB" or min(width, height) < 1:
            raise ValueError("positive RGB target required")
        if image.height < 2 or width == image.width:
            return image.resize(size, filters.BILINEAR)
        middle = image.height//2
        spans = ((0, middle), (middle, image.height))
        futures = [self.pool.submit(image.resize, (width, end-start), filters.BILINEAR,
                                    (0, start, image.width, end)) for start, end in spans]
        horizontal = Image.new("RGB", (width, image.height))
        for (start, _), future in zip(spans, futures):
            horizontal.paste(future.result(), (0, start))
        return horizontal.resize((width, height), filters.BILINEAR)

    def close(self):
        self.pool.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
