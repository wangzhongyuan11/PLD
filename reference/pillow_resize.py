"""Pillow 9.0.1 RGB row access for a zero-copy NEON resize candidate.

The version guard is mandatory: this uses the imaging core row pointer API,
whose RGB pixels have four bytes. The source Image stays alive and unmodified
until both horizontal workers finish. No source pixels are written.
Source: Pillow/9.0.1/src/_imaging.c _getattr_unsafe_ptrs and libImaging/Imaging.h.
"""
import ctypes as ct
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .native_resize import NativeBilinear


class NativePillowBilinear(NativeBilinear):
    def __init__(self, library):
        if Image.__version__ != "9.0.1":
            raise ValueError("direct row resize requires the verified Pillow 9.0.1 ABI")
        super().__init__(library)
        self.info.update(kind="pillow_rows_neon", pillow=Image.__version__,
                         row_wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        self.horizontal_rows=self.library.rgb_horizontal_rows
        self.horizontal_rows.argtypes=[ct.c_void_p,ct.c_void_p,ct.c_uint64,
                ct.c_uint32,ct.c_uint32,ct.c_uint32,ct.c_uint32,ct.c_uint32,
                ct.c_void_p,ct.c_void_p,ct.c_uint32]
        self.horizontal_rows.restype=ct.c_int

    def _rows(self, rows, output, width, height, target, start, stop, coeff):
        bounds,weights,stride=coeff
        code=self.horizontal_rows(rows,output.ctypes.data,output.nbytes,
                width,height,target,start,stop,bounds.ctypes.data,weights.ctypes.data,stride)
        if code: raise ValueError(f"native Pillow row resize rejected arguments: {code}")

    def resize(self, image, size):
        width,height=size
        if image.mode!="RGB" or not 1<=width<=4096 or not 1<=height<=4096 or max(image.size)>16384:
            raise ValueError("RGB dimensions outside native resize contract")
        if image.size==size: return image.copy()
        image.load()
        rows=dict(image.im.unsafe_ptrs)["image32"]
        if not rows: raise ValueError("Pillow RGB row storage unavailable")
        horizontal=np.empty((image.height,width,3),dtype=np.uint8)
        output=np.empty((height,width,3),dtype=np.uint8)
        coeff=self.coeff(image.width,width)
        middle=image.height//2
        futures=[self.pool.submit(self._rows,rows,horizontal,image.width,image.height,width,a,b,coeff)
                 for a,b in ((0,middle),(middle,image.height)) if a<b]
        # Wait for every worker even if one fails, before releasing the Image.
        failure=None
        for future in futures:
            try: future.result()
            except BaseException as exc: failure=exc
        if failure is not None: raise failure
        self._filter(self.vertical,horizontal,output,self.coeff(image.height,height),height)
        return Image.fromarray(output,"RGB")
