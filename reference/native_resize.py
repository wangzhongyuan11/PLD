"""Isolated RGB resize candidate. Not enabled in the board inference runtime."""
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


class NativeBilinear:
    def __init__(self, library):
        path=Path(library).resolve()
        self.library=ct.CDLL(str(path))
        self.library.rgb_resize_abi.restype=ct.c_uint32
        if self.library.rgb_resize_abi()!=1:
            raise ValueError('native RGB resize ABI mismatch')
        self.info={'library_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                   'source_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'board/native/rgb_resize.c').read_bytes()).hexdigest(),
                   'wrapper_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        self.coefficients=self.library.rgb_coefficients
        self.coefficients.argtypes=[ct.c_uint32,ct.c_uint32,ct.c_uint32,ct.c_void_p,ct.c_void_p]
        self.coefficients.restype=ct.c_int
        self.horizontal=self.library.rgb_horizontal
        self.vertical=self.library.rgb_vertical
        for function in (self.horizontal,self.vertical):
            function.argtypes=[ct.c_void_p,ct.c_uint64,ct.c_void_p,ct.c_uint64,
                               ct.c_uint32,ct.c_uint32,ct.c_uint32,ct.c_void_p,ct.c_void_p,ct.c_uint32]
            function.restype=ct.c_int
        self.coeff=lru_cache(maxsize=16)(self._make_coeff)
        self.pool=ThreadPoolExecutor(max_workers=2)

    def _make_coeff(self, source, target):
        stride=2*int(np.ceil(max(1.0,source/target)))+1
        bounds=np.empty((target,2),dtype=np.uint32)
        weights=np.empty((target,stride),dtype=np.int32)
        if self.coefficients(source,target,stride,bounds.ctypes.data,weights.ctypes.data):
            raise ValueError('native resize coefficient rejection')
        bounds.flags.writeable=False; weights.flags.writeable=False
        return bounds,weights,stride

    def _filter(self, function, source, output, coeff, target):
        bounds,weights,stride=coeff
        code=function(source.ctypes.data,source.nbytes,output.ctypes.data,output.nbytes,
                      source.shape[1],source.shape[0],target,bounds.ctypes.data,weights.ctypes.data,stride)
        if code: raise ValueError(f'native RGB filter failed code={code}')

    def resize(self, image, size):
        width,height=size
        if (image.mode!='RGB' or not 1<=width<=4096 or not 1<=height<=4096 or
                max(image.size)>16384):
            raise ValueError('RGB dimensions outside native resize contract')
        if image.size==size: return image.copy()
        source=np.asarray(image,dtype=np.uint8)
        horizontal=np.empty((image.height,width,3),dtype=np.uint8)
        output=np.empty((height,width,3),dtype=np.uint8)
        xcoeff=self.coeff(image.width,width); ycoeff=self.coeff(image.height,height)
        middle=image.height//2
        if middle:
            jobs=[self.pool.submit(self._filter,self.horizontal,source[a:b],horizontal[a:b],xcoeff,width)
                  for a,b in ((0,middle),(middle,image.height))]
            for job in jobs: job.result()
        else:
            self._filter(self.horizontal,source,horizontal,xcoeff,width)
        self._filter(self.vertical,horizontal,output,ycoeff,height)
        return Image.fromarray(output)

    def close(self):
        self.pool.shutdown(wait=True)
        self.coeff.cache_clear()

    def __enter__(self): return self
    def __exit__(self,*args): self.close()
