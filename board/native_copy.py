"""Optional checked gather kernel. Loading it is explicit during evaluation."""
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np


class NativeCopy:
    def __init__(self, path, verify_build=False):
        path = Path(path).resolve()
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        self.info = {"kind": "native", "library_sha256": digest(path), "abi": 1}
        if verify_build:
            manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            source = Path(__file__).resolve().parent/"native/tensor_copy.c"
            if (manifest["library_sha256"] != self.info["library_sha256"] or
                    manifest["source_sha256"] != digest(source) or
                    manifest["wrapper_sha256"] != digest(Path(__file__))):
                raise ValueError("native copy build manifest mismatch")
            self.info["build"] = manifest
        self.library = ctypes.CDLL(str(path))
        self.library.tensor_copy_abi.argtypes = []
        self.library.tensor_copy_abi.restype = ctypes.c_uint32
        if self.library.tensor_copy_abi() != 1:
            raise ValueError("native tensor copy ABI mismatch")
        self.copy_fn = self.library.tensor_copy_i8
        self.copy_fn.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p,
                                ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int64)]
        self.copy_fn.restype = ctypes.c_int

    def copy(self, destination, source):
        if (not isinstance(destination, np.ndarray) or destination.dtype not in (np.dtype("uint8"), np.dtype("int8"))
                or not destination.flags.c_contiguous or not destination.flags.writeable):
            raise ValueError("destination must be a writable contiguous byte array")
        if not isinstance(source, np.ndarray) or source.dtype != np.int8 or source.ndim != 4:
            raise ValueError("source must be a four-dimensional INT8 array")
        if source.nbytes > destination.nbytes or min(source.shape) < 1:
            raise ValueError("source exceeds destination capacity or is empty")
        if np.may_share_memory(source, destination):
            raise ValueError("source and destination must not overlap")
        shape = (ctypes.c_uint64 * 4)(*source.shape)
        strides = (ctypes.c_int64 * 4)(*source.strides)
        result = self.copy_fn(destination.ctypes.data, destination.nbytes, source.ctypes.data, shape, strides)
        if result == 1:
            raise ValueError("input contains -128 outside the symmetric INT8 contract")
        if result:
            raise ValueError("native tensor copy rejected its dimensions or capacity")
