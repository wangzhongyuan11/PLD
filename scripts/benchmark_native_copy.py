"""Isolated gather experiment on a real cacheable PYNQ TX buffer; no PL jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"board"))
from native_copy import NativeCopy


def baseline(dst, src):
    if np.any(src == -128):
        raise ValueError("input outside INT8 contract")
    np.copyto(dst[:src.nbytes].view(np.int8).reshape(src.shape), src, casting="no")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    from pynq import allocate
    kernel = NativeCopy(args.library.resolve())
    rng = np.random.default_rng(27)
    base = rng.integers(-127, 128, (1, 8, 256, 256), dtype=np.int8)
    samples = [base[:, :, y:y+32] for y in range(0, 256, 32)]
    samples += [rng.integers(-127, 128, shape, dtype=np.int8) for shape in
                ((1, 3, 34, 258), (1, 64, 34, 34), (1, 32, 34, 66), (1, 48, 18, 34))]
    report = {"passed": False, "scope": "copy/check only, real cacheable TX mapping; no DMA or network inference",
              "rounds": [], "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest()}
    tx = allocate(shape=(131072,), dtype=np.uint8, cacheable=True)
    try:
        dst = np.asarray(tx)
        for src in samples:
            dst[:] = 0xa5
            kernel.copy(dst, src)
            np.testing.assert_array_equal(dst[:src.nbytes], np.frombuffer(src.tobytes(), dtype=np.uint8))
            assert np.all(dst[src.nbytes:] == 0xa5)
        for index in range(8):
            row = {}
            methods = [("numpy", baseline), ("native", kernel.copy)]
            if index & 1:
                methods.reverse()
            for name, method in methods:
                start = time.perf_counter()
                for _ in range(10):
                    for src in samples:
                        method(dst, src)
                row[name] = time.perf_counter()-start
            if index >= 2:
                report["rounds"].append(row)
        report["median_seconds"] = {name: float(np.median([r[name] for r in report["rounds"]])) for name in ("numpy", "native")}
        report["passed"] = True
    finally:
        tx.freebuffer()
        args.report.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
