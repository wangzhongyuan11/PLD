"""Exercise a separately built gather library against NumPy before integration."""
import argparse
import ctypes
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"board"))
from native_copy import NativeCopy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library", type=Path)
    args = parser.parse_args()
    kernel = NativeCopy(args.library.resolve())
    rng = np.random.default_rng(27)
    source = rng.integers(-127, 128, (2, 9, 38, 266), dtype=np.int8)
    cases = [source, source[:, 1:8, 2:36, 3:261], source[:, ::-1, ::-1, ::-1],
             source[::2, ::2, ::3, ::2], source.transpose(0, 2, 1, 3),
             np.broadcast_to(source[:1, :1, :1, :1], (1, 3, 34, 258))]
    for array in cases:
        dst = np.full(array.nbytes+64, 0xa5, dtype=np.uint8)
        kernel.copy(dst, array)
        np.testing.assert_array_equal(dst[:array.nbytes], np.frombuffer(array.tobytes(), dtype=np.uint8))
        assert np.all(dst[array.nbytes:] == 0xa5)
    for column in (5, 265):  # vector body and scalar tail, including last row
        bad = source.copy(); bad[1, 8, 37, column] = -128
        try:
            kernel.copy(np.empty(bad.nbytes, dtype=np.uint8), bad)
        except ValueError as exc:
            assert "-128" in str(exc)
        else:
            raise AssertionError("invalid signed value accepted")
    dst = np.full(8, 0xa5, dtype=np.uint8)
    # Also validate the C capacity gate, independent of the Python preflight.
    shape = (ctypes.c_uint64 * 4)(1, 1, 1, 9)
    strides = (ctypes.c_int64 * 4)(9, 9, 9, 1)
    assert kernel.copy_fn(dst.ctypes.data, 8, source.ctypes.data, shape, strides) == 2
    assert np.all(dst == 0xa5)
    for destination, data in ((dst, source), (source.reshape(-1), source),
                               (np.empty(source.nbytes, dtype=np.uint8)[::2], source)):
        try:
            kernel.copy(destination, data)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid copy accepted")
    print("NATIVE_COPY_PASS layouts=6 invalid_value capacity overlap strided_destination tail_guard")


if __name__ == "__main__":
    main()
