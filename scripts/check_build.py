"""Reject mismatched PYNQ artifacts or failed routed timing (stage 0 gate)."""
import argparse
import math
from pathlib import Path
import zipfile


def check(directory):
    root = Path(directory)
    lines = (root / "timing_summary.rpt").read_text(encoding="utf-8").splitlines()
    header = next(i for i, line in enumerate(lines) if "WNS(ns)" in line and "WPWS(ns)" in line)
    values = [float(x) for x in lines[header + 2].split()]
    if len(values) != 12 or not all(math.isfinite(v) for v in values) or any(values[i] < 0 for i in (0, 4, 8)) or any(values[i] != 0 for i in (1, 2, 5, 6, 9, 10)):
        raise ValueError(f"routed setup/hold/pulse-width violation: {values}")
    if "All user specified timing constraints are met." not in lines:
        raise ValueError("missing successful timing verdict")
    with zipfile.ZipFile(root / "dma_top.xsa") as archive:
        for suffix in (".bit", ".hwh"):
            names = [name for name in archive.namelist() if name.endswith(suffix)]
            if len(names) != 1 or archive.read(names[0]) != (root / ("dma_top" + suffix)).read_bytes():
                raise ValueError(f"XSA/extracted {suffix} mismatch")
    print(f"BUILD_CHECK_PASS {root}: WNS={values[0]} WHS={values[4]} WPWS={values[8]} ns; bit/hwh match XSA")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    check(parser.parse_args().directory)
