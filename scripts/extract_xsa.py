"""Extract .bit and .hwh from a Vivado .xsa (zip) for PYNQ deployment.

Usage: python extract_xsa.py <file.xsa> <out_prefix>
Writes <out_prefix>.bit and <out_prefix>.hwh (PYNQ requires same stem).
"""
import sys
import zipfile
from pathlib import Path

def main():
    xsa = Path(sys.argv[1])
    prefix = Path(sys.argv[2])
    with zipfile.ZipFile(xsa) as z:
        bits = [n for n in z.namelist() if n.endswith(".bit")]
        hwhs = [n for n in z.namelist() if n.endswith(".hwh")]
        if len(bits) != 1 or len(hwhs) != 1:
            sys.exit(f"expected exactly one .bit/.hwh in {xsa}, got {bits} {hwhs}")
        prefix.parent.mkdir(parents=True, exist_ok=True)
        bit_dst = prefix.with_suffix(".bit")
        hwh_dst = prefix.with_suffix(".hwh")
        bit_dst.write_bytes(z.read(bits[0]))
        hwh_dst.write_bytes(z.read(hwhs[0]))
    print(f"extracted {bit_dst} ({bit_dst.stat().st_size} B), "
          f"{hwh_dst} ({hwh_dst.stat().st_size} B)")

if __name__ == "__main__":
    main()
