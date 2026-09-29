"""Independent Python-integer rational oracle for RTL requantization."""
import argparse
import json
from pathlib import Path
import random


def expected(acc, multiplier):
    numerator = acc * multiplier
    sign = -1 if numerator < 0 else 1
    whole, fraction = divmod(abs(numerator), 1 << 24)
    if fraction > (1 << 23) or (fraction == (1 << 23) and whole % 2):
        whole += 1
    return max(-127, min(127, sign * whole))


def vectors():
    spec = json.loads(Path("role_a_final/int8/model_spec.json").read_text())
    rng = random.Random(20260926)
    # Exact ties, +/- half, saturation neighbors, zero and full INT32 edges.
    for acc in list(range(-259, 260)) + [-(1 << 31), (1 << 31)-1]:
        for mult in (0, 1, (1 << 23)-1, 1 << 23, (1 << 23)+1, 1 << 24, (1 << 31)-1):
            yield acc, mult
    for layer in spec["layers"]:
        for mult in layer["multiplier_q24"]:
            # Real model multipliers, both in and outside the saturation range.
            threshold = (127 << 24) // mult
            for acc in (-threshold-1, -threshold, -1, 0, 1, threshold, threshold+1):
                yield acc, mult
            for _ in range(12):
                yield rng.randrange(-(1 << 23), 1 << 23), mult
    for _ in range(4096):
        yield rng.randrange(-(1 << 31), 1 << 31), rng.randrange(0, 1 << 31)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="out/vectors/requant.txt")
    path = Path(parser.parse_args().output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [f"{a & 0xffffffff:08x} {m:08x} {expected(a, m) & 255:02x}\n" for a, m in vectors()]
    path.write_text("".join(rows), encoding="ascii")
    print(f"REQUANT_VECTORS count={len(rows)} path={path}")
