"""INT8 dot-product stream vectors with independent Python integer oracle."""
from pathlib import Path
import random


def pack(values, width):
    return sum((int(v) & ((1 << width)-1)) << (i*width) for i,v in enumerate(values))


def main():
    rng = random.Random(20260926)
    path = Path("out/vectors/mac.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="ascii") as f:
        for packet in range(132):
            length = [1, 2, 7, 9, 18, 36, 72, 144, 216][packet % 9]
            bias = [rng.randrange(-100000,100001) for _ in range(8)]
            if packet == 130:
                bias = [(1 << 31)-1]*8
            if packet == 131:
                bias = [-(1 << 31)]*8
            accum = bias.copy()
            overflow = False
            for step in range(length):
                x = [rng.randrange(-127,128) for _ in range(4)]
                w = [rng.randrange(-127,128) for _ in range(32)]
                if packet >= 130:
                    x = [127]*4
                    w = [127 if packet == 130 else -127]*32
                for o in range(8):
                    result = accum[o] + sum(x[i]*w[o*4+i] for i in range(4))
                    overflow |= not -(1 << 31) <= result < (1 << 31)
                    accum[o] = ((result + (1 << 31)) & 0xffffffff) - (1 << 31)
                f.write(f"{int(step==0):x} {int(step==length-1):x} {pack(x,8):08x} "
                        f"{pack(w,8):064x} {pack(bias,32):064x} {pack(accum,32):064x} {int(overflow):x}\n")
                count += 1
    print(f"MAC_VECTORS packets=132 beats={count} path={path}")


if __name__ == "__main__":
    main()
