"""Stage 1/2 exact board validation with bounded DMA and recovery checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from simple_dma import DMAFault, SimpleDMA

SIZES = [1, 7, 8, 64, 1000, 65536, 1 << 20]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Loopback:
    def __init__(self, bit, mode, timeout):
        from pynq import Overlay, MMIO
        self.overlay = Overlay(str(bit))
        self.mode, self.timeout = mode, timeout
        self.mmio_type = MMIO
        self.attach()

    def attach(self):
        info = self.overlay.ip_dict["axi_dma"]
        self.dma = SimpleDMA(self.mmio_type(info["phys_addr"], info["addr_range"]),
                             info["parameters"], self.timeout)
        self.dma.reset()
        self.dma.start()

    def reload(self):
        # Soft DMA reset cannot empty an external AXIS FIFO/register slice.
        self.overlay.download()
        self.attach()

    def packet(self, data, receive_only=False):
        from pynq import allocate
        n = len(data)
        source = destination = None
        safe_to_free = True
        try:
            source = allocate(shape=(n,), dtype=np.uint8)
            destination = allocate(shape=(n + 64,), dtype=np.uint8)
            source[:] = data
            destination[:] = 0xA5
            safe_to_free = False
            start = time.perf_counter()
            try:
                self.dma.transfer(source, destination, n, receive_only=receive_only)
                safe_to_free = True
            except BaseException:
                # No active buffer is freed until both channels acknowledge reset.
                self.dma.reset()
                safe_to_free = True
                raise
            elapsed = time.perf_counter() - start
            expected = data if self.mode == "fifo" else (data.astype(np.uint16) + 1).astype(np.uint8)
            bad = np.flatnonzero(destination[:n] != expected)
            if len(bad):
                i = int(bad[0])
                raise DMAFault(f"mismatch index={i} expected={expected[i]} actual={destination[i]}",
                               self.dma.snapshot())
            if not np.all(destination[n:] == 0xA5):
                raise DMAFault("tail guard overwritten", self.dma.snapshot())
            return elapsed
        finally:
            if not safe_to_free:
                # If reset failed, reconfigure the PL before releasing DDR.
                # An unrecoverable device failure needs operator intervention;
                # keep buffer references alive while this process is alive.
                try:
                    self.overlay.download()
                    safe_to_free = True
                except BaseException:
                    self.retained_buffers = (source, destination)
                    print("FATAL: DMA not quiescent; buffers retained, board recovery required", file=sys.stderr)
            if safe_to_free:
                for buffer in (destination, source):
                    if buffer is not None:
                        buffer.freebuffer()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("fifo", "incr"))
    p.add_argument("--bit", default="dma_top.bit")
    p.add_argument("--timeout", type=float, default=2.0)
    p.add_argument("--repeat", type=int, default=1, help="rounds over seven sizes")
    p.add_argument("--random-packets", type=int, default=0)
    p.add_argument("--seed", type=int, default=20260926)
    p.add_argument("--check-recovery", action="store_true")
    p.add_argument("--inject-timeout", action="store_true", help="negative test: recover hardware, exit nonzero")
    p.add_argument("--report", help="JSON evidence, written for success and failure")
    args = p.parse_args()
    if args.repeat < 1 or args.random_packets < 0:
        p.error("repeat must be >=1, random-packets >=0")
    report = {"mode": args.mode, "seed": args.seed, "passed": False, "packets": 0,
              "repeat": args.repeat, "random_packets": args.random_packets, "sizes": SIZES,
              "timing_scope": "source/destination flush + DMA programming/polling + invalidate; excludes allocation, compare, reload",
              "one_mib_MBps": []}
    loop = None
    try:
        bit = Path(args.bit).resolve()
        report["sha256"] = {str(path.name): sha256(path) for path in (
            bit, bit.with_suffix(".hwh"), Path(__file__), Path(__file__).with_name("simple_dma.py"))}
        print("ARTIFACTS " + json.dumps(report["sha256"], sort_keys=True), flush=True)
        loop = Loopback(bit, args.mode, args.timeout)
        rng = np.random.default_rng(args.seed)
        for repeat in range(args.repeat):
            for n in SIZES:
                # Zero, all-255 wrap, byte ramp, then independent seeded random.
                if repeat % 4 == 0:
                    data = rng.integers(0, 256, n, dtype=np.uint8)
                elif repeat % 4 == 1:
                    data = np.zeros(n, dtype=np.uint8)
                elif repeat % 4 == 2:
                    data = np.full(n, 255, dtype=np.uint8)
                else:
                    data = np.arange(n, dtype=np.uint8)
                elapsed = loop.packet(data)
                report["packets"] += 1
                if n == SIZES[-1]:
                    report["one_mib_MBps"].append(n / elapsed / 1e6)
                print(f"PASS round={repeat} bytes={n} ms={elapsed*1000:.3f} MBps={n/elapsed/1e6:.3f}", flush=True)
        for i in range(args.random_packets):
            n = int(rng.integers(1, 65537))
            loop.packet(rng.integers(0, 256, n, dtype=np.uint8))
            report["packets"] += 1
            if (i + 1) % 100 == 0:
                print(f"RANDOM_PASS {i+1}/{args.random_packets}", flush=True)
        if args.check_recovery or args.inject_timeout:
            try:
                loop.packet(np.arange(64, dtype=np.uint8), receive_only=True)
            except DMAFault as error:
                if error.reason != "transfer timeout":
                    raise
                report["injected_timeout"] = error.registers
                print(f"EXPECTED_TIMEOUT {error}", flush=True)
            else:
                raise RuntimeError("injected missing-source transfer unexpectedly completed")
            loop.reload()
            for n in SIZES:
                loop.packet(rng.integers(0, 256, n, dtype=np.uint8))
                report["packets"] += 1
            report["recovery_passed"] = True
            print("RECOVERY_PASS (7 sizes after DMA reset + overlay reload)", flush=True)
            if args.inject_timeout:
                raise RuntimeError("injected timeout: hardware recovered; negative test exits nonzero")
        report["passed"] = True
        print(f"BOARD_PASS mode={args.mode} (7 sizes, {report['packets']} packets)", flush=True)
        return 0
    except Exception as error:
        report["error"] = str(error)
        print(f"BOARD_FAIL {error}", file=sys.stderr, flush=True)
        if loop is not None:
            try:
                report["dma_state"] = loop.dma.snapshot()
                # Clear both DMA and external PL state before returning failure.
                loop.dma.reset()
                loop.reload()
                report["cleanup_recovered"] = True
            except Exception as recovery_error:
                report["cleanup_error"] = str(recovery_error)
                print(f"RECOVERY_FAIL {recovery_error}", file=sys.stderr, flush=True)
                return 2
        return 1
    finally:
        if args.report:
            Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
