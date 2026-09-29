"""Same frozen integer graph on the board CPU, timed without oracle I/O."""
import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.tinyunet import DEFAULT_MODEL_ROOT,Model, digest, require_equal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL_ROOT, type=Path)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("repeat must be positive")
    report = {"passed": False, "frames": [], "warmup_frames": 1,
              "scope": "board CPU NumPy INT64 exact TinyUNet from quantized input to head; excludes model load, JPEG/preprocess and comparison; not an optimized ARM inference library",
              "python": platform.python_version(), "numpy": np.__version__,
              "source_hashes": {str(f): digest(f) for f in (Path(__file__), Path(__file__).resolve().parents[1]/"reference/tinyunet.py")},
              "cpuinfo": Path("/proc/cpuinfo").read_text()}
    try:
        model = Model(args.model)
        report["model_spec_sha256"] = model.spec_hash
        x, expected = model.golden(0, "input"), model.golden(14, "output")
        for index in range(args.repeat+1):
            start = time.perf_counter()
            outputs = model.run(x)
            seconds = time.perf_counter()-start
            require_equal(outputs[-1], expected, "CPU full graph")
            del outputs
            if index:
                report["frames"].append({"frame": index, "seconds": seconds})
            else:
                report["warmup_seconds"] = seconds
            print(f"CPU_FRAME index={index} seconds={seconds:.6f} correct=True", flush=True)
        report["median_seconds"] = float(np.median([r["seconds"] for r in report["frames"]]))
        report["p95_seconds"] = float(np.percentile([r["seconds"] for r in report["frames"]], 95))
        report["passed"] = True
    finally:
        args.report.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
