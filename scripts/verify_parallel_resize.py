"""Compare every resized RGB byte before measuring a preprocessing candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image, features

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from reference.preprocess import decode_rgb
from reference.parallel_resize import ParallelBilinear
from reference.datasets import validate_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--native-library", type=Path, help="isolated native RGB candidate instead of Pillow threads")
    p.add_argument("--pillow-rows", action="store_true")
    p.add_argument("--allow-subset", action="store_true")
    args = p.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    complete=validate_manifest(manifest)
    if not complete and not args.allow_subset:
        raise ValueError("requires a complete named collection")
    if args.pillow_rows and not args.native_library:
        p.error("--pillow-rows requires --native-library")
    images = {f.stem: f for f in args.images.rglob("*.jpg")}
    filters = getattr(Image, "Resampling", Image)
    report = {"passed": False, "frames": [], "pillow": Image.__version__, "jpeg": features.version("jpg"),
              "scope": "decoded RGB resize only; no JPEG decode, quantization, DMA or inference",
              "source_sha256": hashlib.sha256((ROOT/"reference/parallel_resize.py").read_bytes()).hexdigest()}
    candidate=ParallelBilinear
    candidate_name='parallel'
    if args.native_library:
        from reference.native_resize import NativeBilinear
        if args.pillow_rows:
            from reference.pillow_resize import NativePillowBilinear
            NativeBilinear=NativePillowBilinear
        candidate=lambda: NativeBilinear(args.native_library)
        candidate_name='native'
        report['source_sha256']=hashlib.sha256((ROOT/'reference/native_resize.py').read_bytes()).hexdigest()
    try:
        with candidate() as resizer:
            if args.native_library: report['native_build']=resizer.info
            # Warm the workers and both code paths outside measured frames.
            small = Image.new("RGB", (100, 101), (10, 20, 30))
            resizer.resize(small, (32, 33)); small.resize((32, 33), filters.BILINEAR)
            for index, record in enumerate(manifest["records"]):
                path = images[record["id"]]
                if hashlib.sha256(path.read_bytes()).hexdigest() != record["image_sha256"]:
                    raise ValueError("source JPEG hash mismatch")
                image = decode_rgb(path)
                g = record["geometry"]
                size = g["resized_width"], g["resized_height"]
                outputs = {}; times = {}
                methods = [("pillow", lambda: image.resize(size, filters.BILINEAR)),
                           (candidate_name, lambda: resizer.resize(image, size))]
                if index & 1:
                    methods.reverse()
                for name, method in methods:
                    start = time.perf_counter(); outputs[name] = method(); times[name] = time.perf_counter()-start
                if outputs["pillow"].tobytes() != outputs[candidate_name].tobytes():
                    raise ValueError("RGB mismatch: "+record["id"])
                report["frames"].append({"id": record["id"], "source_size": image.size, **times})
                image.close()
                if (index+1) % 20 == 0:
                    print(f"RESIZE_PASS {index+1} {record['id']}", flush=True)
        if len(report["frames"]) != len(manifest["records"]):
            raise ValueError("incomplete resize traversal")
        report['complete_collection']=complete
        report['collection']=manifest.get('collection')
        report["median_seconds"] = {name: float(np.median([f[name] for f in report["frames"]])) for name in ("pillow", candidate_name)}
        report["p95_seconds"] = {name: float(np.percentile([f[name] for f in report["frames"]],95)) for name in ("pillow", candidate_name)}
        report["passed"] = True
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:report[k] for k in ("passed", "median_seconds", "p95_seconds")}), flush=True)


if __name__ == "__main__":
    main()
