"""Run the whole integer graph from the first input and verify all 15 layers."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.tinyunet import Model, DEFAULT_MODEL_ROOT, digest
from reference.tinyunet import conv2d
from reference.tiling import conv2d_tiled


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL_ROOT)
    p.add_argument("--spec-sha256", default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--report", default="out/reference/full_reference.json")
    p.add_argument("--image", help="start from original image; must be the exported golden source for verification")
    p.add_argument("--tiled", action="store_true", help="execute all convs through the bounded halo tile plan")
    args = p.parse_args()
    model = Model(args.model, args.spec_sha256, **({"manifest": args.manifest} if args.manifest else {}))
    start = time.perf_counter()
    records = []
    input_tensor = model.golden(0, "input")
    preprocessing = None
    if args.image:
        import PIL
        from PIL import features
        from reference.preprocess import preprocess
        input_tensor, valid, metadata = preprocess(args.image, model.layers[0]["input_scale"])
        preprocessing = {"image": args.image, "image_sha256": digest(args.image),
                         "pillow": PIL.__version__, "jpeg": features.version_codec("jpg"),
                         "valid_pixels": int(valid.sum()), **metadata}

    def record(i, layer, output):
        records.append({"layer": layer["directory"], "input_mismatch": 0, "output_mismatch": 0,
                        "output_bytes": output.nbytes, **model.bounds[i]})
        print(f"LAYER_PASS {i:02d} {layer['directory']} input/output mismatch=0", flush=True)

    model.run(input_tensor, verify=True, callback=record, conv_backend=conv2d_tiled if args.tiled else conv2d)
    report = {"passed": True, "spec_sha256": model.spec_hash, "elapsed_seconds": time.perf_counter()-start,
              "scope": "one input -> all non-conv operators and 15 integer convs; golden is comparison only",
              "layers": records, "preprocessing": preprocessing, "tiled": args.tiled}
    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"FULL_REFERENCE_PASS 15/15 layers seconds={report['elapsed_seconds']:.3f}")


if __name__ == "__main__":
    main()
