"""Run original JPEGs through the frozen model; emit owned raw INT8 heads.

No GT or intermediate golden tensor is used. C can consume the binary and
JSON pair, or call PLGraph.run with the same input metadata in its process.
"""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
import time

from PIL import Image, features

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.tinyunet import Model, DEFAULT_MODEL_ROOT, digest
from reference.preprocess import decode_rgb, letterbox_rgb, letterbox_int8, quantize_input
from reference.parallel_resize import ParallelBilinear
from tensor_accel import TileAccelerator
from pl_graph import PLGraph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--bit", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-copy")
    parser.add_argument("--native-control")
    parser.add_argument("--tx-range-sync", action="store_true")
    parser.add_argument("--parallel-resize", action="store_true")
    parser.add_argument("--native-resize", help="verified Pillow row resize library")
    parser.add_argument("--fast-quantize", action="store_true", help="pixel-exact RGB8 lookup instead of float32 conversion")
    parser.add_argument("--fuse-maps", action="store_true")
    parser.add_argument("--fuse-decoder", action="store_true")
    args = parser.parse_args()
    if args.parallel_resize and args.native_resize:
        parser.error("choose one resize backend")
    # A result directory is an immutable run, so old outputs cannot be confused
    # with a partially completed retry. Hardware failures propagate nonzero.
    args.output.mkdir(parents=True, exist_ok=False)
    model = Model(args.model)
    with ExitStack() as stack:
        if args.native_resize:
            from native_resize import NativeResize
            resizer=stack.enter_context(NativeResize(args.native_resize))
        else:
            resizer = stack.enter_context(ParallelBilinear()) if args.parallel_resize else None
        device = stack.enter_context(TileAccelerator(args.bit, native_copy=args.native_copy,
                        native_control=args.native_control, tx_range_sync=args.tx_range_sync))
        graph = PLGraph(model, device, fuse_maps=args.fuse_maps, fuse_decoder=args.fuse_decoder)
        for frame, source in enumerate(args.images, 1):
            device.events = []
            begin = time.perf_counter()
            image = decode_rgb(source)
            decode_seconds = time.perf_counter()-begin
            begin = time.perf_counter()
            try:
                if args.fast_quantize:
                    data, valid, geometry = letterbox_int8(
                        image,model.layers[0]["input_scale"],resizer=resizer)
                else:
                    real, valid, geometry = letterbox_rgb(image, resizer=resizer)
                    data = quantize_input(real, model.layers[0]["input_scale"])
            finally:
                image.close()
            preprocess_seconds = time.perf_counter()-begin
            metadata = dict(geometry, model_spec_sha256=model.spec_hash,
                    input_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
                    image_sha256=digest(source), valid_pixels=int(valid.sum()),
                    valid_rect=[geometry[k] for k in ("left", "top", "resized_width", "resized_height")],
                    pillow=Image.__version__, jpeg=features.version("jpg"))
            head, timing = graph.run(data, frame=frame, input_metadata=metadata)
            result = dict(timing["result"], source_name=source.name,
                    jpeg_decode_seconds=decode_seconds, preprocess_seconds=preprocess_seconds,
                    inference_seconds=timing["seconds"],
                    online_b_seconds=preprocess_seconds+timing["seconds"],
                    timing_scope="decoded RGB to raw head; excludes file IO, JPEG decode and C processing/display",
                    bit_sha256=digest(args.bit), hwh_sha256=digest(args.bit.with_suffix(".hwh")),
                    abi_version=device.abi_version, fused_maps=args.fuse_maps,
                    fused_decoder=args.fuse_decoder,
                    quantization_backend="rgb8_lut" if args.fast_quantize else "float32",
                    copy_backend=device.copy_info, control_backend=device.control_info,
                    resize_backend=resizer.info if args.native_resize else {"kind":"parallel_bilinear_2" if args.parallel_resize else "pillow"},
                    tx_sync_backend=device.tx_sync_info)
            stem=args.output/f"frame_{frame:06d}"
            stem.with_suffix(".bin").write_bytes(head.tobytes())
            stem.with_suffix(".json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
            print(f"INFERENCE_PASS frame={frame} source={source.name} bytes={head.nbytes} online_B={result['online_b_seconds']:.6f}s", flush=True)


if __name__ == "__main__":
    main()
