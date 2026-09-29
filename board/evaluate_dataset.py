"""Board dataset verification, latency distribution and bounded stability run.

Input is an audited first-layer tensor, never an intermediate golden feature.
Disk I/O and output comparison are outside inference timing; JPEG/display are
not included. The first traversal supplies test-set metrics; repeats stress it.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np
import PIL
from PIL import features

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.tinyunet import DEFAULT_MODEL_ROOT,Model, digest, require_equal
from reference.evaluation import segmentation_metrics
from reference.datasets import validate_manifest
from reference.preprocess import decode_rgb, letterbox_rgb, letterbox_int8, quantize_input
from tensor_accel import TileAccelerator
from pl_graph import PLGraph
from verify_cnn import fault_checks


def memory_state():
    values = {}
    for path, keys in (("/proc/self/status", ("VmRSS", "VmHWM")),
                       ("/proc/meminfo", ("MemAvailable", "CmaFree"))):
        for line in Path(path).read_text().splitlines():
            name, _, tail = line.partition(":")
            if name in keys:
                values[name+"_KiB"] = int(tail.split()[0])
    return values


def load_case(root, record):
    if not re.fullmatch(r"ISIC_[0-9]+", record["id"]) or record["file"] != record["id"]+".npz":
        raise ValueError("invalid case filename")
    path = root/record["file"]
    if digest(path) != record["sha256"]:
        raise ValueError("case artifact hash mismatch: "+record["id"])
    with np.load(path, allow_pickle=False) as data:
        x, head, valid, target = (data[k].copy() for k in ("input", "head", "valid", "target"))
    for array, key in ((x, "input_sha256"), (head, "head_sha256")):
        if hashlib.sha256(array.tobytes()).hexdigest() != record[key]:
            raise ValueError("tensor hash mismatch: "+key)
    if x.dtype != np.int8 or x.shape != (1, 3, 256, 256) or head.dtype != np.int8 or head.shape != (1, 1, 256, 256):
        raise ValueError("dataset tensor contract mismatch")
    if valid.dtype != np.bool_ or target.dtype != np.bool_ or valid.shape != (256, 256) or target.shape != valid.shape:
        raise ValueError("invalid evaluation masks")
    return x, head, valid, target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bit", required=True, type=Path)
    p.add_argument("--model", default=DEFAULT_MODEL_ROOT, type=Path)
    p.add_argument("--dataset", required=True, type=Path)
    p.add_argument("--report", required=True, type=Path)
    p.add_argument("--min-frames", type=int, help="default: one traversal of the selected collection")
    p.add_argument("--min-seconds", type=float, default=0)
    p.add_argument("--max-frames", type=int, default=3000)
    p.add_argument("--allow-subset", action="store_true", help="smoke only; cannot pass full-set/stability acceptance")
    p.add_argument("--check-recovery", action="store_true")
    p.add_argument("--native-copy", help="verified native copy library")
    p.add_argument("--tx-range-sync", action="store_true", help="XRT TX prefix cache synchronization")
    p.add_argument("--native-control", help="verified native command library")
    p.add_argument("--parallel-resize", action="store_true", help="two-worker pixel-exact bilinear preprocessing")
    p.add_argument("--native-resize", help="verified Pillow row resize library")
    p.add_argument("--decode-images", action="store_true", help="decode original JPEG, compare preprocessing and use its tensor for PL")
    p.add_argument("--fast-quantize", action="store_true", help="pixel-exact RGB8 lookup instead of float32 conversion")
    p.add_argument("--fuse-maps",action="store_true",help="ABI 1.1 PL input-LUT convolution")
    p.add_argument("--fuse-decoder",action="store_true",help="ABI 1.2 PL fused decoder-entry convolution")
    args = p.parse_args()
    if (args.parallel_resize or args.native_resize) and not args.decode_images:
        p.error("resize options require --decode-images")
    if args.fast_quantize and not args.decode_images:
        p.error("fast quantization requires --decode-images")
    if args.parallel_resize and args.native_resize:
        p.error("choose one resize backend")
    if (args.min_frames is not None and (args.min_frames < 1 or args.max_frames < args.min_frames)) or args.max_frames < 1 or not np.isfinite(args.min_seconds) or args.min_seconds < 0:
        p.error("invalid run bounds")
    report = {"passed": False, "frames": [], "scope": "quantized input through PS/PL to head; excludes JPEG/preprocess, comparison, file IO and C/display"}
    report["runtime"] = {"python": platform.python_version(), "numpy": np.__version__,
                         "pillow": PIL.__version__, "jpeg": features.version("jpg")}
    device = None
    resizer = None
    code = 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    stream_path = args.report.with_suffix(".jsonl")
    try:
        report["preprocessing_backend"] = "pillow"
        report["quantization_backend"] = "rgb8_lut" if args.fast_quantize else "float32"
        if args.native_resize:
            from native_resize import NativeResize
            resizer=NativeResize(args.native_resize)
            report["preprocessing_backend"]="pillow_rows_neon"
            report["resize_backend"]=resizer.info
        elif args.parallel_resize:
            from reference.parallel_resize import ParallelBilinear
            resizer = ParallelBilinear()
            report["preprocessing_backend"] = "parallel_bilinear_2"
            report["resizer_sha256"] = digest(Path(__file__).resolve().parents[1]/"reference/parallel_resize.py")
        model = Model(args.model)
        manifest_path = args.dataset/"manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        cases = manifest["records"]
        full = validate_manifest(manifest, model)
        if not full and not args.allow_subset:
            raise ValueError("dataset preparation incomplete or does not identify this model")
        if args.min_frames is None:
            args.min_frames = len(cases)
        if args.min_frames < len(cases):
            raise ValueError("run must cover every provided case")
        if args.max_frames < args.min_frames:
            raise ValueError("maximum frames cannot cover the selected collection")
        report.update(dataset_manifest_sha256=digest(manifest_path), model_spec_sha256=model.spec_hash,
                      model_manifest_sha256=model.manifest_hash, threshold=model.threshold,
                      collection=manifest.get("collection"),
                      primary_evaluation_complete=full and manifest.get("collection", {}).get("primary", False),
                      full_test_set=bool(full), source_hashes={str(f): digest(f) for f in
                      (Path(__file__), Path(__file__).with_name("tensor_accel.py"), Path(__file__).with_name("pl_graph.py"),
                       Path(__file__).with_name("simple_dma.py"), Path(__file__).resolve().parents[1]/"reference/evaluation.py",
                       Path(__file__).resolve().parents[1]/"reference/preprocess.py",
                       Path(__file__).resolve().parents[1]/"reference/tinyunet.py",
                       Path(__file__).resolve().parents[1]/"reference/datasets.py")},
                      bit_sha256=digest(args.bit), hwh_sha256=digest(args.bit.with_suffix(".hwh")))
        device = TileAccelerator(args.bit, native_copy=args.native_copy, tx_range_sync=args.tx_range_sync,
                                 native_control=args.native_control)
        report["copy_backend"] = device.copy_info
        report["tx_sync_backend"] = device.tx_sync_info
        report["control_backend"] = device.control_info
        graph = PLGraph(model, device,fuse_maps=args.fuse_maps,fuse_decoder=args.fuse_decoder)
        report["clock_mhz"] = device.clock_mhz
        report["abi_version"]=device.abi_version
        report["graph_options"]={"fuse_maps":args.fuse_maps,"fuse_decoder":args.fuse_decoder}
        warm, expected, _, _ = load_case(args.dataset, cases[0])
        output, _ = graph.run(warm, frame=0)
        require_equal(output, expected, "warmup")
        device.events = []
        del output, warm, expected
        report["memory_after_warmup"] = memory_state()
        start = time.perf_counter()
        with stream_path.open("w", encoding="utf-8") as stream:
            for index in range(args.max_frames):
                case = cases[index % len(cases)]
                x, expected, valid, target = load_case(args.dataset, case)
                decode_seconds = preprocess_seconds = 0.0
                if args.decode_images:
                    image_path = args.dataset/"images"/(case["id"]+".jpg")
                    if digest(image_path) != case["image_sha256"]:
                        raise ValueError("source JPEG hash mismatch")
                    begin = time.perf_counter()
                    image = decode_rgb(image_path)
                    decode_seconds = time.perf_counter()-begin
                    begin = time.perf_counter()
                    try:
                        if args.fast_quantize:
                            prepared, runtime_valid, geometry = letterbox_int8(
                                image,model.layers[0]["input_scale"],resizer=resizer)
                        else:
                            real, runtime_valid, geometry = letterbox_rgb(image, resizer=resizer)
                            prepared = quantize_input(real, model.layers[0]["input_scale"])
                    finally:
                        image.close()
                    preprocess_seconds = time.perf_counter()-begin
                    require_equal(prepared, x, case["id"]+" JPEG preprocessing")
                    require_equal(runtime_valid, valid, case["id"]+" valid region")
                    if geometry != case["geometry"]:
                        raise ValueError("runtime image geometry differs from reference")
                    x = prepared
                device.events = []
                output, timing = graph.run(x, frame=index+1)
                require_equal(output, expected, case["id"])
                metrics = segmentation_metrics(output[0, 0] >= model.threshold["threshold_q"], target, valid)
                if metrics != case["int8"]:
                    raise ValueError("metric counts disagree with software reference")
                row = {"frame": index+1, "id": case["id"], "seconds": timing["seconds"],
                       "jpeg_decode_seconds": decode_seconds, "preprocess_seconds": preprocess_seconds,
                       "online_b_seconds": preprocess_seconds+timing["seconds"],
                       "output_sha256": hashlib.sha256(output.tobytes()).hexdigest(), "metrics": metrics,
                       "commands": len(device.events), "compute_cycles": sum(e["compute_cycles"] for e in device.events),
                       "dma_wait_seconds": sum(e["dma_seconds"] for e in device.events), "memory": memory_state()}
                report["frames"].append(row)
                stream.write(json.dumps(row)+"\n"); stream.flush()
                if (index+1) % 10 == 0 or index == 0:
                    print(f"DATASET_FRAME {index+1} {case['id']} seconds={row['seconds']:.3f} rss={row['memory']['VmRSS_KiB']}KiB", flush=True)
                if index+1 >= args.min_frames and time.perf_counter()-start >= args.min_seconds:
                    break
            else:
                raise RuntimeError("maximum frames reached before requested duration")
        report["wall_seconds"] = time.perf_counter()-start
        durations = [r["seconds"] for r in report["frames"]]
        report["median_seconds"] = float(np.median(durations))
        report["p95_seconds"] = float(np.percentile(durations, 95))
        report["original_jpeg_inputs"] = args.decode_images
        if args.decode_images:
            online = [r["online_b_seconds"] for r in report["frames"]]
            report["online_b_median_seconds"] = float(np.median(online))
            report["online_b_p95_seconds"] = float(np.percentile(online, 95))
            report["online_b_scope"] = "decoded RGB through preprocessing and PS/PL to head; excludes comparison, JPEG decode, dataset IO and C/display"
        report["inference_fps"] = len(durations)/sum(durations)
        report["dice_mean"] = float(np.mean([r["metrics"]["dice"] for r in report["frames"][:len(cases)]]))
        report["iou_mean"] = float(np.mean([r["metrics"]["iou"] for r in report["frames"][:len(cases)]]))
        rss = [r["memory"]["VmRSS_KiB"] for r in report["frames"]]
        report["rss_tail_minus_early_median_KiB"] = float(np.median(rss[-100:])-np.median(rss[20:120] or rss))
        report["stability_duration_and_frames"] = full and len(durations) >= 1000 and report["wall_seconds"] >= 600
        # Raw memory samples remain authoritative; final review checks drift and
        # allocator plateaus rather than treating a single peak as a leak.
        if args.check_recovery:
            report["faults"] = fault_checks(device, model)
            device.events = []
            output, _ = graph.run(x, frame=len(durations)+1)
            require_equal(output, expected, "post-recovery complete frame")
            report["post_recovery_frame_passed"] = True
        report["passed"] = True
        code = 0
        print(f"DATASET_PASS frames={len(durations)} full_set={full} median={report['median_seconds']:.6f} p95={report['p95_seconds']:.6f}", flush=True)
    except Exception as exc:
        report["error"] = repr(exc)
        print(f"DATASET_FAIL {exc}", file=sys.stderr, flush=True)
    finally:
        if resizer is not None:
            try:
                resizer.close()
            except Exception as exc:
                report["passed"] = False
                report["resizer_close_error"] = repr(exc)
                code = 2
        if device is not None:
            try:
                device.close()
            except Exception as exc:
                report["passed"] = False
                report["close_error"] = repr(exc)
                code = 2
        args.report.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
