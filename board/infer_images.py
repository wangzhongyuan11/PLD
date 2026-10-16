"""Run original JPEGs through the frozen model; emit owned raw INT8 heads.

No GT or intermediate golden tensor is used. C can consume the binary and
JSON pair, or call PLGraph.run with the same input metadata in its process.
"""
import os
import time

MODULE_ENTRY = time.perf_counter()


def _finite_nonnegative(value):
    return value == value and -float("inf") < value < float("inf") and value >= 0.0


def _best_effort_process_entry_age():
    """Return process age at module entry on Linux, otherwise None.

    /proc timestamps have clock-tick granularity and are tied to boot uptime,
    not to time.perf_counter(). They are therefore reported as an age and only
    combined with perf_counter spans as an explicitly best-effort startup total.
    """
    try:
        with open("/proc/self/stat", "rt", encoding="ascii") as stat_file:
            fields = stat_file.read().rstrip("\n").rsplit(")", 1)[1].split()
        starttime_ticks = float(fields[19])
        with open("/proc/uptime", "rt", encoding="ascii") as uptime_file:
            uptime_seconds = float(uptime_file.read().split()[0])
        ticks_per_second = float(os.sysconf("SC_CLK_TCK"))
        age = uptime_seconds - starttime_ticks / ticks_per_second
        if _finite_nonnegative(age):
            return age
    except (OSError, ValueError, IndexError, TypeError, AttributeError):
        pass
    return None


PROCESS_ENTRY_AGE = _best_effort_process_entry_age()

import argparse
import math
from contextlib import ExitStack
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, features

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reference.tinyunet import Model, DEFAULT_MODEL_ROOT, digest
from reference.preprocess import decode_rgb, letterbox_rgb, letterbox_int8, quantize_input
from reference.parallel_resize import ParallelBilinear
from tensor_accel import TileAccelerator
from pl_graph import PLGraph


MODULE_IMPORTS_DONE = time.perf_counter()


class StageTimer:
    """Non-overlapping wall-clock stages with explicit ownership."""

    def __init__(self):
        self.created = time.perf_counter()
        self.stages = {}
        self.elapsed = {}
        self._active = None

    @contextmanager
    def stage(self, name):
        if not isinstance(name, str) or not name:
            raise ValueError("stage name must be a non-empty string")
        if name in self.stages:
            raise ValueError(f"duplicate timing stage: {name}")
        if self._active is not None:
            raise RuntimeError(f"nested timing stage under {self._active}: {name}")
        self._active = name
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - started
            if not math.isfinite(elapsed) or elapsed < 0.0:
                raise RuntimeError("non-finite timing measurement")
            self.stages[name] = elapsed
            self._active = None

    def record_elapsed(self, name, elapsed):
        if not isinstance(name, str) or not name:
            raise ValueError("elapsed name must be a non-empty string")
        if name in self.elapsed:
            raise ValueError(f"duplicate elapsed timing: {name}")
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise RuntimeError("elapsed timing must be finite and nonnegative")
        self.elapsed[name] = elapsed


def timing_profile(startup_timer, accelerator_timing, frame_timer=None):
    """Build the subprocess timing schema without changing legacy semantics."""
    return {
        "timing_clock": "time.perf_counter wall seconds",
        "scope": (
            "Non-overlapping startup, accelerator init/close and frame stages. "
            "A frame JSON profile necessarily ends before its own JSON write; "
            "the structured frame log adds that write. Process-entry age is "
            "best-effort Linux /proc data and is unavailable on Windows."
        ),
        "startup_stage_seconds": dict(startup_timer.stages),
        "accelerator_init_stage_seconds": dict(accelerator_timing["init_stages"]),
        "accelerator_close_stage_seconds": dict(accelerator_timing["close_stages"]),
        "startup_elapsed_seconds": dict(startup_timer.elapsed),
        "accelerator_init_elapsed_seconds": (
            {"tile_accelerator_init": accelerator_timing["elapsed"]["tile_accelerator_init"]}
            if "tile_accelerator_init" in accelerator_timing["elapsed"] else {}
        ),
        "accelerator_close_elapsed_seconds": (
            {"device_close_total": accelerator_timing["elapsed"]["device_close_total"]}
            if "device_close_total" in accelerator_timing["elapsed"] else {}
        ),
        "frame_stage_seconds": dict(frame_timer.stages) if frame_timer is not None else {},
        "frame_elapsed_seconds": dict(frame_timer.elapsed) if frame_timer is not None else {},
        "legacy_fields": {
            "online_b_seconds": "preprocess_seconds + PLGraph timing seconds; unchanged",
            "inference_seconds": "PLGraph timing seconds; unchanged",
        },
    }


def _emit_timing_event(event, profile, **fields):
    payload = {"event": event, **fields, "timing": profile}
    print("B_TIMING_JSON " + json.dumps(payload, sort_keys=True,
                                        separators=(",", ":")), flush=True)


class InferenceSession:
    """Single-owner cache. Only the serial worker may keep this between jobs.

    Full runtime hashes are still checked by the web supervisor per request.
    No cache is reused across a changed manifest or failed inference.
    """
    def __init__(self):
        self.model = None
        self.model_key = None
        self.hardware_key = None
        self.stack = None
        self.device = self.resizer = None
        self.failed = False
        self.active = False
        self.last_reuse = {}

    def release_hardware(self):
        if self.active:
            raise RuntimeError("cannot release hardware during inference")
        stack, self.stack = self.stack, None
        self.device = self.resizer = None
        self.hardware_key = None
        if stack is not None:
            try:
                stack.close()
            except BaseException:
                self.failed = True
                self.model = None
                self.model_key = None
                raise

    def close(self):
        try:
            self.release_hardware()
        finally:
            self.model = None
            self.model_key = None

    @contextmanager
    def resources(self, args, startup):
        if self.failed or self.active:
            raise RuntimeError("inference session failed or already active")
        model_key = str(args.model.resolve())
        hardware_key = (str(args.bit.resolve()), args.native_copy,
                        args.native_control, args.tx_range_sync,
                        args.native_resize, args.parallel_resize)
        model_reused = self.model is not None and self.model_key == model_key
        hardware_reused = self.device is not None and self.hardware_key == hardware_key
        try:
            if not model_reused:
                self.release_hardware()
                hardware_reused = False
                with startup.stage("model_init_and_manifest_validation"):
                    self.model = Model(args.model)
                self.model_key = model_key
            if not hardware_reused:
                self.release_hardware()
                self.stack = ExitStack()
                with startup.stage("resize_backend_init"):
                    if args.native_resize:
                        from native_resize import NativeResize
                        self.resizer = self.stack.enter_context(NativeResize(args.native_resize))
                    else:
                        self.resizer = self.stack.enter_context(ParallelBilinear()) if args.parallel_resize else None
                self.device = self.stack.enter_context(TileAccelerator(
                    args.bit, native_copy=args.native_copy,
                    native_control=args.native_control, tx_range_sync=args.tx_range_sync))
                self.hardware_key = hardware_key
            else:
                # Do not report cold initialization times again on a warm request.
                self.device.timing = {"init_stages": {}, "close_stages": {}, "elapsed": {}}
            self.last_reuse = {"model_reused": model_reused, "hardware_reused": hardware_reused}
            self.active = True
            try:
                yield self.model, self.resizer, self.device
            finally:
                self.active = False
        except BaseException:
            self.failed = True
            self.close()
            raise


def main(argv=None, session=None):
    main_entry = time.perf_counter()
    startup = StageTimer()
    with startup.stage("argument_parse"):
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("images", type=Path, nargs="+")
        parser.add_argument("--warmup-only", action="store_true", help="initialize session without reading images or creating outputs")
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
        args = parser.parse_args(argv)
        if args.parallel_resize and args.native_resize:
            parser.error("choose one resize backend")
    # A result directory is an immutable run, so old outputs cannot be confused
    # with a partially completed retry. Hardware failures propagate nonzero.
    with startup.stage("output_directory_create"):
        if not args.warmup_only:
            args.output.mkdir(parents=True, exist_ok=False)
    # A CLI invocation retains the original close-at-end behavior. Only the
    # worker passes a session and owns its explicit release before HDMI.
    with ExitStack() as ownership:
        if session is None:
            session = InferenceSession()
            ownership.callback(session.close)
        model, resizer, device = ownership.enter_context(session.resources(args, startup))
        with startup.stage("pl_graph_setup"):
            graph = PLGraph(model, device, fuse_maps=args.fuse_maps, fuse_decoder=args.fuse_decoder)
        ready = time.perf_counter()
        startup.record_elapsed("main_entry_to_ready", ready - main_entry)
        startup.record_elapsed("module_entry_to_ready", ready - MODULE_ENTRY)
        startup.record_elapsed("module_entry_to_imports_done", MODULE_IMPORTS_DONE - MODULE_ENTRY)
        if PROCESS_ENTRY_AGE is not None:
            startup.record_elapsed("best_effort_process_entry_to_ready",
                                   PROCESS_ENTRY_AGE + (ready - MODULE_ENTRY))
        _emit_timing_event("startup_ready", timing_profile(startup, device.timing))
        for frame, source in enumerate([] if args.warmup_only else args.images, 1):
            frame_timer = StageTimer()
            frame_started = frame_timer.created
            device.events = []
            with frame_timer.stage("jpeg_decode"):
                image = decode_rgb(source)
            decode_seconds = frame_timer.stages["jpeg_decode"]
            with frame_timer.stage("preprocess"):
                try:
                    if args.fast_quantize:
                        data, valid, geometry = letterbox_int8(
                            image, model.layers[0]["input_scale"], resizer=resizer)
                    else:
                        real, valid, geometry = letterbox_rgb(image, resizer=resizer)
                        data = quantize_input(real, model.layers[0]["input_scale"])
                finally:
                    image.close()
            preprocess_seconds = frame_timer.stages["preprocess"]
            with frame_timer.stage("input_metadata"):
                metadata = dict(geometry, model_spec_sha256=model.spec_hash,
                        input_sha256=hashlib.sha256(data.tobytes()).hexdigest(),
                        image_sha256=digest(source), valid_pixels=int(valid.sum()),
                        valid_rect=[geometry[k] for k in ("left", "top", "resized_width", "resized_height")],
                        pillow=Image.__version__, jpeg=features.version("jpg"))
            with frame_timer.stage("inference"):
                head, timing = graph.run(data, frame=frame, input_metadata=metadata)
            with frame_timer.stage("result_assembly"):
                result = dict(timing["result"], source_name=source.name,
                        jpeg_decode_seconds=decode_seconds, preprocess_seconds=preprocess_seconds,
                        inference_seconds=timing["seconds"],
                        online_b_seconds=preprocess_seconds + timing["seconds"],
                        timing_scope="decoded RGB to raw head; excludes file IO, JPEG decode and C processing/display",
                        bit_sha256=digest(args.bit), hwh_sha256=digest(args.bit.with_suffix(".hwh")),
                        abi_version=device.abi_version, fused_maps=args.fuse_maps,
                        fused_decoder=args.fuse_decoder,
                        quantization_backend="rgb8_lut" if args.fast_quantize else "float32",
                        copy_backend=device.copy_info, control_backend=device.control_info,
                        resize_backend=resizer.info if args.native_resize else {"kind":"parallel_bilinear_2" if args.parallel_resize else "pillow"},
                        tx_sync_backend=device.tx_sync_info)
            stem = args.output / f"frame_{frame:06d}"
            with frame_timer.stage("head_file_write"):
                stem.with_suffix(".bin").write_bytes(head.tobytes())
            frame_timer.record_elapsed("decode_through_head_write",
                                       time.perf_counter() - frame_started)
            with frame_timer.stage("timing_profile_assembly"):
                result["b_timing"] = timing_profile(startup, device.timing, frame_timer)
                result["session_reuse"] = dict(session.last_reuse)
            with frame_timer.stage("json_report_write"):
                stem.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            frame_timer.record_elapsed("decode_through_json_write",
                                       time.perf_counter() - frame_started)
            print(f"INFERENCE_PASS frame={frame} source={source.name} bytes={head.nbytes} online_B={result['online_b_seconds']:.6f}s", flush=True)
            _emit_timing_event("frame_complete", timing_profile(startup, device.timing, frame_timer),
                               frame=frame, source_name=source.name)
    _emit_timing_event("device_closed" if device.closed else "device_retained",
                       timing_profile(startup, device.timing), **session.last_reuse)


if __name__ == "__main__":
    main()
