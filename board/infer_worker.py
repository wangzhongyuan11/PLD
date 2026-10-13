"""Persistent B preloader with a bounded, line-oriented request protocol.

The worker imports PYNQ once but does not instantiate an Overlay or own DMA.
Every request calls infer_images.main(argv), which still constructs and closes
TileAccelerator for that job. Inference stdout/stderr goes to that job's log;
only supervisor protocol lines use the parent pipe.
"""
import argparse
import contextlib
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import traceback


MAX_PROTOCOL_LINE = 64 * 1024
MAX_ARGUMENTS = 64
MAX_ARGUMENT_CHARS = 64 * 1024
REQUIRED_MANIFEST_FILES = (
    "board/infer_worker.py",
    "board/infer_images.py",
    "board/tensor_accel.py",
)
REQUEST_ID = re.compile(r"[0-9a-f]{1,128}\Z")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_manifest_identity(runtime):
    """Return (manifest SHA-256, file hash map) after contract checks."""
    runtime = Path(runtime).resolve()
    manifest_path = runtime / "deployment_manifest.json"
    raw = manifest_path.read_bytes()
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("deployment manifest is too large")
    manifest = json.loads(raw)
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict):
        raise ValueError("deployment manifest has no file hash map")
    missing = [name for name in REQUIRED_MANIFEST_FILES if name not in files]
    if missing:
        raise ValueError("persistent worker files missing from manifest: " + ", ".join(missing))
    return hashlib.sha256(raw).hexdigest(), files


def enable_parent_death_signal():
    """Ask the long-lived Linux launcher thread's process to supervise us.

    PR_SET_PDEATHSIG follows the parent thread, not merely the parent thread
    group. The persistent supervisor therefore launches this process from a
    launcher thread that remains alive between HTTP jobs.
    """
    if sys.platform != "linux":
        return False
    parent = os.getppid()
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, 15, 0, 0, 0) != 0:
            raise OSError("PR_SET_PDEATHSIG failed", ctypes.get_errno())
    except OSError as exc:
        raise RuntimeError("failed to install Linux PR_SET_PDEATHSIG") from exc
    if os.getppid() != parent:
        os._exit(1)
    return True


def emit(event, **fields):
    payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
    line = f"PLD_WORKER_{event} {payload}\n"
    if len(line.encode("utf-8")) > MAX_PROTOCOL_LINE:
        raise RuntimeError("worker protocol response is too long")
    sys.stdout.write(line)
    sys.stdout.flush()


def preload_pynq():
    """Import the expensive PYNQ API without programming a bitstream."""
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            from pynq import Overlay, MMIO, allocate  # noqa: F401


def read_request(stream):
    raw = stream.readline(MAX_PROTOCOL_LINE + 1)
    if not raw:
        return None
    if len(raw.encode("utf-8")) > MAX_PROTOCOL_LINE or not raw.endswith("\n"):
        raise ValueError("request line exceeds the bounded protocol")
    prefix, separator, encoded = raw.rstrip("\n").partition(" ")
    if prefix != "PLD_WORKER_REQUEST" or not separator:
        raise ValueError("request is not PLD_WORKER_REQUEST")
    value = json.loads(encoded)
    if not isinstance(value, dict):
        raise ValueError("request payload must be an object")
    request_id = value.get("request_id")
    argv = value.get("argv")
    log_path = value.get("log_path")
    if not isinstance(request_id, str) or not REQUEST_ID.fullmatch(request_id):
        raise ValueError("invalid request_id")
    if (not isinstance(argv, list) or not 1 <= len(argv) <= MAX_ARGUMENTS or
            any(not isinstance(item, str) for item in argv) or
            sum(len(item.encode("utf-8")) for item in argv) > MAX_ARGUMENT_CHARS):
        raise ValueError("invalid inference argv")
    if not isinstance(log_path, str) or not log_path:
        raise ValueError("invalid log_path")
    return value


def serve_request(inference_main, request):
    log_path = Path(request["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", encoding="utf-8") as job_log:
            with contextlib.redirect_stdout(job_log), contextlib.redirect_stderr(job_log):
                inference_main(list(request["argv"]))
    except BaseException:
        # The per-job ExitStack may have completed, but the worker cannot prove
        # hardware state after an exception. Fail closed and let C require a
        # service restart rather than claiming cleanup.
        try:
            with log_path.open("a", encoding="utf-8") as job_log:
                traceback.print_exc(file=job_log)
        except OSError:
            pass
        emit("FATAL", request_id=request["request_id"], stage="request",
             error="inference exception; device state not claimed")
        return 1
    emit("RESULT", request_id=request["request_id"], status="success", returncode=0)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args(argv)
    if not enable_parent_death_signal():
        pass
    try:
        manifest_sha256, files = runtime_manifest_identity(args.runtime)
        worker = Path(__file__).resolve()
        if worker != (args.runtime.resolve() / "board/infer_worker.py").resolve():
            raise ValueError("worker is not running from the supplied runtime")
        for name in REQUIRED_MANIFEST_FILES:
            if _sha256(args.runtime / name) != files[name]:
                raise ValueError("persistent worker source hash mismatch: " + name)
        import infer_images
        if Path(infer_images.__file__).resolve() != (args.runtime / "board/infer_images.py").resolve():
            raise ValueError("infer_images resolved outside the frozen runtime")
        preload_pynq()
    except BaseException as exc:
        emit("FATAL", stage="startup", error=f"{type(exc).__name__}: {exc}")
        return 1

    emit("READY", manifest_sha256=manifest_sha256)
    while True:
        try:
            request = read_request(sys.stdin)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            emit("FATAL", stage="protocol", error=f"{type(exc).__name__}: {exc}")
            return 1
        if request is None:
            return 0
        status = serve_request(infer_images.main, request)
        if status != 0:
            return status


if __name__ == "__main__":
    raise SystemExit(main())
