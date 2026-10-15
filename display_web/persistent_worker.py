"""Conservative supervisor for the opt-in persistent B inference worker."""
import hashlib
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import uuid


MAX_PROTOCOL_LINE = 64 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
REQUIRED_MANIFEST_FILES = (
    "board/infer_worker.py",
    "board/infer_images.py",
    "board/tensor_accel.py",
)


class PersistentWorkerError(RuntimeError):
    def __init__(self, message, *, hardware_unknown=False, require_restart=False):
        super().__init__(message)
        self.hardware_unknown = hardware_unknown
        self.require_restart = require_restart


def runtime_manifest_sha256(runtime):
    runtime = Path(runtime).resolve()
    raw = (runtime / "deployment_manifest.json").read_bytes()
    if len(raw) > MAX_MANIFEST_BYTES:
        raise ValueError("deployment manifest is too large")
    manifest = json.loads(raw)
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not isinstance(files, dict):
        raise ValueError("deployment manifest has no file hash map")
    missing = [name for name in REQUIRED_MANIFEST_FILES if name not in files]
    if missing:
        raise ValueError("persistent worker files missing from manifest: " + ", ".join(missing))
    return hashlib.sha256(raw).hexdigest()


class PersistentBWorker:
    """Own one worker process; one C job may use it at a time."""

    def __init__(self, runtime, *, startup_timeout=60.0, request_timeout=180.0):
        self.runtime = Path(runtime).resolve()
        self.startup_timeout = startup_timeout
        self.request_timeout = request_timeout
        self.expected_manifest = runtime_manifest_sha256(self.runtime)
        self.process = None
        self.reader = None
        self.events = queue.Queue()
        self.failure = None
        self.close_reason = None
        self.last_startup_seconds = None
        self.last_request_seconds = None
        self._allow_diagnostics = False
        self._launcher = None
        self._launcher_commands = queue.Queue()
        self._lock = threading.RLock()

    @property
    def pid(self):
        return self.process.pid if self.process is not None else None

    @property
    def launcher_thread(self):
        return self._launcher

    def _reader_main(self):
        stream = getattr(self.process.stdout, "buffer", self.process.stdout)
        while True:
            try:
                raw = stream.readline(MAX_PROTOCOL_LINE + 1)
            except (OSError, ValueError):
                self.events.put(("error", "worker output stream failed"))
                break
            if not raw:
                self.events.put(("eof", None))
                break
            if len(raw) > MAX_PROTOCOL_LINE:
                self.events.put(("error", "worker protocol line exceeds 64 KiB"))
                # Do not continue accepting an unbounded stream. The supervisor
                # terminates this process before returning from the deadline.
                break
            try:
                line = raw.decode("utf-8").rstrip("\n")
            except UnicodeDecodeError:
                self.events.put(("error", "worker protocol line is not UTF-8"))
                break
            marker = "PLD_WORKER_"
            if not line.startswith(marker):
                # PYNQ/import diagnostics are intentionally suppressed. Actual
                # inference output is redirected by the worker to its job log.
                if self._allow_diagnostics:
                    continue
                self.events.put(("error", "unexpected non-protocol worker output"))
                break
            event, separator, encoded = line[len(marker):].partition(" ")
            if not separator or event not in ("READY", "RESULT", "FATAL"):
                self.events.put(("error", "unknown worker protocol event"))
                break
            try:
                payload = json.loads(encoded)
                if not isinstance(payload, dict):
                    raise ValueError("payload is not an object")
            except (ValueError, json.JSONDecodeError) as exc:
                self.events.put(("error", f"invalid worker JSON: {exc}"))
                break
            self.events.put(("protocol", (event, payload)))

    def _remaining(self, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PersistentWorkerError("persistent B worker deadline exceeded",
                                        hardware_unknown=True)
        return remaining

    def _wait_event(self, expected, request_id, deadline, *, allow_diagnostics=False):
        while True:
            try:
                kind, value = self.events.get(timeout=self._remaining(deadline))
            except queue.Empty:
                if self.process.poll() is not None:
                    raise PersistentWorkerError(
                        f"persistent B worker exited before reply (code {self.process.returncode})",
                        hardware_unknown=True)
                raise PersistentWorkerError("persistent B worker deadline exceeded",
                                            hardware_unknown=True)
            if kind == "error":
                raise PersistentWorkerError(value, hardware_unknown=True)
            if kind == "eof":
                try:
                    code = self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    code = self.process.poll()
                if code is not None:
                    raise PersistentWorkerError(
                        f"persistent B worker exited before reply (code {code})",
                        hardware_unknown=True)
                raise PersistentWorkerError(
                    f"persistent B worker pipe closed (code {code})",
                    hardware_unknown=True)
            event, payload = value
            if event == "FATAL":
                detail = payload.get("error", "unspecified worker failure")
                raise PersistentWorkerError(
                    f"persistent B worker failed: {detail}",
                    hardware_unknown=request_id is not None)
            if event in expected and (request_id is None or payload.get("request_id") == request_id):
                if self.process.poll() is not None:
                    raise PersistentWorkerError(
                        "persistent B worker died after its reply; reply rejected",
                        hardware_unknown=True)
                return event, payload
            if allow_diagnostics:
                continue
            raise PersistentWorkerError(
                "unexpected/stale persistent B worker reply rejected",
                hardware_unknown=True)

    def _stop(self, reason):
        if self.close_reason is None:
            self.close_reason = reason
        # Stop the launcher first: a timed-out Popen can still assign a process
        # after the caller has observed self.process is None.
        self._shutdown_launcher()
        process = self.process
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            try:
                process.stdin.close()
            except (OSError, AttributeError):
                pass
            try:
                process.stdout.close()
            except (OSError, AttributeError):
                pass
            if self.reader is not None and self.reader is not threading.current_thread():
                self.reader.join(timeout=5)

    def _launcher_main(self):
        while True:
            try:
                command, result = self._launcher_commands.get(timeout=5)
            except queue.Empty:
                continue
            if command == "exit":
                return
            if command != "start":
                result.put(("error", RuntimeError("invalid launcher command")))
                continue
            try:
                # Linux PR_SET_PDEATHSIG follows the parent *thread*. The HTTP
                # job thread that requests inference would otherwise terminate
                # at the end of every job and kill a healthy worker. Keep this
                # launcher alive for the lifetime of PersistentBWorker.
                self.process = subprocess.Popen(
                    [sys.executable, str(self.runtime / "board/infer_worker.py"),
                     "--runtime", str(self.runtime)],
                    cwd=str(self.runtime), stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=-1)
            except BaseException as exc:
                result.put(("error", exc))
                return
            result.put(("started", self.process))

    def _shutdown_launcher(self):
        launcher = self._launcher
        if launcher is None:
            return
        try:
            self._launcher_commands.put_nowait(("exit", None))
        except queue.Full:
            return
        if launcher is not threading.current_thread():
            launcher.join(timeout=5)
        self._launcher = None

    def close(self):
        with self._lock:
            try:
                self.release_hardware(self.runtime / "worker_release.log")
            finally:
                self._stop("closed_by_service")

    def release_hardware(self, log_path):
        with self._lock:
            if self.process is None:
                return
            # A dead process may have left DMA active; do not permit HDMI.
            if self.process.poll() is not None or self.close_reason is not None:
                raise PersistentWorkerError("worker unavailable; hardware release unconfirmed",
                                            hardware_unknown=True)
            return self.run(["release"], log_path, action="release")

    def check_manifest(self):
        current = runtime_manifest_sha256(self.runtime)
        if current != self.expected_manifest:
            self._stop("manifest_changed")
            raise PersistentWorkerError(
                "persistent B worker deployment manifest changed; restart the service before inference",
                hardware_unknown=False, require_restart=True)

    def _ensure_started(self):
        if self.process is not None:
            if self.process.poll() is None:
                return False
            raise PersistentWorkerError("cached worker exited; hardware state unknown; restart service",
                                        hardware_unknown=True, require_restart=True)
        started = time.perf_counter()
        deadline = time.monotonic() + self.startup_timeout
        self.events = queue.Queue()
        self._allow_diagnostics = True
        if self._launcher is None:
            self._launcher = threading.Thread(
                target=self._launcher_main, name="pld-persistent-b-launcher",
                daemon=True)
            self._launcher.start()
        launch_result = queue.Queue(maxsize=1)
        self._launcher_commands.put(("start", launch_result))
        self.reader = threading.Thread(target=self._reader_main,
                                       name="pld-persistent-b-worker", daemon=True)
        try:
            try:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise PersistentWorkerError(
                        "persistent B worker launch thread deadline exceeded",
                        hardware_unknown=False, require_restart=True)
                status, value = launch_result.get(timeout=remaining)
            except queue.Empty:
                raise PersistentWorkerError(
                    "persistent B worker launch thread deadline exceeded",
                    hardware_unknown=False, require_restart=True)
            if status != "started":
                raise RuntimeError(f"worker launch failed: {value}")
            self.reader.start()
            _, payload = self._wait_event({"READY"}, None, deadline, allow_diagnostics=True)
            if payload.get("manifest_sha256") != self.expected_manifest:
                raise PersistentWorkerError(
                    "persistent B worker loaded a different deployment manifest",
                    hardware_unknown=False, require_restart=True)
            # Catch a deployment replacement race between launch and READY.
            self.check_manifest()
            self._allow_diagnostics = False
            self.last_startup_seconds = time.perf_counter() - started
        except BaseException as exc:
            self._stop("startup_failed")
            # No inference request has been submitted yet, so startup failures
            # do not imply unknown accelerator ownership.
            exc = PersistentWorkerError(
                f"persistent B worker startup failed: {exc}",
                hardware_unknown=False, require_restart=True)
            self.failure = str(exc)
            raise exc
        return True

    @staticmethod
    def _write_log_failure(log_path, exc):
        try:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as log:
                log.write(f"PERSISTENT_WORKER_SUPERVISOR_FAILURE {type(exc).__name__}: {exc}\n")
        except OSError:
            pass

    def run(self, argv, log_path, *, action="infer"):
        if action not in ("infer", "release"):
            raise ValueError("invalid worker action")
        argv = list(argv)
        if (not argv or len(argv) > 64 or
                any(not isinstance(item, str) for item in argv)):
            raise ValueError("invalid persistent worker argv")
        with self._lock:
            if self.close_reason is not None:
                raise PersistentWorkerError(
                    f"persistent B worker is not reusable ({self.close_reason})",
                    hardware_unknown=False, require_restart=True)
            request_id = uuid.uuid4().hex
            try:
                self.check_manifest()
                started_worker = self._ensure_started()
                request_started = time.perf_counter()
                payload = {
                    "request_id": request_id,
                    "action": action,
                    "argv": argv,
                    "log_path": str(Path(log_path).resolve()),
                }
                encoded = (b"PLD_WORKER_REQUEST " + json.dumps(
                    payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n")
                if len(encoded) > MAX_PROTOCOL_LINE:
                    raise ValueError("persistent worker request exceeds 64 KiB")
                self.process.stdin.write(encoded)
                self.process.stdin.flush()
                deadline = time.monotonic() + (12.0 if action == "release" else self.request_timeout)
                _, response = self._wait_event({"RESULT"}, request_id, deadline)
                if response.get("status") != "success":
                    raise PersistentWorkerError(
                        "persistent B worker returned failure; device state unknown",
                        hardware_unknown=True)
                if action == "release" and response.get("device_released") is not True:
                    raise PersistentWorkerError("worker did not confirm hardware release", hardware_unknown=True)
                self.last_request_seconds = time.perf_counter() - request_started
                alive_after_reply = self.process.poll() is None
                return dict(response, worker_pid=self.process.pid,
                            startup_seconds=(self.last_startup_seconds
                                             if started_worker else 0.0),
                            request_seconds=self.last_request_seconds,
                            alive_after_reply=alive_after_reply)
            except BaseException as exc:
                if not isinstance(exc, PersistentWorkerError):
                    exc = PersistentWorkerError(
                        f"persistent B worker request failed: {type(exc).__name__}: {exc}",
                        hardware_unknown=True)
                self._stop("request_failed")
                self.failure = str(exc)
                self._write_log_failure(log_path, exc)
                raise exc
