"""Host-only supervision tests for the opt-in persistent B worker."""
import hashlib
import base64
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from persistent_worker import PersistentBWorker, PersistentWorkerError


REPOSITORY = Path(__file__).resolve().parents[2]


def encoded_jpeg():
    raw = io.BytesIO()
    Image.fromarray(np.zeros((4, 5, 3), np.uint8)).save(raw, format="JPEG")
    return base64.b64encode(raw.getvalue()).decode("ascii")


def make_runtime(worker_source):
    root = Path(tempfile.mkdtemp(prefix="pld-worker-runtime-"))
    board = root / "board"
    board.mkdir()
    worker_path = board / "infer_worker.py"
    worker_path.write_text(worker_source, encoding="utf-8")
    (board / "infer_images.py").write_text("# fake frozen entry\n", encoding="utf-8")
    (board / "tensor_accel.py").write_text("# fake driver\n", encoding="utf-8")
    files = {
        "board/infer_worker.py": hashlib.sha256(worker_path.read_bytes()).hexdigest(),
        "board/infer_images.py": hashlib.sha256((board / "infer_images.py").read_bytes()).hexdigest(),
        "board/tensor_accel.py": hashlib.sha256((board / "tensor_accel.py").read_bytes()).hexdigest(),
    }
    (root / "deployment_manifest.json").write_text(
        json.dumps({"files": files}) + "\n", encoding="utf-8")
    return root


FAKE_WORKER_HEADER = r'''
import hashlib, json, pathlib, sys, time
root = pathlib.Path(__file__).resolve().parents[1]
manifest = hashlib.sha256((root / "deployment_manifest.json").read_bytes()).hexdigest()
def emit(event, **fields):
    print("PLD_WORKER_" + event + " " + json.dumps(fields, sort_keys=True, separators=(",", ":")), flush=True)
'''


FAKE_SESSION = (
    "class InferenceSession:\n"
    "    def __init__(self):\n"
    "        self.device = None\n"
    "        self.last_reuse = {'model_reused': True, 'hardware_reused': True}\n"
    "    def release_hardware(self):\n"
    "        self.device = None\n"
    "    def close(self):\n"
    "        self.release_hardware()\n"
)


class PersistentWorkerClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.jobs = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_real_worker_protocol_reuses_process_and_writes_job_logs(self):
        runtime = make_runtime((REPOSITORY / "board/infer_worker.py").read_text(encoding="utf-8"))
        (runtime / "pynq.py").write_text(
            "class Overlay: pass\nclass MMIO: pass\ndef allocate(**kwargs): pass\n",
            encoding="utf-8",
        )
        (runtime / "board/infer_images.py").write_text(
            "import json, pathlib, sys\n"
            "sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))\n"
            + FAKE_SESSION + "def main(argv, session=None):\n"
            "    output = pathlib.Path(argv[argv.index('--output') + 1]); output.mkdir(parents=True)\n"
            "    (output / 'frame_000001.json').write_text('{}')\n"
            "    (output / 'frame_000001.bin').write_bytes(b'head')\n"
            "    print('FAKE_INFER_IMAGES_OUTPUT', argv[0])\n",
            encoding="utf-8",
        )
        manifest = json.loads((runtime / "deployment_manifest.json").read_text(encoding="utf-8"))
        manifest["files"]["board/infer_images.py"] = hashlib.sha256(
            (runtime / "board/infer_images.py").read_bytes()).hexdigest()
        (runtime / "deployment_manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

        worker = PersistentBWorker(runtime, startup_timeout=8, request_timeout=8)
        try:
            pids = []
            responses = []
            for index in range(2):
                destination = self.jobs / f"job{index}" / "inference"
                log = self.jobs / f"job{index}" / "inference.log"
                response = worker.run(
                    ["input.jpg", "--bit", str(runtime / "dma_top.bit"),
                     "--output", str(destination)], log)
                self.assertEqual(response["status"], "success")
                self.assertEqual(response["returncode"], 0)
                self.assertTrue((destination / "frame_000001.json").is_file())
                self.assertIn("FAKE_INFER_IMAGES_OUTPUT", log.read_text(encoding="utf-8"))
                pids.append(response["worker_pid"])
                responses.append(response)
            self.assertEqual(pids[0], pids[1])
            self.assertGreater(responses[0]["startup_seconds"], 0.0)
            self.assertEqual(responses[1]["startup_seconds"], 0.0)
            self.assertGreaterEqual(responses[0]["request_seconds"], 0.0)
            self.assertGreaterEqual(responses[1]["request_seconds"], 0.0)
        finally:
            worker.close()
        self.assertIsNotNone(worker.process.poll())
        self.assertEqual(worker.close_reason, "closed_by_service")
        with self.assertRaisesRegex(PersistentWorkerError, "not reusable") as ctx:
            worker.run(["fake"], self.jobs / "after-close.log")
            self.assertFalse(ctx.exception.hardware_unknown)
            self.assertTrue(ctx.exception.require_restart)

    def test_worker_outlives_requesting_http_thread_and_closes_launcher(self):
        runtime = make_runtime((REPOSITORY / "board/infer_worker.py").read_text(encoding="utf-8"))
        (runtime / "pynq.py").write_text(
            "class Overlay: pass\nclass MMIO: pass\ndef allocate(**kwargs): pass\n",
            encoding="utf-8",
        )
        (runtime / "board/infer_images.py").write_text(
            "import pathlib, sys\n"
            "sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))\n"
            + FAKE_SESSION + "def main(argv, session=None):\n"
            "    output = pathlib.Path(argv[argv.index('--output') + 1])\n"
            "    output.mkdir(parents=True)\n"
            "    (output / 'frame_000001.bin').write_bytes(b'head')\n",
            encoding="utf-8",
        )
        manifest = json.loads((runtime / "deployment_manifest.json").read_text(encoding="utf-8"))
        manifest["files"]["board/infer_images.py"] = hashlib.sha256(
            (runtime / "board/infer_images.py").read_bytes()).hexdigest()
        (runtime / "deployment_manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

        worker = PersistentBWorker(runtime, startup_timeout=8, request_timeout=8)
        result = {}

        def first_request():
            result["response"] = worker.run(
                ["input.jpg", "--output", str(self.jobs / "thread-job")],
                self.jobs / "thread-job.log")

        requester = threading.Thread(target=first_request)
        requester.start()
        requester.join(timeout=8)
        self.assertFalse(requester.is_alive())
        first_pid = result["response"]["worker_pid"]
        self.assertTrue(result["response"]["alive_after_reply"])
        self.assertIsNone(worker.process.poll())
        self.assertTrue(worker.launcher_thread.is_alive())
        self.assertIsNot(worker.launcher_thread, requester)

        second = worker.run(["input.jpg", "--output", str(self.jobs / "thread-job-2")],
                            self.jobs / "thread-job-2.log")
        self.assertEqual(second["worker_pid"], first_pid)
        self.assertEqual(second["startup_seconds"], 0.0)
        launcher = worker.launcher_thread
        worker.close()
        self.assertIsNotNone(worker.process.poll())
        self.assertFalse(launcher.is_alive())

    def test_real_worker_inference_exception_fails_closed_with_traceback(self):
        runtime = make_runtime((REPOSITORY / "board/infer_worker.py").read_text(encoding="utf-8"))
        (runtime / "pynq.py").write_text(
            "class Overlay: pass\nclass MMIO: pass\ndef allocate(**kwargs): pass\n",
            encoding="utf-8",
        )
        (runtime / "board/infer_images.py").write_text(
            "import pathlib, sys\n"
            "sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))\n"
            + FAKE_SESSION + "def main(argv, session=None):\n"
            "    raise RuntimeError('inference exploded')\n",
            encoding="utf-8",
        )
        manifest = json.loads((runtime / "deployment_manifest.json").read_text(encoding="utf-8"))
        manifest["files"]["board/infer_images.py"] = hashlib.sha256(
            (runtime / "board/infer_images.py").read_bytes()).hexdigest()
        (runtime / "deployment_manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

        worker = PersistentBWorker(runtime, startup_timeout=8, request_timeout=8)
        log = self.jobs / "exception.log"
        with self.assertRaisesRegex(
                PersistentWorkerError, "inference exception; device state not claimed") as ctx:
            worker.run(["input.jpg", "--output", str(self.jobs / "exception-out")], log)
        self.assertTrue(ctx.exception.hardware_unknown)
        self.assertIsNotNone(worker.process.poll())
        self.assertEqual(worker.close_reason, "request_failed")
        log_text = log.read_text(encoding="utf-8")
        self.assertIn("RuntimeError: inference exploded", log_text)
        self.assertIn("Traceback (most recent call last)", log_text)

    def test_preload_imports_pynq_without_instantiating_overlay(self):
        class Overlay:
            instantiated = 0

            def __init__(self):
                type(self).instantiated += 1

        fake_module = types.ModuleType("pynq")
        fake_module.Overlay = Overlay
        fake_module.MMIO = object
        fake_module.allocate = lambda **kwargs: None
        spec = importlib.util.spec_from_file_location(
            "pld_infer_worker_preload_test", REPOSITORY / "board/infer_worker.py")
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"pynq": fake_module}):
            spec.loader.exec_module(module)
            module.preload_pynq()
        self.assertEqual(Overlay.instantiated, 0)

    def test_linux_parent_death_signal_installation_failure_is_fatal(self):
        spec = importlib.util.spec_from_file_location(
            "pld_infer_worker_pdeath_test", REPOSITORY / "board/infer_worker.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class FailingLib:
            @staticmethod
            def prctl(*args):
                return -1

        with patch.object(module.sys, "platform", "linux"), \
                patch.object(module.ctypes, "CDLL", return_value=FailingLib()):
            with self.assertRaisesRegex(RuntimeError, "PR_SET_PDEATHSIG"):
                module.enable_parent_death_signal()

    def test_startup_failure_is_fail_closed(self):
        runtime = make_runtime("import sys\nprint('not protocol')\nsys.exit(7)\n")
        worker = PersistentBWorker(runtime, startup_timeout=2, request_timeout=2)
        with self.assertRaisesRegex(PersistentWorkerError, "startup failed") as ctx:
            worker.run(["fake"], self.jobs / "failed.log")
            self.assertFalse(ctx.exception.hardware_unknown)
            self.assertTrue(ctx.exception.require_restart)
        self.assertEqual(worker.close_reason, "startup_failed")
        self.assertIsNotNone(worker.process.poll())

    def test_worker_death_before_reply_rejects_job(self):
        source = FAKE_WORKER_HEADER + "emit('READY', manifest_sha256=manifest)\nline=sys.stdin.readline()\nsys.exit(9)\n"
        runtime = make_runtime(source)
        worker = PersistentBWorker(runtime, startup_timeout=3, request_timeout=3)
        log = self.jobs / "death.log"
        with self.assertRaisesRegex(PersistentWorkerError, "exited before reply") as ctx:
            worker.run(["fake"], log)
            self.assertTrue(ctx.exception.hardware_unknown)
        self.assertIsNotNone(worker.process.poll())
        self.assertTrue(log.read_text(encoding="utf-8").startswith("PERSISTENT_WORKER_SUPERVISOR_FAILURE"))

    def test_request_timeout_terminates_worker_and_fails_closed(self):
        source = FAKE_WORKER_HEADER + "emit('READY', manifest_sha256=manifest)\nsys.stdin.readline()\ntime.sleep(10)\n"
        runtime = make_runtime(source)
        worker = PersistentBWorker(runtime, startup_timeout=3, request_timeout=.15)
        started = time.monotonic()
        with self.assertRaisesRegex(PersistentWorkerError, "deadline exceeded") as ctx:
            worker.run(["fake"], self.jobs / "timeout.log")
            self.assertTrue(ctx.exception.hardware_unknown)
        self.assertLess(time.monotonic() - started, 5)
        self.assertIsNotNone(worker.process.poll())

    def test_manifest_drift_refuses_stale_in_memory_worker(self):
        source = FAKE_WORKER_HEADER + "emit('READY', manifest_sha256=manifest)\nline=sys.stdin.readline()\n"
        runtime = make_runtime(source)
        worker = PersistentBWorker(runtime, startup_timeout=3, request_timeout=3)
        worker._ensure_started()
        manifest = json.loads((runtime / "deployment_manifest.json").read_text(encoding="utf-8"))
        manifest["note"] = "deployment changed"
        (runtime / "deployment_manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(PersistentWorkerError, "manifest changed") as ctx:
            worker.run(["fake"], self.jobs / "drift.log")
            self.assertFalse(ctx.exception.hardware_unknown)
            self.assertTrue(ctx.exception.require_restart)
        self.assertEqual(worker.close_reason, "manifest_changed")
        self.assertIsNotNone(worker.process.poll())
        with self.assertRaisesRegex(PersistentWorkerError, "not reusable") as ctx:
            worker.run(["fake"], self.jobs / "drift2.log")
            self.assertFalse(ctx.exception.hardware_unknown)
            self.assertTrue(ctx.exception.require_restart)

    def test_stale_success_reply_is_rejected(self):
        source = FAKE_WORKER_HEADER + (
            "emit('READY', manifest_sha256=manifest)\n"
            "request=json.loads(sys.stdin.readline().split(' ',1)[1])\n"
            "pathlib.Path(request['log_path']).write_text('job output\\n')\n"
            "emit('RESULT', request_id='wrong', status='success', returncode=0)\n"
            "time.sleep(5)\n"
        )
        runtime = make_runtime(source)
        worker = PersistentBWorker(runtime, startup_timeout=3, request_timeout=3)
        with self.assertRaisesRegex(PersistentWorkerError, "stale") as ctx:
            worker.run(["fake"], self.jobs / "stale.log")
            self.assertTrue(ctx.exception.hardware_unknown)
        self.assertIsNotNone(worker.process.poll())

    def test_inference_fatal_reply_requires_restart(self):
        source = FAKE_WORKER_HEADER + (
            "emit('READY', manifest_sha256=manifest)\n"
            "request=json.loads(sys.stdin.readline().split(' ',1)[1])\n"
            "pathlib.Path(request['log_path']).write_text('fatal job\\n')\n"
            "emit('FATAL', request_id=request['request_id'], stage='request', error='inference exception')\n"
            "time.sleep(5)\n"
        )
        runtime = make_runtime(source)
        worker = PersistentBWorker(runtime, startup_timeout=3, request_timeout=3)
        with self.assertRaisesRegex(PersistentWorkerError, "inference exception") as ctx:
            worker.run(["fake"], self.jobs / "fatal.log")
            self.assertTrue(ctx.exception.hardware_unknown)
        self.assertIsNotNone(worker.process.poll())

    def test_oversized_protocol_line_is_bounded(self):
        source = "import sys\nsys.stdout.write('PLD_WORKER_READY ' + 'x'*70000 + '\\n')\nsys.stdout.flush()\nimport time\ntime.sleep(5)\n"
        runtime = make_runtime(source)
        worker = PersistentBWorker(runtime, startup_timeout=3, request_timeout=3)
        with self.assertRaisesRegex(PersistentWorkerError, "64 KiB") as ctx:
            worker.run(["fake"], self.jobs / "large.log")
            self.assertTrue(ctx.exception.hardware_unknown)
        self.assertIsNotNone(worker.process.poll())


class PersistentConsoleIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def test_console_reuses_worker_and_blocks_hdmi_while_request_owns_dma(self):
        import server
        app = server.Console(self.temp.name, token="test-only-token", persistent_b_worker=True)
        app.board = True
        app.runtime = Path(self.temp.name)
        app.runtime_ok = True
        events = []
        instances = []

        class FakeResult:
            stats = {"area_percent": 10.0, "mask_pixels": 2, "valid_pixels": 64,
                     "area_scope": "test"}
            metadata = {"online_b_seconds": .25}
            rgb = np.zeros((4, 5, 3), np.uint8)

            def original_mask(self):
                return np.zeros((4, 5), bool)

        class FakeWorker:
            def __init__(self, runtime):
                instances.append(self)
                self.closed = False
                self.request_count = 0

            def check_manifest(self):
                events.append("manifest_check")

            def run(self, argv, log_path):
                self.argv = argv
                self.log_path = log_path
                events.append("worker_request")
                try:
                    app.start_hdmi(app.active, "mask")
                except RuntimeError:
                    events.append("hdmi_blocked")
                Path(log_path).parent.mkdir(parents=True, exist_ok=True)
                Path(log_path).write_text("fake persistent inference\n", encoding="utf-8")
                startup_started = time.perf_counter()
                if self.request_count == 0:
                    time.sleep(.01)
                startup = (time.perf_counter() - startup_started
                           if self.request_count == 0 else 0.0)
                request_started = time.perf_counter()
                time.sleep(.005)
                request = time.perf_counter() - request_started
                self.request_count += 1
                return {"status": "success", "returncode": 0, "worker_pid": 101,
                        "startup_seconds": startup, "request_seconds": request,
                        "alive_after_reply": True}

            def close(self):
                self.closed = True

        def fake_stop_hdmi():
            events.append("stop_hdmi")

        def fake_command(runtime, source, destination, portable=False):
            return [sys.executable, "infer_images.py", str(source),
                    "--bit", "dma_top.bit", "--output", str(destination)]

        with patch.object(app, "stop_hdmi", side_effect=fake_stop_hdmi), \
                patch.object(app, "verify_deployment", return_value="full_sha256"), \
                patch("server.inference_command", side_effect=fake_command), \
                patch("server.PersistentBWorker", FakeWorker), \
                patch("server.load_result", return_value=FakeResult()):
            jobs = []
            for _ in range(2):
                ident = app.submit({"mode": "infer", "image": encoded_jpeg()})["id"]
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    current = app.job(ident)
                    if current["state"] != "running":
                        break
                    time.sleep(.01)
                jobs.append(app.job(ident))
            app.close()

        self.assertEqual([job["state"] for job in jobs], ["done", "done"])
        self.assertEqual(events, ["stop_hdmi", "worker_request", "hdmi_blocked",
                                  "stop_hdmi", "manifest_check", "worker_request",
                                  "hdmi_blocked"])
        self.assertEqual(len(instances), 1)
        self.assertTrue(instances[0].closed)
        for job in jobs:
            stages = job["statistics"]["stage_seconds"]
            self.assertEqual(stages["inference_backend"], "persistent_b_worker")
            self.assertIn("persistent_worker_request", stages)
            self.assertGreaterEqual(stages["persistent_worker_total"],
                                    stages["persistent_worker_startup"] +
                                    stages["persistent_worker_request"])
            self.assertGreater(jobs[0]["statistics"]["stage_seconds"]["persistent_worker_startup"], 0)
            self.assertEqual(jobs[1]["statistics"]["stage_seconds"]["persistent_worker_startup"], 0)
            self.assertEqual([job["statistics"]["stage_seconds"]["persistent_worker_pid"]
                              for job in jobs], [101, 101])
            self.assertTrue(all(job["statistics"]["stage_seconds"][
                "persistent_worker_alive_after_reply"] for job in jobs))
        folder = Path(self.temp.name) / jobs[0]["id"]
        self.assertEqual((folder / "inference.log").read_text(encoding="utf-8"),
                         "fake persistent inference\n")

    def test_persistent_worker_failure_sets_hardware_fault(self):
        import server
        app = server.Console(self.temp.name, token="test-only-token", persistent_b_worker=True)
        app.board = True
        app.runtime = Path(self.temp.name)
        app.runtime_ok = True

        class FakeWorker:
            def __init__(self, runtime):
                pass

            def check_manifest(self):
                pass

            def run(self, argv, log_path):
                raise PersistentWorkerError("persistent worker failed; device state unknown",
                                            hardware_unknown=True)

            def close(self):
                pass

        with patch.object(app, "stop_hdmi"), patch.object(app, "verify_deployment", return_value="full_sha256"), \
                patch("server.inference_command", return_value=["python", "infer_images.py", "x"]), \
                patch("server.PersistentBWorker", FakeWorker):
            ident = app.submit({"mode": "infer", "image": encoded_jpeg()})["id"]
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                job = app.job(ident)
                if job["state"] != "running":
                    break
                time.sleep(.01)
        self.assertEqual(app.job(ident)["state"], "error")
        self.assertIn("device state unknown", app.fault)
        with self.assertRaisesRegex(RuntimeError, "device state unknown"):
            app.submit({"mode": "demo"})


class PersistentWorkerPackagingTests(unittest.TestCase):
    def test_board_package_includes_worker_in_shared_file_map(self):
        completed = subprocess.run(
            [sys.executable, str(REPOSITORY / "scripts/package_cnn_board.py"),
             "--list-board-files"],
            cwd=REPOSITORY, text=True, capture_output=True, timeout=20, check=False)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        names = [line for line in completed.stdout.splitlines() if line]
        self.assertIn("board/infer_worker.py", names)
        for name in names:
            self.assertTrue((REPOSITORY / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
