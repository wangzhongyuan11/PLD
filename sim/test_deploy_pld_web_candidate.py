"""Host-only contracts for the versioned pld-web candidate switch.

These tests drive the real deploy_pld_web_candidate.py entry point with a fake
systemctl, a fake /proc tree and a local authenticated HTTP endpoint. They do
not access the board, systemd, PYNQ or any FPGA hardware.
"""
import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import deploy_pld_web_candidate as deploy


TOKEN = "host-only-test-token-0123456789"
PID = "4242"
HISTORY = ["job-old-1", "job-old-2"]


FAKE_SYSTEMCTL = r'''import json
import os
import sys

state_path = os.environ["FAKE_SYSTEMCTL_STATE"]
with open(state_path, encoding="utf-8") as handle:
    state = json.load(handle)


def save():
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle)


def effective():
    return "candidate" if os.path.exists(os.environ["FAKE_DROPIN"]) else "original"


def write_proc():
    which = effective()
    folder = os.path.join(os.environ["FAKE_PROC_ROOT"], os.environ.get("FAKE_PID", "4242"))
    os.makedirs(folder, exist_ok=True)
    exec_start = os.environ["FAKE_{}_EXEC".format(which.upper())]
    with open(os.path.join(folder, "cmdline"), "wb") as handle:
        handle.write(exec_start.encode() + b"\0")
    cwd = os.path.join(folder, "cwd")
    target = os.environ["FAKE_{}_CWD".format(which.upper())]
    if os.path.islink(cwd):
        os.unlink(cwd)
    elif os.path.exists(cwd):
        os.remove(cwd)
    try:
        os.symlink(target, cwd)
    except OSError:
        with open(cwd, "w", encoding="utf-8") as handle:
            handle.write(target)


argv = sys.argv[1:]
state.setdefault("calls", []).append(argv)
if state.get("hang_" + argv[0]):
    save()
    import time
    time.sleep(30)
if argv[:1] == ["is-active"]:
    code = state.get("is_active", 0)
    save()
    raise SystemExit(code)
if argv[:1] == ["show"]:
    which = effective()
    values = {"ExecStart": os.environ["FAKE_{}_EXEC".format(which.upper())],
              "WorkingDirectory": os.environ["FAKE_{}_CWD".format(which.upper())],
              "FragmentPath": os.environ["FAKE_FRAGMENT"],
              "DropInPaths": os.environ["FAKE_DROPIN"] if which == "candidate" else "",
              "MainPID": os.environ.get("FAKE_PID", "4242")}
    properties = [part.split("=", 1)[1] for part in argv if part.startswith("--property=")]
    print("\n".join("{}={}".format(name, values.get(name, "")) for name in properties))
    save()
    raise SystemExit(0)
if argv[:1] == ["daemon-reload"]:
    code = state.get("daemon_reload_code", 0)
    save()
    raise SystemExit(code)
if argv[:1] == ["restart"]:
    write_proc()
    code = state.get("restart_code", 0)
    save()
    raise SystemExit(code)
save()
print("unsupported call: " + " ".join(argv))
raise SystemExit(64)
'''


class FakeAPIHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code, payload):
        data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain" if isinstance(payload, bytes)
                         else "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        api = self.server.api
        if self.path == "/":
            return self._send(api.get("root_code", 200), b"ok")
        if self.headers.get("Authorization", "") != "Bearer " + api["token"]:
            return self._send(401, {"error": "unauthorized"})
        if self.path == "/api/status":
            status = api["status"]() if callable(api["status"]) else api["status"]
            return self._send(api.get("status_code", 200), status)
        if self.path == "/api/history":
            return self._send(200, {"items": [{"id": name} for name in api["history_ids"]]})
        return self._send(404, {"error": "missing"})


def healthy_status():
    return {"mode": "board", "inference_available": True, "active": None,
            "hdmi": "off", "fault": None, "runtime_error": None}


class SwitchFixture:
    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="pld-switch-test-"))
        self.original_app = self.root / "PLD_Web"
        self.runtime_original = self.root / "pld" / "final_mode6_fastq_20260929_v3"
        self.candidate_c = self.root / "PLD_Web_png_worker_candidate_v3_20261014"
        self.candidate_b = self.root / "pld" / "final_fastq_worker_candidate_v2_20261013"
        self.systemd_dir = self.root / "etc" / "systemd" / "system"
        self.record_dir = self.root / "var" / "lib" / "pld-web-switch"
        self.proc_root = self.root / "proc"
        self.data = self.original_app / "data"
        self.python = self.root / "venv" / "bin" / "python3"

        (self.original_app / "static").mkdir(parents=True)
        (self.original_app / "server.py").write_text("original stub\n", encoding="utf-8")
        self.data.mkdir(parents=True)
        (self.data / ".access_token").write_text(TOKEN + "\n", encoding="ascii")
        self.runtime_original.mkdir(parents=True)
        self.python.parent.mkdir(parents=True)
        self.python.write_text("#!stub\n", encoding="utf-8")

        candidate_c_hashes = {}
        for relative in deploy.CANDIDATE_C_SHA256:
            source = ROOT / "display_web" / relative
            target = self.candidate_c / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # These are local source bytes used as a synthetic candidate. The
            # production v3 pins remain frozen; this fixture maps its pins to
            # the bytes it actually copied so drift tests still exercise the
            # checker rather than an unrelated source-version mismatch.
            shutil.copyfile(source, target)
            candidate_c_hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
        self.candidate_c_patch = mock.patch.object(
            deploy, "CANDIDATE_C_SHA256", candidate_c_hashes)
        self.candidate_c_patch.start()

        manifest_files = {}
        contents = {"overlay/dma_top.bit": b"bit-bytes", "overlay/dma_top.hwh": b"hwh-bytes",
                    "board/infer_images.py": b"# infer", "board/infer_worker.py": b"# worker",
                    "reference/final_manifest.json": b"model-manifest"}
        for relative, content in contents.items():
            path = self.candidate_b / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            manifest_files[relative] = hashlib.sha256(content).hexdigest()
        manifest = {"schema": "test", "files": manifest_files}
        (self.candidate_b / "deployment_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest_sha = hashlib.sha256(
            (self.candidate_b / "deployment_manifest.json").read_bytes()).hexdigest()
        self.manifest_patch = mock.patch.object(deploy, "CANDIDATE_B_MANIFEST_SHA256", manifest_sha)
        self.manifest_patch.start()

        self.unit_path = self.systemd_dir / deploy.UNIT_NAME
        self.unit_path.parent.mkdir(parents=True, exist_ok=True)
        self.unit_path.write_text(self.unit_text(), encoding="utf-8")
        self.dropin_dir = self.systemd_dir / (deploy.UNIT_NAME + ".d")
        self.dropin_path = self.dropin_dir / deploy.DROPIN_NAME

        self.fake_systemctl = self.root / "bin" / "fake_systemctl.py"
        self.fake_systemctl.parent.mkdir(parents=True)
        self.fake_systemctl.write_text(FAKE_SYSTEMCTL, encoding="utf-8")
        self.state_path = self.root / "systemctl-state.json"
        self.state_path.write_text(json.dumps({"calls": []}), encoding="utf-8")

        self.api = {"token": TOKEN, "history_ids": list(HISTORY), "status": healthy_status()}
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeAPIHandler)
        self.server.api = self.api
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        base_argv = self.argv("preflight")
        context = deploy.Context(deploy.parse_args(base_argv))
        environment = {
            "FAKE_SYSTEMCTL_STATE": str(self.state_path),
            "FAKE_DROPIN": str(self.dropin_path),
            "FAKE_ORIGINAL_EXEC": self.original_exec(),
            "FAKE_CANDIDATE_EXEC": deploy.candidate_exec_start(context),
            "FAKE_ORIGINAL_CWD": str(self.original_app),
            "FAKE_CANDIDATE_CWD": str(self.candidate_c),
            "FAKE_PROC_ROOT": str(self.proc_root),
            "FAKE_FRAGMENT": str(self.unit_path),
            "FAKE_PID": PID,
        }
        self.environment_patch = mock.patch.dict(os.environ, environment)
        self.environment_patch.start()

    def original_exec(self):
        return ("/bin/bash -lc 'exec {python} {server} --host 0.0.0.0 --port 8765 "
                "--data {data} --runtime {runtime}'").format(
            python=self.python, server=self.original_app / "server.py",
            data=self.data, runtime=self.runtime_original)

    def unit_text(self):
        return ("[Unit]\nDescription=PLD image segmentation web console\n\n"
                "[Service]\nType=simple\nUser=root\n"
                "WorkingDirectory={cwd}\nExecStart={exec_start}\n\n"
                "[Install]\nWantedBy=multi-user.target\n").format(
            cwd=self.original_app, exec_start=self.original_exec())

    def argv(self, action, extra=()):
        arguments = [action,
                     "--original-app", str(self.original_app),
                     "--candidate-c", str(self.candidate_c),
                     "--candidate-b", str(self.candidate_b),
                     "--python", str(self.python),
                     "--systemd-dir", str(self.systemd_dir),
                     "--record-dir", str(self.record_dir),
                     "--proc-root", str(self.proc_root),
                     "--systemctl", '"{}" "{}"'.format(sys.executable, self.fake_systemctl),
                     "--http-base", "http://127.0.0.1:{}".format(self.server.server_port),
                     "--startup-timeout", "3", "--http-timeout", "2",
                     "--no-platform-check"]
        arguments += list(extra)
        return arguments

    def calls(self):
        return json.loads(self.state_path.read_text(encoding="utf-8"))["calls"]

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.environment_patch.stop()
        self.candidate_c_patch.stop()
        self.manifest_patch.stop()
        shutil.rmtree(self.root, ignore_errors=True)


def run_main(argv):
    output, error = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
        code = deploy.main(argv)
    return code, output.getvalue(), error.getvalue()


class DeploySwitchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SwitchFixture()

    def tearDown(self):
        self.fixture.close()

    def assert_no_token(self, *texts):
        for text in texts:
            self.assertNotIn(TOKEN, text)

    def records(self):
        return sorted(self.fixture.record_dir.glob("apply-*.json"))

    def test_preflight_passes_without_side_effects_or_token_leak(self):
        code, out, err = run_main(self.fixture.argv("preflight"))
        self.assertEqual(code, 0)
        self.assertIn("PREFLIGHT_PASS", out)
        self.assertIn("idle, HDMI off, fault empty", out)
        self.assertFalse(self.fixture.dropin_dir.exists())
        self.assertEqual(self.records(), [])
        self.assertEqual(self.fixture.calls()[0][0], "is-active")
        self.assertTrue(all(call[0] != "restart" for call in self.fixture.calls()))
        self.assert_no_token(out, err)

    def test_preflight_rejects_candidate_c_hash_drift(self):
        server = self.fixture.candidate_c / "server.py"
        server.write_bytes(server.read_bytes() + b"\n")
        code, out, err = run_main(self.fixture.argv("preflight"))
        self.assertEqual(code, 1)
        self.assertIn("candidate C checksum mismatch for server.py", err)
        self.assertFalse(self.fixture.dropin_dir.exists())
        self.assertTrue(all(call[0] not in ("restart", "daemon-reload")
                            for call in self.fixture.calls()))
        self.assert_no_token(out, err)

    def test_style_css_pin_targets_actual_candidate_bytes(self):
        source = ROOT / "display_web" / "static" / "style.css"
        content = source.read_bytes()
        self.assertEqual(hashlib.sha256(content).hexdigest(),
                         deploy.CANDIDATE_C_SHA256["static/style.css"])
        # This fixture pins the current checkout's raw bytes (see SwitchFixture).
        # A clean checkout can contain LF; the frozen production v3 pins above
        # intentionally describe a different historical deployment. Hash drift
        # rejection is tested independently and must not assume checkout EOLs.
        self.assertEqual((self.fixture.candidate_c / "static" / "style.css").read_bytes(),
                         content)

    def test_preflight_rejects_candidate_b_manifest_drift(self):
        path = self.fixture.candidate_b / "deployment_manifest.json"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        code, out, err = run_main(self.fixture.argv("preflight"))
        self.assertEqual(code, 1)
        self.assertIn("candidate B manifest checksum mismatch", err)
        self.assert_no_token(out, err)

    def test_preflight_rejects_candidate_b_file_drift(self):
        target = self.fixture.candidate_b / "board" / "infer_worker.py"
        target.write_bytes(b"# drifted\n")
        code, out, err = run_main(self.fixture.argv("preflight"))
        self.assertEqual(code, 1)
        self.assertIn("candidate B checksum mismatch for board/infer_worker.py", err)
        self.assert_no_token(out, err)

    def test_preflight_rejects_busy_hdmi_or_faulted_service(self):
        variants = {"active job": dict(active="job-1"),
                    "hdmi showing": dict(hdmi="showing"),
                    "fault": dict(fault="hardware fault")}
        for name, changes in variants.items():
            with self.subTest(name=name):
                status = healthy_status()
                status.update(changes)
                self.fixture.api["status"] = status
                code, out, err = run_main(self.fixture.argv("preflight"))
                self.assertEqual(code, 1)
                self.assertIn("DEPLOY_FAILED", err)
                self.assertFalse(self.fixture.dropin_dir.exists())
                self.assert_no_token(out, err)

    def test_preflight_rejects_unexpected_effective_exec_start(self):
        drifted = self.fixture.original_exec().replace(
            "--data " + str(self.fixture.data), "--data " + str(self.fixture.root / "other-data"))
        with mock.patch.dict(os.environ, {"FAKE_ORIGINAL_EXEC": drifted}):
            code, out, err = run_main(self.fixture.argv("preflight"))
        self.assertEqual(code, 1)
        self.assertIn("effective ExecStart does not contain --data", err)
        self.assertTrue(all(call[0] not in ("restart", "daemon-reload")
                            for call in self.fixture.calls()))
        self.assert_no_token(out, err)

    def test_foreign_data_directory_is_rejected_before_any_side_effect(self):
        foreign = self.fixture.root / "candidate-owned-data"
        foreign.mkdir()
        code, out, err = run_main(self.fixture.argv("preflight", ["--data", str(foreign)]))
        self.assertEqual(code, 1)
        self.assertIn("--data must stay the original application data directory", err)
        self.assertEqual(self.fixture.calls(), [])
        self.assertFalse(self.fixture.dropin_dir.exists())
        self.assertEqual(self.records(), [])
        self.assertTrue((self.fixture.data / ".access_token").is_file())
        self.assert_no_token(out, err)

    def test_preflight_fails_bounded_when_systemctl_hangs(self):
        state = json.loads(self.fixture.state_path.read_text(encoding="utf-8"))
        state["hang_is-active"] = True
        self.fixture.state_path.write_text(json.dumps(state), encoding="utf-8")
        started = time.monotonic()
        code, out, err = run_main(self.fixture.argv("preflight", ["--systemctl-timeout", "1"]))
        elapsed = time.monotonic() - started
        self.assertEqual(code, 1)
        self.assertIn("systemctl is-active did not finish within 1s", err)
        self.assertLess(elapsed, 20)
        self.assertEqual(self.records(), [])
        self.assertFalse(self.fixture.dropin_dir.exists())
        self.assert_no_token(out, err)

    def test_apply_hanging_restart_fails_bounded_and_reports_unknown(self):
        state = json.loads(self.fixture.state_path.read_text(encoding="utf-8"))
        state["hang_restart"] = True
        self.fixture.state_path.write_text(json.dumps(state), encoding="utf-8")
        started = time.monotonic()
        code, out, err = run_main(self.fixture.argv("apply", ["--restart-timeout", "1"]))
        elapsed = time.monotonic() - started
        self.assertEqual(code, 2)
        self.assertIn("APPLY_FAILED_RESTORE_FAILED", err)
        self.assertIn("systemctl restart did not finish within 1s", err)
        self.assertIn("UNKNOWN", err)
        self.assertLess(elapsed, 20)
        self.assertFalse(self.fixture.dropin_path.exists())
        self.assertEqual(self.fixture.unit_path.read_text(encoding="utf-8"),
                         self.fixture.unit_text())
        record = json.loads(self.records()[0].read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "failed_restore_unknown")
        self.assert_no_token(out, err)

    def test_apply_refuses_existing_dropin(self):
        self.fixture.dropin_dir.mkdir(parents=True)
        self.fixture.dropin_path.write_text("[Service]\nExecStart=/bin/true\n", encoding="utf-8")
        code, out, err = run_main(self.fixture.argv("apply"))
        self.assertEqual(code, 1)
        self.assertIn("candidate drop-in is already active; use rollback instead", err)
        self.assertEqual(self.fixture.dropin_path.read_text(encoding="utf-8"),
                         "[Service]\nExecStart=/bin/true\n")
        self.assertTrue(all(call[0] != "restart" for call in self.fixture.calls()))
        self.assert_no_token(out, err)

    def test_apply_creates_versioned_dropin_and_preserves_original(self):
        self.fixture.dropin_dir.mkdir(parents=True)
        other = self.fixture.dropin_dir / "10-local.conf"
        other.write_text("[Service]\nEnvironment=PLD_KEEP=1\n", encoding="utf-8")
        unit_before = self.fixture.unit_path.read_bytes()
        token_before = (self.fixture.data / ".access_token").read_bytes()
        code, out, err = run_main(self.fixture.argv("apply"))
        self.assertEqual(code, 0)
        self.assertIn("APPLY_PASS", out)
        self.assertIn("rollback --record", out)
        self.assert_no_token(out, err)

        content = self.fixture.dropin_path.read_text(encoding="utf-8")
        self.assertIn("WorkingDirectory={}".format(self.fixture.candidate_c), content)
        self.assertIn("--data {}".format(self.fixture.data), content)
        self.assertIn("--port 8765", content)
        self.assertIn("--runtime {}".format(self.fixture.candidate_b), content)
        self.assertIn("--persistent-b-worker", content)
        self.assertIn(str(self.fixture.python), content)
        self.assertIn("ExecStart=\n", content)
        self.assertEqual(self.fixture.unit_path.read_bytes(), unit_before)
        self.assertEqual((self.fixture.data / ".access_token").read_bytes(), token_before)
        self.assertFalse((self.fixture.candidate_c / "data").exists())
        self.assertEqual(other.read_text(encoding="utf-8"), "[Service]\nEnvironment=PLD_KEEP=1\n")

        records = self.records()
        self.assertEqual(len(records), 1)
        record = json.loads(records[0].read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "applied")
        self.assertEqual(record["candidate"]["dropin_path"], str(self.fixture.dropin_path))
        self.assertEqual(record["history_ids"], HISTORY)
        self.assertIn("backup-", record.get("backup_dir", ""))
        self.assert_no_token(records[0].read_text(encoding="utf-8"))
        calls = self.fixture.calls()
        self.assertEqual([call for call in calls if call[0] == "restart"],
                         [["restart", deploy.UNIT_NAME]])
        self.assertEqual([call for call in calls if call[0] == "daemon-reload"],
                         [["daemon-reload"]])

    def test_apply_failure_restores_original_dropin_and_service(self):
        state = {"restarts_when_faulty": 1}

        def status():
            restarts = sum(1 for call in self.fixture.calls() if call[0] == "restart")
            status_value = healthy_status()
            if restarts == state["restarts_when_faulty"]:
                status_value["fault"] = "candidate failed to become healthy"
            return status_value

        self.fixture.api["status"] = status
        code, out, err = run_main(self.fixture.argv("apply"))
        self.assertEqual(code, 1)
        self.assertIn("APPLY_FAILED_RESTORED", err)
        self.assertIn("candidate failed to become healthy", err)
        self.assertFalse(self.fixture.dropin_path.exists())
        self.assertEqual([call for call in self.fixture.calls() if call[0] == "restart"],
                         [["restart", deploy.UNIT_NAME], ["restart", deploy.UNIT_NAME]])
        self.assertEqual(len([call for call in self.fixture.calls()
                              if call[0] == "daemon-reload"]), 2)
        record = json.loads(self.records()[0].read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "failed_restored")
        self.assert_no_token(out, err)

    def test_apply_failure_with_failed_restore_reports_unknown_ownership(self):
        state = json.loads(self.fixture.state_path.read_text(encoding="utf-8"))
        state["restart_code"] = 1
        self.fixture.state_path.write_text(json.dumps(state), encoding="utf-8")
        code, out, err = run_main(self.fixture.argv("apply"))
        self.assertEqual(code, 2)
        self.assertIn("APPLY_FAILED_RESTORE_FAILED", err)
        self.assertIn("UNKNOWN", err)
        self.assertFalse(self.fixture.dropin_path.exists())
        record = json.loads(self.records()[0].read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "failed_restore_unknown")
        self.assert_no_token(out, err)

    def test_rollback_restores_original_and_updates_record(self):
        code, out, err = run_main(self.fixture.argv("apply"))
        self.assertEqual(code, 0)
        self.assertTrue(self.fixture.dropin_path.exists())
        code, out, err = run_main(self.fixture.argv("rollback"))
        self.assertEqual(code, 0)
        self.assertIn("ROLLBACK_PASS", out)
        self.assertIn("candidate directories were kept", out)
        self.assertFalse(self.fixture.dropin_path.exists())
        self.assertTrue(self.fixture.candidate_c.is_dir())
        self.assertTrue(self.fixture.candidate_b.is_dir())
        self.assertTrue(self.fixture.data.is_dir())
        record = json.loads(self.records()[0].read_text(encoding="utf-8"))
        self.assertEqual(record["state"], "rolled_back")
        restarts = [call for call in self.fixture.calls() if call[0] == "restart"]
        self.assertEqual(restarts, [["restart", deploy.UNIT_NAME], ["restart", deploy.UNIT_NAME]])
        self.assert_no_token(out, err)

    def test_rollback_refuses_busy_service_and_keeps_dropin(self):
        code, out, err = run_main(self.fixture.argv("apply"))
        self.assertEqual(code, 0)
        status = healthy_status()
        status["hdmi"] = "showing"
        self.fixture.api["status"] = status
        code, out, err = run_main(self.fixture.argv("rollback"))
        self.assertEqual(code, 1)
        self.assertIn("HDMI is showing", err)
        self.assertTrue(self.fixture.dropin_path.exists())
        self.assertEqual([call for call in self.fixture.calls() if call[0] == "restart"],
                         [["restart", deploy.UNIT_NAME]])
        self.assert_no_token(out, err)


if __name__ == "__main__":
    unittest.main()
