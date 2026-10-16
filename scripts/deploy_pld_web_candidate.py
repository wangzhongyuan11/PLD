"""Reviewable, reversible switch of pld-web.service to the combined candidate.

Candidate combination (board-isolated, not yet the live service):
  C: /home/xilinx/jupyter_notebooks/PLD_Web_png_worker_candidate_v3_20261014
  B: /home/xilinx/jupyter_notebooks/pld/final_fastq_worker_candidate_v2_20261013

The original unit file, the frozen runtime package and the original data
directory are never modified or replaced. apply adds exactly one versioned
systemd drop-in overriding WorkingDirectory/ExecStart only: same PYNQ venv
Python, same root login-shell environment, same port 8765, the ORIGINAL
PLD_Web/data directory (access token, history, user results), and the explicit
--persistent-b-worker flag. Every step is verified with bounded timeouts; any
failure removes the drop-in, restarts the original service and re-verifies it.

Board use (PYNQ venv Python as root):
  sudo -E /usr/local/share/pynq-venv/bin/python3 deploy_pld_web_candidate.py preflight
  sudo -E /usr/local/share/pynq-venv/bin/python3 deploy_pld_web_candidate.py apply
  sudo -E /usr/local/share/pynq-venv/bin/python3 deploy_pld_web_candidate.py rollback

apply writes a JSON record without the access token under the record directory;
rollback consumes that record. Candidate directories and user data are never
deleted. The access token is read into memory only and never printed or stored.
Every systemctl call and HTTP request is bounded; --restart-timeout defaults to
300 s so it covers the unit's TimeoutStopSec=240 plus restart overhead, while
--startup-timeout is the separate post-restart verification window. The data
directory is pinned to ORIGINAL_APP/data: any other --data value is rejected
before any systemctl call, HTTP request or file modification.
Host regression tests drive the same code with --no-platform-check, a fake
systemctl and a local HTTP endpoint.
"""
import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


UNIT_NAME = "pld-web.service"
CANDIDATE_VERSION = "png-worker-v3-20261014"
DROPIN_NAME = "50-pld-web-" + CANDIDATE_VERSION + ".conf"

DEFAULT_ORIGINAL_APP = Path("/home/xilinx/jupyter_notebooks/PLD_Web")
DEFAULT_CANDIDATE_C = Path("/home/xilinx/jupyter_notebooks/PLD_Web_png_worker_candidate_v3_20261014")
DEFAULT_CANDIDATE_B = Path("/home/xilinx/jupyter_notebooks/pld/final_fastq_worker_candidate_v2_20261013")
DEFAULT_PYTHON = Path("/usr/local/share/pynq-venv/bin/python3")
DEFAULT_SYSTEMD_DIR = Path("/etc/systemd/system")
DEFAULT_RECORD_DIR = Path("/var/lib/pld-web-switch")
DEFAULT_HTTP_BASE = "http://127.0.0.1:8765"
DEFAULT_PORT = 8765
DEFAULT_PID = 0

# SHA256 of the candidate C core source files as actually deployed in the
# board-isolated candidate directory (verified byte-identical to this working
# tree by read-only SFTP). These are REAL FILE BYTES, not Git-normalized blobs:
# static/style.css contains one CRLF among LF lines, so its hash intentionally
# differs from the normalized Git blob (4fa78b1c…32ceb6). Do not "fix" that
# pin by normalizing; update it only for a new reviewed candidate.
CANDIDATE_C_SHA256 = {
    "server.py": "4facbcc5d5e385415ead0529e1db21ce704a3f3d0f2fec8f2c58b343383c8835",
    "contour.py": "e09500096a25ecf40dc613adcaad47c5fd88a23f0975d2b23673627f18f082e2",
    "pld_result.py": "5044b8133f0a19c9ccca0f1702d0f31ab56a95461f6187e34c3a89388dd31b09",
    "infer_then_display.py": "9910df8da68c81c05d533e1476256998d13ae155eceae698b12296dfeb4e7523",
    "persistent_worker.py": "931b4ca4f618e5256c8a1c2a0c8f37cc0f46e6a97e9f22c9601fa7d9b558ed0f",
    "pld_hdmi.py": "d46bb71fec0b7192cfb11e937dc17546e66e58c02dc8444c763cf8824e501466",
    "hdmi_worker.py": "5e36b9d44a1d827aff85c46750d45e905bedb8a5f926298dbdeca5ada658e939",
    "hdmi_buttons.py": "bcd11308daa08f51cbe97086bc23fab172f91e8913efc793e22842a828c898f9",
    "show_result.py": "00246da054a574f9910c64325d95d53dbf1925a854386abbf6fb35427b6e0011",
    "static/app.js": "a9c564c3d83bb8b26cfef7cba2c15e457283c16a0999da057c78afa443fc72de",
    "static/index.html": "b362030e4c084bddae3dfadbbc9f4176f70942b1d37d7746bf1453e871d76c37",
    "static/style.css": "ca38cc0154d41942a17c7d3119f728e03fef1092f5aba7185691e0f2e3931a0a",
    "static/favicon.svg": "19b69d1b2e86b42e32b6d3c213ff0b12cfea9989765ba2b604e09df22d27e417",
}

# SHA256 of the candidate B deployment_manifest.json itself; every file listed
# inside the manifest is then verified exactly like infer_then_display does.
CANDIDATE_B_MANIFEST_SHA256 = "b80c8813dc56f32d1bfbb35161f91a5092cdfb6f27044be9ef998cff3dd1ab55"
REQUIRED_B_PATHS = ("overlay/dma_top.bit", "overlay/dma_top.hwh",
                    "board/infer_images.py", "board/infer_worker.py")


class DeployError(RuntimeError):
    def __init__(self, message, exit_code=1):
        super().__init__(message)
        self.exit_code = exit_code


class Context:
    def __init__(self, args):
        self.args = args
        self.unit = UNIT_NAME
        self.original_app = args.original_app.resolve()
        self.candidate_c = args.candidate_c.resolve()
        self.candidate_b = args.candidate_b.resolve()
        expected_data = (self.original_app / "data").resolve()
        self.data = (args.data.resolve() if args.data is not None else expected_data)
        if self.data != expected_data:
            # Refuse before any systemctl call, HTTP request or file change: the
            # original access token/history directory must never be switched.
            raise DeployError("--data must stay the original application data directory "
                              "{} ; got {}".format(expected_data, self.data))
        self.python = args.python
        self.systemd_dir = args.systemd_dir
        self.record_dir = args.record_dir
        self.systemctl = shlex.split(str(args.systemctl))
        self.http_base = args.http_base
        self.port = args.port
        self.proc_root = args.proc_root
        self.startup_timeout = args.startup_timeout
        self.http_timeout = args.http_timeout
        self.systemctl_timeout = args.systemctl_timeout
        self.restart_timeout = args.restart_timeout
        self.dropin_dir = self.systemd_dir / (self.unit + ".d")
        self.dropin_path = self.dropin_dir / args.dropin_name


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_text(path, text, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_write_json(path, payload):
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def systemctl(ctx, *arguments, check=True):
    command = [*ctx.systemctl, *arguments]
    action = arguments[0] if arguments else "call"
    timeout = ctx.restart_timeout if action == "restart" else ctx.systemctl_timeout
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise DeployError("systemctl {} did not finish within {:.0f}s; service state is "
                          "unclear".format(action, timeout))
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise DeployError("systemctl {} failed ({}): {}".format(
            arguments[0], result.returncode, detail))
    return result


def unit_properties(ctx, properties):
    arguments = ["show", ctx.unit]
    arguments += ["--property=" + name for name in properties]
    result = systemctl(ctx, *arguments)
    values = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            name, value = line.split("=", 1)
            values[name] = value
    return values


def http_get(ctx, path, token=None):
    url = ctx.http_base.rstrip("/") + path
    request = urllib.request.Request(url)
    if token is not None:
        request.add_header("Authorization", "Bearer " + token)
    try:
        with urllib.request.urlopen(request, timeout=ctx.http_timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        raise DeployError("HTTP {} from {}".format(exc.code, url))
    except urllib.error.URLError as exc:
        raise DeployError("HTTP request failed for {}: {}".format(url, exc.reason))


def read_access_token(data_dir):
    path = data_dir / ".access_token"
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise DeployError("cannot read access token file {}: {}".format(path, exc))
    if not value:
        raise DeployError("access token file is empty: {}".format(path))
    return value


def api_status(ctx, token):
    code, raw = http_get(ctx, "/api/status", token)
    try:
        payload = json.loads(raw)
    except ValueError:
        raise DeployError("/api/status returned invalid JSON")
    if not isinstance(payload, dict):
        raise DeployError("/api/status returned unexpected payload")
    return payload


def history_ids(ctx, token):
    code, raw = http_get(ctx, "/api/history", token)
    try:
        payload = json.loads(raw)
        items = payload["items"]
        return [item["id"] for item in items]
    except (ValueError, KeyError, TypeError) as exc:
        raise DeployError("/api/history returned unexpected payload: {}".format(exc))


def require_idle(status):
    if status.get("mode") != "board":
        raise DeployError("web console is not in board mode: {}".format(status.get("mode")))
    if not status.get("inference_available"):
        detail = status.get("fault") or status.get("runtime_error") or "unknown reason"
        raise DeployError("board inference unavailable: {}".format(detail))
    if status.get("active") is not None:
        raise DeployError("a processing job is active ({}); wait for it to finish".format(status.get("active")))
    if status.get("hdmi") != "off":
        raise DeployError("HDMI is {} ; stop it before switching".format(status.get("hdmi")))
    if status.get("fault"):
        raise DeployError("service fault is set: {}".format(status.get("fault")))


def verify_candidate_c(candidate):
    for relative, expected in sorted(CANDIDATE_C_SHA256.items()):
        path = candidate / relative
        if not path.is_file():
            raise DeployError("candidate C is missing {}".format(relative))
        actual = sha256_file(path)
        if actual != expected:
            raise DeployError("candidate C checksum mismatch for {} (expected {}, got {})".format(
                relative, expected[:16] + "…", actual[:16] + "…"))


def verify_candidate_b(candidate):
    manifest_path = candidate / "deployment_manifest.json"
    if not manifest_path.is_file():
        raise DeployError("candidate B is missing deployment_manifest.json")
    actual = sha256_file(manifest_path)
    if actual != CANDIDATE_B_MANIFEST_SHA256:
        raise DeployError("candidate B manifest checksum mismatch (expected {}, got {})".format(
            CANDIDATE_B_MANIFEST_SHA256[:16] + "…", actual[:16] + "…"))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["files"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DeployError("candidate B manifest is unreadable: {}".format(exc))
    if not isinstance(files, dict) or not files:
        raise DeployError("candidate B manifest has no file map")
    root = candidate.resolve()
    for name, expected in sorted(files.items()):
        target = (candidate / name).resolve()
        if root != target and root not in target.parents:
            raise DeployError("candidate B manifest path escapes deployment: {}".format(name))
        if not target.is_file():
            raise DeployError("candidate B is missing manifest entry {}".format(name))
        if sha256_file(target) != expected:
            raise DeployError("candidate B checksum mismatch for {}".format(name))
    for relative in REQUIRED_B_PATHS:
        if not (candidate / relative).is_file():
            raise DeployError("candidate B is missing required {}".format(relative))
    return len(files)


def exec_start_lines(unit_text):
    return [line[len("ExecStart="):].strip() for line in unit_text.splitlines()
            if line.strip().startswith("ExecStart=")]


def check_original_unit(ctx, properties):
    if ctx.dropin_path.name in properties.get("DropInPaths", ""):
        raise DeployError("candidate drop-in is already active; use rollback instead")
    if ctx.dropin_path.exists():
        raise DeployError("drop-in already exists: {}".format(ctx.dropin_path))
    fragment = Path(properties.get("FragmentPath") or (ctx.systemd_dir / ctx.unit))
    if not fragment.is_file():
        raise DeployError("original unit file not found: {}".format(fragment))
    lines = exec_start_lines(fragment.read_text(encoding="utf-8"))
    if not lines:
        raise DeployError("original unit has no ExecStart: {}".format(fragment))
    effective = properties.get("ExecStart", "")
    working_directory = properties.get("WorkingDirectory", "")
    for name, value in (("ExecStart", lines[0]), ("effective ExecStart", effective)):
        for forbidden in (str(ctx.candidate_c), str(ctx.candidate_b), "--persistent-b-worker"):
            if forbidden in value:
                raise DeployError("original {} already references {}; refusing to switch twice".format(name, forbidden))
    for name, value in (("ExecStart", lines[0]), ("effective ExecStart", effective)):
        for needle in (str(ctx.original_app / "server.py"), "--data " + str(ctx.data),
                       "--port " + str(ctx.port)):
            if needle not in value:
                raise DeployError("original {} does not contain {}".format(name, needle))
    if working_directory != str(ctx.original_app):
        raise DeployError("effective WorkingDirectory is {} , expected {}".format(
            working_directory, ctx.original_app))
    return fragment


def existing_dropins(ctx):
    if not ctx.dropin_dir.is_dir():
        return []
    records = []
    for path in sorted(ctx.dropin_dir.glob("*.conf")):
        records.append({"name": path.name, "path": str(path), "sha256": sha256_file(path)})
    return records


def preflight(ctx):
    if not ctx.data.is_dir():
        raise DeployError("original data directory not found: {}".format(ctx.data))
    token = read_access_token(ctx.data)
    if systemctl(ctx, "is-active", "--quiet", ctx.unit, check=False).returncode != 0:
        raise DeployError("{} is not active".format(ctx.unit))
    code, _ = http_get(ctx, "/")
    if code != 200:
        raise DeployError("web console root returned HTTP {}".format(code))
    status = api_status(ctx, token)
    require_idle(status)
    history = history_ids(ctx, token)
    verify_candidate_c(ctx.candidate_c)
    candidate_b_files = verify_candidate_b(ctx.candidate_b)
    properties = unit_properties(ctx, ("ExecStart", "WorkingDirectory", "FragmentPath",
                                       "DropInPaths", "MainPID"))
    fragment = check_original_unit(ctx, properties)
    dropins = existing_dropins(ctx)
    if ctx.candidate_c == ctx.original_app or ctx.data == ctx.candidate_c:
        raise DeployError("candidate C must not replace the original application or data directory")
    if not ctx.python.is_file():
        raise DeployError("PYNQ venv Python not found: {}".format(ctx.python))
    facts = {
        "fragment_path": str(fragment),
        "effective_exec_start": properties.get("ExecStart", ""),
        "effective_working_directory": properties.get("WorkingDirectory", ""),
        "existing_dropins": dropins,
        "history_ids": history,
        "candidate_c_files": len(CANDIDATE_C_SHA256),
        "candidate_b_files": candidate_b_files,
    }
    print("PREFLIGHT_PASS {}".format(CANDIDATE_VERSION))
    print("  unit={} active, HTTP root 200, idle, HDMI off, fault empty".format(ctx.unit))
    print("  original app={} data={}".format(ctx.original_app, ctx.data))
    print("  candidate C files={} candidate B manifest files={}".format(
        facts["candidate_c_files"], facts["candidate_b_files"]))
    print("  fragment={} existing_dropins={} history_items={}".format(
        fragment, len(dropins), len(history)))
    return facts


def candidate_exec_start(ctx):
    inner = ("exec {python} {server} --host 0.0.0.0 --port {port} --data {data} "
             "--runtime {runtime} --persistent-b-worker").format(
        python=ctx.python, server=ctx.candidate_c / "server.py", port=ctx.port,
        data=ctx.data, runtime=ctx.candidate_b)
    return "/bin/bash -lc '" + inner + "'"


def dropin_text(ctx):
    return ("# Generated by deploy_pld_web_candidate.py; version {version}\n"
            "# Rollback removes this file, runs daemon-reload and restarts the original unit.\n"
            "[Service]\n"
            "WorkingDirectory={cwd}\n"
            "ExecStart=\n"
            "ExecStart={exec_start}\n").format(version=CANDIDATE_VERSION,
                                               cwd=ctx.candidate_c,
                                               exec_start=candidate_exec_start(ctx))


def process_cmdline(ctx, pid):
    path = ctx.proc_root / str(pid) / "cmdline"
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise DeployError("cannot read {} : {}".format(path, exc))
    return " ".join(part.decode(errors="replace") for part in raw.split(b"\0") if part)


def process_cwd(ctx, pid):
    path = ctx.proc_root / str(pid) / "cwd"
    try:
        return Path(os.readlink(path))
    except OSError:
        # Boards expose a symlink; host regression tests use a plain file.
        try:
            return Path(path.read_text(encoding="utf-8").strip())
        except OSError as exc:
            raise DeployError("cannot read {} : {}".format(path, exc))


def comparable_path(path):
    """Normalize for comparison; strips Windows extended-length prefixes."""
    text = os.path.normpath(str(path))
    if text.startswith("\\\\?\\UNC\\"):
        return Path("\\\\" + text[8:])
    if text.startswith("\\\\?\\"):
        return Path(text[4:])
    return Path(text)


def candidate_expectations(ctx):
    return {
        "cwd": ctx.candidate_c,
        "required": [str(ctx.candidate_c / "server.py"), "--data " + str(ctx.data),
                     "--port " + str(ctx.port), str(ctx.candidate_b),
                     "--persistent-b-worker"],
        "forbidden": [str(ctx.original_app / "server.py")],
        "exec_start_needle": str(ctx.candidate_c / "server.py"),
    }


def original_expectations(ctx):
    return {
        "cwd": ctx.original_app,
        "required": [str(ctx.original_app / "server.py"), "--data " + str(ctx.data),
                     "--port " + str(ctx.port), str(ctx.original_app)],
        "forbidden": ["--persistent-b-worker", str(ctx.candidate_c), str(ctx.candidate_b)],
        "exec_start_needle": str(ctx.original_app / "server.py"),
    }


def check_service_once(ctx, expectations, history_before, token):
    if systemctl(ctx, "is-active", "--quiet", ctx.unit, check=False).returncode != 0:
        raise DeployError("{} is not active".format(ctx.unit))
    properties = unit_properties(ctx, ("ExecStart", "WorkingDirectory", "MainPID"))
    try:
        pid = int(properties.get("MainPID", "0") or 0)
    except ValueError:
        pid = 0
    if pid <= 0:
        raise DeployError("MainPID is not a running process")
    cmdline = process_cmdline(ctx, pid)
    for needle in expectations["required"]:
        if needle not in cmdline:
            raise DeployError("service cmdline lacks {}".format(needle))
    for needle in expectations["forbidden"]:
        if needle in cmdline:
            raise DeployError("service cmdline unexpectedly contains {}".format(needle))
    cwd = comparable_path(process_cwd(ctx, pid))
    expected = comparable_path(expectations["cwd"].resolve())
    if cwd != expected:
        raise DeployError("service cwd is {} , expected {}".format(cwd, expectations["cwd"]))
    if expectations["exec_start_needle"] not in properties.get("ExecStart", ""):
        raise DeployError("effective ExecStart does not reference {}".format(
            expectations["exec_start_needle"]))
    code, _ = http_get(ctx, "/")
    if code != 200:
        raise DeployError("web console root returned HTTP {}".format(code))
    status = api_status(ctx, token)
    require_idle(status)
    after = set(history_ids(ctx, token))
    missing = set(history_before) - after
    if missing:
        raise DeployError("original history entries disappeared: {}".format(sorted(missing)))


def verify_service(ctx, expectations, history_before, token, timeout):
    deadline = time.monotonic() + timeout
    failure = None
    while True:
        try:
            check_service_once(ctx, expectations, history_before, token)
            return
        except DeployError as exc:
            failure = exc
            if time.monotonic() >= deadline:
                raise DeployError("service verification failed within {:.0f}s: {}".format(timeout, exc))
            time.sleep(1.0)


def restore_original(ctx, record, token):
    dropin = Path(record["candidate"]["dropin_path"])
    if dropin.exists():
        dropin.unlink()
    systemctl(ctx, "daemon-reload")
    systemctl(ctx, "restart", ctx.unit)
    verify_service(ctx, original_expectations(ctx), record.get("history_ids", []),
                   token, ctx.startup_timeout)


def action_apply(ctx):
    facts = preflight(ctx)
    token = read_access_token(ctx.data)
    created_ns = time.time_ns()
    content = dropin_text(ctx)
    record = {
        "schema": 1,
        "state": "prepared",
        "created_ns": created_ns,
        "version": CANDIDATE_VERSION,
        "unit": ctx.unit,
        "original": {
            "app": str(ctx.original_app),
            "data": str(ctx.data),
            "python": str(ctx.python),
            "fragment_path": facts["fragment_path"],
            "exec_start": facts["effective_exec_start"],
            "working_directory": facts["effective_working_directory"],
        },
        "existing_dropins": facts["existing_dropins"],
        "candidate": {
            "c": str(ctx.candidate_c),
            "b": str(ctx.candidate_b),
            "dropin_path": str(ctx.dropin_path),
            "dropin_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "dropin_content": content,
        },
        "history_ids": facts["history_ids"],
    }
    ctx.record_dir.mkdir(parents=True, exist_ok=True)
    record_path = ctx.record_dir / ("apply-{}.json".format(created_ns))
    atomic_write_json(record_path, record)
    backup_dir = ctx.record_dir / ("backup-{}".format(created_ns))
    for entry in facts["existing_dropins"]:
        backup_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(entry["path"], backup_dir / entry["name"])
    if backup_dir.is_dir():
        record["backup_dir"] = str(backup_dir)
        atomic_write_json(record_path, record)
    rollback_command = (sys.executable + " " + str(Path(__file__).resolve()) +
                        " rollback --record " + str(record_path))
    try:
        atomic_write_text(ctx.dropin_path, content)
        systemctl(ctx, "daemon-reload")
        systemctl(ctx, "restart", ctx.unit)
        verify_service(ctx, candidate_expectations(ctx), record["history_ids"],
                       token, ctx.startup_timeout)
    except Exception as exc:
        record["failure"] = str(exc)
        try:
            restore_original(ctx, record, token)
        except Exception as restore_exc:
            record["state"] = "failed_restore_unknown"
            atomic_write_json(record_path, record)
            raise DeployError(
                "APPLY_FAILED_RESTORE_FAILED: candidate verification failed ({}); automatic "
                "rollback also failed ({}). Hardware/DMA ownership is UNKNOWN: do not start "
                "new inference or HDMI, keep the service stopped, and inspect "
                "'journalctl -u {} -n 100 --no-pager' plus leftover infer_worker/hdmi_worker "
                "processes before any recovery.".format(exc, restore_exc, ctx.unit), exit_code=2)
        record["state"] = "failed_restored"
        atomic_write_json(record_path, record)
        raise DeployError(
            "APPLY_FAILED_RESTORED: candidate verification failed ({}); original drop-in state "
            "was restored and re-verified (HTTP 200, idle, original history intact). The "
            "candidate directories were kept for diagnosis.".format(exc))
    record["state"] = "applied"
    atomic_write_json(record_path, record)
    print("APPLY_PASS {}".format(CANDIDATE_VERSION))
    print("  drop-in={}".format(ctx.dropin_path))
    print("  record={}".format(record_path))
    print("  rollback: {}".format(rollback_command))
    return 0


def load_record(ctx):
    if ctx.args.record is not None:
        record_path = ctx.args.record
    else:
        records = sorted(ctx.record_dir.glob("apply-*.json"))
        if not records:
            raise DeployError("no apply record found in {}".format(ctx.record_dir))
        record_path = records[-1]
    try:
        record = json.loads(Path(record_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise DeployError("cannot read record {}: {}".format(record_path, exc))
    if record.get("schema") != 1 or record.get("unit") != ctx.unit:
        raise DeployError("record {} has an unexpected schema or unit".format(record_path))
    if record.get("state") not in ("prepared", "applied", "failed_restored", "failed_restore_unknown"):
        raise DeployError("record {} has state {} and cannot drive rollback".format(
            record_path, record.get("state")))
    for section, key in (("original", "data"), ("candidate", "dropin_path"),
                         ("candidate", "dropin_sha256")):
        if not record.get(section, {}).get(key):
            raise DeployError("record {} lacks {}.{}".format(record_path, section, key))
    print("ROLLBACK_RECORD {} state={}".format(record_path, record.get("state")))
    return record


def action_rollback(ctx):
    record = load_record(ctx)
    data_dir = Path(record["original"]["data"])
    token = read_access_token(data_dir)
    status = api_status(ctx, token)
    require_idle(status)
    dropin = Path(record["candidate"]["dropin_path"])
    if dropin.exists():
        if sha256_file(dropin) != record["candidate"]["dropin_sha256"]:
            raise DeployError("drop-in {} changed since apply; refusing automatic rollback".format(dropin))
        dropin.unlink()
    systemctl(ctx, "daemon-reload")
    systemctl(ctx, "restart", ctx.unit)
    verify_service(ctx, original_expectations(ctx), record.get("history_ids", []),
                   token, ctx.startup_timeout)
    record_path = (ctx.args.record if ctx.args.record is not None
                   else sorted(ctx.record_dir.glob("apply-*.json"))[-1])
    record["state"] = "rolled_back"
    record["rolled_back_ns"] = time.time_ns()
    atomic_write_json(record_path, record)
    print("ROLLBACK_PASS {}".format(CANDIDATE_VERSION))
    print("  original service restored and verified; record={}".format(record_path))
    print("  candidate directories were kept; user data was not modified")
    return 0


def check_platform(args):
    if args.no_platform_check:
        return
    if platform.system() != "Linux" or not platform.machine().lower().startswith("arm"):
        raise DeployError("run on PYNQ-Z2 with the board PYNQ venv Python")
    if os.geteuid() != 0:
        raise DeployError("run as root, e.g. sudo -E /usr/local/share/pynq-venv/bin/python3 ...")
    if not Path("/run/systemd/system").is_dir():
        raise DeployError("systemd is not running; no changes were made")


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("preflight", "apply", "rollback"))
    parser.add_argument("--original-app", type=Path, default=DEFAULT_ORIGINAL_APP)
    parser.add_argument("--candidate-c", type=Path, default=DEFAULT_CANDIDATE_C)
    parser.add_argument("--candidate-b", type=Path, default=DEFAULT_CANDIDATE_B)
    parser.add_argument("--data", type=Path, default=None,
                        help="must resolve to ORIGINAL_APP/data; any other directory is "
                             "rejected before checks or changes")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--systemd-dir", type=Path, default=DEFAULT_SYSTEMD_DIR)
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD_DIR)
    parser.add_argument("--dropin-name", default=DROPIN_NAME)
    parser.add_argument("--systemctl", default="systemctl",
                        help="systemctl command; may include an interpreter prefix for host tests")
    parser.add_argument("--http-base", default=DEFAULT_HTTP_BASE)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--proc-root", type=Path, default=Path("/proc"))
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--systemctl-timeout", type=float, default=30.0,
                        help="bounded wait for is-active/show/daemon-reload")
    parser.add_argument("--restart-timeout", type=float, default=300.0,
                        help="bounded wait for systemctl restart; must cover the unit's "
                             "TimeoutStopSec=240 plus restart overhead (independent of "
                             "--startup-timeout, which only bounds post-restart verification)")
    parser.add_argument("--http-timeout", type=float, default=3.0)
    parser.add_argument("--record", type=Path, default=None,
                        help="rollback: specific apply record (default: newest)")
    parser.add_argument("--no-platform-check", action="store_true",
                        help="host regression tests only; skips board/root/systemd checks")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    try:
        check_platform(args)
        ctx = Context(args)
        if args.action == "preflight":
            preflight(ctx)
            return 0
        if args.action == "apply":
            return action_apply(ctx)
        if args.action == "rollback":
            return action_rollback(ctx)
    except DeployError as exc:
        print("DEPLOY_FAILED: {}".format(exc), file=sys.stderr)
        return exc.exit_code
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
