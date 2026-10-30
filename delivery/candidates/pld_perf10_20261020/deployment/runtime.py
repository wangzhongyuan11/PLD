"""Deployment adapter; imported candidate sources remain byte-identical."""
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
CORE = ROOT / "PLD_refactor_stage8"
sys.path[:0] = [str(CORE), str(ROOT / "extension")]
from refactor_tools.session7 import Session
from refactor_tools.placement8 import install
from fast_device import FastDevice
BASE = ROOT.parent / "pld_dual_20261017"
BUILD = CORE / "out/stage8_route/baseline100"

def verify_payload():
    manifest = json.loads((ROOT / "payload_sha256.json").read_text())
    for name, digest in manifest.items():
        p = ROOT / name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise ValueError("Deployment payload drift: " + name)
    return len(manifest)

def factory(kind):
    roots = {"skin": ROOT / "models/skin/int8", "camus": ROOT / "models/camus/int8_export"}
    session = Session(kind, roots[kind], backend="board", build=BUILD,
                      batch_size=64, operator_placement="ps")
    try:
        old = session.device
        session.device = FastDevice(old.model, session.transport.exchange_into, old.capacity, batch_size=64)
        session.device.clock_mhz = session.clock_mhz
        session.device.evidence = "board_actual_perf10_mixed"
        install(session.device, "ps")
        old.close()
        return session
    except BaseException:
        session.close()
        raise
