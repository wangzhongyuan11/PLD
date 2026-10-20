import json, time
from runtime import ROOT, factory, verify_payload
import numpy as np
report = {"passed": False, "scope": "actual board, 8x8 first-layer input, full 15-layer topology, independent per-layer reference", "models": {}}
verify_payload()
for kind, channels in (("skin", 3), ("camus", 1)):
    session = factory(kind)
    try:
        q = np.random.default_rng(20261020).integers(-127,128,(channels,8,8),dtype=np.int8)
        result = session.device.run(q, verify=True)
        assert result["report"]["all_layers_bit_exact"]
        report["models"][kind] = result["report"]
        print(kind, "SMALL_BOARD_15_LAYER_PASS", flush=True)
    finally:
        session.close()
report["passed"] = True
(ROOT / "evidence").mkdir(exist_ok=True)
(ROOT / "evidence/small_board.json").write_text(json.dumps(report, indent=2))
