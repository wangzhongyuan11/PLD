import json
from pathlib import Path
from runtime import ROOT, BASE, factory, verify_payload
from refactor_tools.workstation7 import Workstation
from refactor_tools.serve_stage7 import make_server, install_shutdown

verify_payload()
config = json.loads((BASE / "hub/config.json").read_text())
token = Path(config["token_file"]).read_text().strip()
app = Workstation(ROOT / "data", factory)
server = None
try:
    app.switch("skin")
    server = make_server(app, token, "0.0.0.0", 8765)
    install_shutdown(server, app)
    print("PERF10_READY port=8765 convolution=FPGA nonconvolution=CPU", flush=True)
    server.serve_forever(poll_interval=.1)
finally:
    app.stop_accepting()
    if server:
        server.server_close()
    app.close()
    print("PERF10_CLOSED hardware_quiesced=true", flush=True)
