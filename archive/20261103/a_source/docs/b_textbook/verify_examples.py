"""Read-only arithmetic checks for the B-group textbook's frozen A handoff."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PACK = ROOT / "deliverables" / "role_a" / "int8"
spec = json.loads((PACK / "model_spec.json").read_text(encoding="utf-8"))
layers = spec["layers"]
weight_bytes = (PACK / spec["weights_int8_file"]).read_bytes()
bias_bytes = (PACK / spec["bias_int32_file"]).read_bytes()

macs = 0
max_bound = ("", 0)
for layer in layers:
    n, c, h, w = layer["output_shape"]
    co, ci, kh, kw = layer["weight_shape"]
    assert n == 1 and co == c
    macs += h * w * co * ci * kh * kw
    bo = layer["bias_offset_bytes"]
    bn = layer["bias_nbytes"] // 4
    bias = np.frombuffer(bias_bytes, dtype="<i4", count=bn, offset=bo)
    bound = int(np.max(np.abs(bias.astype(np.int64)))) + ci * kh * kw * 127 * 127
    if bound > max_bound[1]:
        max_bound = (layer["name"], bound)
    assert bound < 2**31, (layer["name"], bound)

assert len(layers) == 15
assert len(weight_bytes) == 121760
assert len(bias_bytes) == 1412
assert macs == 675282944

layer = layers[0]
folder = PACK / layer["directory"]
config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
x = np.fromfile(folder / "input_full.bin", dtype=np.int8).reshape(config["full_input_shape"])
w = np.fromfile(folder / "weight.bin", dtype=np.int8).reshape(config["weight_shape"])
b = np.fromfile(folder / "bias.bin", dtype="<i4")
y = np.fromfile(folder / "output_full_golden.bin", dtype=np.int8).reshape(config["full_output_shape"])
patch = x[0, :, 127:130, 127:130].astype(np.int64)
channel_sums = np.sum(patch * w[0].astype(np.int64), axis=(1, 2))
acc = int(b[0]) + int(channel_sums.sum())
m = int(config["multiplier_q24"][0])
requant = int(np.rint(acc * m / (1 << 24)))
golden = int(y[0, 0, 128, 128])
assert channel_sums.tolist() == [-4261, 21622, -2698]
assert acc == -3922 and requant == golden == -3

print(f"layers={len(layers)} weight_bytes={len(weight_bytes)} bias_bytes={len(bias_bytes)}")
print(f"macs={macs} max_conservative_acc_bound={max_bound}")
print(f"first_layer_channel_sums={channel_sums.tolist()} bias={int(b[0])}")
print(f"acc={acc} multiplier_q24={m} requant={requant} golden={golden}")
