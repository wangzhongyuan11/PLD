"""Verify exported INT8 artifacts and every layer's integer tile vector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .quant_math import conv2d_int32, requantize_int8
except ImportError:
    from quant_math import conv2d_int32, requantize_int8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(spec_path: Path) -> dict[str, int | str]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != 1:
        raise ValueError("unsupported model_spec schema")
    root = spec_path.parent
    packed = {}
    for kind, name_key, hash_key in (
        ("weight", "weights_int8_file", "weights_int8_sha256"),
        ("bias", "bias_int32_file", "bias_int32_sha256"),
    ):
        path = root / spec[name_key]
        if sha256(path) != spec[hash_key]:
            raise ValueError(f"SHA256 mismatch: {path}")
        packed[kind] = path.read_bytes()
    checked = 0
    for layer in spec["layers"]:
        directory = root / layer["directory"]
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        for filename, expected in config["files_sha256"].items():
            if sha256(directory / filename) != expected:
                raise ValueError(f"SHA256 mismatch: {directory / filename}")
        for kind, filename in (("weight", "weight.bin"), ("bias", "bias.bin")):
            start = layer[f"{kind}_offset_bytes"]
            length = layer[f"{kind}_nbytes"]
            if packed[kind][start:start + length] != (directory / filename).read_bytes():
                raise ValueError(f"packed {kind} mismatch: {layer['name']}")
        x = np.fromfile(directory / "input.bin", dtype=np.int8).reshape(config["input_shape"])
        w = np.fromfile(directory / "weight.bin", dtype=np.int8).reshape(config["weight_shape"])
        b = np.fromfile(directory / "bias.bin", dtype="<i4")
        acc = conv2d_int32(x, w, b, int(config["padding"]))
        saved_acc = np.fromfile(directory / "acc_golden.bin", dtype="<i4").reshape(config["output_shape"])
        if not np.array_equal(acc, saved_acc):
            raise ValueError(f"INT32 golden mismatch: {layer['name']}")
        output = requantize_int8(acc, np.asarray(config["multiplier_q24"], dtype=np.int32))
        saved_output = np.fromfile(directory / "output_golden.bin", dtype=np.int8).reshape(config["output_shape"])
        if not np.array_equal(output, saved_output):
            raise ValueError(f"INT8 golden mismatch: {layer['name']}")
        for filename, shape in ((config["full_input_file"], config["full_input_shape"]),
                                (config["full_output_file"], config["full_output_shape"])):
            if (directory / filename).stat().st_size != int(np.prod(shape)):
                raise ValueError(f"full vector size mismatch: {directory / filename}")
        checked += 1
    return {"layers_verified": checked, "weights_int8_bytes": len(packed["weight"]),
            "bias_int32_bytes": len(packed["bias"]),
            "checkpoint_sha256": spec["checkpoint_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = verify(args.spec)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
