"""A's frozen TinyUNet: integer convolution and explicit inter-layer maps.

The graph runs from ONE input. Exported per-layer inputs are only oracles,
never substituted for values produced by the preceding graph operation.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

BASELINE_SPEC_SHA256 = "759a496708ae3bbde1dd50cfab7343a380fe7204cfd96031dc0855f35250fdab"
BASELINE_MANIFEST = Path(__file__).with_name("baseline_manifest.json")
FINAL_SPEC_SHA256 = "fe137ca5b128fc1983fabad9bee39b3208cda4256e0d04c4e27cef5c07666aef"
FINAL_MANIFEST = Path(__file__).with_name("final_manifest.json")
DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[1] / "role_a_final/int8"
REGISTERED_MODELS = {
    BASELINE_SPEC_SHA256: (BASELINE_MANIFEST, 0.50, "isic2016_baseline"),
    FINAL_SPEC_SHA256: (FINAL_MANIFEST, 0.45, "soup_v5_a050"),
}
_AUTO_MANIFEST = object()
LAYER_NAMES = tuple([f"encoders_{i}_{j}" for i in range(4) for j in (0, 2)] +
                    [f"decoder_blocks_{i}_{j}" for i in range(3) for j in (0, 2)] + ["head"])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def threshold_contract(probability, output_scale):
    """Include equality; thresholds outside the INT8 range mean all/none."""
    if not (np.isfinite(probability) and 0 < probability < 1 and
            np.isfinite(output_scale) and output_scale > 0):
        raise ValueError("threshold requires probability in (0,1) and positive scale")
    logit = math.log(probability / (1 - probability))
    return {"threshold_probability": float(probability), "threshold_logit": logit,
            "threshold_q": math.ceil(logit / output_scale),
            "head_output_scale": float(output_scale), "comparison": ">="}


def require_equal(actual, expected, label):
    if actual.shape != expected.shape:
        raise ValueError(f"{label}: shape {actual.shape} != {expected.shape}")
    bad = np.argwhere(actual != expected)
    if len(bad):
        index = tuple(bad[0])
        raise ValueError(f"{label}: mismatch={len(bad)}, first={index}, "
                         f"actual={actual[index]}, expected={expected[index]}")


def requant_q24(acc, multipliers):
    """INT64 product, arithmetic floor division, symmetric ties-to-even."""
    acc = np.asarray(acc, dtype=np.int64)
    if np.any(acc < -(1 << 31)) or np.any(acc > (1 << 31) - 1):
        raise OverflowError("convolution accumulator outside INT32")
    mult = np.asarray(multipliers, dtype=np.int64)
    if np.any(mult < 0) or np.any(mult > (1 << 31) - 1):
        raise ValueError("Q24 multiplier must be a nonnegative signed INT32")
    product = acc * mult
    quotient, remainder = np.divmod(product, 1 << 24)
    quotient += (remainder > (1 << 23)) | ((remainder == (1 << 23)) & ((quotient & 1) != 0))
    return np.clip(quotient, -127, 127).astype(np.int8)


def rescale_lut(source_scale, target_scale, relu=True):
    """Raw-byte indexed LUT, preserving A's float32 dequantize/quantize order.

    Only parameter preparation uses floating point; PL applies this 256-byte
    table to INT8 activations, including every negative value and zero.
    """
    if not (np.isfinite(source_scale) and np.isfinite(target_scale) and
            source_scale > 0 and target_scale > 0):
        raise ValueError("scales must be finite and positive")
    values = np.arange(256, dtype=np.uint8).view(np.int8)
    if relu:
        values = np.maximum(values, 0)
    values = values.astype(np.float32) * np.float32(source_scale)
    return np.clip(np.rint(values / np.float32(target_scale)), -127, 127).astype(np.int8)


def apply_lut(x, lut):
    return np.asarray(lut, dtype=np.int8)[np.asarray(x, dtype=np.int8).view(np.uint8)]


def maxpool2(x):
    n, c, h, w = x.shape
    if h % 2 or w % 2:
        raise ValueError("maxpool2 requires even spatial dimensions")
    return x.reshape(n, c, h // 2, 2, w // 2, 2).max(axis=(3, 5))


def upsample2(x):
    return x.repeat(2, axis=2).repeat(2, axis=3)


def conv2d(x, weights, bias, multipliers, padding):
    """NCHW/OIHW, stride=1; return requant output before ReLU."""
    x = np.asarray(x, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.int64)
    co, ci, kh, kw = weights.shape
    if x.ndim != 4 or x.shape[1] != ci or kh != kw or padding < 0:
        raise ValueError("invalid convolution shape")
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)))
    h, w = padded.shape[2] - kh + 1, padded.shape[3] - kw + 1
    acc = np.broadcast_to(np.asarray(bias, dtype=np.int64)[None, :, None, None],
                          (x.shape[0], co, h, w)).copy()
    for ky in range(kh):
        for kx in range(kw):
            acc += np.einsum("ncyx,oc->noyx", padded[:, :, ky:ky+h, kx:kx+w],
                             weights[:, :, ky, kx], optimize=True)
    return requant_q24(acc, np.asarray(multipliers)[None, :, None, None])


class Model:
    def __init__(self, root=DEFAULT_MODEL_ROOT, expected_spec=None, manifest=_AUTO_MANIFEST):
        self.root = Path(root)
        self.spec_hash = digest(self.root / "model_spec.json")
        registered = REGISTERED_MODELS.get(self.spec_hash)
        if expected_spec is None:
            if registered is None:
                raise ValueError("unregistered model; explicitly validate/version the new A handoff")
            expected_spec = self.spec_hash
        if self.spec_hash != expected_spec:
            raise ValueError("model_spec hash changed; explicitly validate/version the new A handoff")
        if manifest is _AUTO_MANIFEST:
            if registered is None:
                raise ValueError("unregistered model requires an explicit frozen manifest")
            manifest = registered[0]
        self.manifest_path = Path(manifest) if manifest is not None else None
        self.manifest_hash = digest(manifest) if manifest is not None else None
        self.model_name = registered[2] if registered else "unregistered"
        self.threshold_probability = registered[1] if registered else 0.5
        if manifest is not None:
            frozen = json.loads(Path(manifest).read_text(encoding="utf-8"))
            if frozen["spec_sha256"] != expected_spec:
                raise ValueError("frozen manifest belongs to another model version")
            for name, expected in frozen["files"].items():
                if digest(self.root.parent / name) != expected:
                    raise ValueError(f"frozen handoff manifest mismatch: {name}")
        self.spec = json.loads((self.root / "model_spec.json").read_text(encoding="utf-8"))
        self.layers = self.spec["layers"]
        if tuple(l["directory"] for l in self.layers) != LAYER_NAMES:
            raise ValueError("unsupported graph; update graph contract before using this model")
        for name, key in (("weights_int8.bin", "weights_int8_sha256"), ("bias.bin", "bias_int32_sha256")):
            if digest(self.root / name) != self.spec[key]:
                raise ValueError(f"parameter hash mismatch: {name}")
        if digest(self.root.parent / "best.pt") != self.spec["checkpoint_sha256"]:
            raise ValueError("checkpoint hash mismatch")
        wblob = (self.root / "weights_int8.bin").read_bytes()
        bblob = (self.root / "bias.bin").read_bytes()
        self.parameters, self.configs, self.bounds = [], [], []
        required_files = {"best.pt", "int8/model_spec.json", "int8/weights_int8.bin", "int8/bias.bin"}
        for layer in self.layers:
            folder = self.root / layer["directory"]
            cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
            required_files.add(f"int8/{layer['directory']}/config.json")
            required_files.update(f"int8/{layer['directory']}/{name}" for name in cfg["files_sha256"])
            for filename, expected in cfg["files_sha256"].items():
                if digest(folder / filename) != expected:
                    raise ValueError(f"golden/parameter hash mismatch: {folder / filename}")
            for key in ("weight_shape", "padding", "stride", "input_scale", "output_scale", "multiplier_q24"):
                if cfg[key] != layer[key]:
                    raise ValueError(f"spec/config disagree: {layer['directory']}/{key}")
            if cfg["full_input_shape"] != layer["input_shape"] or cfg["full_output_shape"] != layer["output_shape"]:
                raise ValueError("full tensor shape mismatch")
            if cfg["layout"] != "NCHW" or cfg["weight_layout"] != "OIHW" or cfg["stride"] != 1:
                raise ValueError("unsupported layout or stride")
            wo, wn, bo, bn = (layer[k] for k in ("weight_offset_bytes", "weight_nbytes", "bias_offset_bytes", "bias_nbytes"))
            wbytes, bbytes = wblob[wo:wo+wn], bblob[bo:bo+bn]
            if wbytes != (folder / "weight.bin").read_bytes() or bbytes != (folder / "bias.bin").read_bytes():
                raise ValueError("parameter offsets do not match individual layer files")
            weights = np.frombuffer(wbytes, dtype=np.int8).reshape(layer["weight_shape"])
            if np.any(weights == -128):
                raise ValueError("weight outside the signed symmetric [-127,127] contract")
            bias = np.frombuffer(bbytes, dtype="<i4")
            bounds = np.abs(weights.astype(np.int64)).sum(axis=(1, 2, 3)) * 127 + np.abs(bias.astype(np.int64))
            if np.any(bounds > (1 << 31)-1):
                raise OverflowError("model cannot use INT32 accumulation")
            mult = np.asarray(layer["multiplier_q24"], dtype=np.int64)
            if np.any(mult < 0) or np.any(mult > (1 << 31)-1):
                raise ValueError("unsupported multiplier range")
            self.bounds.append({"acc_abs_max": int(bounds.max()), "product_abs_max": int((bounds * mult).max())})
            self.parameters.append((weights, bias, mult))
            self.configs.append(cfg)
        if manifest is not None and set(frozen["files"]) != required_files:
            raise ValueError("frozen manifest must cover the complete handoff file set")
        self.threshold = threshold_contract(self.threshold_probability, self.layers[-1]["output_scale"])

    def golden(self, index, which):
        layer = self.layers[index]
        name, shape = ("input_full.bin", "input_shape") if which == "input" else ("output_full_golden.bin", "output_shape")
        return np.fromfile(self.root / layer["directory"] / name, dtype=np.int8).reshape(layer[shape])

    def prepare_input(self, index, outputs):
        """Reference non-convolution operators. Hardware backend must replace them."""
        current, previous = self.layers[index], self.layers[index-1]
        x = outputs[index-1]
        main_lut = rescale_lut(previous["output_scale"], current["input_scale"])
        if index in (2, 4, 6):
            x = maxpool2(x)
        if index in (8, 10, 12):
            skip_index = {8: 5, 10: 3, 12: 1}[index]
            skip_lut = rescale_lut(self.layers[skip_index]["output_scale"], current["input_scale"])
            skip = apply_lut(outputs[skip_index], skip_lut)
            x = np.concatenate((skip, upsample2(apply_lut(x, main_lut))), axis=1)
        else:
            x = apply_lut(x, main_lut)
        return x

    def run(self, image, verify=False, callback=None, conv_backend=conv2d):
        if list(image.shape) != self.layers[0]["input_shape"] or image.dtype != np.int8 or np.any(image == -128):
            raise ValueError("input must match model NCHW shape and signed INT8 [-127,127] contract")
        outputs = []
        for i, layer in enumerate(self.layers):
            x = image if i == 0 else self.prepare_input(i, outputs)
            if verify:
                require_equal(x, self.golden(i, "input"), layer["directory"] + "/input")
            weights, bias, multipliers = self.parameters[i]
            out = conv_backend(x, weights, bias, multipliers, layer["padding"])
            if verify:
                require_equal(out, self.golden(i, "output"), layer["directory"] + "/output")
            outputs.append(out)
            if callback:
                callback(i, layer, out)
        return outputs
