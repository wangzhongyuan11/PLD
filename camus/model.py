"""Load and freeze-check the independent CAMUS INT8 candidate package."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .integer import INT32_MAX, INT64_MAX

DEFAULT_MODEL_ROOT = (
    Path(__file__).resolve().parents[1]
    / "camus_candidate_hidden_histogram_int8_20261015"
    / "int8_export"
)
MODEL_SPEC_SHA256 = "68b4f425cc3612ad72d5e83955c2018981edee912fbc0937e9ac8c1718e7b5ae"
_packaged_model = Path(__file__).resolve().parents[1] / 'model/int8_export'
if _packaged_model.is_dir():
    DEFAULT_MODEL_ROOT = _packaged_model
MODEL_MANIFEST_PATH = Path(__file__).with_name("model_manifest.json")
EXPECTED_MEMBER_COUNT = 173
EXPECTED_LAYERS = (
    "encoders.0.0", "encoders.0.2",
    "encoders.1.0", "encoders.1.2",
    "encoders.2.0", "encoders.2.2",
    "encoders.3.0", "encoders.3.2",
    "decoders.0.0", "decoders.0.2",
    "decoders.1.0", "decoders.1.2",
    "decoders.2.0", "decoders.2.2",
    "head",
)
EXPECTED_CONCAT_LEVELS = (2, 1, 0)


class ModelError(ValueError):
    """The candidate package differs from the reviewed frozen contract."""


@dataclass(frozen=True)
class CamusLayer:
    index: int
    name: str
    directory: str
    kernel: int
    padding: int
    input_scale: float
    output_scale: float
    relu: bool
    weight: np.ndarray
    bias: np.ndarray
    multipliers: np.ndarray
    weight_scales: np.ndarray
    full_input_shape: tuple[int, int, int, int]
    full_output_shape: tuple[int, int, int, int]
    conservative_bound: int


@dataclass(frozen=True)
class CamusConcat:
    level: int
    decoder: int
    skip_channels: int
    up_channels: int
    source_scales: tuple[float, float]
    common_scale: float
    multiplier_q24: tuple[int, int]


@dataclass(frozen=True)
class CamusModel:
    root: Path
    spec: dict[str, Any]
    layers: tuple[CamusLayer, ...]
    concats: tuple[CamusConcat, ...]
    input_scale: float
    head_scale: float
    input_shape: tuple[int, int, int, int]
    head_shape: tuple[int, int, int, int]
    model_sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_member(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute() or "\\" in relative:
        raise ModelError(f"unsafe manifest path {relative!r}")
    raw_path = root / relative
    if ".." in Path(relative).parts or any(p.is_symlink() for p in (raw_path, *raw_path.parents)):
        raise ModelError(f"symlink/traversal is not a model member: {relative}")
    path = raw_path.resolve()
    root_resolved = root.resolve()
    try:
        path.relative_to(root_resolved)
    except ValueError as error:
        raise ModelError(f"manifest path escapes model root: {relative}") from error
    if path.is_symlink() or path.parent.is_symlink():
        raise ModelError(f"symlink is not a model member: {relative}")
    if not path.is_file():
        raise ModelError(f"missing model member: {relative}")
    return path


def _load_manifest(root: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ModelError(f"cannot read {MODEL_MANIFEST_PATH}: {error}") from error
    if manifest.get("schema") != "pld-camus-int8-model-v1":
        raise ModelError("unsupported CAMUS model manifest schema")
    if manifest.get("model_spec_sha256") != MODEL_SPEC_SHA256:
        raise ModelError("CAMUS model spec identity mismatch")
    members = manifest.get("members")
    if not isinstance(members, list) or len(members) != EXPECTED_MEMBER_COUNT:
        raise ModelError(f"expected {EXPECTED_MEMBER_COUNT} model members")
    seen: set[str] = set()
    for row in members:
        if not isinstance(row, dict) or set(row) != {"path", "bytes", "sha256"}:
            raise ModelError(f"invalid manifest row: {row!r}")
        relative = row["path"]
        if relative in seen:
            raise ModelError(f"duplicate model member: {relative}")
        seen.add(relative)
        path = _safe_member(root, relative)
        size = path.stat().st_size
        if size != row["bytes"]:
            raise ModelError(f"length mismatch for {relative}: {size} != {row['bytes']}")
        digest = sha256_file(path)
        if digest != row["sha256"]:
            raise ModelError(f"SHA256 mismatch for {relative}: {digest} != {row['sha256']}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if actual != seen:
        missing = sorted(seen - actual)
        extra = sorted(actual - seen)
        raise ModelError(f"model tree differs from manifest; missing={missing}, extra={extra}")
    return manifest


def _finite_positive(value: Any, what: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ModelError(f"{what} is not numeric") from error
    if not np.isfinite(number) or number <= 0:
        raise ModelError(f"{what} must be finite positive, got {number!r}")
    return number


def _read_exact(path: Path, expected: dict[str, Any], root: Path) -> np.ndarray:
    relative = path.relative_to(root).as_posix()
    if path.stat().st_size != expected["bytes"]:
        raise ModelError(f"spec length mismatch for {relative}")
    dtype = np.dtype(expected["dtype"])
    if expected["dtype"] == "int32":
        dtype = np.dtype("<i4")
    elif expected["dtype"] not in ("int8", "float64"):
        raise ModelError(f"unsupported model dtype {expected['dtype']!r} in {relative}")
    count = int(np.prod(expected["shape"], dtype=np.int64))
    raw = path.read_bytes()
    if len(raw) != count * dtype.itemsize:
        raise ModelError(f"dtype/shape length mismatch for {relative}")
    return np.frombuffer(raw, dtype=dtype).reshape(expected["shape"]).copy()


def _read_npy_exact(path: Path, expected: dict[str, Any], root: Path) -> np.ndarray:
    relative = path.relative_to(root).as_posix()
    if path.stat().st_size != expected["bytes"] or expected["dtype"] != "float64":
        raise ModelError(f"spec metadata mismatch for {relative}")
    array = np.load(path, allow_pickle=False)
    if array.dtype != np.float64 or tuple(array.shape) != tuple(expected["shape"]):
        raise ModelError(f"npy dtype/shape mismatch for {relative}")
    return array


def _check_no_minus128(array: np.ndarray, what: str) -> None:
    if array.dtype == np.int8 and array.size and int(np.min(array)) == -128:
        raise ModelError(f"{what} contains forbidden -128")


def _validate_spec_graph(spec: dict[str, Any]) -> None:
    graph = spec.get("graph", {})
    expected_graph = {
        "input_channels": 1,
        "num_classes": 4,
        "base_channels": 16,
        "num_pools": 3,
        "convs": 15,
        "hidden_convs": 14,
        "layer_names": list(EXPECTED_LAYERS),
        "concat_skip_up": [[2, 64, 128], [1, 32, 64], [0, 16, 32]],
        "skip_first": True,
    }
    for key, value in expected_graph.items():
        if graph.get(key) != value:
            raise ModelError(f"unexpected graph.{key}: {graph.get(key)!r}")
    quant = spec.get("quantization", {})
    expected_quant = {
        "signed_range": [-127, 127], "zero_point": 0, "qfrac": 24,
        "rounding": "ties_even", "weight_layout": "OIHW",
        "activation_layout": "NCHW", "shared_head_output_scale": True,
        "no_sigmoid": True, "input_scale": 0.007874015748031496,
    }
    for key, value in expected_quant.items():
        if quant.get(key) != value:
            raise ModelError(f"unexpected quantization.{key}: {quant.get(key)!r}")
    packed = spec.get("packed", {})
    if packed.get("weights_int8", {}).get("bytes") != 486352:
        raise ModelError("unexpected packed weights length")
    if packed.get("bias_int32", {}).get("bytes") != 2832:
        raise ModelError("unexpected packed bias length")


def _load_layer(row: dict[str, Any], root: Path) -> CamusLayer:
    directory = row["directory"]
    base = root / directory
    if base.resolve().parent != root.resolve():
        raise ModelError(f"layer directory escapes root: {directory}")
    weight = _read_exact(base / row["weight_file"]["file"], row["weight_file"], root)
    bias = _read_exact(base / row["bias_file"]["file"], row["bias_file"], root)
    multipliers = _read_exact(base / row["multiplier_file"]["file"], row["multiplier_file"], root)
    scales = _read_npy_exact(base / row["weight_scales_file"]["file"], row["weight_scales_file"], root)
    for name, array in (("weight", weight), ("bias", bias), ("multipliers", multipliers)):
        if not np.all(np.isfinite(array.astype(np.float64))):
            raise ModelError(f"{directory}/{name} is not finite")
    _check_no_minus128(weight, f"{directory}/weight")
    for name in ("input_full", "output_full_golden"):
        item = row["tile"][name]
        golden = _read_exact(base / "tile" / item["file"], item, root)
        _check_no_minus128(golden, f"{directory}/tile/{name}")
    if tuple(weight.shape) != tuple(row["weight_shape"]):
        raise ModelError(f"weight shape metadata mismatch for {directory}")
    if bias.shape != (weight.shape[0],) or multipliers.shape != (weight.shape[0],):
        raise ModelError(f"per-layer vector shape mismatch for {directory}")
    if scales.shape != (weight.shape[0],) or np.any(scales <= 0):
        raise ModelError(f"invalid weight scales for {directory}")
    if np.any(multipliers <= 0) or np.any(multipliers > INT32_MAX):
        raise ModelError(f"invalid Q24 multiplier for {directory}")
    input_scale = _finite_positive(row["input_scale"], f"{directory}.input_scale")
    output_scale = _finite_positive(row["output_scale"], f"{directory}.output_scale")
    for value in scales:
        _finite_positive(value, f"{directory}.weight_scale")
    bound_values = np.abs(bias.astype(np.int64)) + 127 * np.sum(
        np.abs(weight.astype(np.int64)), axis=(1, 2, 3), dtype=np.int64
    )
    if np.any(bound_values > INT32_MAX):
        raise ModelError(f"conservative accumulator exceeds INT32 for {directory}")
    if int(np.max(bound_values)) * int(np.max(multipliers)) > INT64_MAX:
        raise ModelError(f"Q24 product bound exceeds INT64 for {directory}")
    if int(np.max(bound_values)) != int(row["conservative_bound_max"]):
        raise ModelError(f"conservative bound metadata mismatch for {directory}")
    relu = bool(row.get("tile_relu"))
    if row["name"] == "head":
        if relu:
            raise ModelError("head fused ReLU must be false")
    elif not relu:
        raise ModelError(f"hidden layer {row['name']} must fuse ReLU")
    expected_mult = np.rint(input_scale * scales / output_scale * (1 << 24)).astype(np.int64)
    if not np.array_equal(expected_mult, multipliers):
        raise ModelError(f"per-channel Q24 mismatch at {row['name']}")
    for parameter in (weight, bias, multipliers, scales):
        parameter.setflags(write=False)
    return CamusLayer(
        index=int(row["index"]), name=row["name"], directory=directory,
        kernel=int(row["kernel"]), padding=int(row["padding"]),
        input_scale=input_scale, output_scale=output_scale, relu=relu,
        weight=weight, bias=bias, multipliers=multipliers, weight_scales=scales,
        full_input_shape=tuple(row["full_input_shape"]),
        full_output_shape=tuple(row["full_output_shape"]),
        conservative_bound=int(np.max(bound_values)),
    )


def _validate_packed(spec: dict[str, Any], layers: tuple[CamusLayer, ...], root: Path) -> None:
    packed_weight = np.frombuffer(
        (root / "weights_int8.bin").read_bytes(), dtype=np.int8
    )
    packed_bias = np.frombuffer(
        (root / "bias_int32.bin").read_bytes(), dtype="<i4"
    )
    weight_cursor = 0
    bias_cursor = 0
    for row, layer in zip(spec["layers"], layers):
        wo = int(row["packed_weight_offset"]); wc = int(row["packed_weight_count"])
        bo = int(row["packed_bias_offset"]); bc = int(row["packed_bias_count"])
        if wo != weight_cursor or bo != bias_cursor:
            raise ModelError(f"packed offsets are not ordered at {layer.name}")
        if wc != layer.weight.size or bc != layer.bias.size:
            raise ModelError(f"packed counts differ from arrays at {layer.name}")
        if not np.array_equal(packed_weight[wo:wo + wc], layer.weight.reshape(-1)):
            raise ModelError(f"packed weight segment differs at {layer.name}")
        if not np.array_equal(packed_bias[bo:bo + bc], layer.bias.reshape(-1)):
            raise ModelError(f"packed bias segment differs at {layer.name}")
        weight_cursor += wc
        bias_cursor += bc
    if weight_cursor != packed_weight.size or bias_cursor != packed_bias.size:
        raise ModelError("packed files contain unused bytes")


def _load_concat(row: dict[str, Any]) -> CamusConcat:
    source = tuple(float(v) for v in row["source_scales"])
    common = _finite_positive(row["common_scale"], "concat.common_scale")
    if common != max(source):
        raise ModelError("concat common scale is not max(source scales)")
    mult = tuple(int(v) for v in row["multiplier_q24"])
    if len(source) != 2 or len(mult) != 2:
        raise ModelError("CAMUS concat must have two branches")
    for scale, multiplier in zip(source, mult):
        expected = int(np.rint(scale / common * (1 << 24)))
        if multiplier != expected or multiplier <= 0:
            raise ModelError("concat Q24 multiplier mismatch")
    level = int(row["level"])
    expected_channels = {0: (16, 32), 1: (32, 64), 2: (64, 128)}[level]
    output_channels = int(row["output_channels"])
    if sum(expected_channels) != output_channels:
        raise ModelError("concat channel metadata mismatch")
    return CamusConcat(
        level=int(row["level"]), decoder=int(row["decoder"]),
        skip_channels=expected_channels[0], up_channels=expected_channels[1],
        source_scales=source, common_scale=common, multiplier_q24=mult,
    )


def load_model(root: Path | str = DEFAULT_MODEL_ROOT, *, verify=True) -> CamusModel:
    model_root = Path(root).resolve()
    if not model_root.is_dir():
        raise ModelError(f"model root does not exist: {model_root}")
    if verify: _load_manifest(model_root)
    spec_path = model_root / "model_spec.json"
    if verify and sha256_file(spec_path) != MODEL_SPEC_SHA256:
        raise ModelError("model_spec.json identity mismatch")
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ModelError(f"invalid model_spec.json: {error}") from error
    if spec.get("schema") != "camus-int8-export-v1":
        raise ModelError("unsupported model spec schema")
    if verify: _validate_spec_graph(spec)
    rows = spec.get("layers")
    if not isinstance(rows, list) or len(rows) != 15:
        raise ModelError("model spec must contain exactly 15 layers")
    layers = tuple(_load_layer(row, model_root) for row in rows)
    if tuple(layer.name for layer in layers) != EXPECTED_LAYERS:
        raise ModelError("layer order differs from frozen CAMUS topology")
    if tuple(layer.index for layer in layers) != tuple(range(15)):
        raise ModelError("layer indices are not ordered")
    if verify: _validate_packed(spec, layers, model_root)
    concat_rows = spec.get("concat")
    if not isinstance(concat_rows, list) or len(concat_rows) != 3:
        raise ModelError("expected three CAMUS concat nodes")
    concats = tuple(_load_concat(row) for row in concat_rows)
    if tuple(item.level for item in concats) != EXPECTED_CONCAT_LEVELS:
        raise ModelError("concat levels are not decoder order")
    expected_input = float(spec["quantization"]["input_scale"])
    previous_scale = expected_input
    cursor = 0
    skip_scales: list[float] = []
    for _level in range(4):
        for _conv in range(2):
            if layers[cursor].input_scale != previous_scale:
                raise ModelError(f"producer scale mismatch at {layers[cursor].name}")
            previous_scale = layers[cursor].output_scale
            cursor += 1
        skip_scales.append(previous_scale)
    for concat in concats:
        if concat.source_scales[0] != skip_scales[concat.level]:
            raise ModelError(f"concat level {concat.level} skip scale mismatch")
        if concat.source_scales[1] != previous_scale:
            raise ModelError(f"concat level {concat.level} upsample scale mismatch")
        previous_scale = concat.common_scale
        for _conv in range(2):
            if layers[cursor].input_scale != previous_scale:
                raise ModelError(f"producer scale mismatch at {layers[cursor].name}")
            previous_scale = layers[cursor].output_scale
            cursor += 1
    head = layers[-1]
    if cursor != 14 or head.input_scale != previous_scale:
        raise ModelError("head producer scale mismatch")
    if head.output_scale != 0.19834566491795338:
        raise ModelError("unexpected shared head output scale")
    return CamusModel(
        root=model_root, spec=spec, layers=layers, concats=concats,
        input_scale=expected_input, head_scale=head.output_scale,
        input_shape=(1, 1, 256, 256), head_shape=(1, 4, 256, 256),
        model_sha256=MODEL_SPEC_SHA256,
    )


def golden_path(model: CamusModel, layer: CamusLayer, kind: str) -> Path:
    """Return a verified golden path; loading remains explicit in callers."""
    if kind not in ("input_full", "output_full_golden"):
        raise ModelError(f"unsupported golden kind {kind}")
    return model.root / layer.directory / "tile" / f"{kind}.bin"


def load_golden(model: CamusModel, layer: CamusLayer, kind: str) -> np.ndarray:
    row = model.spec["layers"][layer.index]["tile"][kind]
    return _read_exact(model.root / layer.directory / "tile" / row["file"], row, model.root)
