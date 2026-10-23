"""Frozen model topology and Axial-only orchestration contract."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ARCHITECTURE = "ACDC25D-B16-P3-v1"
PROFILE = "axial_only_v1"
PREPROCESSING = "acdc-production-numerics-20261018-v1"
INPUT_SHAPE = (1, 3, 256, 256)
HEAD_SHAPE = (1, 4, 256, 256)
INPUT_BYTES = 196608
HEAD_BYTES = 262144
LABELS = ("background", "RV", "MYO", "LV")
COLORS = ((0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255))
SHAPES = (
    (3, 16, 256, 3), (16, 16, 256, 3), (16, 32, 128, 3),
    (32, 32, 128, 3), (32, 64, 64, 3), (64, 64, 64, 3),
    (64, 128, 32, 3), (128, 128, 32, 3), (192, 64, 64, 3),
    (64, 64, 64, 3), (96, 32, 128, 3), (32, 32, 128, 3),
    (48, 16, 256, 3), (16, 16, 256, 3), (16, 4, 256, 1),
)
NODE_NAMES = ("C01", "C02", "P0", "C03", "C04", "P1", "C05", "C06",
              "P2", "C07", "C08", "CAT1", "C09", "C10", "CAT2", "C11",
              "C12", "CAT3", "C13", "C14", "C15")


class ContractError(ValueError):
    """An input is outside the declared numerical or execution contract."""


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def is_sha(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def checked_int8(value, shape=None) -> np.ndarray:
    a = np.asarray(value)
    if a.dtype != np.int8 or (shape is not None and a.shape != tuple(shape)):
        raise ContractError(f"expected INT8 {shape}, got {a.dtype}/{a.shape}")
    if a.size == 0 or np.any(a == -128):
        raise ContractError("empty tensor or forbidden -128")
    return np.ascontiguousarray(a)


def positive(value, name="scale") -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ContractError(f"invalid {name}")
    v = float(value)
    if not np.isfinite(v) or v <= 0:
        raise ContractError(f"invalid {name}")
    return v


def safe_member(root: Path, name: str) -> Path:
    if (not isinstance(name, str) or not name or "\\" in name or ":" in name
            or Path(name).is_absolute() or ".." in Path(name).parts):
        raise ContractError("unsafe bundle member")
    root = Path(root).resolve()
    raw = root / name
    if any(p.is_symlink() for p in (raw, *raw.parents)):
        raise ContractError("symlink bundle member")
    p = raw.resolve()
    if not p.is_relative_to(root) or not p.is_file():
        raise ContractError(f"missing/escaping member {name}")
    return p


def atomic_json(path: Path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    pending.write_bytes(canonical(value) + b"\n")
    pending.replace(path)


@dataclass(frozen=True)
class Layer:
    index: int
    name: str
    kernel: int
    padding: int
    relu: bool
    input_scale: float
    output_scale: float
    weight: np.ndarray
    bias: np.ndarray
    multipliers: np.ndarray
    weight_scales: np.ndarray
    full_input_shape: tuple
    full_output_shape: tuple


@dataclass(frozen=True)
class Concat:
    level: int
    source_scales: tuple
    common_scale: float
    multiplier_q24: tuple


@dataclass(frozen=True)
class Model:
    root: Path
    spec: dict
    layers: tuple[Layer, ...]
    concats: tuple[Concat, ...]
    input_scale: float
    head_scale: float
    model_sha256: str
    synthetic: bool
    implementation_binding: dict | None = None
    input_shape: tuple = INPUT_SHAPE
    head_shape: tuple = HEAD_SHAPE
