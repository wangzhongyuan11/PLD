"""Strict CAMUS input readers and the exact frozen letterbox preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .integer import quantize_letterbox_input

INPUT_SIZE = 256


class PreprocessError(ValueError):
    """The input source or geometry violates the CAMUS software contract."""


@dataclass(frozen=True)
class LoadedGray:
    data: np.ndarray
    spacing_yx_mm: tuple[float, float] | None
    source_kind: str
    source_path: str


@dataclass(frozen=True)
class PreprocessResult:
    raw_gray: np.ndarray
    normalized_float32: np.ndarray
    canvas_float32: np.ndarray
    input_int8: np.ndarray
    letterbox_meta: dict[str, int | float]
    clipped_low: int
    clipped_high: int
    source: LoadedGray


def _require_2d_gray(array: np.ndarray, source: str) -> np.ndarray:
    data = np.asarray(array)
    if data.ndim != 2 or data.size == 0 or data.size > 8_000_000:
        raise PreprocessError(f"{source} must be a nonempty 2-D scalar frame, got {data.shape}")
    if data.dtype == np.bool_ or np.issubdtype(data.dtype, np.complexfloating):
        raise PreprocessError(f"{source} has non-scalar numeric dtype {data.dtype}")
    if not np.issubdtype(data.dtype, np.number):
        raise PreprocessError(f"{source} has unsupported dtype {data.dtype}")
    if np.issubdtype(data.dtype, np.floating) and not np.all(np.isfinite(data)):
        raise PreprocessError(f"{source} contains NaN/Inf")
    return data


def _load_image(path: Path) -> LoadedGray:
    try:
        with Image.open(path) as image:
            if image.width * image.height > 8_000_000:
                raise PreprocessError('frame exceeds 8 million native pixels')
            if image.mode not in ("L", "I", "I;16", "I;16B", "I;16L", "F"):
                raise PreprocessError(
                    f"image must already be grayscale mode L, got {image.mode}"
                )
            array = np.asarray(image)
    except PreprocessError:
        raise
    except Exception as error:
        raise PreprocessError(f"cannot decode image {path}: {error}") from error
    return LoadedGray(
        _require_2d_gray(array, path.name), None, "image", str(path)
    )


def _load_numeric(path: Path) -> LoadedGray:
    try:
        if path.suffix.lower() == ".npy":
            array = np.load(path, allow_pickle=False)
        else:
            with zipfile.ZipFile(path) as archive:
                if sum(info.file_size for info in archive.infolist()) > 64 * 1024 * 1024:
                    raise PreprocessError('NPZ exceeds 64 MiB decoded size')
            with np.load(path, allow_pickle=False) as payload:
                if not isinstance(payload, np.lib.npyio.NpzFile):
                    raise PreprocessError("unexpected numeric input")
                keys = list(payload.keys())
                if len(keys) != 1:
                    raise PreprocessError(f"NPZ must contain exactly one array, got {keys}")
                array = payload[keys[0]]
    except PreprocessError:
        raise
    except Exception as error:
        raise PreprocessError(f"cannot load numeric input {path}: {error}") from error
    return LoadedGray(
        _require_2d_gray(array, path.name), None, "numeric", str(path)
    )


def _parse_mhd_number(value: str, key: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise PreprocessError(f"{key} is not numeric: {value!r}") from error
    if not math.isfinite(number):
        raise PreprocessError(f"{key} is not finite")
    return number


def _mhd_values(header: dict[str, list[str]], key: str) -> list[str] | None:
    values = header.get(key)
    return values if values else None


def _single_mhd(header: dict[str, list[str]], key: str) -> str | None:
    values = _mhd_values(header, key)
    if values is None:
        return None
    if len(values) != 1:
        raise PreprocessError(f"{key} appears {len(values)} times")
    return values[0]


def _parse_mhd(path: Path) -> LoadedGray:
    try:
        text = path.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise PreprocessError(f"cannot read ASCII MHD {path}: {error}") from error
    header: dict[str, list[str]] = {}
    for line_no, raw_line in enumerate(text.splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise PreprocessError(f"MHD line {line_no} is not Key = Value")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise PreprocessError(f"MHD line {line_no} has empty key/value")
        header.setdefault(key, []).append(value)

    required = ("NDims", "DimSize", "ElementType", "ElementDataFile")
    missing = [key for key in required if _single_mhd(header, key) is None]
    if missing:
        raise PreprocessError(f"MHD missing required fields {missing}")
    ndims = int(_single_mhd(header, "NDims"))
    if ndims not in (2, 3):
        raise PreprocessError(f"CAMUS endpoint requires NDims 2 or Z=1, got {ndims}")
    dims = [int(value) for value in _single_mhd(header, "DimSize").split()]
    if len(dims) != ndims or any(value <= 0 for value in dims):
        raise PreprocessError(f"DimSize {dims} does not match NDims {ndims}")
    if ndims == 3 and dims[2] != 1:
        raise PreprocessError(f"multi-frame MHD is unsupported (Z={dims[2]}); select one endpoint externally")

    element_type = _single_mhd(header, "ElementType")
    dtypes = {
        "MET_UCHAR": np.dtype("u1"),
        "MET_USHORT": np.dtype("u2"),
        "MET_SHORT": np.dtype("i2"),
        "MET_FLOAT": np.dtype("f4"),
    }
    if element_type not in dtypes:
        raise PreprocessError(f"unsupported ElementType {element_type!r}")

    if (_single_mhd(header, "BinaryData") or "True").lower() not in ("true", "1"):
        raise PreprocessError("ASCII MetaImage data is unsupported")
    compressed = _single_mhd(header, "CompressedData") or "False"
    if compressed.lower() not in ("false", "0"):
        raise PreprocessError("compressed MetaImage data is unsupported")
    header_size = _single_mhd(header, "HeaderSize") or "0"
    if int(header_size) != 0:
        raise PreprocessError("external RAW HeaderSize must be zero")
    channels = _single_mhd(header, "ElementNumberOfChannels") or "1"
    if int(channels) != 1:
        raise PreprocessError(f"multi-channel MetaImage is unsupported ({channels})")

    msb_value = _single_mhd(header, "BinaryDataByteOrderMSB")
    alternate_order = _single_mhd(header, "ElementByteOrderMSB")
    if msb_value is not None and alternate_order is not None:
        if (msb_value.lower() in ("true", "1")) != (alternate_order.lower() in ("true", "1")):
            raise PreprocessError("conflicting byte order fields")
    if msb_value is None:
        msb_value = _single_mhd(header, "ElementByteOrderMSB")
    if msb_value is None:
        raise PreprocessError("MHD must explicitly declare BinaryDataByteOrderMSB")
    if msb_value.lower() not in ("true", "false", "1", "0"):
        raise PreprocessError(f"invalid BinaryDataByteOrderMSB {msb_value!r}")
    msb = msb_value.lower() in ("true", "1")
    dtype = dtypes[element_type].newbyteorder(">" if msb else "<")

    data_file = Path(_single_mhd(header, "ElementDataFile"))
    if str(data_file).upper() == 'LOCAL' or str(data_file).upper().startswith('LIST') or '%' in str(data_file):
        raise PreprocessError('embedded/list/pattern MetaImage data is unsupported')
    if data_file.is_absolute() or ".." in data_file.parts or data_file.drive:
        raise PreprocessError(f"ElementDataFile must be a relative sibling path: {data_file}")
    raw_path = (path.parent / data_file).resolve()
    try:
        raw_path.relative_to(path.parent.resolve())
    except ValueError as error:
        raise PreprocessError("ElementDataFile escapes the MHD directory") from error
    if raw_path.is_symlink() or not raw_path.is_file():
        raise PreprocessError(f"RAW data file is missing or a symlink: {raw_path}")
    if math.prod(dims) > 8_000_000:
        raise PreprocessError('MHD frame exceeds 8 million native pixels')
    expected_bytes = math.prod(dims) * dtype.itemsize
    actual_bytes = raw_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise PreprocessError(
            f"RAW length {actual_bytes} does not match DimSize/type ({expected_bytes})"
        )
    # MetaImage DimSize is X,Y[,Z] while its stream is X-fastest.  Reversing
    # the dimensions for C-order reshape directly yields H,W; no transpose or
    # canonical orientation reordering is applied.
    array = np.fromfile(raw_path, dtype=dtype).reshape(tuple(reversed(dims)))
    if ndims == 3:
        array = np.squeeze(array, axis=0)
    display = np.ascontiguousarray(array)
    display = _require_2d_gray(display, path.name)

    spacing_keys = ("ElementSpacing", "Spacing", "PixelSpacing")
    present = [key for key in spacing_keys if _mhd_values(header, key) is not None]
    if len(present) > 1:
        raise PreprocessError(f"multiple/conflicting spacing fields {present}")
    spacing_yx = None
    if present:
        values = _mhd_values(header, present[0])
        if len(values) != 1:
            raise PreprocessError(f"{present[0]} appears {len(values)} times")
        spacing = [_parse_mhd_number(item, present[0]) for item in values[0].split()]
        if len(spacing) not in (2, 3) or any(value <= 0 for value in spacing):
            raise PreprocessError(f"invalid {present[0]} values {spacing}")
        spacing_yx = (spacing[1], spacing[0])
    return LoadedGray(display, spacing_yx, "mhd_raw", str(path))


def _load_nifti(path: Path) -> LoadedGray:
    try:
        import nibabel as nib
    except ImportError as error:
        raise PreprocessError(
            "NIfTI input requires the optional dependency nibabel; it is not installed"
        ) from error
    try:
        handle = nib.load(str(path))
        header_shape = tuple(handle.header.get_data_shape())
        if not (len(header_shape) == 2 or len(header_shape) == 3 and header_shape[2] == 1):
            raise PreprocessError('NIfTI must be 2-D or a single Z slice')
        if math.prod(header_shape) > 8_000_000:
            raise PreprocessError('NIfTI frame exceeds 8 million native pixels')
        array = np.asarray(handle.dataobj)
    except Exception as error:
        raise PreprocessError(f"cannot read NIfTI {path}: {error}") from error
    if array.ndim == 3 and array.shape[2] == 1:
        squeezed = array[:, :, 0]
    elif array.ndim == 2:
        squeezed = array
    else:
        raise PreprocessError("NIfTI must be 2-D or a single Z slice")
    squeezed = _require_2d_gray(squeezed, path.name)
    if squeezed.dtype == np.bool_ or np.issubdtype(squeezed.dtype, np.complexfloating):
        raise PreprocessError("NIfTI frame must be scalar numeric")
    if np.issubdtype(squeezed.dtype, np.floating) and not np.all(np.isfinite(squeezed)):
        raise PreprocessError("NIfTI frame contains NaN/Inf")
    zooms = tuple(abs(float(value)) for value in handle.header.get_zooms()[:2])
    if len(zooms) != 2 or any(value <= 0 or not math.isfinite(value) for value in zooms):
        raise PreprocessError(f"invalid NIfTI XY zooms {zooms!r}")
    display = np.ascontiguousarray(squeezed.T)
    return LoadedGray(display, (zooms[1], zooms[0]), "nifti", str(path))


def load_gray_frame(path: Path | str) -> LoadedGray:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise PreprocessError(f"input is not a regular file: {source}")
    suffix = source.suffix.lower()
    if suffix in (".png", ".jpg", ".jpeg"):
        return _load_image(source)
    if suffix in (".npy", ".npz"):
        return _load_numeric(source)
    if suffix == ".mhd":
        return _parse_mhd(source)
    if suffix == ".nii" or source.name.lower().endswith(".nii.gz"):
        return _load_nifti(source)
    raise PreprocessError(f"unsupported input type {suffix!r}")


@lru_cache(maxsize=64)
def input_axis_plan(source_size: int, new_size: int):
    coordinates = (np.arange(new_size, dtype=np.float64) + 0.5) * (source_size / float(new_size)) - 0.5
    coordinates = np.clip(coordinates, 0.0, source_size - 1)
    lower = np.floor(coordinates).astype(np.int64)
    upper = np.minimum(lower + 1, source_size - 1)
    weight = (coordinates - lower).astype(np.float32)
    for value in (lower, upper, weight): value.flags.writeable = False
    return lower, upper, weight


def _input_bilinear_axis(array: np.ndarray, new_size: int, axis: int, scale: float) -> np.ndarray:
    lower, upper, weight = input_axis_plan(array.shape[axis], new_size)
    # np.take requires native-width indices on 32-bit ARM NumPy 1.21.
    # Keep coordinate/weight arithmetic unchanged, cast only at indexing.
    view_lower = np.take(array, lower.astype(np.intp), axis=axis)
    view_upper = np.take(array, upper.astype(np.intp), axis=axis)
    if axis == 0:
        weight = weight[:, None]
    else:
        weight = weight[None, :]
    return view_lower * (np.float32(1.0) - weight) + view_upper * weight


def _input_resize_bilinear(array: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    source = np.asarray(array, dtype=np.float32)
    if source.ndim != 2:
        raise PreprocessError("input resize expects a 2-D gray frame")
    vertical = _input_bilinear_axis(source, new_h, 0, source.shape[0] / float(new_h))
    return _input_bilinear_axis(vertical, new_w, 1, source.shape[1] / float(new_w))


def normalize_gray(array: np.ndarray) -> tuple[np.ndarray, int, int]:
    raw = np.asarray(array)
    finite_ok = (not np.issubdtype(raw.dtype, np.floating)) or bool(np.all(np.isfinite(raw)))
    if not finite_ok:
        raise PreprocessError("gray input contains NaN/Inf")
    values = np.asarray(raw, dtype=np.float32)
    clipped_low = int(np.count_nonzero(values < np.float32(0.0)))
    clipped_high = int(np.count_nonzero(values > np.float32(255.0)))
    normalized = (np.clip(values, 0.0, 255.0) / 255.0).astype(np.float32)
    return normalized, clipped_low, clipped_high


def letterbox_normalized(normalized: np.ndarray, size: int = INPUT_SIZE):
    array = np.asarray(normalized, dtype=np.float32)
    if array.ndim != 2 or min(array.shape) < 1:
        raise PreprocessError("letterbox expects nonempty 2-D float32")
    height, width = array.shape
    scale = min(size / height, size / width)
    new_h = max(1, min(size, int(round(height * scale))))
    new_w = max(1, min(size, int(round(width * scale))))
    resized = _input_resize_bilinear(array, new_h, new_w)
    canvas = np.zeros((size, size), dtype=np.float32)
    top = (size - new_h) // 2
    left = (size - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    meta = {
        "original_height": height, "original_width": width,
        "scale": scale, "new_height": new_h, "new_width": new_w,
        "top": top, "left": left, "size": size,
    }
    return canvas, meta


def preprocess(path: Path | str) -> PreprocessResult:
    loaded = load_gray_frame(path)
    normalized, low, high = normalize_gray(loaded.data)
    canvas, meta = letterbox_normalized(normalized)
    quantized = quantize_letterbox_input(canvas)
    return PreprocessResult(
        raw_gray=loaded.data.copy(), normalized_float32=normalized,
        canvas_float32=canvas, input_int8=quantized[None, None, :, :],
        letterbox_meta=meta, clipped_low=low, clipped_high=high, source=loaded,
    )
