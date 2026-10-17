"""Validate uploads before touching PL; read the documented Role-A NPZ cache.

The cache contract comes from role_a_camus/data.py in CAMUS_20261016:
native_image is normalized float32 H,W; spacing_mm is (Y, X).
Labels and object metadata are never loaded and pickle remains disabled.
"""
from pathlib import Path
import time
import threading
import zipfile


def install(web):
    import numpy as np
    from camus import preprocess as prep

    original_load = prep.load_gray_frame
    pending = threading.local()

    def load_once(path):
        cached = getattr(pending, "frame", None)
        if cached is not None and Path(path).resolve() == pending.path:
            pending.frame = None
            return cached
        return original_load(path)

    original_numeric = prep._load_numeric
    original_execute = web.CamusServer.execute

    def numeric(path):
        if path.suffix.lower() != '.npz':
            return original_numeric(path)
        try:
            with zipfile.ZipFile(path) as archive:
                entries = archive.infolist()
                if sum(i.file_size for i in entries) > 64 * 1024 * 1024:
                    raise prep.PreprocessError('NPZ exceeds 64 MiB decoded size')
                if len({i.filename for i in entries}) != len(entries):
                    raise prep.PreprocessError('NPZ contains duplicate members')
            with np.load(path, allow_pickle=False) as payload:
                keys = set(payload.files)
                if len(keys) == 1:
                    return original_numeric(path)
                required = {'image', 'valid', 'native_image', 'meta', 'spacing_mm', 'affine', 'quality_source'}
                if not required <= keys or keys - required - {'native_label', 'label'}:
                    raise prep.PreprocessError('Unsupported multi-array NPZ; use a Role-A cache or a single grayscale array')
                native = prep._require_2d_gray(payload['native_image'], 'native_image')
                if native.dtype != np.dtype('float32') or np.any(native < 0) or np.any(native > 1):
                    raise prep.PreprocessError('Role-A native_image must be finite float32 in [0,1]')
                spacing = spacing_pair(payload['spacing_mm'], np, prep.PreprocessError)
                # The existing readers/native preprocessor consume the 0..255
                # intensity domain. Never use the letterboxed image or labels.
                gray = native * np.float32(255.0)
                return prep.LoadedGray(gray, spacing, 'role_a_cache_native', str(path))
        except prep.PreprocessError:
            raise
        except Exception as exc:
            raise prep.PreprocessError('Cannot read NPZ input: ' + str(exc)) from exc

    def execute(server, job, selected, spacing):
        # This block is strictly before stop_hdmi, device reinitialization and
        # network.forward. Only failures here are recoverable input errors.
        try:
            frame = prep.load_gray_frame(server.data / job['id'] / 'input' / selected)
            if spacing is not None:
                supplied = spacing_pair(spacing, np, prep.PreprocessError)
                if frame.spacing_yx_mm is not None and frame.spacing_yx_mm != supplied:
                    raise prep.PreprocessError('spacing conflicts with source')
        except (prep.PreprocessError, ValueError, OSError) as exc:
            with server.guard:
                job.update(state='error', error=str(exc), error_kind='input',
                           server_seconds=time.perf_counter()-job.pop('_started'))
            server.busy.release()
            print('CAMUS_INPUT_REJECTED ' + str(exc), flush=True)
            return
        # One-use, per-thread handoff: validation and inference share the same
        # decoded input. Never retain images between jobs or across failures.
        pending.path = (server.data / job['id'] / 'input' / selected).resolve()
        pending.frame = frame
        try:
            original_execute(server, job, selected, spacing)
        finally:
            pending.__dict__.clear()

    prep.load_gray_frame = load_once
    prep._load_numeric = numeric
    web.CamusServer.execute = execute


def spacing_pair(value, np, error_type):
    try:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (2,) or not np.all(np.isfinite(array)) or np.any(array <= 0):
            raise ValueError('invalid shape/value')
        return tuple(float(v) for v in array)
    except (ValueError, TypeError, OverflowError) as exc:
        raise error_type('spacing must be two positive finite Y/X millimeter values') from exc
