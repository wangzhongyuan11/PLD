"""Recompute both complete frozen networks, feeding calculated outputs forward.

Host NumPy verification only: elapsed seconds are NOT board/RTL performance.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from budget import ROOT, REPO, MODELS
from compile_models import q24, skin_lut


def conv(x, w, bias):
    co, ci, k, _ = w.shape
    h, width = x.shape[1:]
    p = np.pad(x.astype(np.int32), ((0, 0), (k//2, k//2), (k//2, k//2)))
    result = np.zeros((co, h, width), dtype=np.int32)
    # Accumulate kernel positions separately; no giant im2col allocation.
    for ky in range(k):
        for kx in range(k):
            result += np.einsum('oc,chw->ohw', w[:, :, ky, kx].astype(np.int32),
                                p[:, ky:ky+h, kx:kx+width], optimize=False)
    return result.astype(np.int64) + bias[:, None, None]


def main():
    report = {'kind': 'full_graph_numpy_reference_not_rtl_or_board', 'models': {}}
    for name, (spec_path, spec_sha, _) in MODELS.items():
        raw = (REPO/spec_path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == spec_sha
        spec = json.loads(raw)
        compiled = json.loads((ROOT/'out/compiled_models'/name/'manifest.json').read_text())
        assert compiled['spec_sha256'] == spec_sha
        def read(path, dtype):
            raw = path.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == compiled['sources'][str(path.relative_to(REPO))]
            return np.frombuffer(raw, dtype=dtype)
        history = []
        records = []
        for li, layer in enumerate(spec['layers']):
            begin = time.perf_counter()
            directory = (REPO/spec_path).parent/layer['directory']
            tile = directory if name == 'skin' else directory/'tile'
            shape = layer.get('input_shape', layer.get('full_input_shape'))[1:]
            frozen_input = read(tile/'input_full.bin', np.int8).reshape(shape)
            skip = {8: 5, 10: 3, 12: 1}.get(li)
            if li == 0:
                x = frozen_input
            elif skip is not None:
                branches = [history[skip], history[-1].repeat(2, axis=1).repeat(2, axis=2)]
                identity = np.arange(256, dtype=np.uint8).view(np.int8)
                tables = ([skin_lut(spec['layers'][src]['output_scale'], layer['input_scale']) for src in (skip, li-1)]
                          if name == 'skin' else [q24(identity, m) for m in spec['concat'][(li-8)//2]['multiplier_q24']])
                x = np.concatenate([t[b.view(np.uint8)] for b, t in zip(branches, tables)])
            else:
                x = history[-1]
                if li in (2, 4, 6):
                    c, h, width = x.shape
                    x = x.reshape(c, h//2, 2, width//2, 2).max(axis=(2, 4))
                if name == 'skin':
                    x = skin_lut(spec['layers'][li-1]['output_scale'], layer['input_scale'])[x.view(np.uint8)]
            assert np.array_equal(x, frozen_input), (name, li, 'computed input differs')
            w = read(directory/'weight.bin', np.int8).reshape(layer['weight_shape'])
            bias = read(directory/'bias.bin', '<i4')
            acc = conv(x, w, bias)
            assert acc.min() >= -(1<<31) and acc.max() < (1<<31)
            output = np.stack([q24(a, m) for a, m in zip(acc, layer['multiplier_q24'])])
            if name == 'camus' and li < 14:
                output = np.maximum(output, 0)
            golden = read(tile/'output_full_golden.bin', np.int8).reshape(output.shape)
            assert np.array_equal(output, golden), (name, li, int(np.count_nonzero(output != golden)))
            history.append(output)
            records.append({'layer': li, 'equal': True, 'bytes': output.nbytes,
                            'acc_min': int(acc.min()), 'acc_max': int(acc.max()),
                            'output_sha256': hashlib.sha256(output.tobytes()).hexdigest(),
                            'host_reference_seconds': time.perf_counter()-begin})
            print(name, li, 'full output exact', flush=True)
        report['models'][name] = {'spec_sha256': spec_sha, 'layers': records}
    report['tool_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (ROOT/'evidence/full_reference.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
