"""Real model transition crops through spatial frontend and convolution.

Independent 8x8 crop convolution, not full-image boundary goldens or full graph.
"""
import hashlib
import json
import numpy as np
from budget import ROOT, REPO, MODELS
from compile_models import nchwc8, q24
from full_reference import conv
from model_vectors import packed


def main():
    lines, records, sources = [], [], {}
    for name, (spec_path, digest, _) in MODELS.items():
        raw = (REPO/spec_path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        spec = json.loads(raw)
        base = (REPO/spec_path).parent
        compiled = ROOT/'out/compiled_models'/name
        manifest = json.loads((compiled/'manifest.json').read_text())
        assert manifest['spec_sha256'] == digest
        def read(path, dtype):
            raw = path.read_bytes()
            key = str(path.relative_to(REPO))
            assert hashlib.sha256(raw).hexdigest() == manifest['sources'][key]
            sources[key] = hashlib.sha256(raw).hexdigest()
            return np.frombuffer(raw, dtype=dtype)
        def output(index):
            layer = spec['layers'][index]
            directory = base/layer['directory']
            if name == 'camus': directory /= 'tile'
            return read(directory/'output_full_golden.bin', np.int8).reshape(layer.get('output_shape', layer.get('full_output_shape'))[1:])
        for li, layer in enumerate(spec['layers']):
            info = manifest['layers'][li]
            directory = base/layer['directory']
            tile = directory if name == 'skin' else directory/'tile'
            lut_raw = (compiled/f'{li:02d}_luts.bin').read_bytes()
            assert hashlib.sha256(lut_raw).hexdigest() == manifest['artifacts'][f'{li:02d}_luts.bin']['sha256']
            lut = np.frombuffer(lut_raw, np.int8).reshape(2,256)
            skip = info['skip_layer']
            mode = 1 if info['mode'] == 'maxpool2' else 0
            if skip is not None:
                first = output(skip)[:, :8, :8]
                second = output(li-1)[:, :4, :4]
                transformed = np.concatenate((lut[0][first.view(np.uint8)], lut[1][second.repeat(2,axis=1).repeat(2,axis=2).view(np.uint8)]))
            else:
                second = np.zeros((0,4,4), dtype=np.int8)
                if li == 0:
                    first = read(tile/'input_full.bin', np.int8).reshape(info['input_shape'])[:, :8, :8]
                else:
                    size = 16 if mode == 1 else 8
                    first = output(li-1)[:, :size, :size]
                spatial = first.reshape(first.shape[0],8,2,8,2).max(axis=(2,4)) if mode == 1 else first
                transformed = lut[0][spatial.view(np.uint8)]
            actual_input = read(tile/'input_full.bin', np.int8).reshape(info['input_shape'])[:, :8, :8]
            assert np.array_equal(transformed, actual_input), (name, li, 'transition crop')
            weights = read(directory/'weight.bin', np.int8).reshape(layer['weight_shape'])
            bias = read(directory/'bias.bin', '<i4')
            count = min(8, info['co'])
            acc = conv(transformed, weights[:count], bias[:count])
            quant = np.stack([q24(a,m) for a,m in zip(acc,layer['multiplier_q24'])])
            if info['relu_in_quantizer']: quant = np.maximum(quant,0)
            b = np.zeros(8,dtype=np.int32); b[:count] = bias[:count]
            mult = np.zeros(8,dtype=np.int32); mult[:count] = layer['multiplier_q24'][:count]
            a_words = nchwc8(first).reshape(-1,8)
            b_words = nchwc8(second).reshape(-1,8) if second.shape[0] else np.zeros((0,8),np.int8)
            lines.append(f"{info['input_groups']} {info['kernel']} {packed(b,32):064x} {packed(mult,32):064x} {int(info['relu_in_quantizer'])} {mode} {int(skip is not None)} {(first.shape[0]+7)//8} {len(a_words)} {len(b_words)}")
            lines.extend(f'{v:02x}' for v in np.frombuffer(lut_raw,np.uint8))
            lines.extend(f'{packed(v,8):016x}' for v in a_words)
            lines.extend(f'{packed(v,8):016x}' for v in b_words)
            packed_weights = (compiled/f'{li:02d}_weights.bin').read_bytes()
            assert hashlib.sha256(packed_weights).hexdigest() == manifest['artifacts'][f'{li:02d}_weights.bin']['sha256']
            block = np.frombuffer(packed_weights[:info['weight_block_bytes']],'<u8')
            lines.extend(f'{int(v):016x}' for v in block)
            lines.extend(f'{packed(v,8):016x}' for v in nchwc8(quant).reshape(-1,8))
            records.append({'model':name,'layer':li,'mode':info['mode'],'output_channels':count})
    target = ROOT/'out/model_vectors/transformed_cases.txt'
    target.write_text(str(len(records))+'\n'+'\n'.join(lines)+'\n',encoding='ascii')
    report = {'kind':'official_transition_crop_with_independent_convolution_not_full_graph','cases':records,'sources':sources,
              'vector_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    (ROOT/'evidence/transformed_vectors.json').write_text(json.dumps(report,indent=2))
    print('Packed',len(records),'real-model transition/convolution crops')


if __name__ == '__main__': main()
