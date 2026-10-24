"""Pack real frozen 8x8 layer fixtures for RTL; golden comes from model export."""
import hashlib
import json
from pathlib import Path
import numpy as np
from budget import ROOT, REPO, MODELS

def packed(values, bits):
    return sum((int(v) & ((1 << bits)-1)) << (bits*i) for i, v in enumerate(values))

def main():
    out = ROOT / 'out/model_vectors'
    out.mkdir(parents=True, exist_ok=True)
    records, engine_records, cases, sources = [], [], [], {}
    for name, (spec_path, sha, _) in MODELS.items():
        base = (REPO/spec_path).parent
        raw = (REPO/spec_path).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == sha
        spec = json.loads(raw)
        for li, layer in enumerate(spec['layers']):
            directory = base/layer['directory']
            tile = directory if name == 'skin' else directory/'tile'
            co, ci, kh, kw = layer['weight_shape']
            def read(path, dtype, expected):
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                assert digest == expected, path
                sources[str(path.relative_to(REPO))] = digest
                return np.frombuffer(data, dtype=dtype)
            if name == 'skin':
                filehash = json.loads((directory/'config.json').read_text())['files_sha256']
                shas = [filehash[k] for k in ('input.bin','weight.bin','bias.bin','acc_golden.bin','output_golden.bin')]
            else:
                shas = [layer['tile']['input']['sha256'], layer['weight_file']['sha256'], layer['bias_file']['sha256'],
                        layer['tile']['acc_golden']['sha256'],layer['tile']['output_golden']['sha256']]
            inp = read(tile/'input.bin', np.int8, shas[0]).reshape(ci,8,8)
            weights = read(directory/'weight.bin', np.int8, shas[1]).reshape(co,ci,kh,kw)
            bias = read(directory/'bias.bin', '<i4', shas[2])
            golden = read(tile/'acc_golden.bin', '<i4', shas[3]).reshape(co,8,8)
            quant = read(tile/'output_golden.bin', np.int8, shas[4]).reshape(co,8,8)
            # Independently validate fixture interpretation before packaging.
            pad = kh//2
            x = np.pad(inp.astype(np.int64), ((0,0),(pad,pad),(pad,pad)))
            ref = np.broadcast_to(bias[:,None,None].astype(np.int64),(co,8,8)).copy()
            for ky in range(kh):
                for kx in range(kw):
                    ref += np.einsum('oc,cyx->oyx',weights[:,:,ky,kx].astype(np.int64),x[:,ky:ky+8,kx:kx+8])
            assert np.array_equal(ref,golden), (name,li,'fixture accumulator')
            prod = ref*np.array(layer['multiplier_q24'],np.int64)[:,None,None]
            floor = prod >> 24
            rem = prod & ((1<<24)-1)
            rq = np.clip(floor+((rem>(1<<23)) | ((rem==(1<<23)) & ((floor&1)!=0))),-127,127)
            if name == 'camus' and layer['tile_relu']: rq = np.maximum(rq,0)
            assert np.array_equal(rq,quant), (name,li,'fixture requantization')
            groups=(ci+7)//8
            iw=8+2*pad
            for oc in range(0,co,8):
                b=np.zeros(8,dtype=np.int32);b[:min(8,co-oc)]=bias[oc:oc+8]
                mult=np.zeros(8,dtype=np.int32)
                mult[:min(8,co-oc)]=layer['multiplier_q24'][oc:oc+8]
                relu=int(name=='camus' and layer['tile_relu'])
                records.append(f'{iw} {iw} {groups} {kh} {packed(b,32):064x} {packed(mult,32):064x} {relu}')
                engine_records.append(f'{groups} {kh} {packed(b,32):064x} {packed(mult,32):064x} {relu}')
                for cg in range(groups):
                    for yy in range(8):
                        for xx in range(8):
                            vec=[int(inp[c,yy,xx]) if c<ci else 0 for c in range(cg*8,cg*8+8)]
                            engine_records.append(f'{packed(vec,8):016x}')
                # All banks, including odd-width padding, are initialized.
                for y in range(iw):
                    for xp in range((iw+1)//2*2):
                        for cg in range(groups):
                            vec=[int(x[c,y,xp]) if c<ci and xp<iw else 0 for c in range(cg*8,cg*8+8)]
                            addr=(y*((iw+1)//2)+xp//2)*groups+cg
                            records.append(f'{xp%2} {addr} {packed(vec,8):016x}')
                for o in range(8):
                    for ky in range(kh):
                        for kx in range(kw):
                            for cg in range(groups):
                                vec=[int(weights[oc+o,c,ky,kx]) if oc+o<co and c<ci else 0 for c in range(cg*8,cg*8+8)]
                                addr=(ky*kh+kx)*groups+cg
                                records.append(f'{o} {addr} {packed(vec,8):016x}')
                                engine_records.append(f'{packed(vec,8):016x}')
                for y in range(8):
                    for xp in range(0,8,2):
                        vals=[int(golden[oc+o,y,xp+p]) if oc+o<co else 0 for p in range(2) for o in range(8)]
                        quantvals=[int(quant[oc+o,y,xp+p]) if oc+o<co else 0 for p in range(2) for o in range(8)]
                        records.append(f'{packed(vals,32):0128x} {packed(quantvals,8):032x}')
                for yy in range(8):
                    for xx in range(8):
                        vals=[int(quant[oc+o,yy,xx]) if oc+o<co else 0 for o in range(8)]
                        engine_records.append(f'{packed(vals,8):016x}')
                cases.append({'model':name,'layer':li,'output_channel_start':oc,'ci':ci,'co':co})
    payload = str(len(cases))+'\n'+'\n'.join(records)+'\n'
    target=out/'cases.txt'
    target.write_text(payload,encoding='ascii')
    engine_target=out/'engine_cases.txt'
    engine_target.write_text(str(len(cases))+'\n'+'\n'.join(engine_records)+'\n',encoding='ascii')
    receipt={'kind':'frozen_model_layer_fixtures_not_full_graph','cases':cases,'sources':sources,
             'vectors_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
             'engine_vectors_sha256':hashlib.sha256(engine_target.read_bytes()).hexdigest(),'checked_fixture_acc_and_quant':True}
    (ROOT/'evidence/model_vectors.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(f'Packed {len(cases)} real output-channel blocks from 30 frozen layers: {target}')

if __name__ == '__main__': main()
