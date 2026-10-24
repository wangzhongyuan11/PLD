"""DDR initialization and independent golden memory for full-frame RTL tests.

Only first input and weights enter DUT-readable memory. Intermediate goldens
are in a distinct scoreboard memory and never supplied to the AXI reader.
"""
import hashlib
import json
import numpy as np
from budget import ROOT, REPO, MODELS
from compile_models import nchwc8
from model_vectors import packed


def main():
    receipt = {'kind':'full_frame_rtl_vector_generation','models':{}}
    for name, (spec_path, digest, _) in MODELS.items():
        compiled = ROOT/'out/compiled_models'/name
        manifest = json.loads((compiled/'manifest.json').read_text())
        desc = json.loads((compiled/'descriptors.json').read_text())['descriptors']
        plan = json.loads((ROOT/'evidence/stripe_schedule.json').read_text())['models'][name]
        raw = (REPO/spec_path).read_bytes(); assert hashlib.sha256(raw).hexdigest() == digest
        spec = json.loads(raw)
        target = ROOT/'out/model_vectors'/('graph_'+name); target.mkdir(exist_ok=True)
        def artifact(filename):
            data = (compiled/filename).read_bytes()
            assert hashlib.sha256(data).hexdigest() == manifest['artifacts'][filename]['sha256']
            return data
        def emit(fp, address, data):
            assert address % 8 == 0 and len(data) % 8 == 0
            fp.write(f'@{address//8:x}\n')
            fp.writelines(f'{int(v):016x}\n' for v in np.frombuffer(data,'<u8'))
        weight_addresses = []
        with (target/'initial.mem').open('w') as fp:
            emit(fp,0,artifact('first_input_nchwc8.bin'))
            address = 0xc00000
            for li in range(15):
                data = artifact(f'{li:02d}_weights.bin'); weight_addresses.append(address)
                emit(fp,address,data); address += (len(data)+63)//64*64
            assert address < 0x1000000
        with (target/'golden.mem').open('w') as fp:
            for li, layer in enumerate(spec['layers']):
                directory = (REPO/spec_path).parent/layer['directory']
                if name == 'camus': directory /= 'tile'
                path = directory/'output_full_golden.bin'; raw = path.read_bytes()
                assert hashlib.sha256(raw).hexdigest() == manifest['sources'][str(path.relative_to(REPO))]
                tensor = np.frombuffer(raw,np.int8).reshape(manifest['layers'][li]['output_shape'])
                emit(fp,plan['layers'][li]['output_offset'],nchwc8(tensor).tobytes())
        with (target/'commands.txt').open('w') as fp:
            fp.write(str(len(desc))+'\n')
            for d in desc:
                li = d['layer']; b = d['branches']; second = b[1] if len(b)>1 else None
                params = artifact(d['params_file'])[d['params_offset']:d['params_offset']+64]
                bias = packed(np.frombuffer(params[:32],'<i4'),32)
                mult = packed(np.frombuffer(params[32:],'<i4'),32)
                values = [li,d['groups'],d['kernel'],int(d['relu']),d['width'],d['height'],d['y'],d['rows'],
                          b[0]['mode'],int(second is not None),b[0]['groups'],int(d['reuse_features']),d['stripe_epoch_within_frame'],int(d['load_luts'])]
                addresses = [b[0]['source_offset'],second['source_offset'] if second else 0,
                             weight_addresses[li]+d['weights_offset'],d['destination_offset']]
                fp.write(' '.join(map(str,values))+' '+' '.join(f'{v:08x}' for v in addresses)+f' {bias:064x} {mult:064x}\n')
                if d['load_luts']: fp.write(' '.join(f'{v:02x}' for v in artifact(d['lut_file']))+'\n')
        receipt['models'][name] = {'descriptors':len(desc),'files':{
            p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in target.iterdir()}}
        print(name,'full-frame DDR vectors generated',flush=True)
    (ROOT/'evidence/graph_vectors.json').write_text(json.dumps(receipt,indent=2))


if __name__ == '__main__': main()
