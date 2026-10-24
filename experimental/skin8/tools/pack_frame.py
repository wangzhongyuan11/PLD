"""Pack v0 experimental descriptors for single-submit frame_engine simulation."""
import hashlib
import json
import struct
from budget import ROOT


def main():
    report={'kind':'experimental_descriptor_abi_v0','models':{}}
    for name in ('skin','camus'):
        base=ROOT/'out/compiled_models'/name
        desc=json.loads((base/'descriptors.json').read_text())['descriptors']
        manifest=json.loads((base/'manifest.json').read_text())
        def artifact(filename):
            data=(base/filename).read_bytes()
            assert hashlib.sha256(data).hexdigest()==manifest['artifacts'][filename]['sha256']
            return data
        weights=[];address=0xc00000
        for li in range(15):
            weights.append(address);address+=(len(artifact(f'{li:02d}_weights.bin'))+63)//64*64
        assert address<=0xd00000
        binary=bytearray()
        for d in desc:
            li=d['layer'];a=d['branches'][0];b=d['branches'][1] if len(d['branches'])==2 else None
            flags=(d['kernel']==3)|(int(d['relu'])<<1)|(a['table']<<2)|((b['table'] if b else 1)<<3)
            flags|=(d['stripe_epoch_within_frame']%2)<<4
            flags|=(int(b is not None)<<5)|(int(d['reuse_features'])<<6)|(int(d['load_luts'])<<7)
            row=bytearray(128)
            struct.pack_into('<4I4H4B2I',row,0,a['source_offset'],b['source_offset'] if b else 0,
                weights[li]+d['weights_offset'],d['destination_offset'],d['width'],d['height'],d['y'],d['rows'],
                d['groups'],a['groups'],a['mode'],flags,d['stripe_epoch_within_frame'],0xd00000+li*512)
            row[48:112]=artifact(d['params_file'])[d['params_offset']:d['params_offset']+64]
            binary.extend(row)
        assert 0xe00000+len(binary)<=0x1000000
        target=ROOT/'out/model_vectors'/('graph_'+name)
        def emit(fp,address,data):
            fp.write(f'@{address//8:x}\n')
            for v in struct.iter_unpack('<Q',data):fp.write(f'{v[0]:016x}\n')
        with (target/'frame.mem').open('w') as fp:
            fp.write((target/'initial.mem').read_text())
            for li in range(15):emit(fp,0xd00000+li*512,artifact(f'{li:02d}_luts.bin'))
            emit(fp,0xe00000,binary)
        (base/'descriptors.bin').write_bytes(binary)
        report['models'][name]={'count':len(desc),'descriptor_address':0xe00000,
            'binary_sha256':hashlib.sha256(binary).hexdigest(),
            'frame_mem_sha256':hashlib.sha256((target/'frame.mem').read_bytes()).hexdigest()}
        print(name,len(desc),'binary descriptors packed')
    (ROOT/'evidence/packed_frame.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
