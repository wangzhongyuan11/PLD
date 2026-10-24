"""Full-size exported inter-layer tensors -> RTL spatial/LUT row fixtures."""
import hashlib
import json
import numpy as np
from budget import ROOT,REPO,MODELS
from compile_models import nchwc8

def main():
    records=[];cases=[];sources={}
    for model,(path,sha,_) in MODELS.items():
        base=(REPO/path).parent;raw=(REPO/path).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha
        spec=json.loads(raw);compiled=ROOT/'out/compiled_models'/model
        manifest=json.loads((compiled/'manifest.json').read_text())
        assert manifest['spec_sha256']==sha
        history=[]
        for li,layer in enumerate(spec['layers']):
            directory=base/layer['directory'];tile=directory if model=='skin' else directory/'tile'
            def read(name,shape):
                file=tile/name;data=file.read_bytes();key=str(file.relative_to(REPO))
                digest=hashlib.sha256(data).hexdigest();assert digest==manifest['sources'][key]
                sources[key]=digest;return np.frombuffer(data,np.int8).reshape(shape)
            inp=read('input_full.bin',layer.get('input_shape',layer.get('full_input_shape'))[1:])
            out=read('output_full_golden.bin',layer.get('output_shape',layer.get('full_output_shape'))[1:])
            if li:
                skip={8:5,10:3,12:1}.get(li)
                branches=[(history[skip],0),(history[-1],2)] if skip is not None else [(history[-1],1 if li in (2,4,6) else 0)]
                lutname=f'{li:02d}_luts.bin';lutdata=(compiled/lutname).read_bytes()
                assert hashlib.sha256(lutdata).hexdigest()==manifest['artifacts'][lutname]['sha256']
                offset=0
                for branch,(source,mode) in enumerate(branches):
                    c,sh,sw=source.shape;target=inp[offset:offset+c];offset+=c
                    _,th,tw=target.shape;source8=nchwc8(source);target8=nchwc8(target);groups=source8.shape[0]
                    lut=lutdata[branch*256:(branch+1)*256]
                    records.append(f'{groups} {th} {sw} {tw} {mode} {int.from_bytes(lut,"little"):0512x}')
                    for cg in range(groups):
                        for y in range(th):
                            rowmode=3 if mode==2 and y%2 else mode
                            records.append(str(rowmode))
                            rows=[y] if mode==0 else [2*y,2*y+1] if mode==1 else [y//2] if rowmode==2 else []
                            for sy in rows:
                                for word in source8[cg,sy]:records.append(f'{int.from_bytes(word.tobytes(),"little"):016x}')
                            for word in target8[cg,y]:records.append(f'{int.from_bytes(word.tobytes(),"little"):016x}')
                    cases.append({'model':model,'layer':li,'branch':branch,'mode':mode,'groups':groups,'height':th,'width':tw})
            history.append(out)
    target=ROOT/'out/model_vectors/transitions.txt'
    target.write_text(str(len(cases))+'\n'+'\n'.join(records)+'\n',encoding='ascii')
    receipt={'kind':'exported_full_size_transition_fixtures_not_full_graph','cases':cases,'sources':sources,
             'vectors_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
    (ROOT/'evidence/transition_vectors.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print('Compiled',len(cases),'spatial/mapping branches; bytes',target.stat().st_size)

if __name__=='__main__':main()
