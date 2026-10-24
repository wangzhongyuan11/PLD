"""Prepare immutable FPGA parameters and independently verify full-size transitions.

Uses frozen per-layer exported outputs as transition inputs. This is NOT full
network inference: actual RTL full-frame validation remains required.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from budget import ROOT, REPO, MODELS

def sha(data):return hashlib.sha256(data).hexdigest()

def q24(values,multiplier):
    p=values.astype(np.int64)*int(multiplier);q=p>>24;r=p&((1<<24)-1)
    return np.clip(q+((r>(1<<23))|((r==(1<<23))&((q&1)!=0))),-127,127).astype(np.int8)

def skin_lut(source,target):
    x=np.maximum(np.arange(256,dtype=np.uint8).view(np.int8),0)
    return np.clip(np.rint((x.astype(np.float32)*np.float32(source))/np.float32(target)),-127,127).astype(np.int8)

def nchwc8(data):
    c,h,w=data.shape
    padded=np.zeros((math.ceil(c/8)*8,h,w),dtype=np.int8);padded[:c]=data
    return np.ascontiguousarray(padded.reshape(-1,8,h,w).transpose(0,2,3,1))

def main():
    report={'kind':'parameter_compile_and_exported_transition_validation_not_full_inference','models':{}}
    for name,(spec_path,spec_sha,_) in MODELS.items():
        base=(REPO/spec_path).parent;raw=(REPO/spec_path).read_bytes();assert sha(raw)==spec_sha
        spec=json.loads(raw);out=ROOT/'out/compiled_models'/name;out.mkdir(parents=True,exist_ok=True)
        history=[];descriptors=[];source_hashes={};artifacts={}
        def save(path,data):
            (out/path).write_bytes(data);artifacts[path]={'bytes':len(data),'sha256':sha(data)}
        for li,layer in enumerate(spec['layers']):
            directory=base/layer['directory'];tile=directory if name=='skin' else directory/'tile'
            if name=='skin':
                hashes=json.loads((directory/'config.json').read_text())['files_sha256']
            else:
                hashes={'input_full.bin':layer['tile']['input_full']['sha256'],
                        'output_full_golden.bin':layer['tile']['output_full_golden']['sha256'],
                        'weight.bin':layer['weight_file']['sha256'],'bias.bin':layer['bias_file']['sha256']}
            def read(path,dtype):
                data=path.read_bytes();assert sha(data)==hashes[path.name],str(path)
                source_hashes[str(path.relative_to(REPO))]=sha(data)
                return np.frombuffer(data,dtype=dtype)
            ci=layer['weight_shape'][1];co=layer['weight_shape'][0];k=layer['weight_shape'][2]
            in_shape=layer.get('input_shape',layer.get('full_input_shape'))[1:]
            out_shape=layer.get('output_shape',layer.get('full_output_shape'))[1:]
            inp=read(tile/'input_full.bin',np.int8).reshape(in_shape)
            output=read(tile/'output_full_golden.bin',np.int8).reshape(out_shape)
            weights=read(directory/'weight.bin',np.int8).reshape(layer['weight_shape'])
            bias=read(directory/'bias.bin','<i4')
            identity=np.arange(256,dtype=np.uint8).view(np.int8)
            tables=[identity.copy(),identity.copy()]
            skip={8:5,10:3,12:1}.get(li)
            mode='input' if li==0 else 'concat_nearest2' if skip is not None else 'maxpool2' if li in (2,4,6) else 'map'
            if li:
                prev=history[-1]
                if skip is not None:
                    branches=[history[skip],prev.repeat(2,axis=1).repeat(2,axis=2)]
                    if name=='skin':
                        tables=[skin_lut(spec['layers'][source]['output_scale'],layer['input_scale']) for source in (skip,li-1)]
                    else:
                        mults=spec['concat'][(li-8)//2]['multiplier_q24']
                        tables=[q24(identity,m) for m in mults]
                    mapped=np.concatenate([t[b.view(np.uint8)] for b,t in zip(branches,tables)],axis=0)
                else:
                    if mode=='maxpool2':
                        c,h,w=prev.shape;prev=prev.reshape(c,h//2,2,w//2,2).max(axis=(2,4))
                    if name=='skin':tables[0]=skin_lut(spec['layers'][li-1]['output_scale'],layer['input_scale'])
                    mapped=tables[0][prev.view(np.uint8)]
                if not np.array_equal(mapped,inp):
                    mismatch=np.argwhere(mapped!=inp)[0].tolist()
                    raise AssertionError((name,li,'full-size transition mismatch',mismatch))
            else:save('first_input_nchwc8.bin',nchwc8(inp).tobytes())
            assert all(t[0]==0 for t in tables)
            wp=[];params=[]
            groups=(ci+7)//8
            for oc in range(0,co,8):
                b=np.zeros(8,dtype='<i4');m=np.zeros(8,dtype='<i4');valid=min(8,co-oc)
                b[:valid]=bias[oc:oc+valid];m[:valid]=layer['multiplier_q24'][oc:oc+valid]
                params.extend((b.tobytes(),m.tobytes()))
                for o in range(8):
                    for ky in range(k):
                        for kx in range(k):
                            for cg in range(groups):
                                vec=np.zeros(8,dtype=np.int8)
                                for c in range(8):
                                    if oc+o<co and cg*8+c<ci:vec[c]=weights[oc+o,cg*8+c,ky,kx]
                                wp.append(vec.tobytes())
            prefix=f'{li:02d}'
            save(prefix+'_weights.bin',b''.join(wp));save(prefix+'_params.bin',b''.join(params))
            save(prefix+'_luts.bin',b''.join(t.tobytes() for t in tables))
            descriptors.append({'index':li,'name':layer['name'],'mode':mode,'ci':ci,'co':co,'kernel':k,
                'input_shape':in_shape,'output_shape':out_shape,'input_groups':groups,'output_groups':(co+7)//8,
                'weight_block_bytes':8*groups*k*k*8,'parameter_block_bytes':64,'skip_layer':skip,
                'relu_in_quantizer':name=='camus' and li<14,'transition_equal':True if li else None})
            history.append(output)
        save('golden_head_nchwc8.bin',nchwc8(history[-1]).tobytes())
        manifest={'spec_sha256':spec_sha,'layers':descriptors,'artifacts':artifacts,'sources':source_hashes}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        report['models'][name]={'spec_sha256':spec_sha,'transitions_equal':14,'compiled_layers':15,
            'manifest_sha256':sha((out/'manifest.json').read_bytes()),'output_dir':str(out)}
        print(name,'14 full-size transitions exact; 15 layer parameter sets compiled',flush=True)
    (ROOT/'evidence/compiled_models.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':main()
