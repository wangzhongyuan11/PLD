"""Per-channel PTQ, integer-valued execution and exact independent tile oracle.

Accumulators use validated CUDA int8/int32 GEMM where available, or an exact
FP64 integer emulator under the proven <2**31 bound. Activations remain int8
between nodes. This is computer-side reference execution, not PYNQ inference.
"""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .common import DATA,RUN,ARCHITECTURE,save_json,sha256,utc_now,LABELS
from .model import ACDCUNet,LAYER_NAMES
from .geometry import stack
from .contract import parameter_blob,conv_channel_chunks,contract_document
from software.role_a_camus_int8 import integer as kernel

def load_fp32(checkpoint,device='cpu'):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['architecture']!=ARCHITECTURE or state['split_sha256']!=sha256(DATA/'SPLIT.json'):raise ValueError('wrong checkpoint/split')
    model=ACDCUNet();model.load_state_dict(state['model'],strict=True);model.contract();return model.to(device).eval(),state

def calibration_samples(direction):
    split=json.loads((DATA/'SPLIT.json').read_text(encoding='utf-8'))
    # Only training patients; every disease group represented. No label reads.
    selected=[]
    for group,pts in sorted(split['groups'].items()):selected.extend(sorted(set(pts)&set(split['train']))[:6])
    selected.extend(sorted(set(split['train'])-set(selected))[:2])
    assert len(selected)==32 and not set(selected)&set(split['dev'])
    manifest=json.loads((DATA/'CACHE_MANIFEST.json').read_text(encoding='utf-8'))
    axis={'axial':2,'coronal':1,'sagittal':0}[direction]
    for r in manifest['volumes']:
        if r['patient'] not in selected:continue
        x=np.load(DATA/r['relative']/'image.npy',mmap_mode='r')
        for fraction in (.35,.65):
            index=int(round((x.shape[axis]-1)*fraction));yield stack(x,direction,index)[0],{'key':r['key'],'patient':r['patient'],'index':index}

@torch.inference_mode()
def calibrate(model,direction,device,baseline=False):
    maxima={n:0. for n in LAYER_NAMES};handles=[];roster=[]
    def hook(name):
        def apply(_module,_input,out):maxima[name]=max(maxima[name],float(out.detach().abs().amax().item()))
        return apply
    for name,module in model.named_modules():
        if name in maxima:handles.append(module.register_forward_hook(hook(name)))
    for x,meta in calibration_samples(direction):
        if baseline:x=np.repeat(x[1:2],3,axis=0)
        model(torch.from_numpy(x[None]).to(device));roster.append(meta)
    for h in handles:h.remove()
    if len(roster)!=128:raise ValueError('expected 32 patients x 2 phases x 2 stacks')
    outputs={n:max(v/127,1e-8) for n,v in maxima.items()}
    inputs={};previous=1/127;skips=[]
    for level in range(4):
        for conv in (0,2):
            n=f'encoders.{level}.{conv}';inputs[n]=previous;previous=outputs[n]
        skips.append(previous)
    cats=[]
    for decoder,level in enumerate((2,1,0)):
        common=max(skips[level],previous)
        cats.append({'level':level,'skip_scale':skips[level],'deep_scale':previous,'common_scale':common,
                     'skip_multiplier_q24':int(np.rint(skips[level]/common*2**24)),
                     'deep_multiplier_q24':int(np.rint(previous/common*2**24)),'skip_first':True})
        previous=common
        for conv in (0,2):
            n=f'decoders.{decoder}.{conv}';inputs[n]=previous;previous=outputs[n]
    inputs['head']=previous
    return inputs,outputs,cats,{'method':'train-only max-abs symmetric PTQ; per-output-channel weights',
             'roster':roster,'patients':32,'stacks':128,'max_abs':maxima,'dev_or_test_access':False}

def requant_torch(value,multiplier):
    p=value.to(torch.int64)*multiplier.to(torch.int64)
    a=p.abs();q=a>>24;r=a&((1<<24)-1)
    q=q+((r>(1<<23))|((r==(1<<23))&((q&1)==1))).to(torch.int64)
    return torch.where(p<0,-q,q).clamp(-127,127).to(torch.int8)

class IntegerModel(torch.nn.Module):
    def __init__(self,layers,cats,device='cpu',accumulator_backend=None):
        super().__init__();self.layers=layers;self.cats=cats;self.device=torch.device(device)
        self.weights=[torch.from_numpy(l.q_weight.astype(np.float64)).to(device) for l in layers]
        self.biases=[torch.from_numpy(l.q_bias.astype(np.float64)).to(device) for l in layers]
        self.multipliers=[torch.from_numpy(l.multiplier.astype(np.int64)).to(device).view(1,-1,1,1) for l in layers]
        self.head_scale=layers[-1].output_scale
        self.accumulator_backend=accumulator_backend or ('cuda_int8_gemm' if self.device.type=='cuda' and hasattr(torch,'_int_mm') else 'exact_fp64')
        if self.accumulator_backend not in ('cuda_int8_gemm','exact_fp64'):raise ValueError('unknown exact accumulator')
        self.gemm_weights=[];self.int_biases=[]
        if self.accumulator_backend=='cuda_int8_gemm':
            from .int8_gemm import pack_weight
            self.gemm_weights=[pack_weight(torch.from_numpy(l.q_weight.copy()).to(device)) for l in layers]
            self.int_biases=[torch.from_numpy(l.q_bias.copy()).to(device) for l in layers]
    def conv(self,x,index):
        layer=self.layers[index]
        if self.accumulator_backend=='cuda_int8_gemm':
            from .int8_gemm import conv_int32
            acc=conv_int32(x,self.gemm_weights[index],layer.kernel,layer.padding,self.int_biases[index])
        else:acc=F.conv2d(x.to(torch.float64),self.weights[index],self.biases[index],padding=layer.padding)
        q=requant_torch(acc.to(torch.int64),self.multipliers[index])
        return q.clamp_min(0) if index<14 else q
    @torch.inference_mode()
    def quantized(self,x):
        if not x.is_floating_point() or x.ndim!=4 or x.shape[1]!=3 or min(x.shape[-2:])<8 or any(n%8 for n in x.shape[-2:]):raise ValueError('normalized three-channel NCHW entry required')
        if not torch.isfinite(x).all() or x.min()<-1 or x.max()>1:raise ValueError('normalized input outside [-1,1]')
        x=(x.to(torch.float64)*127).round().clamp(-127,127).to(torch.int8);skips=[];i=0
        for level in range(4):
            x=self.conv(x,i);x=self.conv(x,i+1);i+=2;skips.append(x)
            if level<3:
                n,c,h,w=x.shape;x=x.reshape(n,c,h//2,2,w//2,2).amax((3,5))
        for cat,skip in zip(self.cats,reversed(skips[:-1])):
            s=requant_torch(skip,torch.tensor(cat['skip_multiplier_q24'],device=x.device))
            d=requant_torch(x,torch.tensor(cat['deep_multiplier_q24'],device=x.device))
            x=torch.cat((s,d.repeat_interleave(2,2).repeat_interleave(2,3)),1)
            x=self.conv(x,i);x=self.conv(x,i+1);i+=2
        return self.conv(x,14)
    def forward(self,x):return self.quantized(x).float()*self.head_scale

def export(checkpoint,direction,output=None,*,calibration_override=None):
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    checkpoint=Path(checkpoint);model,state=load_fp32(checkpoint,device)
    if calibration_override is None:
        inputs,outputs,cats,calibration=calibrate(model,direction,device,state['config']['baseline'])
    else:
        p=calibration_override
        if (p!=state.get('quantization_profile') or p.get('schema')!='acdc-qat-fixed-ranges-v1'
                or p['direction']!=direction or p['split_sha256']!=sha256(DATA/'SPLIT.json') or state['config']['baseline']):
            raise ValueError('fixed QAT profile/checkpoint mismatch')
        inputs,outputs,cats=p['inputs'],p['outputs'],p['concat']
        if set(inputs)!=set(LAYER_NAMES) or set(outputs)!=set(LAYER_NAMES) or len(cats)!=3:
            raise ValueError('incomplete fixed QAT scale graph')
        if any(not np.isfinite(v) or v<=0 for v in [*inputs.values(),*outputs.values()]):raise ValueError('invalid fixed scales')
        split=json.loads((DATA/'SPLIT.json').read_text(encoding='utf-8'));roster=p['calibration']['roster']
        if len(roster)!=128 or len({r['patient'] for r in roster})!=32 or any(r['patient'] not in split['train'] for r in roster):
            raise ValueError('QAT calibration data boundary mismatch')
        calibration={**p['calibration'],'fixed_qat_ranges':True,'recalibration_at_export':False,
            'quantization_profile_sha256':state['quantization_profile_sha256'],
            'method':'frozen train-only activation ranges used throughout QAT; no new calibration forward at export'}
    convs=dict(model.named_modules());layers=[kernel.build_layer(convs[n],n,input_scale=inputs[n],output_scale=outputs[n]) for n in LAYER_NAMES]
    output=Path(output or RUN/'exports'/state['config']['name']);output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for i,layer in enumerate(layers):
        d=output/f'C{i+1:02}';d.mkdir(exist_ok=True)
        files={}
        for name,array in [('weight.bin',layer.q_weight),('bias.bin',layer.q_bias.astype('<i4')),('multipliers.bin',layer.multiplier.astype('<i4'))]:
            f=d/name;f.write_bytes(array.tobytes());files[name]={'path':f.relative_to(output).as_posix(),'bytes':f.stat().st_size,'sha256':sha256(f)}
        blob=d/'parameters.bin';blob.write_bytes(parameter_blob(layer.q_weight,layer.q_bias,layer.multiplier))
        row=layer.as_dict();row.update(id=f'C{i+1:02}',files=files,parameter_blob_sha256=sha256(blob),
                channel_chunks=conv_channel_chunks(layer.q_weight.shape[1],layer.q_weight.shape[0],layer.kernel),relu=i<14)
        # Independent small-tile NumPy integer convolution vs exact FP64 emulator.
        rng=np.random.default_rng(20261017+i);q=rng.integers(-127,128,(1,layer.q_weight.shape[1],8,8),dtype=np.int8)
        report=kernel.verify_against_float64_graph(layer,kernel.Int8Feature(q,layer.input_scale),relu=i<14)
        if not report['matches_oracle'] or report['mismatch_count']:raise ValueError(f'independent integer oracle failed: {layer.name}')
        row['independent_oracle']=report
        rows.append(row)
    spec={'schema':'acdc-int8-model-v1','source_kind':'trained','architecture':ARCHITECTURE,'direction':direction,
        'profile_id':'acdc_baseline2d_reference' if state['config']['baseline'] else 'acdc_'+direction+'_v1','source_checkpoint_sha256':sha256(checkpoint),'selected_epoch':state['epoch'],
        'source_dev_dice':state['dev_dice'],'split_sha256':state['split_sha256'],'input_shape':[1,3,256,256],
        'input_scale':1/127,'output_shape':[1,4,256,256],'head_scale':outputs['head'],'labels':LABELS,
        'head_activation':'none','layers':rows,'concat':cats,'calibration':calibration,
        'contract':contract_document(),'board_validated':False,'backend':'exact integer arithmetic software emulation'}
    save_json(output/'model_spec.json',spec)
    # Real normalized training input, without GT, for whole-graph golden.
    sample,meta=next(calibration_samples(direction))
    if state['config']['baseline']:sample=np.repeat(sample[1:2],3,axis=0)
    engine=IntegerModel(layers,cats,device)
    with torch.inference_mode():head=engine.quantized(torch.from_numpy(sample[None]).to(device)).cpu().numpy()
    (output/'golden_input.bin').write_bytes(np.clip(np.rint(sample[None].astype(np.float64)*127),-127,127).astype(np.int8).tobytes())
    (output/'golden_head.bin').write_bytes(head.tobytes())
    save_json(output/'golden.json',{'source':meta,'synthetic':False,'backend':'integer software reference; not board',
          'input_sha256':sha256(output/'golden_input.bin'),'head_sha256':sha256(output/'golden_head.bin'),
          'spec_sha256':sha256(output/'model_spec.json'),'head_scale':outputs['head']})
    return output

def load_integer(folder,device='cpu'):
    folder=Path(folder);spec=json.loads((folder/'model_spec.json').read_text(encoding='utf-8'))
    if spec.get('schema')!='acdc-int8-model-v1' or spec.get('source_kind')!='trained' or spec['architecture']!=ARCHITECTURE or spec['input_shape']!=[1,3,256,256] or spec['output_shape']!=[1,4,256,256]:raise ValueError('trained model contract mismatch')
    if tuple(r['name'] for r in spec['layers'])!=LAYER_NAMES:raise ValueError('wrong layer topology')
    layers=[]
    topology=ACDCUNet().contract()
    for row,expected in zip(spec['layers'],topology):
        if row['weight_shape']!=[expected['out_channels'],expected['in_channels'],expected['kernel'],expected['kernel']]:raise ValueError('layer dimensions changed')
        if row['kernel']!=expected['kernel'] or row['padding']!=expected['padding'] or row['relu']!=expected['relu']:raise ValueError('layer semantics changed')
        if not all(np.isfinite(row[n]) and row[n]>0 for n in ('input_scale','output_scale')):raise ValueError('invalid scales')
        a={}
        for n,r in row['files'].items():
            path=folder/r['path']
            if not path.resolve().is_relative_to(folder.resolve()) or path.stat().st_size!=r['bytes'] or sha256(path)!=r['sha256']:raise ValueError('parameter identity mismatch')
            a[n]=np.fromfile(path,dtype=np.int8 if n=='weight.bin' else '<i4')
        w=a['weight.bin'].reshape(row['weight_shape']);b=a['bias.bin'];mult=a['multipliers.bin'];bound=kernel.require_bounds(w,b)
        if np.any(bound*mult.astype(np.int64)>2**63-1):raise ValueError('Q24 product overflow')
        layer=kernel.QuantLayer(row['name'],w,b,mult,row['input_scale'],row['output_scale'],row['padding'],row['kernel'],bound,True)
        kernel.check_layer(layer);layers.append(layer)
    if spec['head_scale']!=layers[-1].output_scale or spec['labels']!=LABELS:raise ValueError('head class/scale mismatch')
    for cat in spec['concat']:
        if any(not isinstance(cat[n],int) or not 1<=cat[n]<=2**31-1 for n in ('skip_multiplier_q24','deep_multiplier_q24')):raise ValueError('invalid CAT multiplier')
    return IntegerModel(layers,spec['concat'],device),spec

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',choices=['baseline2d','axial','coronal','sagittal'],required=True);a=p.parse_args()
    print(export(RUN/'training'/a.name/'best.pt','axial' if a.name=='baseline2d' else a.name))
