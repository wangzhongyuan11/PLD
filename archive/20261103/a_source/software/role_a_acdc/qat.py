"""QAT repair of the frozen topology; original training code remains untouched.

Fake quantization is a training surrogate, never a board/integer accuracy proof.
Final acceptance always executes exported Q24 integer parameters separately.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from .common import RUN,DATA,save_json,sha256
from .model import LAYER_NAMES
from .quantize import load_fp32,calibration_samples,IntegerModel
from software.role_a_camus_int8 import integer as kernel


def propagate_scales(outputs):
    if set(outputs)!=set(LAYER_NAMES) or any(not np.isfinite(v) or v<=0 for v in outputs.values()):
        raise ValueError('all fifteen positive finite output scales required')
    inputs={};previous=1/127;skips=[];cats=[]
    for level in range(4):
        for conv in (0,2):
            name=f'encoders.{level}.{conv}';inputs[name]=previous;previous=outputs[name]
        skips.append(previous)
    for decoder,level in enumerate((2,1,0)):
        common=max(skips[level],previous)
        cats.append({'level':level,'skip_scale':skips[level],'deep_scale':previous,'common_scale':common,
            'skip_multiplier_q24':int(np.rint(skips[level]/common*2**24)),
            'deep_multiplier_q24':int(np.rint(previous/common*2**24)),'skip_first':True})
        previous=common
        for conv in (0,2):
            name=f'decoders.{decoder}.{conv}';inputs[name]=previous;previous=outputs[name]
    inputs['head']=previous
    return inputs,cats


@torch.inference_mode()
def make_profile(base,direction):
    """Only the same fixed 32 train patients / 128 unlabeled stacks calibrate."""
    samples={n:[] for n in LAYER_NAMES};handles=[];generator=torch.Generator(device='cpu').manual_seed(20261018)
    def hook(name):
        def observe(_module,_input,out):
            value=out.detach().float()
            value=(value-value.mean(1,keepdim=True)).abs() if name=='head' else value.clamp_min(0)
            flat=value.flatten()
            if not torch.isfinite(flat).all():raise ValueError('nonfinite calibration output')
            index=torch.randint(flat.numel(),(4096,),generator=generator).to(flat.device)
            samples[name].append(flat[index].cpu().numpy())
        return observe
    modules=dict(base.named_modules())
    for name in LAYER_NAMES:handles.append(modules[name].register_forward_hook(hook(name)))
    roster=[];device=next(base.parameters()).device
    try:
        for x,meta in calibration_samples(direction):
            base(torch.from_numpy(x[None]).to(device));roster.append(meta)
    finally:
        for handle in handles:handle.remove()
    split=__import__('json').loads((DATA/'SPLIT.json').read_text(encoding='utf-8'))
    if len(roster)!=128 or len({r['patient'] for r in roster})!=32 or any(r['patient'] not in split['train'] for r in roster):
        raise ValueError('calibration data boundary mismatch')
    clips={n:max(float(np.percentile(np.concatenate(parts),99.)),1e-5) for n,parts in samples.items()}
    divisor=max(clips['head']/12.,1.)
    outputs={n:clips[n]/127 for n in LAYER_NAMES};outputs['head']=12/127
    inputs,cats=propagate_scales(outputs)
    return {'schema':'acdc-qat-fixed-ranges-v1','direction':direction,'inputs':inputs,'outputs':outputs,'concat':cats,
        'head_temperature_divisor':divisor,'head_clip':12.,'split_sha256':sha256(DATA/'SPLIT.json'),
        'calibration':{'method':'train-only p99 fused ReLU ranges, fixed during QAT; centered temperature-scaled head',
            'roster':roster,'patients':32,'stacks':128,'dev_or_test_access':False,'ground_truth_access':False,
            'sample_seed':20261018,'samples_per_layer_per_stack':4096,'percentile':99.,'original_clips':clips},
        'training_surrogate':'FP32 fake quant, per-output-channel weights, bias rounding, Q24 multiplier approximation; final exact integer evaluation required'}


class QATUNet(torch.nn.Module):
    def __init__(self,base,profile):
        super().__init__();self.base=base;self.profile=profile;self.blend=1.
        if profile['split_sha256']!=sha256(DATA/'SPLIT.json'):raise ValueError('QAT split mismatch')
        inputs,cats=propagate_scales(profile['outputs'])
        if inputs!=profile['inputs'] or cats!=profile['concat']:raise ValueError('QAT scale propagation mismatch')

    def fake(self,x,scale):
        q=torch.fake_quantize_per_tensor_affine(x.float(),float(scale),0,-127,127)
        return x+(q-x)*self.blend

    def conv(self,x,name,index):
        module=dict(self.base.named_modules())[name]
        weight=module.weight.float();bias=module.bias.float()
        weight_scale=weight.detach().abs().amax((1,2,3)).clamp_min(1e-12)/127
        qweight=torch.fake_quantize_per_channel_affine(weight,weight_scale,torch.zeros_like(weight_scale,dtype=torch.int32),0,-127,127)
        used_weight=weight+(qweight-weight)*self.blend
        input_scale=self.profile['inputs'][name];output_scale=self.profile['outputs'][name]
        bias_scale=weight_scale*input_scale
        rounded_bias=(bias/bias_scale).round().clamp(-(2**31),2**31-1)*bias_scale
        used_bias=bias+(rounded_bias-bias).detach()*self.blend
        y=F.conv2d(x,used_weight,used_bias,padding=module.padding)
        # Match the exported Q24 ratio as closely as float training permits.
        multiplier=(bias_scale/output_scale*2**24).round()
        if bool((multiplier<1).any()) or bool((multiplier>2**31-1).any()):raise ValueError('Q24 range in QAT surrogate')
        correction=(multiplier*output_scale/(2**24*bias_scale)).detach()
        y=y*(1+self.blend*(correction[None,:,None,None]-1))
        if index<14:y=y.relu()
        return self.fake(y,output_scale)

    def forward(self,x):
        x=self.fake(x,1/127);skips=[];index=0
        for level in range(4):
            x=self.conv(x,f'encoders.{level}.0',index);x=self.conv(x,f'encoders.{level}.2',index+1)
            index+=2;skips.append(x)
            if level<3:x=F.max_pool2d(x,2)
        for decoder,(cat,skip) in enumerate(zip(self.profile['concat'],reversed(skips[:-1]))):
            # A floating surrogate for each branch's separate Q24 conversion.
            def align(value,source,multiplier):
                ratio=multiplier*cat['common_scale']/(2**24*source)
                return self.fake(value*(1+self.blend*(ratio-1)),cat['common_scale'])
            skip=align(skip,cat['skip_scale'],cat['skip_multiplier_q24'])
            x=align(x,cat['deep_scale'],cat['deep_multiplier_q24'])
            x=torch.cat((skip,F.interpolate(x,scale_factor=2,mode='nearest')),1)
            x=self.conv(x,f'decoders.{decoder}.0',index);x=self.conv(x,f'decoders.{decoder}.2',index+1);index+=2
        return self.conv(x,'head',14)

    def integer_model(self,device='cuda'):
        modules=dict(self.base.named_modules())
        layers=[kernel.build_layer(modules[n],n,input_scale=self.profile['inputs'][n],output_scale=self.profile['outputs'][n]) for n in LAYER_NAMES]
        return IntegerModel(layers,self.profile['concat'],device)


def prepare_qat(checkpoint,direction,device='cuda'):
    base,state=load_fp32(checkpoint,device);profile=make_profile(base,direction)
    with torch.no_grad():
        base.head.weight.sub_(base.head.weight.mean(0,keepdim=True));base.head.bias.sub_(base.head.bias.mean())
        base.head.weight.div_(profile['head_temperature_divisor']);base.head.bias.div_(profile['head_temperature_divisor'])
    # Head centering preserves softmax. Temperature changes probabilities and is
    # accepted only after full retraining/native evaluation, never by assertion.
    return QATUNet(base,profile),state
