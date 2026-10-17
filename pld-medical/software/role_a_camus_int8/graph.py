"""Owner-reviewed CAMUS topology and producer-scale propagation; integer tensors throughout."""
from __future__ import annotations
from typing import Mapping,Sequence,Any
import numpy as np
from . import integer as i8
LAYER_NAMES=tuple([f'encoders.{level}.{conv}' for level in range(4) for conv in (0,2)]+[f'decoders.{level}.{conv}' for level in range(3) for conv in (0,2)]+['head'])

def scales_from_calibration(model,calibration:Mapping[str,Any]):
    names=[n for n,m in model.named_modules() if isinstance(m,__import__('torch').nn.Conv2d)]
    if tuple(names)!=LAYER_NAMES:raise i8.IntegerGraphError('CAMUS requires the exact 15-Conv topology')
    if float(calibration['input_scale'])!=i8.INPUT_SCALE:raise i8.IntegerGraphError('invalid input scale')
    outputs={n:i8.qm.symmetric_scale(float(calibration['layers'][n]['output_max_abs'])) for n in LAYER_NAMES[:-1]}
    outputs['head']=i8.qm.symmetric_scale(float(calibration['head']['output_max_abs']))
    inputs={};previous=i8.INPUT_SCALE;skip=[]
    for level in range(4):
        for conv in (0,2):
            n=f'encoders.{level}.{conv}';inputs[n]=previous;previous=outputs[n]
        skip.append(previous)
    for decoder,level in enumerate((2,1,0)):
        previous=max(skip[level],previous)
        for conv in (0,2):
            n=f'decoders.{decoder}.{conv}';inputs[n]=previous;previous=outputs[n]
    inputs['head']=previous
    return inputs,outputs,outputs['head']

def forward_graph(model,layers:Sequence[i8.QuantLayer],image:np.ndarray,*,valid:np.ndarray|None=None,capture:bool=False,oracle:bool=False):
    if tuple(l.name for l in layers)!=LAYER_NAMES:raise i8.IntegerGraphError('exact 15 ordered layers required')
    image=np.asarray(image)
    if image.ndim!=2 or min(image.shape)<8 or any(s%8 for s in image.shape):raise i8.IntegerGraphError('image dimensions must be multiples of 8')
    if valid is not None and np.asarray(valid).shape!=image.shape:raise i8.IntegerGraphError('valid shape mismatch')
    feature=i8.quantize_input_float64(image);input_feature=feature;captures={};concat_info=[];skips=[];cursor=0;peak=feature.data.nbytes
    def apply(value,index,relu):
        nonlocal peak
        d={} if capture else None
        out=i8.conv2d_int8_float64(value,layers[index],relu=relu,oracle=oracle,capture=d)
        peak=max(peak,value.data.nbytes,out.data.nbytes)
        if capture:captures[layers[index].name]=d
        return out
    for level in range(4):
        feature=apply(feature,cursor,True);feature=apply(feature,cursor+1,True);cursor+=2
        skips.append(feature)
        if level<3:feature=i8.maxpool_int8(feature)
    for decoder,level in enumerate((2,1,0)):
        up=i8.nearest_upsample_int8(feature);skip=skips[level]
        expected=((64,128),(32,64),(16,32))[decoder]
        if (skip.data.shape[1],up.data.shape[1])!=expected or skip.data.shape[2:]!=up.data.shape[2:]:raise i8.IntegerGraphError('concat topology mismatch')
        merged,info=i8.concat_int8([skip,up]);info.update(level=level,decoder=decoder,skip_first=True)
        info['multiplier_q24']=[int(np.rint(s/info['common_scale']*(1<<24))) for s in info['source_scales']]
        concat_info.append(info);peak=max(peak,merged.data.nbytes)
        if capture:
            captures[f'concat_level{level}']={'skip_int8':skip.data.copy(),'up_int8':up.data.copy(),'skip_rescaled_int8':i8.rescale_int8(skip,merged.scale).data,'up_rescaled_int8':i8.rescale_int8(up,merged.scale).data,'output_int8':merged.data.copy(),**info}
        feature=apply(merged,cursor,True);feature=apply(feature,cursor+1,True);cursor+=2
    logits=apply(feature,14,False)
    return {'logits':(logits.data.astype(np.float32)*logits.scale)[0],'quant_logits':logits.data,'captures':captures,'concat_info':concat_info,'head_output_scale':logits.scale,'input_scale':i8.INPUT_SCALE,'peak_int8_tensor_bytes':peak,'full_sample':{'input_scale':i8.INPUT_SCALE,'head_output_scale':logits.scale,'logits_shape':list(logits.data.shape),'input_int8':input_feature.data}}
