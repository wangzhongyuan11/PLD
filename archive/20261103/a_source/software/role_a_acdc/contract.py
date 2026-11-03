"""Public tensor contract, safe PC/board interchange and CAM2-compatible CAT."""
from __future__ import annotations
import hashlib,json,uuid,re
from pathlib import Path
import numpy as np
from .common import ARCHITECTURE,LABELS,save_json,sha256

PROFILES={d:f'acdc_{d}_v1' for d in ('axial','coronal','sagittal')}
INPUT_SHAPE=(1,3,256,256);HEAD_SHAPE=(1,4,256,256)

def quantize_input(x):
    a=np.asarray(x,dtype=np.float64)
    if a.shape!=INPUT_SHAPE or not np.isfinite(a).all() or a.min()<-1 or a.max()>1:raise ValueError('input shape/range invalid')
    return np.ascontiguousarray(np.clip(np.rint(a*127),-127,127).astype(np.int8))

def round_q24(acc,multiplier):
    a=np.asarray(acc,dtype=np.int64);m=np.asarray(multiplier,dtype=np.int64)
    if np.any(m<1) or np.any(m>2**31-1):raise ValueError('Q24 multiplier must be positive int32')
    if a.size and int(np.max(np.abs(a)))*int(np.max(m))>2**63-1:raise ValueError('int64 product overflow')
    product=a*m;absolute=np.abs(product);quotient=absolute>>24;remainder=absolute&((1<<24)-1)
    quotient+=((remainder>(1<<23))|((remainder==(1<<23))&((quotient&1)==1))).astype(np.int64)
    return np.where(product<0,-quotient,quotient)

def cat_mode4(skip,deep,skip_m=1<<24,deep_m=1<<24):
    s=np.asarray(skip);d=np.asarray(deep)
    if s.dtype!=np.int8 or d.dtype!=np.int8 or s.ndim!=3 or d.ndim!=3 or s.shape[1:]!=(2*d.shape[1],2*d.shape[2]):raise ValueError('CAT requires high-resolution skip and half-resolution deep CHW')
    if np.any(s==-128) or np.any(d==-128):raise ValueError('signed127 only')
    s=np.clip(round_q24(s,skip_m),-127,127).astype(np.int8)
    d=np.clip(round_q24(d,deep_m),-127,127).astype(np.int8).repeat(2,1).repeat(2,2)
    return np.ascontiguousarray(np.concatenate((s,d),axis=0))

def parameter_blob(weight,bias,multipliers):
    w=np.asarray(weight);b=np.asarray(bias);m=np.asarray(multipliers)
    if w.dtype!=np.int8 or w.ndim!=4 or np.any(w==-128):raise ValueError('OIHW signed127 required')
    if b.shape!=(w.shape[0],) or m.shape!=b.shape or np.any(m<1) or np.any(m>2**31-1):raise ValueError('invalid parameter vector')
    if np.any(b<-(2**31)) or np.any(b>2**31-1):raise ValueError('bias overflow')
    return w.tobytes(order='C')+b.astype('<i4').tobytes()+m.astype('<i4').tobytes()

def conv_channel_chunks(ci,co,k):
    limit=min(64,(2048//(((ci+3)//4)*k*k))*8)
    if limit<1:raise ValueError('weight bank cannot hold one output group')
    return [(offset,min(limit,co-offset)) for offset in range(0,co,limit)]

def contract_document():
    from .board_adapter import PAIRS,HEIGHTS,NAMES
    layers=[]
    for i,((ci,co),height,name) in enumerate(zip(PAIRS,HEIGHTS,NAMES)):
        k=1 if i==14 else 3
        layers.append({'id':f'C{i+1:02}','name':name,'input_shape':[1,ci,height,height],
            'output_shape':[1,co,height,height],'kernel':k,'stride':1,'padding':0 if i==14 else 1,
            'dilation':1,'groups':1,'bias':True,'relu':i<14,'weight_layout':'OIHW',
            'weight_bytes':ci*co*k*k,'parameter_bytes':ci*co*k*k+8*co})
    return {'schema':'acdc-board-contract-v1','architecture':ARCHITECTURE,'profiles':PROFILES,
        'required_device':{'vlnv':'user:pld:axis_tensor_accel_camus:1.0','device_id':'0x43414D32','abi_version':'0x00020000',
             'rtl':'camus/rtl/axis_tensor_accel_camus.sv','max_ci':192,'reject_skin_abi1':True},
        'convolution_layers':layers,'conv_count':15,'logical_node_order':['C01','C02','P0','C03','C04','P1','C05','C06','P2','C07','C08','CAT1','C09','C10','CAT2','C11','C12','CAT3','C13','C14','C15'],
        'conv_macs_per_stack':sum(ci*co*(1 if i==14 else 9)*h*h for i,((ci,co),h) in enumerate(zip(PAIRS,HEIGHTS))),
        'input':{'shape':list(INPUT_SHAPE),'dtype':'int8','layout':'NCHW','bytes':196608,'scale':1/127,'zero_point':0,
                 'channels':['previous','center','next'],'boundary':'clamp','range':[-127,127]},
        'head':{'shape':list(HEAD_SHAPE),'dtype':'int8','layout':'NCHW','bytes':262144,'zero_point':0,
                'scale':'one positive shared scalar per frozen profile','activation':'none','classes':LABELS},
        'cat':{'mode':4,'channel_axis':1,'order':['skip','nearest2(deep)'],'input_payload':'skip CHW then half-resolution deep CHW',
               'lut_order':['skip 256 bytes','deep 256 bytes'],'lut_index':'raw unsigned byte representation q & 255',
               'splits':[64,32,16],'common_scale':'max(skip_scale,deep_scale)'},
        'normalization':'nonzero native-volume percentiles 1/99 mapped to [-1,1]; clip; no GT crop',
        'geometry':'PC world RAS grid at 2 mm, linear MRI / nearest labels, full-FOV aspect-preserving 256 letterbox',
        'quantization':{'weights':'OIHW signed127 per output channel','bias':'little-endian int32','multiplier':'positive Q24 little-endian int32',
                         'round':'signed ties-to-even','accumulator':'int32 with per-channel proof','product':'int64'},
        'parameter_chunks':{'C07':[[0,64],[64,64]],'C08':[[0,56],[56,56],[112,16]],'C09':[[0,32],[32,32]]},
        'deployment_status':'contract and software only; real PYNQ bit/HWH/model-loader/latency not yet accepted'}

def write_board_request(folder,x,direction,spec_sha256,*,context,head_scale):
    if direction not in PROFILES or not re.fullmatch('[0-9a-f]{64}',spec_sha256):raise ValueError('profile/spec identity required')
    for key in ('run_id','volume_id','phase','center_index','slice_count','geometry_sha256'):
        if key not in context:raise ValueError(f'missing volume context: {key}')
    count,index=context['slice_count'],context['center_index']
    if type(count) is not int or type(index) is not int or not 0<=index<count:raise ValueError('invalid slice index/count')
    if context['phase'] not in ('ED','ES') or not re.fullmatch('[0-9a-f]{64}',context['geometry_sha256']):raise ValueError('invalid phase/geometry identity')
    if any(not isinstance(context[k],str) or not 1<=len(context[k])<=128 for k in ('run_id','volume_id')):raise ValueError('run/volume identity required')
    if isinstance(head_scale,bool) or not np.isfinite(head_scale) or head_scale<=0:raise ValueError('positive shared head scale required')
    q=quantize_input(x)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    f=folder/'input_int8.bin'
    if (folder/'request.json').exists():raise FileExistsError(folder/'request.json')
    with f.open('xb') as handle:handle.write(q.tobytes())
    request={'schema':'acdc-board-request-v1','architecture_id':ARCHITECTURE,'model_id':PROFILES[direction],
             'request_id':str(uuid.uuid4()),
             'model_spec_sha256':spec_sha256,'input':{'path':f.name,'sha256':sha256(f),'bytes':196608},
             'run_id':context['run_id'],'volume_id':context['volume_id'],'phase':context['phase'],
             'orientation':direction,'center_index':index,'neighbor_indices':[max(0,index-1),index,min(count-1,index+1)],
             'geometry_sha256':context['geometry_sha256'],'slice_count':count,
             'input_shape':list(INPUT_SHAPE),'output_shape':list(HEAD_SHAPE),'dtype':'int8','layout':'NCHW',
             'input_scale':1/127,'head_scale':float(head_scale),'zero_point':0,'required_backend':'pynq_pl','synthetic':False}
    save_json(folder/'request.json',request);return request

def read_board_response(folder,request,expected_scale):
    folder=Path(folder);response=json.loads((folder/'response.json').read_text(encoding='utf-8'))
    for key in ('architecture_id','model_id','model_spec_sha256','request_id'):
        if response.get(key)!=request[key]:raise ValueError(f'stale/wrong {key}')
    if response.get('input_sha256')!=request['input']['sha256']:raise ValueError('response belongs to a different input')
    if response.get('backend')!='pynq_pl' or response.get('synthetic') is not False:raise ValueError('real board result required')
    if response.get('head_scale')!=expected_scale or request['head_scale']!=expected_scale or not np.isfinite(expected_scale) or expected_scale<=0:raise ValueError('wrong head scale')
    row=response['output'];relative=Path(row['path'])
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe output path')
    output=folder/relative
    if not output.resolve().is_relative_to(folder.resolve()):raise ValueError('output escapes request directory')
    if row.get('shape')!=list(HEAD_SHAPE) or row.get('dtype')!='int8' or row.get('layout')!='NCHW' or row.get('bytes')!=262144:raise ValueError('head metadata mismatch')
    if output.stat().st_size!=262144 or sha256(output)!=row['sha256']:raise ValueError('head size/hash mismatch')
    head=np.fromfile(output,dtype=np.int8).reshape(HEAD_SHAPE)
    if np.any(head==-128):raise ValueError('head uses forbidden -128')
    return head.astype(np.float32)*expected_scale
