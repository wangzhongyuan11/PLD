"""NumPy-only ACDC loader for the existing generic CAM2 device interface.

This module does not open MMIO, acquire DMA, load an overlay or replace a service.
The caller supplies an explicitly owned/validated device. Emulator results are
marked as such. The old CAMUS package loader and its frozen identity stay intact.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,math,sys,threading
import numpy as np

ARCH='ACDC25D-B16-P3-v1'
NAMES=tuple([f'encoders.{i}.{j}' for i in range(4) for j in (0,2)]+[f'decoders.{i}.{j}' for i in range(3) for j in (0,2)]+['head'])
PAIRS=((3,16),(16,16),(16,32),(32,32),(32,64),(64,64),(64,128),(128,128),(192,64),(64,64),(96,32),(32,32),(48,16),(16,16),(16,4))
HEIGHTS=(256,256,128,128,64,64,32,32,64,64,128,128,256,256,256)
_LOCK_INIT=threading.Lock()

def _hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def _finite_scale(value):
    if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value) or value<=0:raise ValueError('positive finite scale required')
    return float(value)

def _cam2_imports():
    # Both the existing workspace (A nested) and the self-contained release layout.
    roots=[Path(__file__).resolve().parents[i] for i in (2,3)]
    root=next((p for p in roots if (p/'camus/graph.py').is_file() and (p/'reference/tiling.py').is_file()),None)
    if root is None:raise ImportError('paired generic CAM2 source runtime missing')
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    from camus.graph import CamusGraph
    from camus.device import EmulatedTileAccelerator,output_channel_chunks,choose_camus_tile
    from camus.contract import tensor_tile_shape
    return CamusGraph,EmulatedTileAccelerator,output_channel_chunks,choose_camus_tile,tensor_tile_shape

@dataclass(frozen=True)
class Layer:
    name:str;weight:np.ndarray;bias:np.ndarray;multipliers:np.ndarray
    padding:int;relu:bool;input_scale:float;output_scale:float;kernel:int

@dataclass(frozen=True)
class Concat:
    level:int;common_scale:float;multiplier_q24:tuple[int,int]

@dataclass(frozen=True)
class Model:
    root:Path;spec:dict;layers:tuple;concats:tuple;input_scale:float;head_scale:float
    model_sha256:str
    input_shape:tuple=(1,3,256,256);head_shape:tuple=(1,4,256,256)

def load_model(folder,expected_sha256,expected_profile):
    folder=Path(folder).resolve();file=folder/'model_spec.json'
    if _hash(file)!=expected_sha256:raise ValueError('spec SHA mismatch')
    spec=json.loads(file.read_text(encoding='utf-8'))
    if spec.get('schema')!='acdc-int8-model-v1' or spec.get('architecture')!=ARCH or spec.get('source_kind')!='trained':raise ValueError('trained ACDC export required')
    if spec.get('profile_id')!=expected_profile or expected_profile not in ('acdc_axial_v1','acdc_coronal_v1','acdc_sagittal_v1'):raise ValueError('wrong profile')
    if spec['input_shape']!=[1,3,256,256] or spec['output_shape']!=[1,4,256,256] or spec['input_scale']!=1/127:raise ValueError('tensor contract changed')
    if spec['labels']!={'0':'background','1':'RV_cavity','2':'LV_myocardium','3':'LV_cavity'} or spec['head_activation']!='none':raise ValueError('class/head semantics changed')
    if tuple(r['name'] for r in spec['layers'])!=NAMES:raise ValueError('unexpected topology')
    layers=[]
    for i,(row,(ci,co)) in enumerate(zip(spec['layers'],PAIRS)):
        k=1 if i==14 else 3;padding=0 if i==14 else 1
        if row['weight_shape']!=[co,ci,k,k] or row['padding']!=padding or row['kernel']!=k or row['relu']!=(i<14):raise ValueError('layer contract changed')
        arrays={}
        for name,count,dtype in [('weight.bin',co*ci*k*k,np.int8),('bias.bin',co,'<i4'),('multipliers.bin',co,'<i4')]:
            r=row['files'][name];p=(folder/r['path']).resolve()
            if not p.is_relative_to(folder) or p.stat().st_size!=np.dtype(dtype).itemsize*count or _hash(p)!=r['sha256']:raise ValueError('parameter size/path/SHA mismatch')
            arrays[name]=np.fromfile(p,dtype=dtype)
        w=arrays['weight.bin'].reshape(co,ci,k,k);b=arrays['bias.bin'];m=arrays['multipliers.bin']
        if np.any(w==-128) or np.any(m<1):raise ValueError('illegal INT8/Q24 value')
        bounds=np.abs(b.astype(np.int64))+127*np.abs(w.astype(np.int64)).sum((1,2,3))
        if np.any(bounds>2**31-1) or np.any(bounds*m.astype(np.int64)>2**63-1):raise ValueError('integer bound exceeded')
        for a in (w,b,m):a.setflags(write=False)
        layers.append(Layer(row['name'],w,b,m,padding,i<14,_finite_scale(row['input_scale']),_finite_scale(row['output_scale']),k))
    if layers[-1].output_scale!=spec['head_scale']:raise ValueError('head scale mismatch')
    cats=[]
    if len(spec['concat'])!=3:raise ValueError('three CAT nodes required')
    previous=1/127
    for i in range(8):
        if layers[i].input_scale!=previous:raise ValueError('encoder scale discontinuity')
        previous=layers[i].output_scale
    for j,(cat,level) in enumerate(zip(spec['concat'],(2,1,0))):
        skip=layers[2*level+1].output_scale;common=max(skip,previous)
        expected=tuple(int(np.rint(v/common*2**24)) for v in (skip,previous))
        actual=(cat['skip_multiplier_q24'],cat['deep_multiplier_q24'])
        if cat['level']!=level or cat['common_scale']!=common or actual!=expected or not cat['skip_first'] or min(actual)<1:raise ValueError('CAT scale/order changed')
        cats.append(Concat(level,common,actual));previous=common
        for index in (8+2*j,9+2*j):
            if layers[index].input_scale!=previous:raise ValueError('decoder scale discontinuity')
            previous=layers[index].output_scale
    if layers[14].input_scale!=previous:raise ValueError('head scale discontinuity')
    return Model(folder,spec,tuple(layers),tuple(cats),1/127,_finite_scale(spec['head_scale']),expected_sha256)

def hardware_plan(model):
    _,_,chunks,choose,tensor=_cam2_imports();rows=[]
    for i,(layer,height) in enumerate(zip(model.layers,HEIGHTS)):
        co,ci,k,_=layer.weight.shape;groups=chunks(ci,co,k);ht,wt=choose(ci,max(groups),height,height,k)
        rows.append({'id':f'C{i+1:02}','input_channels':ci,'output_channels':co,'spatial':height,
           'output_channel_chunks':list(groups),'output_tile_hw':[ht,wt],
           'run_commands':len(groups)*math.ceil(height/ht)*math.ceil(height/wt)})
    tensors=[]
    for ch,h in ((16,128),(32,64),(64,32)):
        ht,wt=tensor(2,ch,h,h);tensors.append({'mode':2,'channels':ch,'output_tile_hw':[ht,wt]})
    for cs,cd,h in ((64,128,64),(32,64,128),(16,32,256)):
        ht,wt=tensor(4,cs+cd,h,h,cs);tensors.append({'mode':4,'channels':cs+cd,'split':cs,'output_tile_hw':[ht,wt]})
    return {'architecture':ARCH,'device_id':'0x43414D32','abi_version':'0x00020000','conv':rows,'tensor':tensors,
            'arithmetic_ops_in_device':True,'physical_board_verified':False,'note':'capacity/scheduling proof only; no bit/HWH/DDR/latency acceptance'}

def forward(model,input_int8,device):
    Graph,Emulator,_,_,_=_cam2_imports();q=np.asarray(input_int8)
    if q.shape!=(1,3,256,256) or q.dtype!=np.int8 or np.any(q==-128):raise ValueError('first-layer NCHW signed127 input only')
    with _LOCK_INIT:
        if not hasattr(device,'_acdc_serial_lock'):device._acdc_serial_lock=threading.Lock()
    if not device._acdc_serial_lock.acquire(blocking=False):raise RuntimeError('device already has an active ACDC frame')
    try:
        if getattr(device,'_acdc_active_sha',None)!=model.model_sha256:
            for name in ('_conv_plans','_lut_cache','_parameter_blobs'):
                cache=getattr(device,name,None)
                if cache is not None:cache.clear()
            device._acdc_active_sha=model.model_sha256
        # Hold immutable arrays while pointer-based underlying plans are cached.
        device._acdc_active_model=model
        result=Graph(model,device).forward(q)
        result['backend']='cam2_emulator' if type(device) is Emulator else 'caller_supplied_device_pending_independent_acceptance'
        result['profile_id']=model.spec['profile_id'];return result
    finally:device._acdc_serial_lock.release()
