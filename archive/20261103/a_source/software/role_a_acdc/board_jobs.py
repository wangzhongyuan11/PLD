"""Filesystem exchange: PC creates all spatial-stack jobs, B returns real heads.

No implicit CPU fallback and no network/DMA/overlay operations. Synthetic heads
are rejected. Reported PL provenance still needs separate physical acceptance.
"""
from __future__ import annotations
import argparse,json,uuid,shutil,time
from pathlib import Path
import numpy as np
import nibabel as nib
from .common import RUN,save_json,sha256,ARCHITECTURE
from .geometry import prepare_volume,AXES,stack,letterbox,undo_letterbox,put_plane,probabilities_to_native
from .contract import write_board_request,read_board_response
from .board_adapter import load_model

def export_jobs(image,output,models,*,phase,geometry_policy='strict',volume_id=None):
    output=Path(output)
    if output.exists():raise FileExistsError('choose a new job directory')
    x,_,metadata=prepare_volume(image,geometry_policy=geometry_policy)
    validated={}
    for direction in AXES:
        d=Path(models)/direction/'int8'
        if not d.is_dir():d=Path(models)/direction
        spec_hash=sha256(d/'model_spec.json');model=load_model(d,spec_hash,f'acdc_{direction}_v1')
        validated[direction]=(model,spec_hash)
    output.mkdir(parents=True);save_json(output/'geometry.json',metadata)
    geometry_sha=sha256(output/'geometry.json');run_id=str(uuid.uuid4());volume_id=volume_id or Path(image).name.removesuffix('.nii.gz')
    shutil.copy2(image,output/'source.nii.gz');jobs=[];profiles={}
    for direction,axis in AXES.items():
        model,spec_hash=validated[direction];profiles[direction]={'model_id':model.spec['profile_id'],'spec_sha256':spec_hash,'head_scale':model.head_scale}
        for index in range(x.shape[axis]):
            relative=f'{direction}/{index:04}';d=output/relative;inputs,_=stack(x,direction,index)
            context={'run_id':run_id,'volume_id':volume_id,'phase':phase,'center_index':index,
                     'slice_count':x.shape[axis],'geometry_sha256':geometry_sha}
            request=write_board_request(d,inputs[None],direction,spec_hash,context=context,head_scale=model.head_scale)
            jobs.append({'relative':relative,'direction':direction,'center_index':index,'request_sha256':sha256(d/'request.json')})
    manifest={'schema':'acdc-volume-jobs-v1','architecture':ARCHITECTURE,'run_id':run_id,'volume_id':volume_id,'phase':phase,
           'geometry_sha256':geometry_sha,'source_sha256':sha256(output/'source.nii.gz'),'profiles':profiles,'jobs':jobs,
           'status':'WAITING_FOR_BOARD_RESPONSES','required_backend':'pynq_pl','synthetic':False}
    save_json(output/'VOLUME_JOBS.json',manifest);return manifest

def collect_jobs(folder,output):
    folder=Path(folder).resolve();output=Path(output)
    if output.exists():raise FileExistsError('choose a new result directory')
    manifest=json.loads((folder/'VOLUME_JOBS.json').read_text(encoding='utf-8'))
    if manifest['schema']!='acdc-volume-jobs-v1' or manifest['synthetic'] is not False or manifest['architecture']!=ARCHITECTURE:raise ValueError('wrong job manifest')
    if sha256(folder/'geometry.json')!=manifest['geometry_sha256'] or sha256(folder/'source.nii.gz')!=manifest['source_sha256']:raise ValueError('volume geometry/source changed')
    metadata=json.loads((folder/'geometry.json').read_text(encoding='utf-8'));shape=metadata['working_shape'];fused=None;timing=[]
    expected_count=sum(shape[a] for a in AXES.values())
    if len(manifest['jobs'])!=expected_count:raise ValueError('incomplete job list')
    seen=set()
    for direction,axis in AXES.items():
        t=time.perf_counter();profile=manifest['profiles'][direction];p=np.empty((4,*shape),np.float32)
        hw=(shape[1],shape[0]) if axis==2 else (shape[2],shape[0]) if axis==1 else (shape[2],shape[1])
        _,transform=letterbox(np.zeros(hw,np.float32))
        jobs=sorted((j for j in manifest['jobs'] if j['direction']==direction),key=lambda j:j['center_index'])
        if [j['center_index'] for j in jobs]!=list(range(shape[axis])):raise ValueError('slice index gap/duplicate')
        for job in jobs:
            relative=Path(job['relative']);d=(folder/relative).resolve()
            if relative.is_absolute() or not d.is_relative_to(folder) or sha256(d/'request.json')!=job['request_sha256']:raise ValueError('request changed/unsafe path')
            req=json.loads((d/'request.json').read_text(encoding='utf-8'))
            if req['request_id'] in seen:raise ValueError('duplicate request ID')
            seen.add(req['request_id'])
            checks={'run_id':manifest['run_id'],'volume_id':manifest['volume_id'],'phase':manifest['phase'],
                    'orientation':direction,'center_index':job['center_index'],'geometry_sha256':manifest['geometry_sha256'],
                    'model_spec_sha256':profile['spec_sha256'],'model_id':profile['model_id']}
            if any(req.get(k)!=v for k,v in checks.items()):raise ValueError('job context/profile mismatch')
            input_path=Path(req['input']['path']);input_file=(d/input_path).resolve()
            if input_path.is_absolute() or not input_file.is_relative_to(d) or input_file.stat().st_size!=196608:raise ValueError('invalid input path/size')
            if sha256(input_file)!=req['input']['sha256']:raise ValueError('input stack changed')
            logits=read_board_response(d,req,profile['head_scale'])[0]
            prob=np.exp(logits-logits.max(axis=0,keepdims=True));prob/=prob.sum(axis=0,keepdims=True)
            full=undo_letterbox(prob,transform)
            for c in range(4):put_plane(p[c],axis,job['center_index'],full[c])
        native=probabilities_to_native(p,metadata);fused=native if fused is None else fused+native
        timing.append({'direction':direction,'collection_seconds':time.perf_counter()-t,'backend_claimed':'pynq_pl'})
    labels=(fused/3).argmax(0).astype(np.uint8)
    from .pipeline import save_prediction
    return save_prediction(folder/'source.nii.gz',output,labels,metadata,timing,
               {d:p['spec_sha256'] for d,p in manifest['profiles'].items()},'pynq_pl_claimed',board_inference=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['export','collect']);p.add_argument('--image');p.add_argument('--jobs');p.add_argument('--models');p.add_argument('--output',required=True)
    p.add_argument('--phase',choices=['ED','ES']);p.add_argument('--geometry-policy',default='strict');a=p.parse_args()
    if a.action=='export':
        if not a.image or not a.models or not a.phase:p.error('export requires image, models and ED/ES phase')
        r=export_jobs(a.image,a.output,a.models,phase=a.phase,geometry_policy=a.geometry_policy)
        print(json.dumps({'jobs':len(r['jobs']),'status':r['status']}))
    else:
        if not a.jobs:p.error('collect requires --jobs')
        print(json.dumps(collect_jobs(a.jobs,a.output)))
