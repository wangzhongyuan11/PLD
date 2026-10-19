"""Strict Axial adapter for A's filesystem exchange and B's runtime protocol.

A geometry SHA is an opaque digest of A's sidecar bytes; it is preserved rather
than relabelled as a B-canonical geometry hash. No preprocessing is repeated.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from .contract import (ARCHITECTURE,PROFILE,PREPROCESSING,INPUT_SHAPE,HEAD_SHAPE,
                       INPUT_BYTES,HEAD_BYTES,ContractError,atomic_json,file_sha,safe_member,is_sha,checked_int8)
from .protocol import validate_request


def translate(request,model_sha,head_scale,runtime_sha):
    if (request.get('schema')!='acdc-board-request-v1' or request.get('architecture_id')!=ARCHITECTURE
            or request.get('model_id')!='acdc_axial_v1' or request.get('orientation')!='axial'
            or request.get('model_spec_sha256')!=model_sha or request.get('head_scale')!=head_scale
            or request.get('input_shape')!=list(INPUT_SHAPE) or request.get('output_shape')!=list(HEAD_SHAPE)
            or request.get('dtype')!='int8' or request.get('layout')!='NCHW'
            or request.get('input_scale')!=1/127 or request.get('zero_point')!=0
            or request.get('required_backend')!='pynq_pl' or request.get('synthetic') is not False
            or not is_sha(runtime_sha)):
        raise ContractError('A exchange numerical/model/backend contract mismatch')
    for key in ('request_id','volume_id'):
        if not isinstance(request.get(key),str) or not 1<=len(request[key])<=128:
            raise ContractError('A exchange missing '+key)
    metadata=dict(architecture_id=ARCHITECTURE,execution_profile=PROFILE,preprocessing_id=PREPROCESSING,
        model_sha256=model_sha,runtime_sha256=runtime_sha,run_id=request['run_id'],
        request_id=request['request_id'],volume_id=request['volume_id'],phase=request['phase'],orientation='axial',
        center_index=request['center_index'],neighbor_indices=request['neighbor_indices'],
        total_slices=request['slice_count'],geometry_sha256=request['geometry_sha256'],
        geometry_digest_kind='A sidecar file SHA256',shape=list(INPUT_SHAPE),dtype='int8',layout='NCHW',input_scale=1/127)
    return validate_request(metadata)


def execute(folder,backend):
    folder=Path(folder).resolve();request_file=safe_member(folder,'request.json')
    if request_file.stat().st_size>65536:raise ContractError('A request JSON exceeds metadata bound')
    request=json.loads(request_file.read_text('utf-8'))
    metadata=translate(request,backend.model_sha,backend.head_scale,backend.runtime_sha)
    if backend.synthetic or backend.kind not in ('pl','remote'):raise ContractError('A exchange requires actual PL backend')
    if (folder/'response.json').exists() or (folder/'head_int8.bin').exists():
        raise ContractError('response already exists; refusing to overwrite another result')
    row=request['input'];source=safe_member(folder,row['path'])
    if row.get('bytes')!=INPUT_BYTES or source.stat().st_size!=INPUT_BYTES or file_sha(source)!=row.get('sha256'):
        raise ContractError('A exchange input bytes/hash mismatch')
    x=checked_int8(np.fromfile(source,np.int8).reshape(INPUT_SHAPE),INPUT_SHAPE)
    head,result=backend.infer(x,metadata)
    if (result.get('backend')!='pl' or result.get('synthetic') is not False
            or result.get('head_scale')!=backend.head_scale or result.get('model_sha256')!=backend.model_sha
            or result.get('runtime_sha256')!=backend.runtime_sha):
        raise ContractError('A exchange did not receive authenticated PL provenance')
    for key in ('run_id','request_id','volume_id','phase','orientation','center_index','neighbor_indices','geometry_sha256'):
        if result.get(key)!=metadata[key]:raise ContractError('A exchange response context mismatch: '+key)
    head=checked_int8(head,HEAD_SHAPE)
    target=folder/'head_int8.bin'
    with target.open('xb') as stream:stream.write(head.tobytes())
    response={key:request[key] for key in ('architecture_id','model_id','model_spec_sha256','request_id',
        'run_id','volume_id','phase','orientation','center_index','neighbor_indices','geometry_sha256','slice_count')}
    response.update(input_sha256=row['sha256'],backend='pynq_pl',synthetic=False,head_scale=backend.head_scale,
        runtime_sha256=backend.runtime_sha,output=dict(path=target.name,sha256=file_sha(target),shape=list(HEAD_SHAPE),
        dtype='int8',layout='NCHW',bytes=HEAD_BYTES))
    atomic_json(folder/'response.json',response)
    return response


if __name__=='__main__':
    from .cli import backend
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--model-spec-sha',required=True);p.add_argument('--runtime',type=Path)
    p.add_argument('--board-url');a=p.parse_args()
    a.backend='remote' if a.board_url else 'pl';a.allow_synthetic=False
    runner=backend(a)
    try:print(json.dumps(execute(a.folder,runner)))
    finally:runner.close()
