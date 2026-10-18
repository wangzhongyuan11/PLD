"""Length-delimited JSON metadata + raw NCHW tensor, never pickle/base64 tensors."""
from __future__ import annotations

import hashlib
import json
import struct

from .contract import (ARCHITECTURE, PROFILE, PREPROCESSING, INPUT_SHAPE, INPUT_BYTES,
                       HEAD_SHAPE, HEAD_BYTES, ContractError, canonical, is_sha, checked_int8)

MAX_META=65536


def encode(metadata,payload):
    payload=bytes(payload)
    m=dict(metadata,payload_bytes=len(payload),payload_sha256=hashlib.sha256(payload).hexdigest())
    header=canonical(m)
    if len(header)>MAX_META or len(payload)>HEAD_BYTES:
        raise ContractError('exchange frame too large')
    return struct.pack('<I',len(header))+header+payload


def decode(data,expected_bytes):
    if len(data)<4:
        raise ContractError('truncated exchange header')
    n=struct.unpack('<I',data[:4])[0]
    if not 0<n<=MAX_META or len(data)!=4+n+expected_bytes:
        raise ContractError('exchange length mismatch')
    try:
        meta=json.loads(data[4:4+n])
    except (ValueError,UnicodeError) as exc:
        raise ContractError('invalid exchange JSON') from exc
    payload=data[4+n:]
    if not isinstance(meta,dict) or meta.get('payload_bytes')!=expected_bytes or meta.get('payload_sha256')!=hashlib.sha256(payload).hexdigest():
        raise ContractError('exchange payload identity mismatch')
    return meta,payload


def validate_request(meta):
    if (meta.get('architecture_id')!=ARCHITECTURE or meta.get('execution_profile')!=PROFILE
            or meta.get('preprocessing_id')!=PREPROCESSING or meta.get('orientation')!='axial'
            or meta.get('phase') not in ('ED','ES') or meta.get('shape')!=list(INPUT_SHAPE)
            or meta.get('dtype')!='int8' or meta.get('layout')!='NCHW'
            or meta.get('input_scale')!=1/127):
        raise ContractError('request numerical/profile contract mismatch')
    for field in ('model_sha256','geometry_sha256'):
        if not is_sha(meta.get(field)):
            raise ContractError('missing request '+field)
    nz,z=meta.get('total_slices'),meta.get('center_index')
    if (type(nz) is not int or type(z) is not int or not 1<=nz<=4096 or not 0<=z<nz
            or meta.get('neighbor_indices')!=[max(z-1,0),z,min(z+1,nz-1)]):
        raise ContractError('request Axial center/neighbors invalid')
    if not isinstance(meta.get('run_id'),str) or not 1<=len(meta['run_id'])<=128:
        raise ContractError('missing run_id')
    return meta


def response_meta(request,head_scale,backend,synthetic):
    return {**request,'shape':list(HEAD_SHAPE),'head_scale':head_scale,
            'backend':backend,'synthetic':synthetic,'status':'completed'}
