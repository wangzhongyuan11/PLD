"""Frozen-PC generation and identity-checked board loading of derived constants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from acdc.contract import ContractError,atomic_json,file_sha,is_sha,checked_int8,INPUT_SHAPE,INPUT_BYTES
from .reconstruct import make_exp_table


def prepare(model,directory,*,selftest_input=None,selftest_head_sha=None):
    directory=Path(directory)
    if directory.exists():raise ContractError('choose a new V2 assets directory')
    expected=(model.implementation_binding or {}).get('versions',{}).get('numpy')
    if expected is None or np.__version__!=expected:
        raise ContractError('generate constants in the frozen A NumPy reference environment')
    directory.mkdir(parents=True)
    table=make_exp_table(model.head_scale)
    np.save(directory/'exp.npy',table,allow_pickle=False)
    record=dict(schema='pld-acdc-appliance-assets-v2',model_sha256=model.model_sha256,
                head_scale=model.head_scale,numpy_reference_version=np.__version__,
                generator_sha256=file_sha(Path(__file__).with_name('reconstruct.py')),
                files={'exp.npy':file_sha(directory/'exp.npy')})
    if selftest_input is not None:
        if not is_sha(selftest_head_sha):raise ContractError('independent selftest head SHA required')
        raw=Path(selftest_input).read_bytes()
        if len(raw)!=INPUT_BYTES:raise ContractError('invalid selftest input length')
        checked_int8(np.frombuffer(raw,np.int8).reshape(INPUT_SHAPE),INPUT_SHAPE)
        (directory/'selftest_input.i8').write_bytes(raw)
        record['files']['selftest_input.i8']=file_sha(directory/'selftest_input.i8')
        record['selftest_head_sha256']=selftest_head_sha
    atomic_json(directory/'manifest.json',record)
    return file_sha(directory/'manifest.json')


def load(model,directory,expected_sha,*,verify=True):
    directory=Path(directory)
    if verify and file_sha(directory/'manifest.json')!=expected_sha:raise ContractError('V2 asset manifest drift')
    record=json.loads((directory/'manifest.json').read_text('utf-8'))
    if (record.get('schema')!='pld-acdc-appliance-assets-v2'
            or record.get('model_sha256')!=model.model_sha256 or record.get('head_scale')!=model.head_scale):
        raise ContractError('V2 assets belong to another model')
    if set(record.get('files',{})) not in ({'exp.npy'},{'exp.npy','selftest_input.i8'}):
        raise ContractError('unexpected V2 asset members')
    for name,sha in record['files'].items():
        if verify and file_sha(directory/name)!=sha:raise ContractError('V2 asset drift: '+name)
    table=np.load(directory/'exp.npy',allow_pickle=False)
    if table.dtype!=np.float32 or table.shape!=(255,255) or not table.flags.c_contiguous:
        raise ContractError('invalid V2 exp table representation')
    return table


def selftest(backend,directory):
    """Invoke only after load() has authenticated this asset manifest and files."""
    import hashlib
    from acdc.contract import ARCHITECTURE,PROFILE,PREPROCESSING,digest
    directory=Path(directory);record=json.loads((directory/'manifest.json').read_text('utf-8'))
    if not is_sha(record.get('selftest_head_sha256')):raise ContractError('production selftest vector missing')
    raw=(directory/'selftest_input.i8').read_bytes()
    x=checked_int8(np.frombuffer(raw,np.int8).reshape(INPUT_SHAPE),INPUT_SHAPE)
    request=dict(architecture_id=ARCHITECTURE,execution_profile=PROFILE,preprocessing_id=PREPROCESSING,
        run_id='startup-selftest',phase='ED',orientation='axial',center_index=0,neighbor_indices=[0,0,0],
        total_slices=1,geometry_sha256=digest({'selftest_tensor_sha256':record['files']['selftest_input.i8']}),
        model_sha256=backend.model_sha,runtime_sha256=backend.runtime_sha,shape=list(INPUT_SHAPE),
        dtype='int8',layout='NCHW',input_scale=1/127)
    head,meta=backend.infer(x,request)
    if hashlib.sha256(head.tobytes()).hexdigest()!=record['selftest_head_sha256']:
        raise ContractError('startup PL selftest differs from independent integer reference')
    return dict(head_equal=True,head_sha256=record['selftest_head_sha256'],backend=meta['backend'])


if __name__=='__main__':
    from acdc.model import load_bundle
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--model-sha',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selftest-input',type=Path);parser.add_argument('--selftest-head-sha')
    args=parser.parse_args()
    model=load_bundle(args.model,expected_sha=args.model_sha)
    print(prepare(model,args.output,selftest_input=args.selftest_input,selftest_head_sha=args.selftest_head_sha))
