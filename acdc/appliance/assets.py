"""Frozen-PC generation and identity-checked board loading of derived constants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from acdc.contract import ContractError,atomic_json,file_sha
from .reconstruct import make_exp_table


def prepare(model,directory):
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
    atomic_json(directory/'manifest.json',record)
    return file_sha(directory/'manifest.json')


def load(model,directory,expected_sha):
    directory=Path(directory)
    if file_sha(directory/'manifest.json')!=expected_sha:raise ContractError('V2 asset manifest drift')
    record=json.loads((directory/'manifest.json').read_text('utf-8'))
    if (record.get('schema')!='pld-acdc-appliance-assets-v2'
            or record.get('model_sha256')!=model.model_sha256 or record.get('head_scale')!=model.head_scale):
        raise ContractError('V2 assets belong to another model')
    if record.get('files')!={'exp.npy':file_sha(directory/'exp.npy')}:
        raise ContractError('V2 exp table drift')
    table=np.load(directory/'exp.npy',allow_pickle=False)
    if table.dtype!=np.float32 or table.shape!=(255,255) or not table.flags.c_contiguous:
        raise ContractError('invalid V2 exp table representation')
    return table


if __name__=='__main__':
    from acdc.model import load_bundle
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True);parser.add_argument('--model-sha',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    model=load_bundle(args.model,expected_sha=args.model_sha)
    print(prepare(model,args.output))
