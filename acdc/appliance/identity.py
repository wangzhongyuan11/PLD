"""Separate V2 application certificate; the frozen P6 core remains immutable."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from acdc.contract import ContractError,atomic_json,file_sha,safe_member


def create(root):
    root=Path(root).resolve();config=json.loads((root/'v2_config.json').read_text('utf-8'))
    names={'v2_config.json'}
    for sub in ('acdc/appliance','v2_assets','v2_deps'):
        for path in (root/sub).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                names.add(path.relative_to(root).as_posix())
    names.add(config['mesh_binary'])
    record=dict(schema='pld-acdc-appliance-application-v2',model_sha256=config['model_sha256'],
                core_runtime_sha256=file_sha(root/'runtime_manifest.json'),
                files={name:file_sha(safe_member(root,name)) for name in sorted(names)})
    atomic_json(root/'v2_application.json',record)
    return file_sha(root/'v2_application.json')


def verify(root,expected_sha):
    root=Path(root).resolve()
    if file_sha(root/'v2_application.json')!=expected_sha:raise ContractError('V2 application manifest drift')
    record=json.loads((root/'v2_application.json').read_text('utf-8'))
    if record.get('schema')!='pld-acdc-appliance-application-v2':raise ContractError('invalid V2 application identity')
    for name,sha in record['files'].items():
        if file_sha(safe_member(root,name))!=sha:raise ContractError('V2 application member drift: '+name)
    if file_sha(root/'runtime_manifest.json')!=record['core_runtime_sha256']:
        raise ContractError('V2 P6 core identity mismatch')
    config=json.loads((root/'v2_config.json').read_text('utf-8'))
    if config['model_sha256']!=record['model_sha256']:raise ContractError('V2 model identity mismatch')
    return config,record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    args=parser.parse_args();print(create(args.root))
