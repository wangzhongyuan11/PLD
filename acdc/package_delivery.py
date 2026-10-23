"""Create/verify a local, hash-bound delivery without publishing or deployment."""
from __future__ import annotations
import argparse
import json
from pathlib import Path,PurePosixPath
import shutil
import subprocess
import zipfile

from .contract import ContractError,atomic_json,file_sha,safe_member
from .package import ROOT,source_package


def verify(root):
    root=Path(root).resolve();manifest=json.loads((root/'MANIFEST.json').read_text('utf-8'))
    if manifest.get('schema')!='pld-acdc-delivery-v1':raise ContractError('unknown delivery format')
    for name,row in manifest['files'].items():
        path=safe_member(root,name)
        if path.stat().st_size!=row['bytes'] or file_sha(path)!=row['sha256']:raise ContractError('delivery member drift: '+name)
    for name in ('source.zip','runtime.zip'):
        with zipfile.ZipFile(root/name) as z:
            if z.testzip() is not None:raise ContractError(name+' CRC failed')
            record=json.loads(z.read('source_manifest.json' if name=='source.zip' else 'runtime_manifest.json'))
            files=record['files']
            import hashlib
            for member,sha in files.items():
                if hashlib.sha256(z.read(member)).hexdigest()!=sha:raise ContractError(name+' inner member drift: '+member)
            if name=='runtime.zip' and hashlib.sha256(z.read('runtime_manifest.json')).hexdigest()!=manifest['runtime_manifest_sha256']:
                raise ContractError('runtime identity mismatch')
    return manifest


def create(inputs,output):
    inputs=Path(inputs).resolve();spec=json.loads(inputs.read_text('utf-8'));output=Path(output).resolve()
    if output.exists() or output.with_suffix('.zip').exists() or output.is_relative_to(ROOT/'acdc'):
        raise ContractError('choose a new output outside source tree')
    records=spec['inputs'];names=set()
    for row in records:
        name=row['destination'];pure=PurePosixPath(name)
        if pure.is_absolute() or '..' in pure.parts or '\\' in name or name in names or name in ('source.zip','MANIFEST.json'):
            raise ContractError('unsafe/duplicate delivery name')
        names.add(name);source=safe_member(ROOT,row['source'])
        if file_sha(source)!=row['sha256']:raise ContractError('delivery input drift: '+row['source'])
    if 'runtime.zip' not in names:raise ContractError('runtime archive required')
    gates=spec['gates']
    accepted=all(value=='passed' for value in gates.values())
    # The input document declares each required gate; pending gates stay visible.
    if not gates:raise ContractError('nonempty explicit acceptance gates required')
    output.mkdir(parents=True)
    source_package(output/'source.zip')
    for row in records:
        target=output/row['destination'];target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(safe_member(ROOT,row['source']),target)
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,check=True,capture_output=True,text=True).stdout.strip()
    files={p.relative_to(output).as_posix():dict(bytes=p.stat().st_size,sha256=file_sha(p)) for p in output.rglob('*') if p.is_file()}
    manifest=dict(schema='pld-acdc-delivery-v1',version=spec['version'],execution_profile='axial_only_v1',
        source_commit=commit,inputs_document_sha256=file_sha(inputs),runtime_manifest_sha256=spec['runtime_manifest_sha256'],
        model_spec_sha256=spec['model_spec_sha256'],acceptance_complete=accepted,gates=gates,files=files,
        notice='Local engineering/research delivery. No upload/publication performed. Consult original model/data license.')
    atomic_json(output/'MANIFEST.json',manifest);verify(output)
    archive=output.with_suffix('.zip')
    if archive.exists():raise ContractError('delivery archive already exists')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(output).as_posix(),compress_type=zipfile.ZIP_STORED if p.suffix=='.zip' else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ContractError('outer delivery CRC failed')
    return dict(directory=str(output),archive=str(archive),archive_sha256=file_sha(archive),manifest_sha256=file_sha(output/'MANIFEST.json'),acceptance_complete=accepted)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='action',required=True)
    c=sub.add_parser('create');c.add_argument('--inputs',type=Path,required=True);c.add_argument('--output',type=Path,required=True)
    v=sub.add_parser('verify');v.add_argument('directory',type=Path)
    a=p.parse_args();r=create(a.inputs,a.output) if a.action=='create' else verify(a.directory)
    print(json.dumps(r if a.action=='create' else dict(version=r['version'],members=len(r['files']),acceptance_complete=r['acceptance_complete'])))
