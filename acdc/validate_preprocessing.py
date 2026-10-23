"""Compare all development MRI inputs directly with frozen A preprocessing.

No neural-network inference is performed or counted here. The independently
prepared input bytes are also compared with the recorded PL/CPU checkpoints.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np

from .contract import ContractError,atomic_json,file_sha,safe_member
from .model import load_bundle
from .release import MANIFEST_SHA
from .volume import load_volume,AxialInputs


def run(model_root,release,results,output,model_sha):
    import torch
    if Path(output).exists():raise ContractError('choose a new input audit')
    torch.set_num_threads(4)
    model=load_bundle(model_root,expected_sha=model_sha);release=Path(release).resolve()
    results=Path(results).resolve()
    if file_sha(release/'MANIFEST.json')!=MANIFEST_SHA:raise ContractError('unknown release')
    identities={r['path']:r['sha256'] for r in json.loads((release/'MANIFEST.json').read_text('utf-8'))['files']}
    member='software/role_a_acdc/geometry.py';source=safe_member(release,member)
    if file_sha(source)!=identities[member]:raise ContractError('A implementation source drift')
    spec=importlib.util.spec_from_file_location('frozen_A_preprocessing',source)
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    cases=json.loads((results/'cases.json').read_text('utf-8'))['cases']
    rows=[];seen=set()
    for case in cases:
        patient,phase=case['patient'],case['phase']
        if (patient,phase) in seen:raise ContractError('duplicate patient-phase')
        seen.add((patient,phase));root=Path(case['result_dir'])
        if not root.is_absolute():root=results/root
        result=json.loads((root/'manifest.json').read_text('utf-8'))
        if result['model_sha256']!=model_sha:raise ContractError('result/model mismatch')
        geometry=root/'geometry'/f'{phase}.json'
        if file_sha(geometry)!=result['files'][f'geometry/{phase}.json']:raise ContractError('geometry hash mismatch')
        member=json.loads(geometry.read_text('utf-8'))['source_release_member'];image=safe_member(release,member)
        if file_sha(image)!=identities[member]:raise ContractError('MRI hash mismatch')
        original,_,metadata=a.prepare_volume(image,geometry_policy='acdc-direction-only-sform-v1')
        volume=load_volume(image,policy='acdc-direction-only-sform-v1',binding=model.implementation_binding)
        if not np.array_equal(volume.affine,np.asarray(metadata['native_affine'])):
            raise ContractError('adopted physical affine differs')
        if not np.array_equal(np.asarray(volume.geometry['working_affine']),np.asarray(metadata['working_affine'])):
            raise ContractError('working affine differs')
        if not np.array_equal(original,volume.working):
            raise ContractError('float32 working voxels differ from A')
        inputs=AxialInputs(volume);hashes=[]
        for z in range(original.shape[2]):
            stack,box=a.stack(original,'axial',z)
            if box!=inputs.spec:raise ContractError('letterbox geometry differs')
            with torch.inference_mode():
                expected=(torch.from_numpy(stack).to(torch.float64)*127).round().clamp(-127,127).to(torch.int8).numpy()[None]
            actual,_=inputs.get(z)
            if not np.array_equal(actual,expected):
                coordinate=tuple(np.argwhere(actual!=expected)[0])
                raise ContractError(f'input mismatch {patient}/{phase}/{z}/{coordinate}')
            digest=hashlib.sha256(expected.tobytes()).hexdigest()
            saved=json.loads((root/'checkpoint'/phase/f'{z:04}.json').read_text('utf-8'))
            if saved['input_sha256']!=digest:raise ContractError('actual inference checkpoint input differs from A')
            hashes.append(digest)
        rows.append(dict(patient=patient,phase=phase,slices=len(hashes),float32_working_voxels_equal=True,
            physical_affines_equal=True,all_integer_inputs_equal=True,input_sha256=hashes))
        atomic_json(output,dict(schema='pld-acdc-input-audit-v1',complete=False,rows=rows))
        print(patient,phase,'A/B working float32 and INT8 input exact',len(hashes),flush=True)
    split=json.loads((model.root/'docs/acdc/PATIENT_SPLIT.json').read_text('utf-8'))
    if seen!={(patient,phase) for patient in split['dev'] for phase in ('ED','ES')}:
        raise ContractError('fixed development cohort incomplete')
    report=dict(schema='pld-acdc-input-audit-v1',complete=True,model_spec_sha256=model_sha,
        A_geometry_sha256=file_sha(source),manifest_sha256=MANIFEST_SHA,patients=20,volumes=len(rows),
        slices=sum(r['slices'] for r in rows),input_mismatches=0,fresh_inference_count=0,
        scope='direct frozen A preprocessing/quantization versus B and actual recorded inference input hashes',
        torch_threads=torch.get_num_threads(),rows=rows)
    atomic_json(output,report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','release','results','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model-spec-sha',required=True);a=p.parse_args()
    run(a.model,a.release,a.results,a.output,a.model_spec_sha)
