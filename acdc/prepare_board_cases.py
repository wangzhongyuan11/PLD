"""Prepare bounded board cases from already verified CPU development heads."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from .contract import ContractError,atomic_json,file_sha,safe_member
from .model import load_bundle
from .volume import load_volume,AxialInputs
from .verify import reference_case


def prepare(model_root,dataset,development,output,sha,patients=10):
    model=load_bundle(model_root,expected_sha=sha)
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    development=Path(development).resolve();dataset=Path(dataset).resolve()
    report=json.loads((development/'evaluation.json').read_text('utf-8'))
    if (report['model_sha256']!=sha or report['patient_count']!=20
            or report['A_report_comparison']['max_abs_dice_delta']!=0):
        raise ContractError('development oracle has not reproduced A')
    split=json.loads(safe_member(model.root,'docs/acdc/PATIENT_SPLIT.json').read_text('utf-8'))
    cases=[]
    def append(x,head,context):
        i=len(cases);row=dict(context)
        for key,value in [('input',x),('head',head)]:
            path=output/f'{i:03}.{key}.bin';path.write_bytes(value.tobytes())
            row[key]=dict(path=path.name,sha256=file_sha(path),bytes=path.stat().st_size)
        cases.append(row)
    x,oracle=reference_case(model)
    append(x,oracle['head'],dict(source='A delivered fixed integer vector, historical cache input'))
    from .witness import forward
    independently=forward(model,x)
    for name,a in independently.items():
        if not np.array_equal(a,oracle['captures'][name]):raise ContractError('node witness mismatch: '+name)
    nodes=output/'nodes.npz'
    np.savez_compressed(nodes,**oracle['captures'],model_spec_sha256=np.array(sha),
                        input_sha256=np.array(hashlib.sha256(x.tobytes()).hexdigest()))
    del independently,oracle
    for patient in split['dev'][:patients]:
        cfg=dict(line.split(':',1) for line in (dataset/patient/'Info.cfg').read_text().splitlines() if ':' in line)
        for phase in ('ED','ES'):
            image=dataset/patient/f'{patient}_frame{int(cfg[phase]):02}.nii.gz'
            volume=load_volume(image,policy='acdc-direction-only-sform-v1',binding=model.implementation_binding)
            center=volume.working.shape[2]//2;input_tensor,_=AxialInputs(volume).get(center)
            folder=development/patient/'checkpoint'/phase
            previous=json.loads((folder/f'{center:04}.json').read_text('utf-8'))
            expected=safe_member(folder,f'{center:04}.bin')
            if (previous['model_sha256']!=sha or previous['geometry_sha256']!=volume.geometry_sha
                    or previous['input_sha256']!=hashlib.sha256(input_tensor.tobytes()).hexdigest()
                    or previous['head_sha256']!=file_sha(expected)):
                raise ContractError('development checkpoint identity mismatch')
            head=np.fromfile(expected,np.int8).reshape(1,4,256,256)
            append(input_tensor,head,dict(source='float32 production preprocessing and independently evaluated CPU head',
                                          patient=patient,phase=phase,center_index=center,geometry_sha256=volume.geometry_sha))
    atomic_json(output/'cases.json',dict(schema='pld-acdc-board-cases-v1',model_spec_sha256=sha,cases=cases,
        node_oracle=dict(path=nodes.name,sha256=file_sha(nodes),origin='B INT64 vs independent Torch; final head matches A delivered golden'),
        synthetic=False,physical_board_accepted=False))
    return output/'cases.json'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=Path,required=True)
    p.add_argument('--model-spec-sha',required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--development',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--patients',type=int,default=10);a=p.parse_args()
    if not 1<=a.patients<=20:p.error('patients must be 1..20')
    print(prepare(a.model,a.dataset,a.development,a.output,a.model_spec_sha,a.patients))
