"""Reproduce the pinned 20-patient Axial software development evaluation.

The default is an explicit CPU reference run. --board-url instead requires a
certified physical PL worker and the independent CPU checkpoint root. Every
MRI/GT is authenticated by the frozen A release manifest; oracles are compared
with fresh worker heads and never substituted for inference.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

from .contract import atomic_json,file_sha,ContractError,safe_member
from .model import load_bundle
from .backend import TorchReferenceBackend,RemoteBackend
from .pipeline import Pipeline
from .evaluate import evaluate


def run(model_root,dataset,output,expected_sha,*,board_url=None,oracle_root=None):
    model=load_bundle(model_root,expected_sha=expected_sha)
    if model.synthetic or not model.spec.get('release_members'):raise ContractError('authenticated research model required')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    dataset=Path(dataset).resolve()
    manifest=json.loads(safe_member(model.root,'MANIFEST.json').read_text('utf-8'))
    identity={r['path']:r['sha256'] for r in manifest['files']}
    split=json.loads(safe_member(model.root,'docs/acdc/PATIENT_SPLIT.json').read_text('utf-8'))
    expected=json.loads(safe_member(model.root,'validation/profiles/axial/int8.json').read_text('utf-8'))
    target={(r['patient'],r['phase']):r for r in expected['rows']}
    if board_url and oracle_root is None:raise ContractError('PL volume run requires independent CPU oracle root')
    backend=(RemoteBackend(board_url,model_sha=model.model_sha256,head_scale=model.head_scale,
                           binding=model.implementation_binding) if board_url else TorchReferenceBackend(model))
    cases=[];comparisons=[]
    try:
        for patient in split['dev']:
            folder=dataset/patient
            info=folder/'Info.cfg'
            if file_sha(info)!=identity[f'datasets/acdc/training/{patient}/Info.cfg']:
                raise ContractError('patient Info.cfg hash mismatch')
            cfg=dict(line.split(':',1) for line in info.read_text().splitlines() if ':' in line)
            images={}
            for phase in ('ED','ES'):
                images[phase]=folder/f'{patient}_frame{int(cfg[phase]):02}.nii.gz'
                for path in (images[phase],images[phase].with_name(images[phase].name.replace('.nii.gz','_gt.nii.gz'))):
                    rel=f'datasets/acdc/training/{patient}/{path.name}'
                    if file_sha(path)!=identity[rel]:raise ContractError('development MRI/GT hash mismatch')
            result=output/patient
            Pipeline(backend,policy='acdc-direction-only-sform-v1').run(images['ED'],images['ES'],result,
                oracle_from=Path(oracle_root)/patient if oracle_root is not None else None)
            stats=json.loads((result/'statistics.json').read_text('utf-8'))
            for phase in ('ED','ES'):
                gt=images[phase].with_name(images[phase].name.replace('.nii.gz','_gt.nii.gz'))
                cases.append(dict(patient=patient,phase=phase,result_dir=str(result),gt=str(gt),gt_sha256=file_sha(gt)))
                for c,name in ((1,'RV'),(2,'MYO'),(3,'LV')):
                    actual=stats['phases'][phase]['classes'][name]['mL']
                    wanted=target[patient,phase]['metrics'][str(c)]['pred_volume_ml']
                    comparisons.append(dict(patient=patient,phase=phase,structure=name,volume_delta_mL=actual-wanted))
            atomic_json(output/'progress.json',dict(patients_completed=len(cases)//2,total_patients=20,
                        execution_backend=backend.kind,PL_numerics_checked=bool(board_url),last_patient=patient))
            print(patient,'completed',flush=True)
        case_path=output/'cases.json'
        atomic_json(case_path,dict(schema='pld-acdc-evaluation-cases-v1',cohort='fixed A development split, used for model selection',
                                  release_manifest=str(model.root/'MANIFEST.json'),cases=cases))
        report=evaluate(case_path,expected_sha)
        deltas=[]
        for row in report['volumes']:
            for c,name in ((1,'RV'),(2,'MYO'),(3,'LV')):
                a=row['metrics'][name]['dice'];b=target[row['patient'],row['phase']]['metrics'][str(c)]['dice']
                deltas.append(a-b)
        report['A_report_comparison']=dict(max_abs_dice_delta=float(np.max(np.abs(deltas))),
            max_abs_volume_delta_mL=max(abs(r['volume_delta_mL']) for r in comparisons),volumes=40,
            source_report_sha256=file_sha(safe_member(model.root,'validation/profiles/axial/int8.json')),
            volume_comparisons=comparisons)
        report['physical_board_accepted']=False
        report['PL_volume_numerics_passed']=bool(board_url)
        report['runtime_manifest_sha256']=backend.runtime_sha
        atomic_json(output/'evaluation.json',report)
        if max(abs(x) for x in deltas)>1e-12 or max(abs(r['volume_delta_mL']) for r in comparisons)>1e-6:
            raise ContractError('development reproduction differs from frozen A report; inspect evaluation.json')
        return report
    finally:backend.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--model-spec-sha',required=True)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--board-url');p.add_argument('--oracle-root',type=Path)
    a=p.parse_args();r=run(a.model,a.dataset,a.output,a.model_spec_sha,board_url=a.board_url,oracle_root=a.oracle_root)
    print(json.dumps(dict(patients=r['patient_count'],macro_dice=r['macro_dice'],physical_board_accepted=False)))
