"""Audit V2 native math against authenticated V1 cases; never executes PL.

Saved heads are deliberately replayed for reconstruction qualification only.
No frame or timing from this tool constitutes a fresh PL performance sample.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from acdc.contract import ContractError,atomic_json,file_sha,digest
from acdc.model import load_bundle
from acdc.volume import load_volume as reference_volume
from .native import load_native
from .volume import load_volume,AxialInputs
from .reconstruct import Collector,make_exp_table


def run(model_root,results,output,model_sha,*,limit=None):
    import nibabel as nib
    model=load_bundle(model_root,expected_sha=model_sha)
    kernel=load_native();table=make_exp_table(model.head_scale)
    cases=json.loads((results/'cases.json').read_text('utf-8'))['cases']
    rows=[];seen=set();began=time.perf_counter()
    for case in cases[:limit]:
        phase=case['phase'];patient=case['patient'];root=Path(case['result_dir'])
        if not root.is_absolute():root=results/root
        manifest=json.loads((root/'manifest.json').read_text('utf-8'))
        if manifest['model_sha256']!=model_sha:raise ContractError('model identity mismatch')
        def asset(rel):
            path=root/rel
            if file_sha(path)!=manifest['files'][rel]:raise ContractError('reference asset hash mismatch: '+rel)
            return path
        geometry=json.loads(asset(f'geometry/{phase}.json').read_text('utf-8'))
        source=Path(case['gt']).with_name(Path(case['gt']).name.replace('_gt.nii','.nii'))
        if file_sha(source)!=geometry['input_sha256']:raise ContractError('source MRI mismatch')
        started=time.perf_counter()
        volume=load_volume(source,kernel,policy='acdc-direction-only-sform-v1',binding=model.implementation_binding)
        preparation=time.perf_counter()-started
        old=reference_volume(source,policy='acdc-direction-only-sform-v1',binding=model.implementation_binding)
        np.testing.assert_array_equal(volume.native,old.native)
        np.testing.assert_array_equal(volume.working,old.working)
        for field in ('native_shape','native_affine','working_shape','working_affine','normalization','geometry_decision'):
            if volume.geometry[field]!=geometry[field]:raise ContractError('geometry mismatch: '+field)
        del old
        inputs=AxialInputs(volume,kernel);collector=Collector(volume,kernel,table)
        nz=volume.working.shape[2];input_time=0.;restore_time=0.
        for z in range(nz):
            tick=time.perf_counter();x,_=inputs.get(z);input_time+=time.perf_counter()-tick
            checkpoint=root/'checkpoint'/phase/f'{z:04}.json'
            saved=json.loads(checkpoint.read_text('utf-8'))
            if hashlib.sha256(x.tobytes()).hexdigest()!=saved['input_sha256']:
                raise ContractError(f'INT8 input differs: {patient}/{phase}/{z}')
            raw=checkpoint.with_suffix('.bin').read_bytes()
            if hashlib.sha256(raw).hexdigest()!=saved['head_sha256']:
                raise ContractError('reference head checkpoint mismatch')
            if (saved['phase']!=phase or saved['center_index']!=z or saved['model_sha256']!=model_sha
                    or saved['geometry_sha256']!=digest({k:v for k,v in geometry.items() if k!='letterbox'})):
                raise ContractError('checkpoint identity mismatch')
            head=np.frombuffer(raw,np.int8).reshape(1,4,256,256)
            tick=time.perf_counter();collector.add(head,z,inputs.spec);restore_time+=time.perf_counter()-tick
        tick=time.perf_counter();_,labels=collector.finish();restore_time+=time.perf_counter()-tick
        expected=np.asarray(nib.load(str(asset(f'labels/{phase}.nii.gz'))).dataobj,np.uint8)
        np.testing.assert_array_equal(labels,expected)
        pair=(patient,phase)
        if pair in seen:raise ContractError('duplicate patient/phase')
        seen.add(pair)
        rows.append(dict(patient=patient,phase=phase,source_sha256=file_sha(source),native_shape=list(volume.native.shape),
                         working_shape=list(volume.working.shape),slices=nz,working_float32_equal=True,
                         integer_inputs_equal=True,native_label_mismatches=0,prepare_seconds=preparation,
                         input_seconds=input_time,restore_seconds=restore_time))
        atomic_json(output,dict(schema='pld-acdc-appliance-native-audit-v2',complete=False,rows=rows))
        print(patient,phase,nz,'inputs and labels exact',flush=True)
    split=json.loads((model.root/'docs/acdc/PATIENT_SPLIT.json').read_text('utf-8'))
    complete=seen=={(patient,phase) for patient in split['dev'] for phase in ('ED','ES')}
    report=dict(schema='pld-acdc-appliance-native-audit-v2',complete=complete,rows=rows,
                model_sha256=model_sha,native_build=kernel.info,exp_table_sha256=hashlib.sha256(table.tobytes()).hexdigest(),
                source_results=str(results),volumes=len(rows),slices=sum(r['slices'] for r in rows),
                fresh_pl_inferences=0,measured_on='development_host',total_audit_seconds=time.perf_counter()-began,
                scope='native preprocessing and saved-head restoration; NOT an ARM/PL performance measurement')
    atomic_json(output,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('model','results','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--model-sha',required=True);parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    run(args.model,args.results,args.output,args.model_sha,limit=args.limit)
