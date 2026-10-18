"""Patient-paired native-space evaluation. Never downloads or selects a cohort."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .contract import ContractError,LABELS,file_sha
from .statistics import class_metrics,volumes,paired
from .volume import adopt_geometry


def evaluate(cases_path,expected_model_sha):
    import nibabel as nib
    cases_path=Path(cases_path).resolve()
    spec=json.loads(cases_path.read_text('utf-8'))
    if spec.get('schema')!='pld-acdc-evaluation-cases-v1' or not spec.get('cases'):
        raise ContractError('nonempty evaluation case list required')
    seen=set();rows=[];per_patient={}
    for case in spec['cases']:
        patient,phase=case['patient'],case['phase']
        if not isinstance(patient,str) or not patient or phase not in ('ED','ES') or (patient,phase) in seen:
            raise ContractError('duplicate/invalid evaluation patient phase')
        seen.add((patient,phase))
        root=(cases_path.parent/case['result_dir']).resolve()
        manifest=json.loads((root/'manifest.json').read_text('utf-8'))
        if manifest.get('model_sha256')!=expected_model_sha or manifest.get('execution_profile')!='axial_only_v1':
            raise ContractError('evaluation result/model identity mismatch')
        pred_path=root/'labels'/f'{phase}.nii.gz'
        if file_sha(pred_path)!=manifest['files'][f'labels/{phase}.nii.gz']:
            raise ContractError('evaluation label hash mismatch')
        pred_image=nib.load(pred_path);pred=np.asanyarray(pred_image.dataobj)
        gt_path=(cases_path.parent/case['gt']).resolve()
        if case.get('gt_sha256')!=file_sha(gt_path):raise ContractError('evaluation GT must be hash pinned')
        gt_image=nib.load(gt_path);gt=np.asanyarray(gt_image.dataobj)
        if (pred.shape!=gt.shape or pred.ndim!=3 or not np.issubdtype(pred.dtype,np.integer)
                or not np.issubdtype(gt.dtype,np.integer) or np.any(pred<0) or np.any(pred>3) or np.any(gt<0) or np.any(gt>3)):
            raise ContractError('invalid/different native label grids')
        geom_path=root/'geometry'/f'{phase}.json'
        if file_sha(geom_path)!=manifest['files'][f'geometry/{phase}.json']:
            raise ContractError('evaluation geometry hash mismatch')
        geom=json.loads(geom_path.read_text('utf-8'))
        ga,_=adopt_geometry(gt_image,geom['geometry_policy'],recognized_acdc=geom['geometry_policy']=='acdc-direction-only-sform-v1')
        if not np.allclose(pred_image.affine,ga,rtol=1e-4,atol=1e-5):raise ContractError('GT/prediction physical grids differ')
        metrics={LABELS[c]:class_metrics(pred,gt,pred_image.affine,c) for c in (1,2,3)}
        pv,gv=volumes(pred.astype(np.uint8),pred_image.affine),volumes(gt.astype(np.uint8),ga)
        for name in ('RV','MYO','LV'):
            metrics[name]['absolute_volume_error_mL']=abs(pv['classes'][name]['mL']-gv['classes'][name]['mL'])
        rows.append(dict(patient=patient,phase=phase,metrics=metrics,result_manifest_sha256=file_sha(root/'manifest.json'),
                         gt_sha256=file_sha(gt_path),synthetic=manifest['synthetic']))
        per_patient.setdefault(patient,{})[phase]=(pv,gv,metrics)
    if any(set(v)!={'ED','ES'} for v in per_patient.values()):raise ContractError('evaluation requires complete ED/ES per patient')
    patient_dice={p:{name:float(np.mean([v[t][2][name]['dice'] for t in ('ED','ES')]))
                     for name in ('RV','MYO','LV')} for p,v in per_patient.items()}
    mean_dice={name:float(np.mean([r[name] for r in patient_dice.values()])) for name in ('RV','MYO','LV')}
    ef_errors={}
    for p,v in per_patient.items():
        predicted=paired(v['ED'][0],v['ES'][0]);truth=paired(v['ED'][1],v['ES'][1])
        ef_errors[p]={name:(abs(predicted['cavities'][name]['EF_percent']-truth['cavities'][name]['EF_percent'])
                          if predicted['cavities'][name]['valid'] and truth['cavities'][name]['valid'] else None)
                      for name in ('LV','RV')}
    return dict(schema='pld-acdc-evaluation-v1',model_sha256=expected_model_sha,patient_count=len(per_patient),
                cohort_declared_by_caller=spec.get('cohort'),official_challenge_score=False,
                class_mean_dice=mean_dice,macro_dice=float(np.mean(list(mean_dice.values()))),
                patient_dice=patient_dice,ef_absolute_error_percentage_points=ef_errors,volumes=rows,
                fp32_quantization_gate_evaluated=False,notice='Quantization acceptance additionally requires A FP32 comparison/oracle evidence')
