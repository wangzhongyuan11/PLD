"""Compare frozen A softmax/restoration with B on authenticated saved heads.

This is postprocessing evidence only. It performs no model inference and never
counts checkpoints as fresh PL samples.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np

from .contract import ContractError,atomic_json,file_sha,safe_member
from .model import load_bundle
from .release import MANIFEST_SHA
from .reconstruct import Collector,plane_probabilities
from .volume import load_volume,AxialInputs


def compare_probabilities(actual,expected,*,tolerance=1e-6):
    if actual.shape!=expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ContractError('invalid/different restored probability grids')
    error=float(np.max(np.abs(actual-expected)))
    differences=int(np.count_nonzero(actual.argmax(0)!=expected.argmax(0)))
    if error>tolerance or differences:
        raise ContractError(f'restoration mismatch: max_probability_error={error}, label_differences={differences}')
    return dict(max_probability_error=error,label_differences=differences)


def run(model_root,release,results,output,model_sha):
    import nibabel as nib
    import torch
    if Path(output).exists():raise ContractError('choose a new restoration audit')
    model=load_bundle(model_root,expected_sha=model_sha)
    release=Path(release).resolve();results=Path(results).resolve()
    if file_sha(release/'MANIFEST.json')!=MANIFEST_SHA:raise ContractError('unrecognized A release')
    manifest=json.loads((release/'MANIFEST.json').read_text('utf-8'))
    identities={r['path']:r['sha256'] for r in manifest['files']}
    member='software/role_a_acdc/geometry.py';a_file=safe_member(release,member)
    if file_sha(a_file)!=identities[member]:raise ContractError('A geometry source hash mismatch')
    module_spec=importlib.util.spec_from_file_location('frozen_acdc_a_geometry',a_file)
    a_geometry=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(a_geometry)
    cases=json.loads((results/'cases.json').read_text('utf-8'))['cases']
    rows=[]
    for case in cases:
        patient,phase=case['patient'],case['phase'];root=Path(case['result_dir'])
        if not root.is_absolute():root=results/root
        result_manifest=json.loads((root/'manifest.json').read_text('utf-8'))
        if result_manifest['model_sha256']!=model_sha:raise ContractError('result/model mismatch')
        geom_file=root/'geometry'/f'{phase}.json'
        if file_sha(geom_file)!=result_manifest['files'][f'geometry/{phase}.json']:raise ContractError('geometry hash mismatch')
        geom=json.loads(geom_file.read_text('utf-8'))
        image=safe_member(release,geom['source_release_member'])
        if file_sha(image)!=identities[geom['source_release_member']]:raise ContractError('MRI hash mismatch')
        volume=load_volume(image,policy='acdc-direction-only-sform-v1',binding=model.implementation_binding)
        if any(geom.get(key)!=value for key,value in volume.geometry.items()):
            raise ContractError('geometry identity mismatch')
        inputs=AxialInputs(volume);collector=Collector(volume,model_sha,model.head_scale,phase)
        a_working=np.empty((4,*volume.working.shape),np.float32);plane_error=0.
        for z in range(volume.working.shape[2]):
            path=root/'checkpoint'/phase/f'{z:04}.json';meta=json.loads(path.read_text('utf-8'))
            binary=path.with_suffix('.bin')
            if file_sha(binary)!=meta['head_sha256']:raise ContractError('checkpoint head hash mismatch')
            head=np.fromfile(binary,np.int8).reshape(1,4,256,256)
            collector.add(head,meta,inputs.spec)
            with torch.inference_mode():
                probabilities=torch.softmax(torch.from_numpy(head.astype(np.float32))*model.head_scale,1)[0].numpy()
            a_plane=a_geometry.undo_letterbox(probabilities,inputs.spec)
            b_plane=plane_probabilities(head,model.head_scale,inputs.spec)
            plane_error=max(plane_error,float(np.max(np.abs(a_plane-b_plane))))
            for c in range(4):a_geometry.put_plane(a_working[c],2,z,a_plane[c])
        b_native,labels=collector.finish();a_native=a_geometry.probabilities_to_native(a_working,geom)
        compared=compare_probabilities(b_native,a_native)
        label_file=root/'labels'/f'{phase}.nii.gz'
        if file_sha(label_file)!=result_manifest['files'][f'labels/{phase}.nii.gz']:raise ContractError('label hash mismatch')
        saved=nib.load(label_file)
        if not np.array_equal(labels,np.asanyarray(saved.dataobj)):raise ContractError('saved label differs from checked restoration')
        if not np.allclose(saved.affine,volume.affine,rtol=1e-6,atol=1e-5):raise ContractError('saved affine differs')
        if plane_error>1e-6:raise ContractError('working plane probability tolerance exceeded')
        rows.append(dict(patient=patient,phase=phase,slices=len(collector.seen),
            result_manifest_sha256=file_sha(root/'manifest.json'),max_plane_probability_error=plane_error,**compared))
        print(patient,phase,'restore exact labels; probability error',compared['max_probability_error'],flush=True)
        atomic_json(output,dict(schema='pld-acdc-restoration-audit-v1',complete=False,rows=rows))
        del collector,a_working,a_native,b_native,volume,inputs
    report=dict(schema='pld-acdc-restoration-audit-v1',complete=True,model_spec_sha256=model_sha,
        A_geometry_sha256=file_sha(a_file),release_manifest_sha256=MANIFEST_SHA,rows=rows,
        volumes=len(rows),slices=sum(r['slices'] for r in rows),
        max_native_probability_error=max(r['max_probability_error'] for r in rows),
        max_plane_probability_error=max(r['max_plane_probability_error'] for r in rows),
        label_differences=sum(r['label_differences'] for r in rows),fresh_inference_count=0,
        scope='frozen A Torch softmax/geometry versus B NumPy softmax/Collector using saved integer heads')
    atomic_json(output,report);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('model','release','results','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--model-spec-sha',required=True);a=p.parse_args()
    run(a.model,a.release,a.results,a.output,a.model_spec_sha)
