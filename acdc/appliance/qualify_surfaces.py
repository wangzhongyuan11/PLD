"""PC qualification of a cheaper physical surface profile against frozen V1.

Both configurations run in the PC VTK reference. This is geometry evidence,
not an ARM timing measurement and not segmentation/clinical accuracy evidence.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np

from acdc.contract import ContractError,atomic_json,file_sha


def run(results,output):
    import nibabel as nib
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    import acdc.surface as reference
    cases=json.loads((results/'cases.json').read_text('utf-8'))['cases'];rows=[]
    original=dict(reference.PRESENTATION)
    try:
        for case in cases:
            root=Path(case['result_dir']);phase=case['phase'];patient=case['patient']
            manifest=json.loads((root/'manifest.json').read_text('utf-8'))
            label=root/'labels'/(phase+'.nii.gz')
            if file_sha(label)!=manifest['files'][f'labels/{phase}.nii.gz']:raise ContractError('native label hash mismatch')
            image=nib.load(str(label));labels=np.asarray(image.dataobj,np.uint8)
            with tempfile.TemporaryDirectory(prefix='acdc-geometry-') as directory:
                directory=Path(directory)
                reference.PRESENTATION.update(subdivisions=1,iterations=80,pass_band=.005)
                reference.export_surfaces(labels,image.affine,directory/'reference',smooth=True)
                reference.PRESENTATION.update(subdivisions=0,iterations=40,pass_band=.02)
                reference.export_surfaces(labels,image.affine,directory/'candidate',smooth=True)
                for structure in ('RV','MYO','LV'):
                    meshes=[]
                    for name in ('reference','candidate'):
                        reader=vtk.vtkXMLPolyDataReader();reader.SetFileName(str(directory/name/(structure+'.vtp')));reader.Update()
                        mesh=vtk.vtkPolyData();mesh.DeepCopy(reader.GetOutput());meshes.append(mesh)
                    expected,actual=meshes
                    if not expected.GetNumberOfPoints() or not actual.GetNumberOfPoints():
                        same=expected.GetNumberOfPoints()==actual.GetNumberOfPoints()
                        row=dict(patient=patient,phase=phase,structure=structure,empty=True,passed=same)
                    else:
                        # Explicit serial implicit-distance batches avoid the
                        # distance filter's crashing pipeline on this VTK wheel.
                        both=[]
                        for source,target in ((actual,expected),(expected,actual)):
                            distance=vtk.vtkImplicitPolyDataDistance();distance.SetInput(target)
                            values=vtk.vtkDoubleArray();values.SetNumberOfComponents(1)
                            values.SetNumberOfTuples(source.GetNumberOfPoints())
                            distance.FunctionValue(source.GetPoints().GetData(),values)
                            both.append(np.abs(vtk_to_numpy(values)).copy())
                        distances=np.concatenate(both)
                        volumes=[]
                        for mesh in meshes:
                            volume=vtk.vtkMassProperties();volume.SetInputData(mesh);volume.Update();volumes.append(volume.GetVolume())
                        ratio=volumes[1]/volumes[0] if volumes[0]>0 else None
                        p95=float(np.percentile(distances,95));maximum=float(distances.max())
                        row=dict(patient=patient,phase=phase,structure=structure,p95_mm=p95,max_mm=maximum,
                            volume_ratio=ratio,reference_points=expected.GetNumberOfPoints(),candidate_points=actual.GetNumberOfPoints(),
                            passed=p95<=.5 and maximum<=2 and ratio is not None and abs(ratio-1)<=.01)
                    rows.append(row)
            atomic_json(output,dict(complete=False,rows=rows))
            print(patient,phase,'geometry',all(r['passed'] for r in rows[-3:]),flush=True)
    finally:reference.PRESENTATION.clear();reference.PRESENTATION.update(original)
    report=dict(schema='pld-acdc-surface-profile-qualification-v2',complete=True,volumes=len(cases),rows=rows,
        reference=dict(subdivisions=1,iterations=80,pass_band=.005),candidate=dict(subdivisions=0,iterations=40,pass_band=.02),
        reference_source_sha256=file_sha(Path(reference.__file__)),vtk_version=vtk.vtkVersion.GetVTKVersion(),
        passed=all(r['passed'] for r in rows),scope='PC geometry comparison only; no PL or ARM performance samples')
    atomic_json(output,report);return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.results,args.output)
