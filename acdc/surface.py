"""Native label surfaces and separate, PC-generated presentation meshes."""
from __future__ import annotations

from pathlib import Path
import json
import tempfile
import time
import numpy as np

from .contract import COLORS,LABELS,atomic_json,digest,file_sha,ContractError


PRESENTATION = dict(version='pc-sinc-v1',generator='local_pc',subdivisions=1,
                    method='windowed-sinc',iterations=80,pass_band=0.005,
                    quantitative=False,changes_labels=False)


def export_surfaces(labels,affine,directory,*,smooth=False):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk,vtk_to_numpy
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    surfaces=[]
    for c in (1,2,3):
        mask=np.asarray(labels==c,np.uint8)
        # Zero padding caps the finite labelmap at a half-voxel boundary; clipping is reported.
        clipped=bool(any(np.any(np.take(mask,i,axis=ax)) for ax in range(3) for i in (0,-1)))
        padded=np.pad(mask,1)
        image=vtk.vtkImageData();image.SetDimensions(*padded.shape);image.SetOrigin(-1,-1,-1)
        image.GetPointData().SetScalars(numpy_to_vtk(padded.ravel(order='F'),deep=True,array_type=vtk.VTK_UNSIGNED_CHAR))
        contour=vtk.vtkDiscreteMarchingCubes();contour.SetInputData(image);contour.SetValue(0,1);contour.ComputeNormalsOff();contour.Update()
        matrix=vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):matrix.SetElement(i,j,float(affine[i,j]))
        transform=vtk.vtkTransform();transform.SetMatrix(matrix)
        convert=vtk.vtkTransformPolyDataFilter();convert.SetInputConnection(contour.GetOutputPort());convert.SetTransform(transform)
        geometry=convert
        if smooth and contour.GetOutput().GetNumberOfPoints():
            # Work in physical coordinates. This is a display derivative, never
            # fed back to labels, probabilities, volumes or the integer model.
            subdivide=vtk.vtkLinearSubdivisionFilter();subdivide.SetInputConnection(convert.GetOutputPort())
            subdivide.SetNumberOfSubdivisions(PRESENTATION['subdivisions'])
            relax=vtk.vtkWindowedSincPolyDataFilter();relax.SetInputConnection(subdivide.GetOutputPort())
            relax.SetNumberOfIterations(PRESENTATION['iterations']);relax.SetPassBand(PRESENTATION['pass_band'])
            relax.BoundarySmoothingOff();relax.FeatureEdgeSmoothingOff();relax.NonManifoldSmoothingOff()
            relax.NormalizeCoordinatesOn()
            geometry=relax
        normals=vtk.vtkPolyDataNormals();normals.SetInputConnection(geometry.GetOutputPort());normals.SplittingOff();normals.ConsistencyOn();normals.Update()
        mesh=normals.GetOutput()
        writer=vtk.vtkXMLPolyDataWriter();writer.SetFileName(str(directory/(LABELS[c]+'.vtp')));writer.SetInputData(mesh)
        if writer.Write()!=1:raise RuntimeError('VTK mesh export failed')
        ply_path=directory/(LABELS[c]+'.ply')
        if mesh.GetNumberOfPoints():
            ply=vtk.vtkPLYWriter();ply.SetFileName(str(ply_path));ply.SetInputData(mesh)
            ply.SetFileTypeToBinary();ply.SetColorModeToUniformColor();ply.SetColor(*COLORS[c])
            if ply.Write()!=1:raise RuntimeError('PLY mesh export failed')
        else:
            ply_path.write_text('ply\nformat ascii 1.0\nelement vertex 0\nproperty float x\nproperty float y\nproperty float z\nelement face 0\nproperty list uchar int vertex_indices\nend_header\n',encoding='ascii')
        if mesh.GetNumberOfPoints():
            # Browser uses de-indexed triangles so WebGL 1 does not require uint32 index extensions.
            pts=vtk_to_numpy(mesh.GetPoints().GetData()).astype(np.float32)
            nrm=vtk_to_numpy(mesh.GetPointData().GetNormals()).astype(np.float32)
            faces=vtk_to_numpy(mesh.GetPolys().GetConnectivityArray()).reshape(-1,3)
            vertices=pts[faces].reshape(-1,3);n=nrm[faces].reshape(-1,3)
        else:vertices=n=np.empty((0,3),np.float32)
        vertices.astype('<f4').tofile(directory/(LABELS[c]+'.vertices.f32'))
        n.astype('<f4').tofile(directory/(LABELS[c]+'.normals.f32'))
        surfaces.append(dict(label=c,name=LABELS[c],color=COLORS[c],vertices=len(vertices),units='mm',
                             ply_file=LABELS[c]+'.ply',vtp_file=LABELS[c]+'.vtp',
                             coordinate_system='RAS',fov_truncated=clipped,smoothed=smooth,
                             generator='local_pc',quantitative=False))
    atomic_json(directory/'meshes.json',surfaces)
    return surfaces


def prepare_presentation(result,cache,phase):
    """Derive a cached view from an authenticated completed result, unchanged.

    Kept outside the result directory: historical ZIP/hash/timing evidence and
    original native surfaces remain intact. Caller serializes cache creation.
    """
    import nibabel as nib
    result=Path(result);cache=Path(cache)
    if phase not in ('ED','ES'):raise ContractError('ED or ES required')
    status=json.loads((result/'status.json').read_text('utf-8'))
    if status.get('status')!='completed':raise ContractError('complete result required for presentation')
    manifest=json.loads((result/'manifest.json').read_text('utf-8'))
    rel=f'labels/{phase}.nii.gz';source=result/rel;source_sha=file_sha(source)
    if manifest.get('files',{}).get(rel)!=source_sha:raise ContractError('native label hash mismatch')
    key=digest(dict(source_sha256=source_sha,settings=PRESENTATION))
    destination=cache/key/phase;description=destination/'presentation.json'
    if not description.is_file():
        begin=time.perf_counter();image=nib.load(source)
        labels=np.asarray(image.dataobj)
        if labels.dtype!=np.uint8 or labels.ndim!=3 or np.any(labels>3):
            raise ContractError('native uint8 labels 0..3 required')
        destination.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='surface-',dir=destination.parent) as temporary:
            staging=Path(temporary)/phase
            meshes=export_surfaces(labels,image.affine,staging,smooth=True)
            report=dict(settings=PRESENTATION,phase=phase,source_label_sha256=source_sha,
                        source_result=result.name,meshes=meshes,generation_seconds=time.perf_counter()-begin)
            atomic_json(staging/'presentation.json',report)
            staging.replace(destination)
    return key,json.loads(description.read_text('utf-8'))
