"""Unsmoothed label surfaces, transformed by the adopted physical affine."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from .contract import COLORS,LABELS,atomic_json


def export_surfaces(labels,affine,directory):
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
        normals=vtk.vtkPolyDataNormals();normals.SetInputConnection(convert.GetOutputPort());normals.SplittingOff();normals.ConsistencyOn();normals.Update()
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
                             coordinate_system='RAS',fov_truncated=clipped,smoothed=False))
    atomic_json(directory/'meshes.json',surfaces)
    return surfaces
