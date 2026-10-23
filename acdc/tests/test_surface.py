"""Presentation derivatives must leave quantitative, authenticated results intact."""
import json
from pathlib import Path
import tempfile
import unittest

import nibabel as nib
import numpy as np

from acdc.contract import ContractError,atomic_json,file_sha
from acdc.surface import export_surfaces,prepare_presentation


class PresentationTest(unittest.TestCase):
    def test_physical_smoothing_empty_class_and_closed_surface(self):
        import vtk
        # An anisotropic, translated and reflected grid exercises physical placement.
        labels=np.zeros((12,13,5),np.uint8);labels[2:10,3:11,1:4]=2
        before=labels.copy();affine=np.diag([-2.,3.,10.,1.]);affine[:3,3]=[60.,-80.,30.]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            raw=export_surfaces(labels,affine,root/'raw')
            smooth=export_surfaces(labels,affine,root/'smooth',smooth=True)
            np.testing.assert_array_equal(labels,before)
            self.assertEqual(smooth[0]['vertices'],0)
            self.assertTrue(all(m['smoothed'] and not m['quantitative'] for m in smooth))
            p=np.fromfile(root/'smooth/MYO.vertices.f32',dtype='<f4').reshape(-1,3)
            n=np.fromfile(root/'smooth/MYO.normals.f32',dtype='<f4').reshape(-1,3)
            self.assertTrue(np.isfinite(p).all());self.assertTrue(np.isfinite(n).all())
            self.assertLess(p[:,0].max(),61);self.assertGreater(p[:,0].min(),35)
            self.assertLess(p[:,1].max(),-40);self.assertGreater(p[:,1].min(),-80)
            # Sinc can overshoot the label's box; check the physical field of
            # view (including half voxels) and preserved translated center.
            self.assertGreater(p[:,2].min(),25);self.assertLess(p[:,2].max(),75)
            np.testing.assert_allclose((p.min(axis=0)+p.max(axis=0))/2,[49.,-60.5,50.],atol=0.02)
            self.assertGreater(smooth[1]['vertices'],raw[1]['vertices'])
            np.testing.assert_allclose(np.linalg.norm(n,axis=1),1,atol=1e-5)
            reader=vtk.vtkXMLPolyDataReader();reader.SetFileName(str(root/'smooth/MYO.vtp'));reader.Update()
            edges=vtk.vtkFeatureEdges();edges.SetInputConnection(reader.GetOutputPort())
            edges.BoundaryEdgesOn();edges.NonManifoldEdgesOn();edges.FeatureEdgesOff();edges.ManifoldEdgesOff();edges.Update()
            self.assertEqual(edges.GetOutput().GetNumberOfCells(),0)

    def test_cache_preserves_result_and_rejects_changed_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);result=root/'job';(result/'labels').mkdir(parents=True)
            labels=np.zeros((8,8,4),np.uint8);labels[2:6,2:6,:]=3
            image=nib.Nifti1Image(labels,np.diag([2.,2.,8.,1.]))
            path=result/'labels/ED.nii.gz';nib.save(image,path)
            atomic_json(result/'status.json',dict(status='completed'))
            atomic_json(result/'manifest.json',dict(files={'labels/ED.nii.gz':file_sha(path)}))
            (result/'statistics.json').write_text('{"unchanged":true}',encoding='utf-8')
            before={p.relative_to(result).as_posix():file_sha(p) for p in result.rglob('*') if p.is_file()}
            key,report=prepare_presentation(result,root/'cache','ED')
            self.assertTrue(report['meshes'][2]['fov_truncated'])
            self.assertFalse(report['settings']['changes_labels'])
            self.assertEqual((key,report),prepare_presentation(result,root/'cache','ED'))
            after={p.relative_to(result).as_posix():file_sha(p) for p in result.rglob('*') if p.is_file()}
            self.assertEqual(before,after)
            labels[0,0,0]=1;nib.save(nib.Nifti1Image(labels,image.affine),path)
            with self.assertRaisesRegex(ContractError,'hash mismatch'):
                prepare_presentation(result,root/'cache','ED')

    def test_incomplete_result_is_not_presented(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);atomic_json(root/'status.json',dict(status='running'))
            with self.assertRaisesRegex(ContractError,'complete result'):
                prepare_presentation(root,root/'cache','ED')
