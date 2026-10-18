import tempfile
import unittest
from pathlib import Path
import nibabel as nib
import numpy as np

from acdc.contract import ContractError
from acdc.volume import normalize,grid,adopt_geometry,load_volume,AxialInputs,letterbox_spec
from acdc.reconstruct import normalize_probabilities,Collector
from acdc.statistics import volumes,ef,class_metrics


class VolumeTest(unittest.TestCase):
    def test_normalization_zero_is_not_forced_to_minus_one(self):
        a=np.linspace(-20,20,27,dtype=np.float32).reshape(3,3,3);a[1,1,1]=0
        u,m=normalize(a)
        self.assertAlmostEqual(float(u[1,1,1]),0.,places=6)
        self.assertEqual(u.dtype,np.float32)
        with self.assertRaises(ContractError):normalize(np.ones((3,3,3),np.float32))
        a[0,0,0]=np.nan
        with self.assertRaises(ContractError):normalize(a)

    def test_center_corner_count_and_oblique(self):
        a=np.diag([1.5,1.5,5,1])
        self.assertEqual(grid((256,256,12),a)[0],(193,193,29))
        theta=.4;rot=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
        a[:3,:3]=rot@a[:3,:3]
        self.assertGreater(grid((256,256,12),a)[0][2],29)

    def test_geometry_repair_is_constrained(self):
        image=nib.Nifti1Image(np.ones((3,3,3),np.float32),np.eye(4))
        image.header.set_xyzt_units('mm');image.header.set_zooms((2,3,5))
        with self.assertRaises(ContractError):adopt_geometry(image)
        with self.assertRaises(ContractError):adopt_geometry(image,'acdc-direction-only-sform-v1')
        a,decision=adopt_geometry(image,'acdc-direction-only-sform-v1',recognized_acdc=True)
        np.testing.assert_array_equal(np.diag(a),[2,3,5,1]);self.assertNotEqual(decision,'unchanged')
        image.header['qform_code']=1
        with self.assertRaises(ContractError):adopt_geometry(image,'acdc-direction-only-sform-v1',recognized_acdc=True)

    def test_four_dimensional_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            image=nib.Nifti1Image(np.ones((3,3,3,2),np.float32),np.eye(4));image.header.set_xyzt_units('mm')
            p=Path(d)/'v.nii.gz';nib.save(image,p)
            with self.assertRaises(ContractError):load_volume(p)

    def test_axial_stack_cache_boundaries_and_axis_order(self):
        from types import SimpleNamespace
        v=np.arange(5*7*4,dtype=np.float32).reshape(5,7,4)/140*2-1
        inputs=AxialInputs(SimpleNamespace(working=v))
        x,ids=inputs.get(0);self.assertEqual(ids,[0,0,1]);np.testing.assert_array_equal(x[0,0],x[0,1])
        for z in range(1,4):inputs.get(z)
        self.assertEqual(inputs.resized_planes,4);self.assertLessEqual(len(inputs.cache),3)
        self.assertEqual(inputs.spec['original_hw'],[7,5])
        self.assertEqual(letterbox_spec([7,5])['resized_hw'],[256,183])

    def test_probability_fallback_preserves_other_channels(self):
        p=np.zeros((4,1,1,2),np.float32);p[1,0,0,0]=1e-9
        a=normalize_probabilities(p)
        self.assertGreater(float(a[1,0,0,0]),0);np.testing.assert_allclose(a.sum(0),1)

    def test_statistics_and_invalid_ef(self):
        label=np.zeros((4,4,4),np.uint8);label[:2,:2,:2]=3
        r=volumes(label,np.diag([2.,3.,5.,1.]))
        self.assertAlmostEqual(r['classes']['LV']['mL'],.240)
        self.assertEqual(ef(100,40)['EF_percent'],60)
        self.assertIsNone(ef(0,0)['EF_percent']);self.assertIsNone(ef(40,100)['EF_percent'])
        metric=class_metrics(label,label,np.diag([2,3,5,1]),3)
        self.assertEqual(metric['dice'],1);self.assertEqual(metric['HD95_mm'],0)
        self.assertIsNone(class_metrics(label,np.zeros_like(label),np.eye(4),3)['HD95_mm'])
