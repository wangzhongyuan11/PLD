import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import nibabel as nib
import numpy as np

from acdc.contract import ContractError
from acdc.model import load_bundle
from acdc.release import import_release,AXIAL_SHA
from acdc.volume import adopt_geometry,check_binding

SOURCE=Path(os.environ.get('ACDC_RELEASE_ROOT','out/acdc/release_review_20261019/extracted/ACDC_TriPlanar_B16_P3_v1_Models_and_Dataset_20261018'))


class ReleaseBoundaryTest(unittest.TestCase):
    def test_wheel_suffix_records_build_but_compares_numerical_release(self):
        versions=dict(numpy='2.3.5',scipy='1.18.1',nibabel='5.4.2',torch='2.11.0+cpu')
        with patch('acdc.volume.environment',return_value=versions):
            self.assertEqual(check_binding({'versions':dict(versions,torch='2.11.0+cu128')}),versions)
            with self.assertRaises(ContractError):check_binding({'versions':dict(versions,torch='2.12.0')})

    def test_unknown_unit_is_not_enabled_by_operator_attestation(self):
        image=nib.Nifti1Image(np.ones((2,2,2),np.float32),np.eye(4))
        with self.assertRaises(ContractError):adopt_geometry(image,'acdc-direction-only-sform-v1',recognized_acdc=True)
        a,_=adopt_geometry(image,source_is_pinned=True)
        np.testing.assert_array_equal(a,np.eye(4))

    @unittest.skipUnless((SOURCE/'MANIFEST.json').is_file(),'set ACDC_RELEASE_ROOT to the authenticated fixture')
    def test_real_release_subset_identity_and_tamper(self):
        with self.assertRaises(ContractError):load_bundle(SOURCE)
        with self.assertRaises(ContractError):load_bundle(SOURCE,expected_sha='0'*64)
        with self.assertRaises(ContractError):load_bundle(SOURCE/'models/coronal/int8',expected_sha=AXIAL_SHA)
        with tempfile.TemporaryDirectory() as d:
            model=import_release(SOURCE,Path(d)/'model')
            self.assertEqual(model.model_sha256,AXIAL_SHA)
            self.assertFalse(model.synthetic)
            self.assertFalse(model.spec['qualification']['physical_board_accepted'])
            self.assertEqual(len(model.layers),15)
            self.assertEqual(len(model.implementation_binding['recognized_acdc_images']),200)
            self.assertFalse((model.root/'datasets').exists())
            again=load_bundle(model.root,expected_sha=AXIAL_SHA)
            np.testing.assert_array_equal(again.layers[-1].multipliers,model.layers[-1].multipliers)
            p=model.root/'models/axial/int8/C01/weight.bin'
            with p.open('r+b') as stream:stream.write(b'\x80')
            with self.assertRaises(ContractError):load_bundle(model.root,expected_sha=AXIAL_SHA)
