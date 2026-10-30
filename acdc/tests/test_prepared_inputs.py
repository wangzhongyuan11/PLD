"""Prepared uploads keep tensors/geometry and bypass the board preprocessing."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import nibabel as nib
import numpy as np
from PIL import Image


def test_camus_prepared_pipeline_bypasses_native_preprocess(tmp_path):
    from camus.native import load_native
    from camus.prepared import prepare,load
    from camus.pipeline import CamusPipeline
    native=load_native();source=tmp_path/'gray.png'
    Image.fromarray(np.arange(1200,dtype=np.uint8).reshape(30,40)).save(source)
    expected=native.preprocess(source);package=tmp_path/'gray.pldc';prepare(source,package)
    prepared=load(package)
    np.testing.assert_array_equal(prepared.input_int8,expected.input_int8)
    np.testing.assert_array_equal(prepared.raw_gray,expected.raw_gray)
    pipeline=CamusPipeline.__new__(CamusPipeline)
    pipeline.model=SimpleNamespace(root=tmp_path/'model',head_shape=[1,4,256,256],head_scale=.125,input_shape=[1,1,256,256])
    pipeline.backend='reference';pipeline.native=native;pipeline.jobs=0;pipeline.initialization_seconds=0
    def forward(q):
        np.testing.assert_array_equal(q,expected.input_int8)
        return dict(head=np.zeros((1,4,256,256),np.int8))
    pipeline.network=SimpleNamespace(forward=forward)
    with patch.object(native,'preprocess',side_effect=AssertionError('preprocessing ran on prepared input')):
        result=pipeline.run(package,tmp_path/'result')
    assert 'prepared_input_load' in result['stages_seconds'] and 'input_read_preprocess' not in result['stages_seconds']


def test_acdc_prepared_bypasses_resampling_and_preserves_geometry(tmp_path):
    from acdc.appliance.native import load_native
    from acdc.appliance.prepared import prepare,load
    from acdc.appliance.volume import load_volume,AxialInputs
    from acdc.appliance.pipeline import NumericalPipeline
    from acdc.appliance.reconstruct import make_exp_table
    from acdc.backend import DemoBackend
    kernel=load_native();source=tmp_path/'phase.nii.gz'
    image=nib.Nifti1Image(np.arange(12*13*3,dtype=np.float32).reshape(12,13,3),np.diag([2.,2.,2.,1.]))
    image.header.set_xyzt_units('mm');nib.save(image,source)
    original=load_volume(source,kernel);axial=AxialInputs(original,kernel)
    package=tmp_path/'phase.plda';prepare(source,package,kernel,binding={},policy='strict')
    volume,inputs=load(package)
    np.testing.assert_array_equal(volume.native,original.native)
    np.testing.assert_array_equal(volume.affine,original.affine)
    for z in range(original.working_shape[2]):
        expected,neighbors=axial.get(z);actual,loaded_neighbors=inputs.get(z)
        np.testing.assert_array_equal(expected,actual);assert loaded_neighbors==neighbors
    backend=DemoBackend();backend.runtime_sha=None
    pipeline=NumericalPipeline(backend,kernel,make_exp_table(backend.head_scale),binding={})
    with patch('acdc.appliance.pipeline.load_volume',side_effect=AssertionError('volume preprocessing ran')), \
         patch('acdc.appliance.pipeline.AxialInputs',side_effect=AssertionError('slice resizing ran')):
        report=pipeline.run(package,package,tmp_path/'output')
    assert all(row['input_mode']=='pc_prepared_int8' and row['resized_planes']==0 for row in report['phases'].values())
