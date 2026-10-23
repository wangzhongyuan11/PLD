"""The numerical scheduler must produce the same outputs with/without overlap."""
import json
import threading

import nibabel as nib
import numpy as np
import pytest

from acdc.backend import DemoBackend
from acdc.appliance.native import load_native
from acdc.appliance.pipeline import NumericalPipeline
from acdc.appliance.reconstruct import make_exp_table


def test_bounded_overlap_matches_serial_and_cancel(tmp_path):
    kernel=load_native();backend=DemoBackend();backend.runtime_sha=None
    rng=np.random.default_rng(123)
    paths=[]
    for phase in ('ED','ES'):
        path=tmp_path/(phase+'.nii.gz')
        image=nib.Nifti1Image(rng.integers(1,250,(19,17,7)).astype(np.float32),np.diag([2.,2.,2.,1.]))
        image.header.set_xyzt_units('mm');nib.save(image,path);paths.append(path)
    pipeline=NumericalPipeline(backend,kernel,make_exp_table(backend.head_scale),binding={})
    serial=pipeline.run(*paths,tmp_path/'serial',overlap=False)
    parallel=pipeline.run(*paths,tmp_path/'parallel',overlap=True)
    assert serial['status']==parallel['status']=='numerics_completed'
    assert serial['synthetic'] and parallel['synthetic']
    assert [(r['input_sha256'],r['head_sha256']) for r in serial['records']]==[
        (r['input_sha256'],r['head_sha256']) for r in parallel['records']]
    for phase in ('ED','ES'):
        a=nib.load(tmp_path/'serial'/'labels'/(phase+'.nii.gz'))
        b=nib.load(tmp_path/'parallel'/'labels'/(phase+'.nii.gz'))
        np.testing.assert_array_equal(a.dataobj,b.dataobj)
    assert json.loads((tmp_path/'serial/statistics.json').read_text())==json.loads(
        (tmp_path/'parallel/statistics.json').read_text())
    event=threading.Event();event.set()
    with pytest.raises(InterruptedError):pipeline.run(*paths,tmp_path/'cancelled',cancel=event)
    assert json.loads((tmp_path/'cancelled/status.json').read_text())['status']=='cancelled'
