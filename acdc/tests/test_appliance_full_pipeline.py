"""Check full-result readiness/export gating; mesh quality has a separate oracle."""
import json
import threading
import zipfile

import nibabel as nib
import numpy as np

from acdc.backend import DemoBackend
from acdc.appliance.native import load_native
from acdc.appliance.pipeline import NumericalPipeline
from acdc.appliance.full_pipeline import FullPipeline
from acdc.appliance.reconstruct import make_exp_table


class MeshFixture:
    """A test double only; it makes no geometry or board-readiness claim."""
    info={'test_double':True}
    def generate(self,labels,affine,directory,*,cancel):
        directory.mkdir(parents=True)
        (directory/'fixture.json').write_text('{"test_double":true}')
        return []


def test_assets_ready_requires_both_phases_mpr_and_zip(tmp_path):
    kernel=load_native();backend=DemoBackend();backend.runtime_sha=None
    rng=np.random.default_rng(600)
    paths=[]
    for phase in ('ED','ES'):
        p=tmp_path/(phase+'.nii.gz')
        image=nib.Nifti1Image(rng.integers(1,250,(9,11,5)).astype(np.float32),np.diag([2.,2.,2.,1.]))
        image.header.set_xyzt_units('mm');nib.save(image,p);paths.append(p)
    numerical=NumericalPipeline(backend,kernel,make_exp_table(backend.head_scale),binding={})
    out=tmp_path/'job';events=[]
    def progress(row):
        events.append(row['status'])
        if row['status']=='assets_ready':
            assert (out/'result.zip').is_file()
            for phase in ('ED','ES'):
                assert (out/'viewer'/phase/'volume.json').is_file()
                assert all((out/'mpr'/phase/(plane+'.png')).is_file() for plane in ('axial','coronal','sagittal'))
    result=FullPipeline(numerical,MeshFixture()).run(*paths,out,progress=progress,upload_seconds=.25)
    assert events[-1]=='assets_ready' and result['server_upload_to_assets_seconds']>=.25
    assert result['browser_interactive_seconds'] is None
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['synthetic'] is True and manifest['backend']=='demo'
    with zipfile.ZipFile(out/'result.zip') as archive:
        assert archive.testzip() is None
        names=set(archive.namelist())
        assert {'statistics.csv','cardiac_function.csv','labels/ED.nii.gz','labels/ES.nii.gz'}<=names
        assert not any(name.endswith('.npy') for name in names)
    assert json.loads((out/'status.json').read_text())['status']=='assets_ready'
