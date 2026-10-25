"""Compare optimized MPR to the frozen C9 general sampler on signed fixtures.

No PL inference or ARM performance claim. PNG decoding uses independent Pillow.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile

import nibabel as nib
import numpy as np
from PIL import Image

from acdc.contract import atomic_json,file_sha
from acdc.appliance.native import load_native
from acdc.appliance.mpr import MPRVolume


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    source=Path(__file__).resolve().parents[1]/'appliance/native/image.c'
    text=source.read_text('utf-8');start=text.index('static uint8_t clamped_byte')
    legacy=text[start:text.index('typedef struct { size_t low',start)].strip()
    legacy_sha=hashlib.sha256(legacy.encode()).hexdigest()
    if legacy_sha!='cff77c70f741b18bd977ca199fb7cdbcaea4caef5321f5a63caa0ad45b053011':
        raise AssertionError('C9 general reference sampler changed')
    fast=load_native();general=load_native()
    general.library.acdc_mpr_fast=general.library.acdc_mpr
    cases=json.loads((a.results/'cases.json').read_text())['cases'];rows=[]
    with tempfile.TemporaryDirectory(prefix='acdc-mpr-qualification-') as tmp:
        tmp=Path(tmp)
        for case in cases:
            phase=case['phase'];root=Path(case['result_dir'])
            manifest=json.loads((root/'manifest.json').read_text())
            def asset(name):
                f=root/name
                if file_sha(f)!=manifest['files'][name]:raise AssertionError('reference asset drift')
                return f
            geometry=json.loads(asset(f'geometry/{phase}.json').read_text())
            mri=Path(case['gt']);mri=mri.with_name(mri.name.replace('_gt.nii','.nii'))
            if file_sha(mri)!=geometry['input_sha256']:raise AssertionError('reference MRI drift')
            gray=np.asarray(nib.load(mri).get_fdata(dtype=np.float32),order='C')
            labels=np.asarray(nib.load(asset(f'labels/{phase}.nii.gz')).dataobj,np.uint8)
            np.save(tmp/'gray.npy',gray);np.save(tmp/'labels.npy',labels)
            atomic_json(tmp/'geometry.json',dict(shape=list(gray.shape),affine=geometry['native_affine']))
            new=MPRVolume(tmp,fast);old=MPRVolume(tmp,general);count=0
            for plane in ('axial','coronal','sagittal'):
                for i in range(2):
                    position=old.bounds[0]+(.35+.3*i)*(old.bounds[1]-old.bounds[0])
                    kw=dict(level=100.,window=250.,alpha=.5,outline=bool(i))
                    x,mx=old.render(plane,position,**kw);y,my=new.render(plane,position,**kw)
                    if mx!=my:raise AssertionError('MPR metadata differs')
                    with Image.open(io.BytesIO(x)) as ix,Image.open(io.BytesIO(y)) as iy:
                        np.testing.assert_array_equal(np.asarray(ix),np.asarray(iy))
                    count+=1
            rows.append(dict(patient=case['patient'],phase=phase,frames=count,pixel_mismatches=0,metadata_equal=True))
            print(case['patient'],phase,'MPR pixels exact',flush=True)
            # Release read-only Windows mappings before replacing the fixture.
            del new,old
    atomic_json(a.output,dict(schema='pld-acdc-mpr-qualification-v2',complete=True,volumes=len(rows),
        frames=sum(r['frames'] for r in rows),rows=rows,native_build=fast.info,reference_sampler_sha256=legacy_sha,
        fresh_pl_inferences=0,scope='development host MPR pixels vs frozen C9 general sampler, independent PNG decoding; no ARM/HTTP/GPU timing'))


if __name__=='__main__':main()
