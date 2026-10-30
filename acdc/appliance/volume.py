"""Board preprocessing with portable native math and separate provenance."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import importlib.metadata

import numpy as np

from acdc.contract import ContractError, PREPROCESSING, digest, file_sha
from acdc.volume import adopt_geometry, grid, letterbox_spec
from . import IMPLEMENTATION_ID

MAX_NATIVE_VOXELS=2_000_000
MAX_WORKING_VOXELS=4_000_000


@dataclass
class Volume:
    native: np.ndarray
    affine: np.ndarray
    working: np.ndarray
    geometry: dict

    @property
    def geometry_sha(self):return digest(self.geometry)

    @property
    def working_shape(self):
        return self.working.shape if self.working is not None else tuple(self.geometry['working_shape'])


def load_volume(path,kernel,*,policy='strict',binding=None):
    import nibabel as nib
    path=Path(path)
    image=nib.load(str(path))
    if len(image.shape)!=3 or min(image.shape)<1 or int(np.prod(image.shape))>MAX_NATIVE_VOXELS:
        raise ContractError('V2 requires a bounded nonempty 3D NIfTI')
    source_sha=file_sha(path)
    pinned=source_sha in (binding or {}).get('recognized_acdc_images',{})
    affine,decision=adopt_geometry(image,policy,recognized_acdc=pinned,source_is_pinned=pinned)
    shape,working_affine=grid(image.shape,affine)
    if int(np.prod(shape))>MAX_WORKING_VOXELS:raise ContractError('V2 working memory limit exceeded')
    native=np.asarray(image.get_fdata(dtype=np.float32,caching='unchanged'),dtype=np.float32,order='C')
    normalized,normalization=kernel.normalize(native)
    # Same mapping as nibabel.processing.resample_from_to: inv(source) @ target.
    transform=np.linalg.inv(affine)@working_affine
    working=kernel.affine(normalized,shape,transform[:3,:3],transform[:3,3],-1)
    geometry=dict(schema='pld-acdc-geometry-v2',preprocessing_id=PREPROCESSING,
        implementation_id=IMPLEMENTATION_ID,input_sha256=source_sha,
        native_shape=list(native.shape),native_affine=affine.tolist(),
        source_affine=np.asarray(image.affine).tolist(),source_pixdim=list(map(float,image.header.get_zooms()[:3])),
        source_qform_code=int(image.header['qform_code']),source_sform_code=int(image.header['sform_code']),
        working_shape=list(shape),working_affine=working_affine.tolist(),working_spacing_mm=[2,2,2],
        normalization=normalization,geometry_policy=policy,geometry_decision=decision,units='mm',
        source_units=image.header.get_xyzt_units()[0],
        unit_interpretation='mm from pinned ACDC release acquisition metadata' if pinned else 'explicit source mm',
        source_release_member=(binding or {}).get('recognized_acdc_images',{}).get(source_sha),
        versions={name:importlib.metadata.version(name) for name in ('numpy','nibabel')},
        native_library_sha256=kernel.info['library_sha256'],axial_plane='V[:,:,z].T',
        phase_independently_normalized=True)
    return Volume(native,affine,working,geometry)


class AxialInputs:
    def __init__(self,volume,kernel):
        self.volume=volume;self.kernel=kernel;self.cache=OrderedDict();self.resized_planes=0
        self.spec=letterbox_spec((volume.working.shape[1],volume.working.shape[0]))

    def _plane(self,z):
        if z not in self.cache:
            s=self.spec
            plane=self.kernel.resize(self.volume.working[:,:,z].T,s['resized_hw'])
            boxed=np.full((256,256),-1,np.float32)
            y,x=s['top'],s['left'];h,w=s['resized_hw']
            boxed[y:y+h,x:x+w]=plane
            self.cache[z]=self.kernel.quantize(boxed);self.resized_planes+=1
        self.cache.move_to_end(z)
        return self.cache[z]

    def get(self,z):
        nz=self.volume.working.shape[2]
        if type(z) is not int or not 0<=z<nz:raise ContractError('invalid center slice')
        neighbors=[max(z-1,0),z,min(z+1,nz-1)]
        result=np.stack([self._plane(i) for i in neighbors])[None]
        while len(self.cache)>3:self.cache.popitem(last=False)
        return result,neighbors
