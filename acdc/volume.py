"""A's frozen per-phase normalization, adopted geometry and Axial input path."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from collections import OrderedDict
from pathlib import Path
import importlib.metadata

import numpy as np

from .contract import ContractError, PREPROCESSING, digest, file_sha

MAX_NATIVE_VOXELS = 16_777_216
MAX_WORKING_VOXELS = 33_554_432


def environment():
    return {name: importlib.metadata.version(name) for name in ('numpy','scipy','nibabel','torch')}


def check_binding(binding=None):
    actual = environment()
    required = {'nibabel': '5.4.2', 'torch': '2.11.0'}
    if binding is not None:
        required.update(binding.get('versions', {}))
    for name, wanted in required.items():
        # +cpu is a wheel build suffix, not a different torch numerical release.
        if actual.get(name, '').split('+')[0] != wanted:
            raise ContractError(f'preprocessing version mismatch: {name} {actual.get(name)} != {wanted}')
    return actual


def normalize(data):
    data = np.asarray(data, np.float32)
    if data.ndim != 3 or not np.all(np.isfinite(data)):
        raise ContractError('finite 3D MRI required')
    values = data[data != 0]
    if values.size < 8:
        raise ContractError('fewer than eight nonzero native voxels')
    lo, hi = np.percentile(values.astype(np.float64), [1,99], method='linear')
    if hi <= lo:
        raise ContractError('degenerate normalization range')
    u = np.clip(2*(data.astype(np.float64)-lo)/(hi-lo)-1,-1,1).astype(np.float32)
    return u, dict(low=float(lo),high=float(hi),method='nonzero-p1-p99-linear-native-phase')


def adopt_geometry(image, policy='strict', *, recognized_acdc=False):
    a = np.asarray(image.affine, np.float64).copy()
    pixdim = np.asarray(image.header.get_zooms()[:3], np.float64)
    units = image.header.get_xyzt_units()[0]
    if units != 'mm':
        raise ContractError('explicit mm spatial units required; unknown units are not guessed')
    if (not np.all(np.isfinite(a)) or not np.all(np.isfinite(pixdim)) or np.any(pixdim <= 0)
            or not np.allclose(a[3], [0,0,0,1]) or abs(np.linalg.det(a[:3,:3])) < 1e-10):
        raise ContractError('invalid physical affine/pixdim')
    if policy not in ('strict','acdc-direction-only-sform-v1'):
        raise ContractError('unknown geometry policy')
    norms=np.linalg.norm(a[:3,:3],axis=0)
    decision='unchanged'
    if not np.allclose(norms,pixdim,rtol=1e-4,atol=1e-5):
        valid=(policy=='acdc-direction-only-sform-v1' and recognized_acdc
               and int(image.header['qform_code'])==0 and int(image.header['sform_code'])==2
               and np.allclose(norms,1,rtol=0,atol=1e-6)
               and np.allclose(a[:3,:3].T@a[:3,:3],np.eye(3),rtol=0,atol=1e-5))
        if not valid:
            raise ContractError('affine/pixdim mismatch; constrained repair not authorized/applicable')
        a[:3,:3] *= pixdim[None,:]
        decision='direction-columns-scaled-by-source-pixdim'
    return a, decision


def grid(shape, affine):
    import nibabel as nib
    corners=np.array(list(product(*[(0,n-1) for n in shape])),np.float64)
    points=nib.affines.apply_affine(affine,corners)
    origin=points.min(axis=0)
    size=np.ceil((points.max(axis=0)-origin)/2).astype(np.int64)+1
    if np.any(size < 1) or int(np.prod(size,dtype=np.int64)) > MAX_WORKING_VOXELS:
        raise ContractError('working grid exceeds memory limit')
    a=np.diag([2.,2.,2.,1.]);a[:3,3]=origin
    return tuple(int(v) for v in size),a


@dataclass
class Volume:
    path: Path
    native: np.ndarray
    affine: np.ndarray
    working: np.ndarray
    geometry: dict

    @property
    def geometry_sha(self):
        return digest(self.geometry)


def load_volume(path, *, policy='strict', recognized_acdc=False, binding=None):
    import nibabel as nib
    from nibabel.processing import resample_to_output
    versions=check_binding(binding)
    path=Path(path)
    image=nib.load(str(path))
    if len(image.shape)!=3 or min(image.shape)<1 or int(np.prod(image.shape,dtype=np.int64))>MAX_NATIVE_VOXELS:
        raise ContractError('only bounded, nonempty 3D NIfTI is accepted; 4D requires a new contract')
    a,decision=adopt_geometry(image,policy,recognized_acdc=recognized_acdc)
    native=image.get_fdata(dtype=np.float32,caching='unchanged')
    u,normalization=normalize(native)
    expected_shape,expected_affine=grid(native.shape,a)
    source=nib.Nifti1Image(u,a)
    result=resample_to_output(source,voxel_sizes=(2,2,2),order=1,mode='constant',cval=-1)
    if result.shape!=expected_shape or not np.allclose(result.affine,expected_affine,rtol=0,atol=1e-8):
        raise ContractError('resampler disagrees with frozen center-corner grid')
    geom=dict(schema='pld-acdc-geometry-v1',preprocessing_id=PREPROCESSING,
              input_sha256=file_sha(path),native_shape=list(native.shape),native_affine=a.tolist(),
              source_affine=np.asarray(image.affine).tolist(),source_pixdim=list(map(float,image.header.get_zooms()[:3])),
              source_qform_code=int(image.header['qform_code']),source_sform_code=int(image.header['sform_code']),
              working_shape=list(expected_shape),working_affine=expected_affine.tolist(),working_spacing_mm=[2,2,2],
              normalization=normalization,geometry_policy=policy,geometry_decision=decision,units='mm',
              versions=versions,axial_plane='V[:,:,z].T',phase_independently_normalized=True)
    return Volume(path,native,a,np.asarray(result.dataobj,dtype=np.float32),geom)


def letterbox_spec(hw, size=256):
    h,w=map(int,hw)
    if min(h,w)<1:
        raise ContractError('empty plane')
    ratio=min(size/h,size/w)
    hh=max(1,min(size,round(h*ratio)));ww=max(1,min(size,round(w*ratio)))
    return dict(original_hw=[h,w],resized_hw=[hh,ww],top=(size-hh)//2,left=(size-ww)//2,size=size)


def resize_torch(value, hw):
    import torch
    import torch.nn.functional as f
    a=np.asarray(value,np.float32)
    with torch.inference_mode():
        t=torch.from_numpy(np.ascontiguousarray(a))
        return f.interpolate(t,size=tuple(hw),mode='bilinear',align_corners=False,antialias=False).numpy()


class AxialInputs:
    """Three-plane LRU: resize/quantize a working plane once for adjacent centers."""
    def __init__(self, volume):
        self.volume=volume
        self.spec=letterbox_spec((volume.working.shape[1],volume.working.shape[0]))
        self.cache=OrderedDict()
        self.resized_planes=0

    def _plane(self,z):
        if z not in self.cache:
            spec=self.spec
            a=resize_torch(self.volume.working[:,:,z].T[None,None],spec['resized_hw'])[0,0]
            boxed=np.full((256,256),-1,np.float32)
            y,x=spec['top'],spec['left'];h,w=spec['resized_hw']
            boxed[y:y+h,x:x+w]=a
            self.cache[z]=np.clip(np.rint(boxed.astype(np.float64)*127),-127,127).astype(np.int8)
            self.resized_planes+=1
        self.cache.move_to_end(z)
        return self.cache[z]

    def get(self,z):
        nz=self.volume.working.shape[2]
        if type(z) is not int or not 0<=z<nz:
            raise ContractError('center index outside Axial grid')
        ids=[max(z-1,0),z,min(z+1,nz-1)]
        data=np.stack([self._plane(i) for i in ids])[None]
        while len(self.cache)>3:
            self.cache.popitem(last=False)
        return data,ids


def make_phantom(directory):
    """Explicit synthetic MRI pair for installation/UI checks, never an ACDC patient."""
    import nibabel as nib
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    x,y,z=np.meshgrid(np.linspace(-1,1,64),np.linspace(-1,1,64),np.linspace(-1,1,12),indexing='ij')
    for phase,scale in [('ED',1.),('ES',.72)]:
        r=np.sqrt(((x+.18)/(.38*scale))**2+(y/(.4*scale))**2+(z/.92)**2)
        labels=np.zeros(x.shape,np.uint8)
        labels[(r<1.28)&(r>=1)]=2;labels[r<1]=3
        rv=((x-.28)/(.31*scale))**2+(y/(.35*scale))**2+(z/.82)**2<1
        labels[rv&(labels==0)]=1
        data=np.choose(labels,[10.,45.,85.,130.]).astype(np.float32)
        data+=((x+y+z)*.7).astype(np.float32)
        a=np.diag([2.,2.,5.,1.]);a[:3,3]=[-63.,-63.,-27.5]
        for name,value in [(phase,data),(phase+'_gt',labels)]:
            image=nib.Nifti1Image(value,a);image.header.set_xyzt_units('mm')
            nib.save(image,str(directory/(name+'.nii.gz')))
    return directory
