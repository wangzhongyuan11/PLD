"""Softmax before interpolation; Axial working probabilities back to adopted native space."""
from __future__ import annotations

import numpy as np

from .contract import HEAD_SHAPE, ContractError, checked_int8, positive
from .volume import resize_torch


def plane_probabilities(head,scale,box):
    q=checked_int8(head,HEAD_SHAPE)
    logits=q[0].astype(np.float32)*np.float32(positive(scale))
    logits-=logits.max(axis=0,keepdims=True)
    p=np.exp(logits)
    p/=p.sum(axis=0,keepdims=True)
    y,x=box['top'],box['left'];h,w=box['resized_hw']
    return resize_torch(p[None,:,y:y+h,x:x+w],box['original_hw'])[0]


def normalize_probabilities(p):
    p=np.asarray(p,np.float32).copy()
    if p.ndim!=4 or p.shape[0]!=4 or not np.all(np.isfinite(p)) or np.any(p<0):
        raise ContractError('invalid class probabilities')
    total=p.sum(axis=0)
    p[0,total<=1e-8]=1
    p/=np.maximum(p.sum(axis=0),np.float32(1e-8))[None]
    return p


class Collector:
    def __init__(self,volume,model_sha,head_scale,phase):
        self.volume=volume;self.model_sha=model_sha;self.head_scale=head_scale;self.phase=phase
        self.seen=set()
        self.probabilities=np.empty((4,*volume.working.shape),np.float32)

    def add(self,head,metadata,box):
        nz=self.volume.working.shape[2]
        z=metadata.get('center_index')
        if (type(z) is not int or not 0<=z<nz or z in self.seen
                or metadata.get('orientation')!='axial' or metadata.get('phase')!=self.phase
                or metadata.get('model_sha256')!=self.model_sha or metadata.get('head_scale')!=self.head_scale
                or metadata.get('geometry_sha256')!=self.volume.geometry_sha):
            raise ContractError('duplicate/missing/misidentified Axial response')
        p=plane_probabilities(head,self.head_scale,box)
        self.probabilities[:,:,:,z]=p.transpose(0,2,1)
        self.seen.add(z)

    def finish(self):
        import nibabel as nib
        from nibabel.processing import resample_from_to
        if self.seen!=set(range(self.volume.working.shape[2])):
            raise ContractError('incomplete Axial volume; refusing background substitution')
        working_a=np.asarray(self.volume.geometry['working_affine'])
        target=(self.volume.native.shape,self.volume.affine)
        p=np.stack([np.asarray(resample_from_to(nib.Nifti1Image(c,working_a),target,
                      order=1,mode='constant',cval=0).dataobj,np.float32) for c in self.probabilities])
        p=normalize_probabilities(p)
        return p,np.argmax(p,axis=0).astype(np.uint8)
