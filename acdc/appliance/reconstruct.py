"""Bounded native probability reconstruction; never interpolate argmax labels."""
from __future__ import annotations

import numpy as np
import time

from acdc.contract import ContractError


def make_exp_table(scale):
    """Run only in the qualified PC build environment; ship with a SHA manifest."""
    if not np.isfinite(scale) or scale<=0:raise ContractError('invalid head scale')
    values=np.arange(-127,128,dtype=np.float32)*np.float32(scale)
    # Rows are the maximum, columns the input. Invalid q>max entries aren't used.
    delta=values[None,:]-values[:,None]
    delta=np.minimum(delta,np.float32(0))
    return np.exp(delta)


class Collector:
    def __init__(self,volume,kernel,table,*,block_x=8):
        self.volume=volume;self.kernel=kernel;self.table=table;self.seen=set()
        if block_x<1:raise ContractError('positive restoration block size required')
        self.block_x=block_x
        self.timings=dict(softmax_seconds=0.,plane_resize_seconds=0.,native_restore_seconds=0.)
        self.probabilities=np.empty((4,*volume.working.shape),np.float32)

    def add(self,head,z,box):
        nz=self.volume.working.shape[2]
        if type(z) is not int or not 0<=z<nz or z in self.seen:
            raise ContractError('duplicate or invalid probability plane')
        tick=time.perf_counter();p=self.kernel.softmax(head,self.table)
        self.timings['softmax_seconds']+=time.perf_counter()-tick
        y,x=box['top'],box['left'];h,w=box['resized_hw']
        tick=time.perf_counter();restored=self.kernel.resize(p[:,y:y+h,x:x+w],box['original_hw'])
        self.timings['plane_resize_seconds']+=time.perf_counter()-tick
        self.probabilities[:,:,:,z]=restored.transpose(0,2,1)
        self.seen.add(z)

    def finish(self,*,save_probabilities=False):
        tick=time.perf_counter()
        if self.seen!=set(range(self.volume.working.shape[2])):
            raise ContractError('incomplete working probability volume')
        shape=self.volume.native.shape
        labels=np.empty(shape,np.uint8)
        full=np.empty((4,*shape),np.float32) if save_probabilities else None
        transform=np.linalg.inv(np.asarray(self.volume.geometry['working_affine']))@self.volume.affine
        samplers=[self.kernel.prepare_affine(c) for c in self.probabilities]
        for start in range(0,shape[0],self.block_x):
            end=min(start+self.block_x,shape[0]);block_shape=(end-start,*shape[1:])
            offset=transform[:3,3]+transform[:3,0]*start
            p=np.stack([sampler.apply(block_shape,transform[:3,:3],offset,0) for sampler in samplers])
            total=p.sum(axis=0)
            p[0,total<=1e-8]=1
            p/=np.maximum(p.sum(axis=0),np.float32(1e-8))[None]
            labels[start:end]=np.argmax(p,axis=0).astype(np.uint8)
            if full is not None:full[:,start:end]=p
        self.timings['native_restore_seconds']+=time.perf_counter()-tick
        return full,labels
