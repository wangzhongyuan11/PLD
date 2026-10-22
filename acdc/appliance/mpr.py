"""Server-side RAS MPR. Clients receive pixels, not medical volume computation."""
from __future__ import annotations

from itertools import product
import io
import json
from pathlib import Path

import numpy as np

from acdc.contract import ContractError


class MPRVolume:
    def __init__(self,directory,kernel):
        directory=Path(directory);self.kernel=kernel
        geometry=json.loads((directory/'geometry.json').read_text('utf-8'))
        self.gray=np.load(directory/'gray.npy',allow_pickle=False,mmap_mode='r')
        self.labels=np.load(directory/'labels.npy',allow_pickle=False,mmap_mode='r')
        if (self.gray.dtype!=np.float32 or self.labels.dtype!=np.uint8 or self.gray.shape!=self.labels.shape
                or self.gray.ndim!=3 or list(self.gray.shape)!=geometry['shape']):
            raise ContractError('MPR result representation mismatch')
        self.affine=np.asarray(geometry['affine'],np.float64)
        if self.affine.shape!=(4,4) or not np.all(np.isfinite(self.affine)):
            raise ContractError('invalid MPR affine')
        self.inverse=np.linalg.inv(self.affine)
        corners=np.array(list(product(*[(0,n-1) for n in self.gray.shape])),np.float64)
        world=corners@self.affine[:3,:3].T+self.affine[:3,3]
        self.bounds=np.array([world.min(axis=0),world.max(axis=0)])

    def render(self,plane,position,*,shape=(400,400),level=100.,window=200.,alpha=.5,visible=14,outline=False):
        axes={'axial':(0,1,2),'coronal':(0,2,1),'sagittal':(1,2,0)}
        if plane not in axes:raise ContractError('unknown MPR plane')
        position=np.asarray(position,np.float64)
        if position.shape!=(3,) or not np.all(np.isfinite(position)):
            raise ContractError('invalid RAS cursor')
        if np.any(position<self.bounds[0]) or np.any(position>self.bounds[1]):
            raise ContractError('RAS cursor outside result bounds')
        h,w=map(int,shape);u,v,n=axes[plane]
        if min(h,w)<1 or max(h,w)>768:raise ContractError('invalid MPR raster size')
        extent=np.maximum(self.bounds[1]-self.bounds[0],1.)
        scale=min(w/extent[u],h/extent[v])*.94
        ox=(w-extent[u]*scale)/2;oy=(h-extent[v]*scale)/2
        world=position.copy();world[u]=self.bounds[0,u]+(.5-ox)/scale
        world[v]=self.bounds[1,v]-(.5-oy)/scale
        origin=self.inverse[:3,:3]@world+self.inverse[:3,3]
        dx=self.inverse[:3,u]/scale;dy=-self.inverse[:3,v]/scale
        pixels=self.kernel.mpr(self.gray,self.labels,origin,dx,dy,(h,w),level=level,window=window,
                               alpha=alpha,visible=visible,outline=outline)
        from PIL import Image
        stream=io.BytesIO();Image.fromarray(pixels).save(stream,format='PNG',compress_level=1)
        return stream.getvalue(),dict(bounds=self.bounds.tolist(),position=position.tolist(),plane=plane,
            normal_axis=n,scale=scale,ox=ox,oy=oy,u=u,v=v,width=w,height=h,generator='pynq_ps',
            level=float(level),window=float(window),alpha=float(alpha),visible=int(visible),outline=bool(outline))
