"""Offline four-panel viewer: three linked physical-grid slices and RAS mesh."""
from __future__ import annotations
import base64,json
from pathlib import Path
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
from skimage.measure import marching_cubes
from .geometry import prepare_volume

def create_viewer(image,prediction,metadata,output):
    x,_,m=prepare_volume(image,geometry_policy=metadata['geometry_policy'])
    pred=nib.load(str(prediction))
    labels=np.asarray(resample_from_to(pred,(tuple(m['working_shape']),np.array(m['working_affine'])),order=0,cval=0).dataobj,dtype=np.uint8)
    meshes=[]
    native=np.asarray(pred.dataobj,dtype=np.uint8)
    for c in (1,2,3):
        if not (native==c).any():continue
        v,f,_,_=marching_cubes(np.pad((native==c).astype(np.uint8),1),.5,allow_degenerate=False)
        v=nib.affines.apply_affine(pred.affine,v-1)
        meshes.append({'c':c,'v':np.round(v,3).tolist(),'f':f.tolist()})
    nonzero=np.argwhere(labels>0);center=np.round(nonzero.mean(0)).astype(int).tolist() if len(nonzero) else [s//2 for s in x.shape]
    data={'shape':list(x.shape),'center':center,'image':base64.b64encode(np.clip(np.rint((x+1)*127.5),0,255).astype(np.uint8).tobytes()).decode(),
          'labels':base64.b64encode(labels.tobytes()).decode(),'meshes':meshes,
          'source':Path(image).name,'spacing':m['working_spacing_mm'],'native_spacing':metadata['native_spacing_mm']}
    template=Path(__file__).with_name('viewer.html').read_text(encoding='utf-8')
    text=template.replace('/*DATA_JSON*/',json.dumps(data,separators=(',',':')).replace('</','<\\/'))
    Path(output).write_text(text,encoding='utf-8')
