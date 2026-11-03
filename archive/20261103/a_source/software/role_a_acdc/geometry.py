"""RAS physical resampling, full-FOV letterbox and invertible prediction geometry."""
from __future__ import annotations
import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output,resample_from_to
import torch
import torch.nn.functional as F

AXES={'axial':2,'coronal':1,'sagittal':0}
SPACING=(2.0,2.0,2.0)
ACDC_GEOMETRY_POLICY='acdc-direction-only-sform-v1'

def physical_affine(source,policy='strict'):
    affine=np.array(source.affine,dtype=float,copy=True)
    actual=nib.affines.voxel_sizes(affine);zooms=np.array(source.header.get_zooms()[:3],float)
    if np.allclose(actual,zooms,rtol=1e-4,atol=1e-5):return affine,'source affine and pixdim agree'
    direction=affine[:3,:3]/actual
    if (policy==ACDC_GEOMETRY_POLICY and int(source.header['qform_code'])==0 and int(source.header['sform_code'])==2
            and np.allclose(actual,1,atol=1e-6) and np.allclose(direction.T@direction,np.eye(3),atol=1e-5)
            and np.isfinite(zooms).all() and np.all(zooms>0)):
        affine[:3,:3]=direction*zooms
        return affine,'explicit ACDC provider repair: direction-only sform columns multiplied by header pixdim; source header preserved in sidecar'
    raise ValueError('NIfTI affine/pixdim mismatch; review source geometry or explicitly choose the ACDC provider policy')

def normalize(image):
    image=np.asarray(image,dtype=np.float32)
    if not np.isfinite(image).all():raise ValueError('non-finite MRI')
    sample=image[image!=0]
    if sample.size<8: raise ValueError('empty MRI')
    lo,hi=np.percentile(sample,[1.0,99.0]).astype(float)
    if hi<=lo:raise ValueError('constant MRI')
    value=np.clip(2*(image-lo)/(hi-lo)-1,-1,1).astype(np.float32)
    return value,{'method':'nonzero-volume-percentile-1-99-to-minus1-plus1-v1','low':lo,'high':hi,
                  'background_fill':-1.0,'labels_used':False}

def prepare_volume(image,label=None,spacing=SPACING,geometry_policy='strict'):
    source=nib.load(str(image)) if not isinstance(image,nib.spatialimages.SpatialImage) else image
    if len(source.shape)!=3 or abs(np.linalg.det(source.affine[:3,:3]))<1e-8:raise ValueError('invalid 3D affine')
    affine,geometry_decision=physical_affine(source,geometry_policy)
    values,norm=normalize(source.get_fdata(dtype=np.float32))
    normalized=nib.Nifti1Image(values,affine)
    ras=resample_to_output(normalized,voxel_sizes=spacing,order=1,mode='constant',cval=-1)
    x=np.asarray(ras.dataobj,dtype=np.float32)
    y=None
    if label is not None:
        gt=nib.load(str(label)) if not isinstance(label,nib.spatialimages.SpatialImage) else label
        if gt.shape!=source.shape or not np.allclose(gt.affine,source.affine,atol=1e-4):raise ValueError('image/label geometry mismatch')
        gt_affine,_=physical_affine(gt,geometry_policy)
        if not np.allclose(gt_affine,affine,atol=1e-4):raise ValueError('image/label physical spacing mismatch')
        data=np.asarray(gt.dataobj)
        if not np.isin(data,[0,1,2,3]).all():raise ValueError('ACDC labels must be 0/1/2/3')
        physical_gt=nib.Nifti1Image(data.astype(np.uint8),gt_affine)
        y=np.asarray(resample_from_to(physical_gt,(ras.shape,ras.affine),order=0,cval=0).dataobj,dtype=np.uint8)
    metadata={'native_shape':list(source.shape),'native_affine':affine.tolist(),
        'source_affine':source.affine.tolist(),'source_pixdim':list(map(float,source.header.get_zooms()[:3])),
        'geometry_policy':geometry_policy,'geometry_decision':geometry_decision,
        'native_spacing_mm':nib.affines.voxel_sizes(affine).tolist(),
        'native_axis_codes':list(nib.aff2axcodes(affine)),
        'native_voxel_volume_mm3':float(abs(np.linalg.det(affine[:3,:3]))),
        'native_spatial_unit':source.header.get_xyzt_units()[0],
        'spatial_unit_interpretation':'millimetres per ACDC acquisition metadata; source unit tag may be unknown',
        'orientation_limit':'axes follow declared NIfTI orientation; missing original acquisition pose cannot be reconstructed',
        'working_shape':list(ras.shape),'working_affine':ras.affine.tolist(),'working_axes':'RAS',
        'working_spacing_mm':list(spacing),'normalization':norm,
        'interpolation':'image linear, labels nearest; resampling does not add acquired resolution'}
    return x,y,metadata

def plane(volume,axis,index):
    return np.take(volume,index,axis=axis).T

def letterbox(array,size=256,is_label=False):
    a=np.asarray(array)
    original=a.shape[-2:];h,w=original
    scale=min(size/h,size/w); nh=max(1,min(size,int(round(h*scale))));nw=max(1,min(size,int(round(w*scale))))
    top=(size-nh)//2;left=(size-nw)//2
    t=torch.from_numpy(np.ascontiguousarray(a,dtype=np.float32)).reshape(-1,1,h,w)
    resized=F.interpolate(t,size=(nh,nw),mode='nearest' if is_label else 'bilinear',
                          **({} if is_label else {'align_corners':False}))
    out=F.pad(resized,(left,size-nw-left,top,size-nh-top),value=0 if is_label else -1)
    return out.numpy().reshape(*a.shape[:-2],size,size),{'original_hw':[h,w],'resized_hw':[nh,nw],'top':top,'left':left,'size':size}

def stack(volume,direction,index,baseline=False,size=256):
    axis=AXES[direction];last=volume.shape[axis]-1
    indices=[index]*3 if baseline else [max(0,index-1),index,min(last,index+1)]
    x=np.stack([plane(volume,axis,i) for i in indices])
    out,transform=letterbox(x,size)
    return out.astype(np.float32),transform

def undo_letterbox(prob,transform):
    top,left=transform['top'],transform['left'];h,w=transform['resized_hw']
    crop=np.ascontiguousarray(prob[:,top:top+h,left:left+w],dtype=np.float32)
    t=torch.from_numpy(crop)[None]
    return F.interpolate(t,size=transform['original_hw'],mode='bilinear',align_corners=False)[0].numpy()

def put_plane(volume,axis,index,value):
    selector=[slice(None)]*volume.ndim;selector[axis]=index
    volume[tuple(selector)]=value.T

def probabilities_to_native(probabilities,metadata):
    shape=tuple(metadata['native_shape']);affine=np.array(metadata['native_affine'])
    out=[]
    for p in probabilities:
        image=nib.Nifti1Image(p.astype(np.float32),np.array(metadata['working_affine']))
        out.append(np.asarray(resample_from_to(image,(shape,affine),order=1,cval=0).dataobj,dtype=np.float32))
    p=np.stack(out);den=p.sum(0);p[0,den<=1e-8]=1
    p/=np.maximum(p.sum(0,keepdims=True),1e-8)
    return p
