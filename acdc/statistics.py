"""A-defined class volumes, EF validity and patient/physical-surface metrics."""
from __future__ import annotations

import numpy as np

from .contract import ContractError,LABELS


def volumes(labels,affine):
    labels=np.asarray(labels)
    if labels.ndim!=3 or labels.dtype!=np.uint8 or np.any(labels>3):
        raise ContractError('native uint8 class labelmap required')
    a=np.asarray(affine,np.float64)
    if a.shape!=(4,4) or not np.all(np.isfinite(a)):
        raise ContractError('invalid statistics affine')
    vv=float(abs(np.linalg.det(a[:3,:3])))
    if vv<=0:
        raise ContractError('singular statistics affine')
    return dict(voxel_volume_mm3=vv,classes={LABELS[c]:dict(
        voxels=int(np.count_nonzero(labels==c)),mL=float(np.count_nonzero(labels==c)*vv/1000)) for c in (1,2,3)})


def ef(edv,esv):
    if not np.isfinite(edv) or not np.isfinite(esv) or edv<=0 or not 0<=esv<=edv:
        return dict(EDV_mL=edv,ESV_mL=esv,SV_mL=edv-esv,EF_percent=None,valid=False,
                    reason='requires EDV>0 and 0<=ESV<=EDV')
    return dict(EDV_mL=edv,ESV_mL=esv,SV_mL=edv-esv,EF_percent=100*(edv-esv)/edv,valid=True,reason=None)


def paired(ed,es):
    return dict(phases=dict(ED=ed,ES=es),cavities={name:ef(ed['classes'][name]['mL'],
                 es['classes'][name]['mL']) for name in ('LV','RV')})


def class_metrics(pred,gt,affine,c):
    from scipy.ndimage import binary_erosion,generate_binary_structure
    from scipy.spatial import cKDTree
    p,g=np.asarray(pred)==c,np.asarray(gt)==c
    if p.shape!=g.shape:
        raise ContractError('metric grids differ')
    pn,gn=int(p.sum()),int(g.sum())
    dice=2*int((p&g).sum())/(pn+gn) if pn+gn else 1.
    result=dict(dice=dice,HD_mm=None,HD95_mm=None,ASSD_mm=None,distance_reason=None)
    if not pn or not gn:
        if pn==gn:result.update(HD_mm=0.,HD95_mm=0.,ASSD_mm=0.)
        else:result['distance_reason']='one class is empty'
        return result
    a=np.asarray(affine)
    struct=generate_binary_structure(3,1)
    pts=[]
    for mask in (p,g):
        ijk=np.argwhere(mask & ~binary_erosion(mask,structure=struct,border_value=0))
        pts.append(ijk@a[:3,:3].T+a[:3,3])
    distances=np.concatenate([cKDTree(pts[1]).query(pts[0])[0],cKDTree(pts[0]).query(pts[1])[0]])
    result.update(HD_mm=float(distances.max()),HD95_mm=float(np.percentile(distances,95,method='linear')),
                  ASSD_mm=float(distances.mean()))
    return result
