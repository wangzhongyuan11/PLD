"""Strict aggregate metrics; never resize labels or silently recode classes."""
import numpy as np


def confusion(prediction,truth,classes,valid=None):
    p=np.asarray(prediction);t=np.asarray(truth)
    if p.shape!=t.shape or p.ndim!=2:raise ValueError('labels must be same-size 2-D arrays')
    if p.dtype.kind not in 'uib' or t.dtype.kind not in 'uib':raise ValueError('integer labels required')
    if type(classes) is not int or classes<2:raise ValueError('classes >=2 required')
    keep=np.ones(t.shape,bool) if valid is None else np.asarray(valid)
    if keep.dtype!=np.bool_ or keep.shape!=t.shape or not keep.any():raise ValueError('nonempty boolean valid mask required')
    p=p[keep].astype(np.int64);t=t[keep].astype(np.int64)
    if np.any(p<0) or np.any(t<0) or np.any(p>=classes) or np.any(t>=classes):raise ValueError('labels outside declared class IDs; do not pass display RGB/0,85,170,255 masks')
    return np.bincount(t*classes+p,minlength=classes*classes).reshape(classes,classes)


def scores(matrix):
    c=np.asarray(matrix)
    if c.ndim!=2 or c.shape[0]!=c.shape[1] or c.dtype.kind not in 'ui' or np.any(c<0):raise ValueError('integer square confusion matrix required')
    rows=[]
    for i in range(len(c)):
        tp=int(c[i,i]);gt=int(c[i].sum());pr=int(c[:,i].sum());union=gt+pr-tp
        rows.append(dict(class_id=i,true_pixels=gt,predicted_pixels=pr,intersection=tp,
            dice=None if gt+pr==0 else 2*tp/(gt+pr),iou=None if union==0 else tp/union))
    result=dict(per_class=rows,confusion=c.tolist(),evaluated_pixels=int(c.sum()),
        absent_class_policy='both-empty class is null, excluded from macro; report excluded-class counts')
    for metric in ('dice','iou'):
        values=[r[metric] for r in rows[1:] if r[metric] is not None]
        result['foreground_macro_'+metric]=sum(values)/len(values) if values else None
        result['foreground_classes_in_macro']=len(values)
    return result


def array_difference(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if a.shape!=b.shape:raise ValueError('shape mismatch')
    if a.dtype!=b.dtype:raise ValueError('dtype mismatch; floating/quantized heads cannot be byte-compared')
    if a.dtype.kind not in 'iu':raise ValueError('integer outputs only for exact comparison')
    different=a!=b
    return dict(elements=int(a.size),mismatched_elements=int(different.sum()),
        exact=bool(not different.any()),max_abs_error=int(np.abs(a.astype(np.int64)-b.astype(np.int64)).max(initial=0)))
