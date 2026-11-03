"""Image/model-guided seeded minimum cut, not a prompt-trained neural network.

Only image features, frozen prediction probabilities and user prompts enter this
module. No GT, training data or rectangle-filled segmentation fallback is used.
"""
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow, breadth_first_order
from .contracts import InputError

GRID_SIDE = 256


def resize_float(array, shape):
    return np.asarray(Image.fromarray(np.asarray(array, np.float32)).resize(
        (shape[1],shape[0]), Image.Resampling.BILINEAR), dtype=np.float64)


def seeded_cut(image, probability, points, box=None):
    """Return a native-size binary mask and truthful refinement metadata.

    points are (x,y,foreground) in the crop coordinate system. A box is a hard
    outside-background constraint, never a foreground-filled output rectangle.
    """
    probability = np.asarray(probability, dtype=np.float64)
    h,w = probability.shape
    if not np.isfinite(probability).all() or np.any((probability < 0)|(probability > 1)):
        raise InputError('模型概率必须有限且位于0..1')
    scale = min(1., GRID_SIDE/max(h,w))
    gh,gw = max(2,round(h*scale)),max(2,round(w*scale))
    q = np.clip(resize_float(probability,(gh,gw)), 1e-6,1-1e-6)
    rgb = np.asarray(image, dtype=np.float64)/255.
    if rgb.ndim == 2:
        rgb = rgb[...,None]
    features = np.stack([resize_float(rgb[...,i],(gh,gw)) for i in range(rgb.shape[-1])], axis=-1)
    fg, bg = np.zeros((gh,gw),bool), np.zeros((gh,gw),bool)
    # One grid pixel is a seed. Conflicts after downsampling are rejected rather
    # than silently overwritten; a smaller ROI provides finer control.
    for x,y,foreground in points:
        gx,gy = min(gw-1,int(x*gw/w)),min(gh-1,int(y*gh/h))
        if (bg if foreground else fg)[gy,gx]:
            raise InputError('前景/背景点在计算网格中重合，请缩小 ROI 或移动提示点')
        (fg if foreground else bg)[gy,gx] = True
    box_mask = np.ones((gh,gw),bool)
    if box is not None:
        x0,y0,x1,y1 = box
        # Centre-based membership respects a half-open rectangle.
        xs,ys = (np.arange(gw)+.5)*w/gw,(np.arange(gh)+.5)*h/gh
        box_mask = (ys[:,None]>=y0)&(ys[:,None]<y1)&(xs[None,:]>=x0)&(xs[None,:]<x1)
        if not box_mask.any():
            raise InputError('提示框在计算网格中过小，请缩小 ROI')
        if np.any(fg & ~box_mask):
            raise InputError('前景点距框边过近，请扩大提示框或缩小 ROI')
        bg |= ~box_mask
    if np.any(fg & bg):
        raise InputError('前景和背景约束冲突')
    # Preserve a strong model prior; sparse colour samples must not replace an
    # already confident prediction over the entire image. Hard clicked seeds
    # still override the model, and appearance resolves uncertain regions.
    log_odds = np.clip(np.log(q/(1-q)), -3.,3.)
    fg_samples = features[ndimage.binary_dilation(fg,iterations=1)&~bg] if fg.any() else features[(q>.75)&box_mask]
    bg_samples = features[ndimage.binary_dilation(bg,iterations=1)&~fg] if bg.any() else features[q<.2]
    appearance_used = len(fg_samples)>=2 and len(bg_samples)>=2
    if appearance_used:
        def likelihood(samples):
            mean = samples.mean(axis=0)
            variance = np.maximum(samples.var(axis=0), .012**2)
            return -.5*np.sum((features-mean)**2/variance + np.log(variance),axis=-1)
        appearance = np.clip(likelihood(fg_samples)-likelihood(bg_samples), -6.,6.)
        log_odds += .2*appearance
    q = 1/(1+np.exp(-log_odds))
    cost_fg = np.rint(-np.log(np.clip(q,1e-6,1))*100).astype(np.int64)
    cost_bg = np.rint(-np.log(np.clip(1-q,1e-6,1))*100).astype(np.int64)
    # A 100000 seed penalty exceeds the sum of all incident pairwise/unary
    # costs at one node, so violating a seed cannot minimize the energy.
    cost_bg[fg] = 100000; cost_fg[fg] = 0
    cost_fg[bg] = 100000; cost_bg[bg] = 0
    ids = np.arange(gh*gw).reshape(gh,gw)
    u = np.r_[ids[:,:-1].ravel(),ids[:-1,:].ravel()]
    v = np.r_[ids[:,1:].ravel(),ids[1:,:].ravel()]
    flat = features.reshape(-1,features.shape[-1])
    difference = np.sum((flat[u]-flat[v])**2,axis=1)
    beta = 1/max(2*float(difference.mean()),1e-8)
    pairwise = np.maximum(1,np.rint(200*np.exp(-beta*difference))).astype(np.int64)
    n = gh*gw; source,sink = n,n+1
    nodes = ids.ravel()
    # Reachable/source-side nodes are foreground: S->p costs background,
    # p->T costs foreground. Two directed links express Potts smoothness.
    graph = coo_matrix((np.r_[cost_bg.ravel(),cost_fg.ravel(),pairwise,pairwise],
        (np.r_[np.full(n,source),nodes,u,v],np.r_[nodes,np.full(n,sink),v,u])),shape=(n+2,n+2)).tocsr()
    flow = maximum_flow(graph,source,sink,method='dinic')
    residual = graph-flow.flow
    residual.data = (residual.data>0).astype(np.int8)
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual,source,directed=True,return_predecessors=False)
    mask = np.zeros(n+2,bool); mask[reachable] = True
    small = mask[:n].reshape(gh,gw)
    if np.any(~small[fg]) or np.any(small[bg]):
        raise RuntimeError('图割未满足硬约束，拒绝返回结果')
    native = np.asarray(Image.fromarray(small.astype(np.uint8)).resize((w,h),Image.Resampling.NEAREST),bool)
    if box is not None:
        x0,y0,x1,y1 = box
        allowed = np.zeros((h,w),bool); allowed[y0:y1,x0:x1] = True
        native &= allowed
    # Preserve exact clicked pixels after grid-to-native reconstruction.
    for x,y,foreground in points:
        native[y,x] = foreground
    return native, {'algorithm':'seeded_min_cut','grid_size':[gw,gh],
        'appearance_used':appearance_used,'flow_value':int(flow.flow_value),
        'appearance_weight':.2,'model_log_odds_limit':3.,
        'foreground_points':sum(bool(p[2]) for p in points),
        'background_points':sum(not p[2] for p in points),
        'box_used_as_background_constraint':box is not None,
        'network_prompt_conditioned':False}
