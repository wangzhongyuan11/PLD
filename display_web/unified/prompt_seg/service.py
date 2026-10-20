"""Native-coordinate ROI, thresholding and model-assisted prompt segmentation."""
import base64
import hashlib
import struct
import time
import uuid
import zlib
import numpy as np
from scipy import ndimage
from .contracts import Request,Rect,InputError
from .refinement import seeded_cut, build_context, image_sha256



def png_data(array):
    # Same lossless Sub + RLE strategy already used by camus.png. Avoid
    # Pillow's adaptive filter search on the Cortex-A9; pixels are unchanged.
    a=np.asarray(array,dtype=np.uint8)
    if not (a.ndim==2 or a.ndim==3 and a.shape[2]==3):
        raise ValueError('PNG requires gray or RGB pixels')
    h,w=a.shape[:2];channels=1 if a.ndim==2 else 3
    raw=np.ascontiguousarray(a).reshape(h,w*channels)
    rows=np.empty((h,w*channels+1),np.uint8);rows[:,0]=1
    rows[:,1:1+channels]=raw[:,:channels]
    np.subtract(raw[:,channels:],raw[:,:-channels],out=rows[:,1+channels:])
    def chunk(kind,data):
        return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(data,zlib.crc32(kind))&0xffffffff)
    compressor=zlib.compressobj(level=1,strategy=zlib.Z_RLE)
    compressed=compressor.compress(rows.tobytes())+compressor.flush()
    header=struct.pack('!IIBBBBB',w,h,8,0 if channels==1 else 2,0,0,0)
    out=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',header)+chunk(b'IDAT',compressed)+chunk(b'IEND',b'')
    return 'data:image/png;base64,'+base64.b64encode(out).decode('ascii')


def _copy_statistics(statistics):
    copy=dict(statistics)
    copy['per_class_pixels']=dict(statistics['per_class_pixels'])
    return copy


def pixel_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(array,np.uint8)).tobytes()).hexdigest()


def png_sha256(data_url):
    """SHA-256 over the decoded PNG bytes (not the data-URL text)."""
    return hashlib.sha256(base64.b64decode(data_url.split(',',1)[1])).hexdigest()


class SegmentationService:
    """One baseline slot per (image, ROI, model, target, threshold, scope).

    The baseline is the fixed post-processed, pre-graph-cut result: prompts never
    modify it and a rectangle is never filled into it. Point/box-only changes
    reuse the cached baseline instead of re-encoding initial labels/mask/overlay.
    """
    def __init__(self,predictor=None):
        if predictor is None: raise ValueError("PL predictor required")
        self.predictor = predictor
        self._baseline = None
        self._cut = None

    def clear(self):
        self._baseline = None
        self._cut = None

    def _key(self,model,image,req,scope,identity):
        roi = None if req.roi is None else (req.roi.x0,req.roi.y0,req.roi.x1,req.roi.y1)
        # The probability version matters: a rebuilt prediction (new source_job or
        # spec) must never reuse a baseline computed from older probabilities.
        version = None if identity is None else (
            identity.get('spec_sha256'),identity.get('source_job'))
        return (model,pixel_sha256(image),roi,req.target_class,req.threshold,req.variant,
                req.backend,None if scope is None else tuple(scope),version)

    def _postprocess(self,model,labels):
        if model=='camus':
            from camus.postprocess import apply_island5_hole1
            labels = apply_island5_hole1(labels).labels.astype(np.uint8)
        return labels

    def _compose(self,image,req,native,labels,nclasses):
        """Statistics and display arrays only; PNG encoding is timed separately."""
        selected = native==req.target_class
        boundary = selected & ~ndimage.binary_erosion(selected)
        overlay = image.copy().astype(float)
        colors = ((0,0,0),(25,174,114),(232,150,43),(98,107,213))
        # Display only the selected structure; labels.png still contains all
        # classes. The drawn rectangle itself is never part of this overlay.
        color = np.asarray(colors[req.target_class],float)
        overlay[selected] = .6*overlay[selected]+.4*color
        outline = ndimage.binary_dilation(boundary,iterations=1)
        overlay[outline] = color
        roi = req.roi
        domain_pixels = (roi.y1-roi.y0)*(roi.x1-roi.x0) if roi else native.size
        statistics = {'target_pixels':int(selected.sum()),'processed_pixels':int(domain_pixels),
            'area_ratio':int(selected.sum())/domain_pixels,
            'per_class_pixels':{str(i):int((labels==i).sum()) for i in range(nclasses)}}
        overlay_png = np.rint(overlay).clip(0,255).astype(np.uint8)
        return overlay_png, selected.astype(np.uint8)*255, statistics

    def _encode(self,native,overlay_rgb,mask):
        return {'labels_png':png_data(native),'target_mask_png':png_data(mask),
            'overlay_png':png_data(overlay_rgb)}

    def process(self,model,image,parameters,scope=None):
        began = time.monotonic()
        image = np.asarray(image)
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:
            raise InputError('输入图像必须为 uint8 RGB')
        h,w = image.shape[:2]
        req = Request.parse(model,parameters,w,h)
        roi = req.roi or Rect(0,0,w,h)
        crop = image[roi.y0:roi.y1,roi.x0:roi.x1]
        prepared = time.monotonic()
        probabilities,identity = self.predictor.predict(crop,model,req.variant,req.backend)
        predicted = time.monotonic()
        nclasses = 2 if model=='skin' else 4
        if probabilities.shape!=(nclasses,*crop.shape[:2]):
            raise RuntimeError('模型输出尺寸或类别数不匹配')
        target = probabilities[req.target_class]
        other = probabilities.copy(); other[req.target_class] = -1
        alternatives = other.argmax(axis=0).astype(np.uint8)
        if req.threshold is None:
            labels = probabilities.argmax(axis=0).astype(np.uint8)
            # Pairwise target-vs-strongest-other odds match multiclass argmax.
            prior = target/np.maximum(target+other.max(axis=0),1e-8)
        else:
            labels = np.where(target>=req.threshold,req.target_class,alternatives).astype(np.uint8)
            epsilon = 1e-6
            p = np.clip(target,epsilon,1-epsilon)
            odds = np.log(p/(1-p))-np.log(req.threshold/(1-req.threshold))
            prior = 1/(1+np.exp(-np.clip(odds,-30,30)))
        labels = self._postprocess(model,labels)
        decided = time.monotonic()

        key = self._key(model,image,req,scope,identity)
        cache = self._baseline
        baseline_hit = bool(cache is not None and cache['key']==key
                            and cache['crop_shape']==crop.shape[:2])
        if baseline_hit:
            baseline_native = cache['native'].copy()
            baseline_outputs = dict(cache['outputs'])
            baseline_statistics = _copy_statistics(cache['statistics'])
            baseline_hashes = dict(cache['hashes'])
            baseline_compose_ms = baseline_png_ms = 0.0
        else:
            baseline_native = np.zeros((h,w),np.uint8)
            baseline_native[roi.y0:roi.y1,roi.x0:roi.x1] = labels
            baseline_compose_started = time.monotonic()
            baseline_overlay,baseline_mask,baseline_statistics = self._compose(
                image,req,baseline_native,labels,nclasses)
            baseline_composed = time.monotonic()
            baseline_outputs = self._encode(baseline_native,baseline_overlay,baseline_mask)
            baseline_encoded = time.monotonic()
            baseline_compose_ms = (baseline_composed-baseline_compose_started)*1000
            baseline_png_ms = (baseline_encoded-baseline_composed)*1000
            baseline_statistics['changed_pixels_by_refinement'] = 0
            baseline_hashes = {'labels_png':png_sha256(baseline_outputs['labels_png']),
                'target_mask_png':png_sha256(baseline_outputs['target_mask_png']),
                'labels':pixel_sha256(baseline_native)}
            self._baseline = {'key':key,'crop_shape':crop.shape[:2],'native':baseline_native.copy(),
                'outputs':dict(baseline_outputs),'statistics':_copy_statistics(baseline_statistics),
                'hashes':dict(baseline_hashes)}
        baseline_labels = labels.copy()
        cached = time.monotonic()

        refinement = None
        graph_cut_ms = final_compose_ms = final_png_ms = cut_context_ms = 0.0
        if req.interactive:
            points = [(p.x-roi.x0,p.y-roi.y0,p.foreground) for p in req.points]
            box = None if req.box is None else (req.box.x0-roi.x0,req.box.y0-roi.y0,req.box.x1-roi.x0,req.box.y1-roi.y0)
            cut_key = (image_sha256(crop),crop.shape[:2])
            cut_context = self._cut[1] if self._cut is not None and self._cut[0]==cut_key else None
            context_hit = cut_context is not None
            if cut_context is None:
                context_started = time.monotonic()
                cut_context = build_context(crop,crop.shape[:2])
                cut_context_ms = (time.monotonic()-context_started)*1000
                self._cut = (cut_key,cut_context)
            else:
                cut_context_ms = 0.0
            cut_started = time.monotonic()
            mask,refinement = seeded_cut(crop,prior,points,box,context=cut_context)
            refinement['context_cache_hit']=context_hit
            graph_cut_ms = (time.monotonic()-cut_started)*1000
            # Other classes retain their original decision unless the chosen
            # target is added/removed; removed pixels select a non-target class.
            labels[(labels==req.target_class)&~mask] = alternatives[(labels==req.target_class)&~mask]
            labels[mask] = req.target_class
            final_native = np.zeros((h,w),np.uint8)
            final_native[roi.y0:roi.y1,roi.x0:roi.x1] = labels
            compose_started = time.monotonic()
            final_overlay,final_mask,statistics = self._compose(
                image,req,final_native,labels,nclasses)
            final_composed = time.monotonic()
            outputs = self._encode(final_native,final_overlay,final_mask)
            final_encoded = time.monotonic()
            final_compose_ms = (final_composed-compose_started)*1000
            final_png_ms = (final_encoded-final_composed)*1000
            statistics['changed_pixels_by_refinement'] = int((labels!=baseline_labels).sum())
            native = final_native
            selected = final_mask.astype(bool)
        else:
            # Nothing was clicked: the cached baseline already is the final result.
            outputs = dict(baseline_outputs)
            statistics = _copy_statistics(baseline_statistics)
            statistics['changed_pixels_by_refinement'] = 0
            native = baseline_native
            selected = baseline_native==req.target_class
        refined = time.monotonic()

        warnings = []
        if not selected.any():
            warnings.append('当前条件下未提取到目标，可调整提示或阈值')
        if req.roi:
            warnings.append('仅处理 ROI；区域外未推理并输出背景，面积占比以 ROI 为分母')
        if req.interactive:
            warnings.append('提示由模型辅助图割处理；原网络未经过提示式训练，交互结果的精度需另行验证')
        result = {'request_id':uuid.uuid4().hex,'model':model,'identity':identity,
            'image':{'width':w,'height':h,'sha256':pixel_sha256(image)},
            'parameters':{'variant':req.variant,'backend':req.backend,'target_class':req.target_class,
                'threshold':req.threshold,'roi':None if req.roi is None else req.roi.as_dict(),
                'box':None if req.box is None else req.box.as_dict(),
                'points':[{'x':p.x,'y':p.y,'label':'foreground' if p.foreground else 'background'} for p in req.points]},
            'operation':'prompt_refinement' if req.interactive else ('roi_model_inference' if req.roi else 'model_inference'),
            'refinement':refinement,'network_prompt_conditioned':False,
            'postprocess':'island5_hole1_before_refinement' if model=='camus' else 'none',
            'outputs':outputs,'statistics':statistics,
            'baseline':{'outputs':dict(baseline_outputs),'statistics':_copy_statistics(baseline_statistics),
                'hashes':baseline_hashes,'cache_hit':baseline_hit,
                'note':'固定后处理、图割之前的无提示结果；提示不会修改它'},
            'reused_baseline_as_final':not req.interactive,
            'label_hashes':{'baseline':baseline_hashes['labels'],
                'final':pixel_sha256(native)},
            'timings_ms':dict(input_validation=(prepared-began)*1000,prediction=(predicted-prepared)*1000,
                decision_postprocess=(decided-predicted)*1000,
                baseline_compose=baseline_compose_ms,baseline_png=baseline_png_ms,
                cut_context=cut_context_ms,
                prompt_refinement=graph_cut_ms,render_statistics=final_compose_ms,result_png=final_png_ms),
            'timing_note':'baseline_compose/baseline_png only on cache miss; cut_context is the one-off colour/edge/pairwise build for a crop; prompt_refinement is the whole PS graph-cut call (unary costs, hard constraints, CSR assembly, maximum_flow and native mask rebuild) and excludes the reusable context build; render_statistics and result_png are the final composition and PNG encoding',
            'reference_available':False,'accuracy_metrics':None,'warnings':warnings,
            'elapsed_ms':round((time.monotonic()-began)*1000,2)}
        return result
